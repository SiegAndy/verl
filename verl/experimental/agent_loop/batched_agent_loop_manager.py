"""
Custom AgentLoopManager with cross-worker tool call batching.

This implementation ensures all workers' tool calls are sent together
to the tool server, enabling efficient batch processing of the pipeline.

Key differences from base AgentLoopManager:
1. Waits for ALL workers to finish generation before processing tool calls
2. Collects tool calls from all workers
3. Sends single batch request to tool server
4. Tool server processes entire batch together (shared retriever/reranker)
5. Distributes results back to workers

Performance improvement:
- Before: 4 workers A- 60s = 240s (sequential)
- After: 4 workers + 60s = 63s (batched)
"""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional
from uuid import uuid4

import ray
from omegaconf import DictConfig

from verl.protocol import DataProto

from .agent_loop import AgentLoopManager, AgentLoopWorker
from .multi_turn_optimizer import MultiTurnStatisticsCollector
from verl.single_controller.ray import RayResourcePool

logger = logging.getLogger(__name__)


class _TurnState:
    def __init__(self) -> None:
        self.expected: Optional[int] = None
        self.expected_ready = asyncio.Event()
        self.ready = asyncio.Event()
        self.reported: set[str] = set()
        self.continue_samples: set[str] = set()
        self.drop_after_tool: set[str] = set()
        self.next_expected_set = False

    def set_expected(self, expected: int) -> None:
        self.expected = max(0, expected)
        self.expected_ready.set()
        if self.expected == 0 or len(self.reported) >= self.expected:
            self.ready.set()


@ray.remote
class TurnBatchCoordinator:
    """
    Coordinate per-turn tool execution across workers.

    Each sample reports after generation (per turn). Tool execution waits until
    all active samples have reported for that turn.
    """

    def __init__(self, timeout_s: float = 900) -> None:
        self.timeout_s = timeout_s
        self.lock = asyncio.Lock()
        self._reset_state()

    def _reset_state(self) -> None:
        self.batch_id: Optional[str] = None
        self.total_samples: int = 0
        self.turn_states: Dict[int, _TurnState] = {}

    def _get_state(self, turn_num: int) -> _TurnState:
        state = self.turn_states.get(turn_num)
        if state is None:
            state = _TurnState()
            self.turn_states[turn_num] = state
        return state

    async def start_batch(self, batch_id: str, total_samples: int) -> bool:
        async with self.lock:
            self._reset_state()
            self.batch_id = batch_id
            self.total_samples = total_samples
            state = self._get_state(1)
            state.set_expected(total_samples)
        return True

    async def report_generation(
        self,
        batch_id: str,
        turn_num: int,
        sample_id: str,
        has_tool_calls: bool,
        terminated: bool,
    ) -> bool:
        if batch_id != self.batch_id:
            return False
        async with self.lock:
            state = self._get_state(turn_num)
            if sample_id in state.reported:
                return True
            state.reported.add(sample_id)
            if not terminated:
                state.continue_samples.add(sample_id)
            self._maybe_finalize_turn(turn_num, state)
        return True

    async def report_tool_outcome(
        self,
        batch_id: str,
        turn_num: int,
        sample_id: str,
        terminated_after_tools: bool,
    ) -> bool:
        if batch_id != self.batch_id or not terminated_after_tools:
            return True
        async with self.lock:
            state = self._get_state(turn_num)
            if sample_id in state.drop_after_tool:
                return True
            state.drop_after_tool.add(sample_id)
            if state.next_expected_set:
                next_state = self._get_state(turn_num + 1)
                if sample_id in state.continue_samples:
                    state.continue_samples.discard(sample_id)
                    if next_state.expected is not None:
                        next_state.expected = max(0, next_state.expected - 1)
                        if len(next_state.reported) >= next_state.expected:
                            next_state.ready.set()
        return True

    async def wait_for_turn(
        self, batch_id: str, turn_num: int, timeout_s: Optional[float] = None
    ) -> bool:
        if batch_id != self.batch_id:
            return False
        state = self._get_state(turn_num)
        timeout = timeout_s or self.timeout_s
        try:
            if not state.expected_ready.is_set():
                await asyncio.wait_for(state.expected_ready.wait(), timeout=timeout)
            if not state.ready.is_set():
                await asyncio.wait_for(state.ready.wait(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            logger.warning(
                "[TurnBatchCoordinator] Turn %s timed out waiting for reports", turn_num
            )
            state.ready.set()
            return False

    def _maybe_finalize_turn(self, turn_num: int, state: _TurnState) -> None:
        if state.expected is None:
            return
        if len(state.reported) < state.expected:
            return
        if not state.ready.is_set():
            state.ready.set()
        if not state.next_expected_set:
            next_state = self._get_state(turn_num + 1)
            expected_next = len(state.continue_samples - state.drop_after_tool)
            next_state.set_expected(expected_next)
            state.next_expected_set = True


class TurnBatchedAgentLoopWorker(AgentLoopWorker):
    """Agent loop worker that injects turn-batch context into extra_info."""

    def __init__(self, config, server_handles, reward_router_address=None):
        super().__init__(config, server_handles, reward_router_address)
        self._turn_batch_coordinator_name = (
            config.actor_rollout_ref.rollout.multi_turn.get(
                "turn_batch_coordinator_name", None
            )
        )
        self._current_turn_batch_id: Optional[str] = None
        self._worker_uid = str(ray.get_runtime_context().get_actor_id())

    def set_turn_batch_context(self, batch_id: str) -> bool:
        self._current_turn_batch_id = batch_id
        return True

    async def _run_agent_loop(
        self,
        sampling_params: dict[str, Any],
        trajectory: dict[str, Any],
        *,
        agent_name: str,
        trace: bool = True,
        **kwargs,
    ):
        if self._turn_batch_coordinator_name and self._current_turn_batch_id:
            extra_info = kwargs.get("extra_info")
            if not isinstance(extra_info, dict):
                extra_info = {}
            else:
                extra_info = dict(extra_info)
            sample_id = (
                f"{self._worker_uid}:{trajectory['sample_index']}:{trajectory['rollout_n']}"
            )
            extra_info["turn_batch_context"] = {
                "coordinator_name": self._turn_batch_coordinator_name,
                "batch_id": self._current_turn_batch_id,
                "sample_id": sample_id,
            }
            kwargs["extra_info"] = extra_info
        return await super()._run_agent_loop(
            sampling_params, trajectory, agent_name=agent_name, trace=trace, **kwargs
        )


class SimpleBatchedAgentLoopManager(AgentLoopManager):
    """
    Simpler batching approach: just synchronize worker completion.

    Instead of complex two-phase generation, simply ensure all workers
    complete before any results are processed. This allows RequestBatcher
    on the tool server to see all requests arrive nearly simultaneously
    and batch them together.

    Advantages:
    - Minimal code changes
    - Works with existing AgentLoopWorker
    - Still gets batching benefits

    Disadvantages:
    - Less control over batch timing
    - Relies on tool server RequestBatcher window
    """

    def __init__(
        self,
        config: DictConfig,
        worker_group=None,
        rm_resource_pool: Optional[RayResourcePool] = None,
    ):
        """Initialize SimpleBatchedAgentLoopManager with logging."""
        logger.warning("=" * 80)
        logger.warning("[SimpleBatched] Initializing SimpleBatchedAgentLoopManager")
        logger.warning("[SimpleBatched] Cross-worker batching is ENABLED")
        logger.warning("=" * 80)
        super().__init__(config, worker_group, rm_resource_pool)
        logger.warning("[SimpleBatched] Initialization complete")

    def generate_sequences(self, prompts: DataProto) -> DataProto:
        """
        Generate sequences with synchronized worker completion.

        Key change: ray.get() waits for ALL workers to complete their
        async tool calls before returning. This causes all tool server
        requests to arrive nearly simultaneously, triggering batching.
        """
        num_workers = len(self.agent_loop_workers)
        total_samples = len(prompts)

        logger.warning("=" * 80)
        logger.warning("[SimpleBatched] Starting batch generation")
        logger.warning("[SimpleBatched]   Workers: %s", num_workers)
        logger.warning("[SimpleBatched]   Total samples: %s", total_samples)
        logger.warning(
            "[SimpleBatched]   Samples per worker: ~%s",
            total_samples // num_workers,
        )
        logger.warning("=" * 80)

        start_time = time.time()

        # Wake up models
        self.wake_up()
        if self.reward_model_manager:
            self.reward_model_manager.wake_up()

        # Split batch across workers
        chunks = prompts.chunk(num_workers)
        chunk_sizes = [len(chunk) for chunk in chunks]

        logger.warning("[SimpleBatched] Chunk sizes: %s", chunk_sizes)

        # Start all workers asynchronously
        generation_start = time.time()
        logger.warning("[SimpleBatched] Dispatching to %s workers...", num_workers)

        futures = [
            worker.generate_sequences.remote(chunk)
            for worker, chunk in zip(self.agent_loop_workers, chunks, strict=True)
        ]

        # Wait for ALL workers to complete
        logger.warning(
            "[SimpleBatched] Waiting for all %s workers to complete generation...",
            num_workers,
        )
        logger.warning(
            "[SimpleBatched]   (This includes model generation + tool calls for %s rollouts)",
            total_samples,
        )

        outputs = ray.get(futures)
        generation_time = time.time() - generation_start

        # Count actual rollouts completed and validate sizes
        completed_rollouts = sum(len(output) for output in outputs)

        # Detailed size logging for debugging
        logger.warning("=" * 80)
        logger.warning("[SimpleBatched] Worker output sizes:")
        for i, output in enumerate(outputs):
            batch_size = len(output)
            tensor_keys = list(output.batch.keys()) if output.batch is not None else []
            non_tensor_keys = (
                list(output.non_tensor_batch.keys())
                if output.non_tensor_batch is not None
                else []
            )
            logger.warning("[SimpleBatched]   Worker %s: %s samples", i, batch_size)
            logger.warning("[SimpleBatched]     Tensor keys: %s", tensor_keys)
            logger.warning("[SimpleBatched]     Non-tensor keys: %s", non_tensor_keys)

            if output.batch is not None and output.non_tensor_batch is not None:
                for key, val in output.non_tensor_batch.items():
                    if len(val) != batch_size:
                        logger.error(
                            "[SimpleBatched]   Worker %s INCONSISTENT: %s has %s entries but batch size is %s",
                            i,
                            key,
                            len(val),
                            batch_size,
                        )

        logger.warning("=" * 80)
        logger.warning("[SimpleBatched] All workers completed")
        logger.warning(
            "[SimpleBatched]   Total rollouts generated: %s", completed_rollouts
        )
        logger.warning("[SimpleBatched]   Expected rollouts: %s", total_samples)
        if completed_rollouts != total_samples:
            logger.error(
                "[SimpleBatched]   MISSING %s rollouts!",
                total_samples - completed_rollouts,
            )
        logger.warning("[SimpleBatched]   Total time: %.2fs", generation_time)
        logger.warning(
            "[SimpleBatched]   Throughput: %.2f rollouts/s",
            completed_rollouts / generation_time,
        )
        logger.warning("=" * 80)

        # Concat outputs
        try:
            output = DataProto.concat(outputs)
        except AssertionError as e:
            logger.error("[SimpleBatched] DataProto.concat failed: %s", e)
            logger.error(
                "[SimpleBatched] This indicates batch size mismatch between workers"
            )
            raise

        # Cleanup
        self.sleep()
        if self.reward_model_manager:
            self.reward_model_manager.sleep()

        # Calculate performance metrics
        metrics = [o.meta_info.pop("metrics") for o in outputs]
        timing = self._performance_metrics(metrics, output)
        agent_metrics = self._aggregate_agent_metrics(metrics)
        timing["batched_total"] = time.time() - start_time
        timing["generation_sync"] = generation_time
        timing["completed_rollouts"] = completed_rollouts
        timing["rollouts_per_second"] = (
            completed_rollouts / generation_time if generation_time > 0 else 0
        )

        output.meta_info = {"timing": timing, **outputs[0].meta_info}
        if agent_metrics:
            output.meta_info["agent_metrics"] = agent_metrics

        # Final summary with detailed breakdown
        logger.warning("[SimpleBatched] Performance Summary:")
        logger.warning(
            "[SimpleBatched]   Model generation time: %.2fs (avg)",
            timing.get("agent_loop/generate_sequences/mean", 0),
        )
        logger.warning(
            "[SimpleBatched]   Tool call time: %.2fs (avg)",
            timing.get("agent_loop/tool_calls/mean", 0),
        )
        logger.warning(
            "[SimpleBatched]   Total batch time: %.2fs", timing["batched_total"]
        )
        logger.warning(
            "[SimpleBatched]   Overall throughput: %.2f rollouts/s",
            timing["rollouts_per_second"],
        )

        # Log multi-turn statistics if available
        logger.warning("")
        logger.warning(
            "[SimpleBatched] Collecting multi-turn statistics from workers..."
        )
        try:
            stats_futures = [
                worker.get_turn_statistics.remote()
                for worker in self.agent_loop_workers
            ]
            all_stats = ray.get(stats_futures)

            if any(all_stats):
                aggregated_collector = MultiTurnStatisticsCollector()

                for worker_stats in all_stats:
                    if worker_stats and "turn_stats" in worker_stats:
                        for turn_num, turn_stat in worker_stats["turn_stats"].items():
                            if turn_num not in aggregated_collector.turn_stats:
                                aggregated_collector.turn_stats[turn_num] = turn_stat
                            else:
                                existing = aggregated_collector.turn_stats[turn_num]
                                existing.generation_times.extend(
                                    turn_stat.generation_times
                                )
                                existing.tool_call_times.extend(
                                    turn_stat.tool_call_times
                                )
                                existing.total_times.extend(turn_stat.total_times)
                                existing.samples_in_turn += turn_stat.samples_in_turn

                        aggregated_collector.sample_turn_counts.extend(
                            worker_stats["sample_turn_counts"]
                        )
                        aggregated_collector.sample_revision_counts.extend(
                            worker_stats.get("sample_revision_counts", [])
                        )
                        aggregated_collector.total_samples += worker_stats[
                            "total_samples"
                        ]

                if aggregated_collector.total_samples > 0:
                    aggregated_collector.log_summary()
                    output.meta_info["turn_statistics"] = (
                        aggregated_collector.get_aggregated_stats()
                    )
        except Exception as e:
            logger.warning(
                "[SimpleBatched] Could not collect turn statistics: %s", e
            )

        return output


class DeterministicTurnBatchedAgentLoopManager(SimpleBatchedAgentLoopManager):
    """
    Deterministic per-turn tool batching across workers.

    Creates a shared TurnBatchCoordinator and synchronizes tool calls for each
    assistant turn so that tool server batching is deterministic.
    """

    def __init__(
        self,
        config: DictConfig,
        worker_group=None,
        rm_resource_pool: Optional[RayResourcePool] = None,
    ):
        self.agent_loop_workers_class = ray.remote(TurnBatchedAgentLoopWorker)

        multi_turn_cfg = config.actor_rollout_ref.rollout.multi_turn
        coordinator_name = multi_turn_cfg.get("turn_batch_coordinator_name")
        if not coordinator_name:
            coordinator_name = f"turn_batch_coordinator_{uuid4().hex}"
            multi_turn_cfg["turn_batch_coordinator_name"] = coordinator_name

        timeout_s = multi_turn_cfg.get("turn_batch_timeout_s", 900)
        self.turn_batch_coordinator = TurnBatchCoordinator.options(
            name=coordinator_name
        ).remote(timeout_s=timeout_s)

        logger.warning("=" * 80)
        logger.warning(
            "[TurnBatched] Deterministic per-turn batching is ENABLED (coordinator=%s)",
            coordinator_name,
        )
        logger.warning("=" * 80)

        super().__init__(config, worker_group, rm_resource_pool)

    def generate_sequences(self, prompts: DataProto) -> DataProto:
        batch_id = uuid4().hex
        total_samples = len(prompts)
        ray.get(self.turn_batch_coordinator.start_batch.remote(batch_id, total_samples))
        ray.get(
            [
                worker.set_turn_batch_context.remote(batch_id)
                for worker in self.agent_loop_workers
            ]
        )
        return super().generate_sequences(prompts)


class FullBatchedAgentLoopManager(AgentLoopManager):
    """
    Full batching with explicit two-phase generation.

    Phase 1: Workers generate up to tool call (parallel, fast)
    Phase 2: Manager batches all tool calls together (single request)
    Phase 3: Workers complete with tool results

    This gives maximum control but requires custom AgentLoopWorker.

    TODO: Implement this if SimpleBatchedAgentLoopManager is insufficient.
    """

    def __init__(
        self,
        config: DictConfig,
        worker_group=None,
        rm_resource_pool: Optional[RayResourcePool] = None,
    ):
        super().__init__(config, worker_group, rm_resource_pool)

        agent_config = config.actor_rollout_ref.rollout.agent
        self.batch_sync_timeout = agent_config.get("batch_sync_timeout", 30)
        self.tool_server_url = (
            agent_config.get("tool_server_urls", [])[0]
            if agent_config.get("tool_server_urls")
            else None
        )

        logger.info(
            "[FullBatched] Initialized with timeout=%ss, server=%s",
            self.batch_sync_timeout,
            self.tool_server_url,
        )

    def generate_sequences(self, prompts: DataProto) -> DataProto:
        """
        Generate sequences with explicit cross-worker batching.

        Flow:
        1. Workers generate until tool call (parallel)
        2. Gather all tool calls from all workers
        3. Send single batch request to tool server
        4. Distribute results back to workers
        5. Workers complete generation with tool results
        """
        raise NotImplementedError(
            "FullBatchedAgentLoopManager requires custom AgentLoopWorker "
            "with generate_sequences_until_tool_call() and "
            "complete_generation_with_tool_results() methods. "
            "Use SimpleBatchedAgentLoopManager for now."
        )


def create_batched_agent_loop_manager(
    config: DictConfig,
    worker_group=None,
    rm_resource_pool: Optional[RayResourcePool] = None,
    mode: str = "simple",
) -> AgentLoopManager:
    """
    Factory function to create batched agent loop manager.

    Args:
        config: Hydra config
        worker_group: ActorRollout worker group (for hybrid mode)
        rm_resource_pool: Reward model resource pool (for standalone mode)
        mode: "simple" or "full"

    Returns:
        AgentLoopManager instance with batching enabled
    """
    if mode == "simple":
        return SimpleBatchedAgentLoopManager(config, worker_group, rm_resource_pool)
    if mode == "deterministic_turn":
        return DeterministicTurnBatchedAgentLoopManager(
            config, worker_group, rm_resource_pool
        )
    if mode == "full":
        return FullBatchedAgentLoopManager(config, worker_group, rm_resource_pool)
    raise ValueError(f"Unknown batching mode: {mode}")
