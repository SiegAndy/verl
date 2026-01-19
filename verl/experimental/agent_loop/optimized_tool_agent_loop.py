"""
Optimized ToolAgentLoop with multi-turn batching and statistics collection.

This extends VERL's ToolAgentLoop to add:
1. Comprehensive turn statistics logging
2. Optional turn-level batch synchronization (local or cross-worker)
3. Per-turn timing breakdown
"""

import logging
import os
import time
from typing import Any, Optional

import ray
from transformers import AutoProcessor, AutoTokenizer

from verl.utils.rollout_trace import rollout_trace_op

from .agent_loop import (
    AgentLoopOutput,
    AsyncLLMServerManager,
    DictConfigWrap,
    register,
)
from .multi_turn_optimizer import (
    EnhancedAgentData,
    MultiTurnBatchSynchronizer,
    MultiTurnStatisticsCollector,
)
from .tool_agent_loop import (
    AgentData,
    AgentState,
    ToolAgentLoop,
)

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@register("optimized_tool_agent")
class OptimizedToolAgentLoop(ToolAgentLoop):
    """
    Enhanced ToolAgentLoop with multi-turn optimization and statistics.

    Configuration options (in rollout.multi_turn):
        enable_turn_batching: bool = True
            Enable synchronization of samples at turn boundaries
        log_turn_statistics: bool = True
            Enable detailed turn-by-turn statistics logging
        enable_cross_worker_turn_batching: bool = False
            Enable deterministic per-turn batching across workers
        turn_batch_timeout_s: int = 900
            Timeout for cross-worker turn batching
    """

    def __init__(
        self,
        trainer_config: DictConfigWrap,
        server_manager: AsyncLLMServerManager,
        tokenizer: AutoTokenizer,
        processor: AutoProcessor,
        **kwargs,
    ):
        super().__init__(trainer_config, server_manager, tokenizer, processor, **kwargs)

        config = trainer_config.config
        multi_turn_config = config.actor_rollout_ref.rollout.multi_turn

        # Configuration
        self.enable_turn_batching = multi_turn_config.get("enable_turn_batching", True)
        self.log_turn_statistics = multi_turn_config.get("log_turn_statistics", True)
        self.enable_cross_worker_turn_batching = multi_turn_config.get(
            "enable_cross_worker_turn_batching", False
        )
        self.turn_batch_timeout_s = multi_turn_config.get("turn_batch_timeout_s", 900)
        self.turn_batch_coordinator_name = multi_turn_config.get(
            "turn_batch_coordinator_name"
        )
        self._turn_batch_coordinator = None

        # Statistics collector (shared across all samples in worker)
        if self.log_turn_statistics:
            self.stats_collector = MultiTurnStatisticsCollector()
        else:
            self.stats_collector = None

        # Batch synchronizer (shared across all samples in worker)
        if self.enable_turn_batching and not self.enable_cross_worker_turn_batching:
            self.batch_synchronizer = MultiTurnBatchSynchronizer(
                enable_synchronization=True
            )
        else:
            self.batch_synchronizer = None

        logger.info(
            "OptimizedToolAgentLoop initialized: "
            "turn_batching=%s, turn_statistics=%s, cross_worker_turn_batching=%s",
            "ENABLED" if self.enable_turn_batching else "DISABLED",
            "ENABLED" if self.log_turn_statistics else "DISABLED",
            "ENABLED" if self.enable_cross_worker_turn_batching else "DISABLED",
        )

    def _get_turn_batch_context(
        self, agent_data: AgentData
    ) -> tuple[Optional[str], Optional[str]]:
        context = agent_data.extra_fields.get("turn_batch_context", {})
        if not isinstance(context, dict):
            return None, None
        return context.get("batch_id"), context.get("sample_id")

    def _get_turn_batch_coordinator(self):
        if (
            not self.enable_cross_worker_turn_batching
            or not self.turn_batch_coordinator_name
        ):
            return None
        if self._turn_batch_coordinator is None:
            try:
                self._turn_batch_coordinator = ray.get_actor(
                    self.turn_batch_coordinator_name
                )
            except Exception as exc:
                logger.warning(
                    "Failed to resolve turn batch coordinator '%s': %s",
                    self.turn_batch_coordinator_name,
                    exc,
                )
                self._turn_batch_coordinator = None
        return self._turn_batch_coordinator

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        """
        Run agent loop with enhanced tracking.

        This wraps the base ToolAgentLoop.run() with statistics collection.
        """
        # Get sample index for tracking
        sample_idx = kwargs.get("index", 0)

        # Store sample_idx in kwargs for state handlers
        kwargs["_sample_idx"] = sample_idx

        # Run base agent loop (calls our overridden state handlers)
        output = await super().run(sampling_params, **kwargs)

        # Log completion for this sample
        if self.log_turn_statistics and self.stats_collector:
            # Extract turn count from output
            num_turns = output.extra_fields.get("_actual_turns", output.num_turns)

            # Count revisions from tool traces
            num_revisions = 0
            tool_traces = output.extra_fields.get("tool_traces", [])
            for trace in tool_traces:
                if trace.get("needs_revision", False):
                    num_revisions += 1

            self.stats_collector.record_sample_completion(
                sample_idx, num_turns, num_revisions
            )

            # Log per-sample statistics if detailed logging enabled
            if (
                hasattr(output, "extra_fields")
                and "_turn_timings" in output.extra_fields
            ):
                turn_timings = output.extra_fields["_turn_timings"]
                logger.debug(
                    "Sample %s completed %s turns, %s revisions:",
                    sample_idx,
                    num_turns,
                    num_revisions,
                )
                for timing in turn_timings:
                    logger.debug(
                        "  Turn %s: gen=%.2fs, tool=%.2fs, total=%.2fs",
                        timing["turn"],
                        timing["generation_time"],
                        timing["tool_call_time"],
                        timing["total_time"],
                    )

        return output

    async def _handle_generating_state(
        self,
        agent_data: AgentData,
        sampling_params: dict[str, Any],
        ignore_termination: bool = False,
    ) -> AgentState:
        """
        Override generation state handler to add timing and synchronization.
        """
        # Initialize timing data in extra_fields if not present
        if "_timing_data" not in agent_data.extra_fields:
            agent_data.extra_fields["_timing_data"] = {
                "turn_start_times": [],
                "turn_generation_times": [],
                "turn_tool_call_times": [],
                "current_generation_start": None,
                "current_tool_start": None,
            }

        timing_data = agent_data.extra_fields["_timing_data"]

        # Start turn tracking if this is a new turn
        if not agent_data.extra_fields.get("_turn_started", False):
            timing_data["turn_start_times"].append(time.time())
            agent_data.extra_fields["_turn_started"] = True

        # Start generation timing
        timing_data["current_generation_start"] = time.time()

        # Call parent implementation
        next_state = await super()._handle_generating_state(
            agent_data, sampling_params, ignore_termination
        )

        # End generation timing
        timing_data = agent_data.extra_fields["_timing_data"]
        if timing_data["current_generation_start"] is not None:
            gen_time = time.time() - timing_data["current_generation_start"]
            timing_data["turn_generation_times"].append(gen_time)
            timing_data["current_generation_start"] = None

        # Record turn statistics if we're moving to tool processing
        if next_state == AgentState.PROCESSING_TOOLS and self.log_turn_statistics:
            turn_num = agent_data.assistant_turns
            agent_data.extra_fields["_pending_turn_record"] = turn_num

        # Report turn generation outcome for cross-worker batching
        if self.enable_cross_worker_turn_batching:
            coordinator = self._get_turn_batch_coordinator()
            if coordinator:
                batch_id, sample_id = self._get_turn_batch_context(agent_data)
                if batch_id and sample_id:
                    turn_num = agent_data.assistant_turns
                    has_tool_calls = next_state == AgentState.PROCESSING_TOOLS
                    terminated = next_state == AgentState.TERMINATED
                    await coordinator.report_generation.remote(
                        batch_id=batch_id,
                        turn_num=turn_num,
                        sample_id=sample_id,
                        has_tool_calls=has_tool_calls,
                        terminated=terminated,
                    )

        return next_state

    async def _handle_processing_tools_state(self, agent_data: AgentData) -> AgentState:
        """
        Override tool processing state handler to add timing and synchronization.
        """
        # Get timing data from extra_fields
        timing_data = agent_data.extra_fields.get("_timing_data", {})

        # Wait for turn batch synchronization if enabled
        if self.enable_cross_worker_turn_batching:
            coordinator = self._get_turn_batch_coordinator()
            if coordinator:
                batch_id, sample_id = self._get_turn_batch_context(agent_data)
                if batch_id and sample_id:
                    turn_num = agent_data.assistant_turns
                    sample_idx = agent_data.extra_fields.get("_sample_idx", 0)
                    logger.debug(
                        "Sample %s waiting for cross-worker turn %s batch...",
                        sample_idx,
                        turn_num,
                    )
                    await coordinator.wait_for_turn.remote(
                        batch_id=batch_id,
                        turn_num=turn_num,
                        timeout_s=self.turn_batch_timeout_s,
                    )
                    logger.debug(
                        "Sample %s proceeding with cross-worker turn %s batch",
                        sample_idx,
                        turn_num,
                    )
        elif self.enable_turn_batching and self.batch_synchronizer:
            turn_num = agent_data.assistant_turns
            total_in_turn = (
                len(agent_data.tool_calls) if hasattr(agent_data, "tool_calls") else 1
            )
            sample_idx = agent_data.extra_fields.get("_sample_idx", 0)
            logger.debug(
                "Sample %s waiting for turn %s batch...",
                sample_idx,
                turn_num,
            )
            await self.batch_synchronizer.wait_for_turn_batch(
                sample_idx, turn_num, total_in_turn
            )
            logger.debug(
                "Sample %s proceeding with turn %s batch",
                sample_idx,
                turn_num,
            )

        # Start tool call timing
        timing_data["current_tool_start"] = time.time()

        # Call parent implementation
        next_state = await super()._handle_processing_tools_state(agent_data)

        # End tool call timing
        if timing_data.get("current_tool_start") is not None:
            tool_time = time.time() - timing_data["current_tool_start"]
            timing_data["turn_tool_call_times"].append(tool_time)
            timing_data["current_tool_start"] = None
        else:
            # No tool calls in this turn
            timing_data["turn_tool_call_times"].append(0.0)

        # Record turn statistics
        if self.log_turn_statistics and self.stats_collector:
            turn_num = agent_data.extra_fields.get(
                "_pending_turn_record", agent_data.assistant_turns
            )
            if "_pending_turn_record" in agent_data.extra_fields:
                del agent_data.extra_fields["_pending_turn_record"]

            # Build turn timing from timing_data
            num_turns = len(timing_data["turn_start_times"])
            if num_turns >= turn_num:
                idx = turn_num - 1
                gen_time = timing_data["turn_generation_times"][idx] if idx < len(timing_data["turn_generation_times"]) else 0.0
                tool_time = timing_data["turn_tool_call_times"][idx] if idx < len(timing_data["turn_tool_call_times"]) else 0.0
                
                sample_idx = agent_data.extra_fields.get("_sample_idx", 0)
                self.stats_collector.record_turn(
                    sample_idx=sample_idx,
                    turn_number=turn_num,
                    generation_time=gen_time,
                    tool_call_time=tool_time,
                    total_time=gen_time + tool_time,
                )

        # Mark that we need a new turn start if continuing
        if next_state == AgentState.GENERATING:
            agent_data.extra_fields["_turn_started"] = False

        if self.enable_cross_worker_turn_batching and next_state == AgentState.TERMINATED:
            coordinator = self._get_turn_batch_coordinator()
            if coordinator:
                batch_id, sample_id = self._get_turn_batch_context(agent_data)
                if batch_id and sample_id:
                    turn_num = agent_data.assistant_turns
                    await coordinator.report_tool_outcome.remote(
                        batch_id=batch_id,
                        turn_num=turn_num,
                        sample_id=sample_id,
                        terminated_after_tools=True,
                    )

        return next_state

    def log_batch_statistics(self):
        """
        Log aggregated statistics for the entire batch.

        Should be called after all samples in a batch complete.
        """
        if self.log_turn_statistics and self.stats_collector:
            self.stats_collector.log_summary()
            return self.stats_collector.get_aggregated_stats()
        return {}

    def reset_batch_state(self):
        """
        Reset state for next batch.

        Should be called between batches.
        """
        if self.stats_collector:
            self.stats_collector = MultiTurnStatisticsCollector()

        if self.batch_synchronizer:
            self.batch_synchronizer.reset()
