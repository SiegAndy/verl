"""
Optimized ToolAgentLoop with multi-turn batching and statistics collection.

This extends VERL's ToolAgentLoop to add:
1. Comprehensive turn statistics logging
2. Optional turn-level batch synchronization
3. Per-turn timing breakdown
"""

import logging
import os
from typing import Any, Optional

from .tool_agent_loop import (
    AgentData,
    AgentState,
    ToolAgentLoop,
)
from .agent_loop import (
    AgentLoopOutput,
    AsyncLLMServerManager,
    DictConfigWrap,
    register,
)
from verl.utils.rollout_trace import rollout_trace_op
from transformers import AutoProcessor, AutoTokenizer

from .multi_turn_optimizer import (
    EnhancedAgentData,
    MultiTurnBatchSynchronizer,
    MultiTurnStatisticsCollector,
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

        # Statistics collector (shared across all samples in worker)
        if self.log_turn_statistics:
            self.stats_collector = MultiTurnStatisticsCollector()
        else:
            self.stats_collector = None

        # Batch synchronizer (shared across all samples in worker)
        if self.enable_turn_batching:
            self.batch_synchronizer = MultiTurnBatchSynchronizer(
                enable_synchronization=True
            )
        else:
            self.batch_synchronizer = None

        logger.info(
            f"OptimizedToolAgentLoop initialized: "
            f"turn_batching={'ENABLED' if self.enable_turn_batching else 'DISABLED'}, "
            f"turn_statistics={'ENABLED' if self.log_turn_statistics else 'DISABLED'}"
        )

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        """
        Run agent loop with enhanced tracking.

        This wraps the base ToolAgentLoop.run() with statistics collection.
        """
        # Get sample index for tracking
        sample_idx = kwargs.get("index", 0)

        # Run base agent loop (calls our overridden state handlers)
        output = await super().run(sampling_params, **kwargs)

        # Log completion for this sample
        if self.log_turn_statistics and self.stats_collector:
            # Extract turn count from output
            num_turns = output.extra_fields.get("_actual_turns", output.num_turns)
            self.stats_collector.record_sample_completion(sample_idx, num_turns)

            # Log per-sample statistics if detailed logging enabled
            if (
                hasattr(output, "extra_fields")
                and "_turn_timings" in output.extra_fields
            ):
                turn_timings = output.extra_fields["_turn_timings"]
                logger.debug(f"Sample {sample_idx} completed {num_turns} turns:")
                for timing in turn_timings:
                    logger.debug(
                        f"  Turn {timing['turn']}: "
                        f"gen={timing['generation_time']:.2f}s, "
                        f"tool={timing['tool_call_time']:.2f}s, "
                        f"total={timing['total_time']:.2f}s"
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
        # Wrap agent_data if not already wrapped
        if not isinstance(agent_data, EnhancedAgentData):
            sample_idx = agent_data.extra_fields.get("index", 0)
            agent_data = EnhancedAgentData(agent_data, sample_idx)

        # Start turn tracking if this is a new turn
        if agent_data.assistant_turns == 0 or not hasattr(agent_data, "_turn_started"):
            agent_data.start_turn()
            agent_data._turn_started = True

        # Start generation timing
        agent_data.start_generation()

        # Call parent implementation
        next_state = await super()._handle_generating_state(
            agent_data, sampling_params, ignore_termination
        )

        # End generation timing
        agent_data.end_generation()

        # Record turn statistics if we're moving to tool processing
        if next_state == AgentState.PROCESSING_TOOLS and self.log_turn_statistics:
            turn_num = agent_data.assistant_turns
            # We'll record tool timing after tool processing completes
            agent_data._pending_turn_record = turn_num

        return next_state

    async def _handle_processing_tools_state(self, agent_data: AgentData) -> AgentState:
        """
        Override tool processing state handler to add timing and synchronization.
        """
        # agent_data should already be wrapped from _handle_generating_state
        # If not, wrap it now (shouldn't happen in normal flow)
        if not isinstance(agent_data, EnhancedAgentData):
            sample_idx = agent_data.extra_fields.get("index", 0)
            agent_data = EnhancedAgentData(agent_data, sample_idx)

        # Wait for turn batch synchronization if enabled
        if self.enable_turn_batching and self.batch_synchronizer:
            turn_num = agent_data.assistant_turns
            # Count how many samples will be in this turn
            # (This is approximate - in practice we'd need cross-worker coordination)
            total_in_turn = (
                len(agent_data.tool_calls) if hasattr(agent_data, "tool_calls") else 1
            )

            logger.debug(
                f"Sample {agent_data.sample_idx} waiting for turn {turn_num} batch..."
            )
            await self.batch_synchronizer.wait_for_turn_batch(
                agent_data.sample_idx, turn_num, total_in_turn
            )
            logger.debug(
                f"Sample {agent_data.sample_idx} proceeding with turn {turn_num} batch"
            )

        # Start tool call timing
        agent_data.start_tool_calls()

        # Call parent implementation
        next_state = await super()._handle_processing_tools_state(agent_data)

        # End tool call timing
        agent_data.end_tool_calls()

        # Record turn statistics
        if self.log_turn_statistics and self.stats_collector:
            turn_num = getattr(
                agent_data, "_pending_turn_record", agent_data.assistant_turns
            )
            if hasattr(agent_data, "_pending_turn_record"):
                delattr(agent_data, "_pending_turn_record")

            turn_timings = agent_data.get_turn_timings()
            if turn_timings and len(turn_timings) >= turn_num:
                timing = turn_timings[turn_num - 1]
                self.stats_collector.record_turn(
                    sample_idx=agent_data.sample_idx,
                    turn_number=turn_num,
                    generation_time=timing["generation_time"],
                    tool_call_time=timing["tool_call_time"],
                    total_time=timing["total_time"],
                )

        # Mark that we need a new turn start if continuing
        if next_state == AgentState.GENERATING:
            agent_data._turn_started = False

        return next_state

    def log_batch_statistics(self):
        """
        Log aggregated statistics for the entire batch.

        Should be called after all samples in a batch complete.
        """
        if self.log_turn_statistics and self.stats_collector:
            self.stats_collector.log_summary()

            # Return stats for potential W&B logging
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
