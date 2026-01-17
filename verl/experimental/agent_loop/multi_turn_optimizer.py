"""
Multi-turn batching optimizer for agent loops.

Synchronizes samples at each turn boundary to enable batch processing
of tool server requests across multiple turns.
"""

import asyncio
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


@dataclass
class TurnStatistics:
    """Statistics for a single turn across all samples."""
    
    turn_number: int
    samples_in_turn: int
    generation_times: List[float] = field(default_factory=list)
    tool_call_times: List[float] = field(default_factory=list)
    total_times: List[float] = field(default_factory=list)
    
    def add_sample(
        self, 
        generation_time: float, 
        tool_call_time: float, 
        total_time: float
    ):
        """Add timing data for a sample in this turn."""
        self.generation_times.append(generation_time)
        self.tool_call_times.append(tool_call_time)
        self.total_times.append(total_time)
    
    def get_summary(self) -> Dict[str, float]:
        """Get statistical summary of this turn."""
        if not self.generation_times:
            return {}
        
        return {
            "turn": self.turn_number,
            "samples": len(self.generation_times),
            "gen_mean": sum(self.generation_times) / len(self.generation_times),
            "gen_min": min(self.generation_times),
            "gen_max": max(self.generation_times),
            "tool_mean": sum(self.tool_call_times) / len(self.tool_call_times) if self.tool_call_times else 0.0,
            "tool_min": min(self.tool_call_times) if self.tool_call_times else 0.0,
            "tool_max": max(self.tool_call_times) if self.tool_call_times else 0.0,
            "total_mean": sum(self.total_times) / len(self.total_times),
            "total_min": min(self.total_times),
            "total_max": max(self.total_times),
        }


@dataclass
class MultiTurnStatisticsCollector:
    """Collects and aggregates statistics across all turns and samples."""
    
    turn_stats: Dict[int, TurnStatistics] = field(default_factory=dict)
    sample_turn_counts: List[int] = field(default_factory=list)
    total_samples: int = 0
    
    def record_turn(
        self,
        sample_idx: int,
        turn_number: int,
        generation_time: float,
        tool_call_time: float,
        total_time: float,
    ):
        """Record timing for a specific turn of a sample."""
        if turn_number not in self.turn_stats:
            self.turn_stats[turn_number] = TurnStatistics(
                turn_number=turn_number,
                samples_in_turn=0
            )
        
        self.turn_stats[turn_number].add_sample(
            generation_time, tool_call_time, total_time
        )
        self.turn_stats[turn_number].samples_in_turn += 1
    
    def record_sample_completion(self, sample_idx: int, num_turns: int):
        """Record that a sample completed with given number of turns."""
        self.sample_turn_counts.append(num_turns)
        self.total_samples += 1
    
    def get_aggregated_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics across all turns."""
        if not self.sample_turn_counts:
            return {}
        
        # Turn distribution
        turn_distribution = defaultdict(int)
        for count in self.sample_turn_counts:
            turn_distribution[count] += 1
        
        # Calculate revision statistics
        samples_with_revisions = sum(1 for c in self.sample_turn_counts if c > 1)
        revision_rate = samples_with_revisions / self.total_samples if self.total_samples > 0 else 0.0
        
        # Per-turn summaries
        turn_summaries = {}
        for turn_num, stats in sorted(self.turn_stats.items()):
            turn_summaries[f"turn_{turn_num}"] = stats.get_summary()
        
        return {
            "total_samples": self.total_samples,
            "turn_distribution": dict(turn_distribution),
            "samples_with_revisions": samples_with_revisions,
            "revision_rate": revision_rate,
            "avg_turns_per_sample": sum(self.sample_turn_counts) / self.total_samples,
            "max_turns": max(self.sample_turn_counts),
            "min_turns": min(self.sample_turn_counts),
            "turn_details": turn_summaries,
        }
    
    def log_summary(self):
        """Log comprehensive summary of all statistics."""
        stats = self.get_aggregated_stats()
        if not stats:
            return
        
        logger.warning("=" * 80)
        logger.warning("📊 MULTI-TURN STATISTICS SUMMARY")
        logger.warning("=" * 80)
        logger.warning(f"Total Samples: {stats['total_samples']}")
        logger.warning(f"Average Turns per Sample: {stats['avg_turns_per_sample']:.2f}")
        logger.warning(f"Turn Range: {stats['min_turns']} - {stats['max_turns']}")
        logger.warning("")
        
        logger.warning("🔄 Revision Statistics:")
        logger.warning(f"  Samples with Revisions: {stats['samples_with_revisions']} / {stats['total_samples']}")
        logger.warning(f"  Revision Rate: {stats['revision_rate']:.1%}")
        logger.warning("")
        
        logger.warning("📈 Turn Distribution:")
        for turns, count in sorted(stats['turn_distribution'].items()):
            pct = 100.0 * count / stats['total_samples']
            logger.warning(f"  {turns} turn(s): {count:4d} samples ({pct:5.1f}%)")
        logger.warning("")
        
        logger.warning("⏱️  Per-Turn Timing Breakdown:")
        for turn_key, turn_data in sorted(stats['turn_details'].items()):
            turn_num = turn_data['turn']
            samples = turn_data['samples']
            logger.warning(f"  Turn {turn_num} ({samples} samples):")
            logger.warning(f"    Generation:  avg={turn_data['gen_mean']:.2f}s, min={turn_data['gen_min']:.2f}s, max={turn_data['gen_max']:.2f}s")
            if turn_data['tool_mean'] > 0:
                logger.warning(f"    Tool Calls:  avg={turn_data['tool_mean']:.2f}s, min={turn_data['tool_min']:.2f}s, max={turn_data['tool_max']:.2f}s")
            logger.warning(f"    Total:       avg={turn_data['total_mean']:.2f}s, min={turn_data['total_min']:.2f}s, max={turn_data['total_max']:.2f}s")
        logger.warning("=" * 80)


class MultiTurnBatchSynchronizer:
    """
    Synchronizes samples at turn boundaries for batch processing.
    
    Instead of letting samples run independently through multiple turns,
    this synchronizer groups samples by turn and processes them in batches:
    - Turn 1: All samples generate and call tools together
    - Turn 2: Only samples needing revision generate and call tools together
    - Turn 3+: Further subsets processed together
    
    This maximizes tool server batching efficiency.
    """
    
    def __init__(self, enable_synchronization: bool = True):
        self.enable_synchronization = enable_synchronization
        self.turn_barriers: Dict[int, asyncio.Event] = {}
        self.turn_sample_sets: Dict[int, Set[int]] = defaultdict(set)
        self.turn_ready_counts: Dict[int, int] = defaultdict(int)
        self.lock = asyncio.Lock()
        
        logger.info(
            f"MultiTurnBatchSynchronizer initialized "
            f"(synchronization={'ENABLED' if enable_synchronization else 'DISABLED'})"
        )
    
    async def wait_for_turn_batch(
        self, 
        sample_idx: int, 
        turn_number: int, 
        total_samples_in_turn: int
    ):
        """
        Wait for all samples in this turn to reach the same point.
        
        Args:
            sample_idx: Index of this sample
            turn_number: Current turn number
            total_samples_in_turn: Expected number of samples in this turn
        """
        if not self.enable_synchronization:
            return  # Skip synchronization
        
        async with self.lock:
            # Register this sample for this turn
            self.turn_sample_sets[turn_number].add(sample_idx)
            self.turn_ready_counts[turn_number] += 1
            
            # Create barrier event if it doesn't exist
            if turn_number not in self.turn_barriers:
                self.turn_barriers[turn_number] = asyncio.Event()
            
            barrier = self.turn_barriers[turn_number]
            current_count = self.turn_ready_counts[turn_number]
            
            # Check if all samples have arrived
            if current_count >= total_samples_in_turn:
                logger.debug(
                    f"Turn {turn_number} batch complete: "
                    f"{current_count}/{total_samples_in_turn} samples ready"
                )
                barrier.set()
        
        # Wait for barrier (outside lock)
        await barrier.wait()
    
    def reset(self):
        """Reset synchronizer for next batch."""
        self.turn_barriers.clear()
        self.turn_sample_sets.clear()
        self.turn_ready_counts.clear()


class EnhancedAgentData:
    """
    Extended AgentData with turn tracking for optimization and statistics.
    
    This wrapper adds timing and turn metadata without modifying the
    original AgentData class from VERL.
    """
    
    def __init__(self, original_agent_data, sample_idx: int):
        self.agent_data = original_agent_data
        self.sample_idx = sample_idx
        
        # Turn tracking
        self.current_turn = 0
        self.turn_start_times: List[float] = []
        self.turn_generation_times: List[float] = []
        self.turn_tool_call_times: List[float] = []
        
        # Current turn timing
        self.current_generation_start: Optional[float] = None
        self.current_tool_start: Optional[float] = None
    
    def start_turn(self):
        """Mark the start of a new turn."""
        self.current_turn += 1
        self.turn_start_times.append(time.perf_counter())
    
    def start_generation(self):
        """Mark the start of generation phase."""
        self.current_generation_start = time.perf_counter()
    
    def end_generation(self):
        """Mark the end of generation phase."""
        if self.current_generation_start is not None:
            gen_time = time.perf_counter() - self.current_generation_start
            self.turn_generation_times.append(gen_time)
            self.current_generation_start = None
    
    def start_tool_calls(self):
        """Mark the start of tool call phase."""
        self.current_tool_start = time.perf_counter()
    
    def end_tool_calls(self):
        """Mark the end of tool call phase."""
        if self.current_tool_start is not None:
            tool_time = time.perf_counter() - self.current_tool_start
            self.turn_tool_call_times.append(tool_time)
            self.current_tool_start = None
        else:
            # No tool calls in this turn
            self.turn_tool_call_times.append(0.0)
    
    def get_turn_timings(self) -> List[Dict[str, float]]:
        """Get timing data for all turns."""
        timings = []
        for i in range(len(self.turn_start_times)):
            turn_data = {
                "turn": i + 1,
                "generation_time": self.turn_generation_times[i] if i < len(self.turn_generation_times) else 0.0,
                "tool_call_time": self.turn_tool_call_times[i] if i < len(self.turn_tool_call_times) else 0.0,
            }
            turn_data["total_time"] = turn_data["generation_time"] + turn_data["tool_call_time"]
            timings.append(turn_data)
        return timings
    
    def __getattr__(self, name):
        """Proxy attribute access to original agent_data."""
        return getattr(self.agent_data, name)
