# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Zero Advantage Soft Filtering Module.

This module implements a streak-based soft filtering mechanism for samples with zero advantages.
When a sample has zero advantage repeatedly, the probability of selecting it decreases exponentially
using p^{z_i}, where p is a base probability and z_i is the streak counter.

The filtering distinguishes between:
- Good zero advantages (positive rewards but zero advantage)
- Bad zero advantages (negative or zero rewards with zero advantage)

Reference:
    Based on ideas from https://arxiv.org/pdf/2506.02177
"""

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from verl import DataProto

logger = logging.getLogger(__name__)

__all__ = ["ZeroAdvantageFilterConfig", "ZeroAdvantageFilter"]


@dataclass
class ZeroAdvantageFilterConfig:
    """Configuration for zero advantage soft filtering.

    Args:
        enable (bool): Whether to enable zero advantage filtering. Default: False
        p_good_zero (float): Base probability for good zero advantages (positive reward).
            Probability decreases as p_good_zero^{streak_count}. Range: (0, 1].
            Default: 0.9 (10% dropout at first zero, 19% at second, etc.)
        p_bad_zero (float): Base probability for bad zero advantages (non-positive reward).
            Probability decreases as p_bad_zero^{streak_count}. Range: (0, 1].
            Default: 0.7 (30% dropout at first zero, 51% at second, etc.)
        zero_threshold (float): Threshold for considering an advantage as zero.
            Default: 1e-6 (effectively zero)
        max_streak (int): Maximum streak count to track. Prevents unbounded growth.
            Default: 10 (p_bad=0.7 gives ~97% dropout at max_streak)
        reset_streak_on_nonzero (bool): Whether to reset streak when advantage is non-zero.
            Default: True
    """

    enable: bool = False
    p_good_zero: float = 0.9
    p_bad_zero: float = 0.7
    zero_threshold: float = 1e-6
    max_streak: int = 10
    reset_streak_on_nonzero: bool = True

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.enable:
            assert (
                0 < self.p_good_zero <= 1
            ), f"p_good_zero must be in (0, 1], got {self.p_good_zero}"
            assert (
                0 < self.p_bad_zero <= 1
            ), f"p_bad_zero must be in (0, 1], got {self.p_bad_zero}"
            assert (
                self.zero_threshold >= 0
            ), f"zero_threshold must be non-negative, got {self.zero_threshold}"
            assert (
                self.max_streak > 0
            ), f"max_streak must be positive, got {self.max_streak}"


class ZeroAdvantageFilter:
    """Implements streak-based soft filtering for samples with zero advantages.

    This filter tracks consecutive zero advantages for each sample (identified by UID)
    and applies exponential probability filtering: p^{z_i} where z_i is the streak count.

    The filter distinguishes between:
    - Good zeros: reward > 0 but advantage ≈ 0 (model already good)
    - Bad zeros: reward ≤ 0 and advantage ≈ 0 (no learning signal)

    Usage:
        config = ZeroAdvantageFilterConfig(enable=True, p_good_zero=0.9, p_bad_zero=0.7)
        filter = ZeroAdvantageFilter(config)
        filtered_batch, stats = filter.filter_batch(batch)
    """

    def __init__(self, config: ZeroAdvantageFilterConfig):
        """Initialize the zero advantage filter.

        Args:
            config (ZeroAdvantageFilterConfig): Filter configuration.
        """
        self.config = config
        # Track streak counts: uid -> streak_count
        self.streak_tracker: Dict[str, int] = defaultdict(int)
        # Track whether last visit was good/bad zero: uid -> is_good
        self.zero_type_tracker: Dict[str, bool] = defaultdict(bool)

    def _compute_sequence_level_advantages(
        self, advantages: torch.Tensor, response_mask: torch.Tensor
    ) -> torch.Tensor:
        """Compute sequence-level advantages by averaging over valid tokens.

        Args:
            advantages (torch.Tensor): Token-level advantages. Shape: (batch_size, seq_len)
            response_mask (torch.Tensor): Response mask. Shape: (batch_size, seq_len)

        Returns:
            torch.Tensor: Sequence-level advantages. Shape: (batch_size,)
        """
        # Mask out non-response tokens
        masked_advantages = advantages * response_mask
        # Sum over sequence, divide by number of valid tokens
        seq_len_sum = response_mask.sum(dim=1)  # (batch_size,)
        # Avoid division by zero
        seq_len_sum = torch.clamp(seq_len_sum, min=1.0)
        seq_advantages = masked_advantages.sum(dim=1) / seq_len_sum  # (batch_size,)
        return seq_advantages

    def _compute_sequence_level_rewards(
        self, rewards: torch.Tensor, response_mask: torch.Tensor
    ) -> torch.Tensor:
        """Compute sequence-level rewards by summing over valid tokens.

        Args:
            rewards (torch.Tensor): Token-level rewards. Shape: (batch_size, seq_len)
            response_mask (torch.Tensor): Response mask. Shape: (batch_size, seq_len)

        Returns:
            torch.Tensor: Sequence-level rewards. Shape: (batch_size,)
        """
        # Mask out non-response tokens and sum
        masked_rewards = rewards * response_mask
        seq_rewards = masked_rewards.sum(dim=1)  # (batch_size,)
        return seq_rewards

    def _identify_zero_advantages(
        self,
        seq_advantages: torch.Tensor,
        seq_rewards: torch.Tensor,
        uids: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Identify samples with zero advantages and classify as good/bad.

        Args:
            seq_advantages (torch.Tensor): Sequence-level advantages. Shape: (batch_size,)
            seq_rewards (torch.Tensor): Sequence-level rewards. Shape: (batch_size,)
            uids (np.ndarray): Sample UIDs. Shape: (batch_size,)

        Returns:
            Tuple containing:
                - is_zero_advantage (np.ndarray): Boolean mask for zero advantages
                - is_good_zero (np.ndarray): Boolean mask for good zeros (subset of zero_advantage)
                - is_bad_zero (np.ndarray): Boolean mask for bad zeros (subset of zero_advantage)
        """
        # Convert to numpy for easier processing
        seq_advantages_np = seq_advantages.detach().cpu().numpy()
        seq_rewards_np = seq_rewards.detach().cpu().numpy()

        # Identify zero advantages
        is_zero_advantage = (
            np.abs(seq_advantages_np) < self.config.zero_threshold
        )  # (batch_size,)

        # Among zero advantages, classify as good (reward > 0) or bad (reward <= 0)
        is_good_zero = is_zero_advantage & (seq_rewards_np > 0)  # (batch_size,)
        is_bad_zero = is_zero_advantage & (seq_rewards_np <= 0)  # (batch_size,)

        return is_zero_advantage, is_good_zero, is_bad_zero

    def _update_streak_counters(
        self,
        uids: np.ndarray,
        is_zero_advantage: np.ndarray,
        is_good_zero: np.ndarray,
    ) -> None:
        """Update streak counters for each sample based on current advantages.

        Args:
            uids (np.ndarray): Sample UIDs. Shape: (batch_size,)
            is_zero_advantage (np.ndarray): Boolean mask for zero advantages. Shape: (batch_size,)
            is_good_zero (np.ndarray): Boolean mask for good zeros. Shape: (batch_size,)
        """
        for i, uid in enumerate(uids):
            uid_str = str(uid)
            if is_zero_advantage[i]:
                # Increment streak counter, cap at max_streak
                self.streak_tracker[uid_str] = min(
                    self.streak_tracker[uid_str] + 1, self.config.max_streak
                )
                # Track whether this is a good or bad zero
                self.zero_type_tracker[uid_str] = bool(is_good_zero[i])
            else:
                # Non-zero advantage
                if self.config.reset_streak_on_nonzero:
                    self.streak_tracker[uid_str] = 0
                # No change to zero_type_tracker (keep last known type)

    def _compute_filter_probabilities(
        self,
        uids: np.ndarray,
        is_zero_advantage: np.ndarray,
    ) -> np.ndarray:
        """Compute filtering probabilities for each sample based on streak count.

        For samples with zero advantage, probability = p^{streak_count},
        where p is p_good_zero or p_bad_zero depending on the type.

        Args:
            uids (np.ndarray): Sample UIDs. Shape: (batch_size,)
            is_zero_advantage (np.ndarray): Boolean mask for zero advantages. Shape: (batch_size,)

        Returns:
            np.ndarray: Selection probabilities. Shape: (batch_size,)
                1.0 for non-zero advantages, p^{streak} for zero advantages.
        """
        batch_size = len(uids)
        selection_probs = np.ones(batch_size, dtype=np.float32)

        for i, uid in enumerate(uids):
            if is_zero_advantage[i]:
                uid_str = str(uid)
                streak = self.streak_tracker[uid_str]
                # Use previous zero type (from last visit) for filtering
                is_good = self.zero_type_tracker.get(uid_str, False)
                base_prob = (
                    self.config.p_good_zero if is_good else self.config.p_bad_zero
                )
                # Exponential decay: p^{streak}
                selection_probs[i] = base_prob**streak

        return selection_probs

    def update_streaks_from_batch(self, batch: DataProto) -> Dict:
        """Update streak counters based on batch advantages without filtering.

        This method is used for epoch-level filtering: collect statistics during
        the current epoch to determine filtering for the next epoch.

        Args:
            batch (DataProto): Input batch with advantages, rewards, UIDs, and response_mask.

        Returns:
            Dict: Statistics about zero advantages in this batch
        """
        if not self.config.enable:
            return {
                "total_samples": batch.batch.batch_size[0],
                "zero_advantage_total": 0,
                "good_zero_count": 0,
                "bad_zero_count": 0,
            }

        # Extract data
        advantages = batch.batch["advantages"]
        response_mask = batch.batch["response_mask"]
        uids = batch.non_tensor_batch["uid"]
        rewards = batch.batch.get(
            "token_level_rewards", batch.batch.get("token_level_scores")
        )

        if rewards is None:
            raise ValueError(
                "Batch must contain either 'token_level_rewards' or 'token_level_scores'"
            )

        # Compute sequence-level metrics
        seq_advantages = self._compute_sequence_level_advantages(
            advantages, response_mask
        )
        seq_rewards = self._compute_sequence_level_rewards(rewards, response_mask)

        # Identify zero advantages
        is_zero_advantage, is_good_zero, is_bad_zero = self._identify_zero_advantages(
            seq_advantages, seq_rewards, uids
        )

        # Update streak counters
        self._update_streak_counters(uids, is_zero_advantage, is_good_zero)

        # Return statistics
        return {
            "total_samples": len(uids),
            "zero_advantage_total": int(is_zero_advantage.sum()),
            "good_zero_count": int(is_good_zero.sum()),
            "bad_zero_count": int(is_bad_zero.sum()),
        }

    def compute_epoch_filter_mask(
        self, uids: List[str], random_state: Optional[np.random.RandomState] = None
    ) -> Tuple[np.ndarray, Dict]:
        """Compute which samples should be included in the next epoch based on current streaks.

        This is used for epoch-level filtering: before an epoch starts, determine
        which samples to include based on their streak counts from previous epochs.

        Args:
            uids (List[str]): List of all sample UIDs in the dataset.
            random_state (Optional[np.random.RandomState]): Random state for reproducibility.

        Returns:
            Tuple containing:
                - keep_mask (np.ndarray): Boolean mask indicating which samples to keep
                - stats (Dict): Statistics about filtering decision
        """
        if not self.config.enable:
            return np.ones(len(uids), dtype=bool), {
                "total_samples": len(uids),
                "samples_to_filter": 0,
                "samples_to_keep": len(uids),
                "avg_streak_of_filtered": 0.0,
            }

        if random_state is None:
            random_state = np.random.RandomState()

        total_samples = len(uids)
        keep_mask = np.ones(total_samples, dtype=bool)
        filtered_streaks = []

        for i, uid in enumerate(uids):
            uid_str = str(uid)
            if uid_str in self.streak_tracker and self.streak_tracker[uid_str] > 0:
                # Has non-zero streak, apply filtering probability
                streak = self.streak_tracker[uid_str]
                is_good = self.zero_type_tracker.get(uid_str, False)
                base_prob = (
                    self.config.p_good_zero if is_good else self.config.p_bad_zero
                )
                selection_prob = base_prob**streak

                # Filter: keep if random < p^{streak}
                if random_state.rand() >= selection_prob:
                    keep_mask[i] = False
                    filtered_streaks.append(streak)

        samples_to_keep = int(keep_mask.sum())
        samples_to_filter = total_samples - samples_to_keep
        avg_streak_of_filtered = (
            float(np.mean(filtered_streaks)) if filtered_streaks else 0.0
        )

        stats = {
            "total_samples": total_samples,
            "samples_to_filter": samples_to_filter,
            "samples_to_keep": samples_to_keep,
            "avg_streak_of_filtered": avg_streak_of_filtered,
        }

        return keep_mask, stats

    def filter_batch(
        self, batch: DataProto, random_state: Optional[np.random.RandomState] = None
    ) -> Tuple[DataProto, Dict]:
        """Apply soft filtering to batch based on zero advantage streaks (step-level filtering).

        NOTE: For epoch-level filtering, use update_streaks_from_batch() during training
        and compute_epoch_filter_mask() before the next epoch starts. This filter_batch()
        method is kept for backward compatibility and step-level filtering use cases.

        Args:
            batch (DataProto): Input batch with advantages, rewards, UIDs, and response_mask.
            random_state (Optional[np.random.RandomState]): Random state for reproducibility.
                If None, uses default numpy random.

        Returns:
            Tuple containing:
                - filtered_batch (DataProto): Filtered batch (subset of input)
                - stats (Dict): Statistics about filtering
                    - total_samples: Total samples in input batch
                    - zero_advantage_total: Number of samples with zero advantage
                    - good_zero_count: Number of good zeros
                    - bad_zero_count: Number of bad zeros
                    - samples_used: Number of samples after filtering
                    - avg_good_zero_streak: Average streak for good zeros
                    - avg_bad_zero_streak: Average streak for bad zeros
        """
        if not self.config.enable:
            # Return original batch without filtering
            return batch, {
                "total_samples": batch.batch.batch_size[0],
                "zero_advantage_total": 0,
                "good_zero_count": 0,
                "bad_zero_count": 0,
                "samples_used": batch.batch.batch_size[0],
                "avg_good_zero_streak": 0.0,
                "avg_bad_zero_streak": 0.0,
            }

        if random_state is None:
            random_state = np.random.RandomState()

        # Extract data
        advantages = batch.batch["advantages"]  # (batch_size, seq_len)
        response_mask = batch.batch["response_mask"]  # (batch_size, seq_len)
        uids = batch.non_tensor_batch["uid"]  # (batch_size,)

        # Get rewards (use token_level_rewards if available, otherwise token_level_scores)
        rewards = batch.batch.get(
            "token_level_rewards", batch.batch.get("token_level_scores")
        )
        if rewards is None:
            raise ValueError(
                "Batch must contain either 'token_level_rewards' or 'token_level_scores'"
            )

        # Compute sequence-level metrics
        seq_advantages = self._compute_sequence_level_advantages(
            advantages, response_mask
        )
        seq_rewards = self._compute_sequence_level_rewards(rewards, response_mask)

        # Identify zero advantages
        is_zero_advantage, is_good_zero, is_bad_zero = self._identify_zero_advantages(
            seq_advantages, seq_rewards, uids
        )

        # Compute statistics BEFORE updating streaks (for current batch)
        total_samples = len(uids)
        zero_advantage_total = int(is_zero_advantage.sum())
        good_zero_count = int(is_good_zero.sum())
        bad_zero_count = int(is_bad_zero.sum())

        # Compute average streaks for current zero samples
        good_zero_streaks = [
            self.streak_tracker[str(uid)]
            for i, uid in enumerate(uids)
            if is_good_zero[i]
        ]
        bad_zero_streaks = [
            self.streak_tracker[str(uid)]
            for i, uid in enumerate(uids)
            if is_bad_zero[i]
        ]
        avg_good_zero_streak = (
            float(np.mean(good_zero_streaks)) if good_zero_streaks else 0.0
        )
        avg_bad_zero_streak = (
            float(np.mean(bad_zero_streaks)) if bad_zero_streaks else 0.0
        )

        # Compute filter probabilities BEFORE updating streaks
        selection_probs = self._compute_filter_probabilities(uids, is_zero_advantage)

        # Update streak counters for next iteration
        self._update_streak_counters(uids, is_zero_advantage, is_good_zero)

        # Apply soft filtering: select if random > p^{streak}
        # Actually, we select with probability p^{streak}, so: random < p^{streak}
        random_values = random_state.rand(total_samples)
        keep_mask = random_values < selection_probs  # True if we keep the sample

        # Count samples used
        samples_used = int(keep_mask.sum())

        # Filter batch
        keep_indices = np.where(keep_mask)[0].tolist()
        if len(keep_indices) == 0:
            # Edge case: no samples kept, keep at least one to avoid empty batch
            logger.warning(
                "Zero advantage filter removed all samples! Keeping original batch."
            )
            keep_indices = list(range(total_samples))
            samples_used = total_samples

        filtered_batch = batch.select(keep_indices)

        # Compile statistics
        stats = {
            "total_samples": total_samples,
            "zero_advantage_total": zero_advantage_total,
            "good_zero_count": good_zero_count,
            "bad_zero_count": bad_zero_count,
            "samples_used": samples_used,
            "avg_good_zero_streak": avg_good_zero_streak,
            "avg_bad_zero_streak": avg_bad_zero_streak,
        }

        return filtered_batch, stats

    def reset(self):
        """Reset all streak trackers. Useful for starting a new epoch."""
        self.streak_tracker.clear()
        self.zero_type_tracker.clear()

    def get_stats(self) -> Dict:
        """Get current statistics about streak trackers.

        Returns:
            Dict: Statistics including total tracked UIDs and average streaks.
        """
        total_uids = len(self.streak_tracker)
        if total_uids == 0:
            return {
                "total_tracked_uids": 0,
                "avg_streak": 0.0,
                "max_streak": 0,
            }

        streaks = list(self.streak_tracker.values())
        return {
            "total_tracked_uids": total_uids,
            "avg_streak": float(np.mean(streaks)),
            "max_streak": int(np.max(streaks)),
        }

    def state_dict(self) -> Dict:
        """Get state dictionary for checkpointing.

        Returns:
            Dict: State dictionary containing config and tracker states.
        """
        return {
            "config": {
                "enable": self.config.enable,
                "p_good_zero": self.config.p_good_zero,
                "p_bad_zero": self.config.p_bad_zero,
                "zero_threshold": self.config.zero_threshold,
                "max_streak": self.config.max_streak,
                "reset_streak_on_nonzero": self.config.reset_streak_on_nonzero,
            },
            "streak_tracker": dict(self.streak_tracker),
            "zero_type_tracker": dict(self.zero_type_tracker),
        }

    def load_state_dict(self, state_dict: Dict) -> None:
        """Load state from checkpoint.

        Args:
            state_dict (Dict): State dictionary from checkpoint.
        """
        # Verify config compatibility (warn if different)
        saved_config = state_dict.get("config", {})
        if saved_config.get("enable") != self.config.enable:
            logger.warning(
                f"Zero advantage filter enable mismatch: "
                f"saved={saved_config.get('enable')}, current={self.config.enable}"
            )
        if saved_config.get("p_good_zero") != self.config.p_good_zero:
            logger.warning(
                f"Zero advantage filter p_good_zero mismatch: "
                f"saved={saved_config.get('p_good_zero')}, current={self.config.p_good_zero}"
            )
        if saved_config.get("p_bad_zero") != self.config.p_bad_zero:
            logger.warning(
                f"Zero advantage filter p_bad_zero mismatch: "
                f"saved={saved_config.get('p_bad_zero')}, current={self.config.p_bad_zero}"
            )

        # Load tracker states
        self.streak_tracker = defaultdict(
            int, state_dict.get("streak_tracker", {})
        )
        self.zero_type_tracker = defaultdict(
            bool, state_dict.get("zero_type_tracker", {})
        )

        logger.info(
            f"Loaded zero advantage filter state: "
            f"{len(self.streak_tracker)} tracked UIDs"
        )
