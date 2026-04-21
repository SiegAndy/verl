import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict

from src.agent.trainer.decomposer.verl_integration.best_turn_truncation import (
    accumulate_best_turn_epoch_stats,
    apply_best_turn_truncation,
    build_best_turn_epoch_metrics,
    select_best_turn,
)
from src.agent.trainer.decomposer.verl_integration.custom_reward import (
    FinalRewardFunction,
)
from verl import DataProto
from verl.utils.model import compute_position_id_with_mask


def _config(rollout_n=1):
    return OmegaConf.create(
        {
            "best_turn_truncation": {
                "enable": True,
                "metric_priority": ["mrr", "recall@20", "ndcg@20"],
                "tie_epsilon": 1e-6,
                "min_positive_signal": 1e-6,
                "tie_break": "earliest",
                "fail_on_missing_metrics": True,
                "train_only": True,
            },
            "actor_rollout_ref": {
                "actor": {"ppo_mini_batch_size": 16},
                "rollout": {"n": rollout_n},
            },
        }
    )


def _feedback(mrr, recall20, ndcg20, turn):
    return {
        "turn": turn,
        "mrr": mrr,
        "recall@20": recall20,
        "ndcg@20": ndcg20,
        "performance_metrics": {
            "recall@10": recall20,
            "mrr": mrr,
            "recall@20": recall20,
            "ndcg@20": ndcg20,
        },
        "format_penalties": {
            "num_queries": 2,
            "dag_depth": 1,
            "num_refines": 0,
            "refine_actions_by_subquery": {},
            "breadth": 2,
            "hard_gate_failed": False,
        },
    }


def _batch(feedbacks):
    if feedbacks and isinstance(feedbacks[0], dict):
        feedback_rows = [feedbacks]
    else:
        feedback_rows = feedbacks

    batch_size = len(feedback_rows)
    response_len = 8
    prompts = torch.tensor([[1, 2, 3, 4]]).repeat(batch_size, 1)
    responses = torch.tensor([[10, 11, 12, 13, 14, 15, 0, 0]]).repeat(batch_size, 1)
    response_mask = torch.tensor([[1, 0, 1, 0, 1, 0, 0, 0]]).repeat(batch_size, 1)
    attention_mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]]).repeat(
        batch_size, 1
    )
    input_ids = torch.cat([prompts, responses], dim=1)
    position_ids = compute_position_id_with_mask(attention_mask)
    tensors = {
        "prompts": prompts,
        "responses": responses,
        "response_mask": response_mask,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "rm_scores": torch.zeros((batch_size, response_len), dtype=torch.float32),
    }
    non_tensors = {
        "uid": np.array([f"qid-{i+1}" for i in range(batch_size)], dtype=object),
        "data_source": np.array(
            [f"qid-{i+1}" for i in range(batch_size)], dtype=object
        ),
        "reward_model": np.array(
            [{"ground_truth": {}} for _ in range(batch_size)], dtype=object
        ),
        "extra_info": np.array([{} for _ in range(batch_size)], dtype=object),
        "__num_turns__": np.array(
            [int(feedback_row[-1]["turn"]) * 2 + 1 for feedback_row in feedback_rows],
            dtype=np.int32,
        ),
        "tool_iteration_feedbacks": np.array(feedback_rows, dtype=object),
        "tool_turn_boundaries": np.array(
            [
                [
                    {"turn": 1, "response_end": 2},
                    {"turn": 2, "response_end": 4},
                    {"turn": 3, "response_end": 6},
                ]
                for _ in range(batch_size)
            ],
            dtype=object,
        ),
        "performance_metrics": np.array(
            [feedback_row[-1]["performance_metrics"] for feedback_row in feedback_rows],
            dtype=object,
        ),
        "format_penalties": np.array(
            [feedback_row[-1]["format_penalties"] for feedback_row in feedback_rows],
            dtype=object,
        ),
        "request_id": np.array([f"req-{i+1}" for i in range(batch_size)], dtype=object),
        "assistant_turns": np.array(
            [feedback_row[-1]["turn"] for feedback_row in feedback_rows], dtype=object
        ),
        "user_turns": np.array(
            [feedback_row[-1]["turn"] for feedback_row in feedback_rows], dtype=object
        ),
    }
    return DataProto(
        batch=TensorDict(tensors, batch_size=batch_size),
        non_tensor_batch=non_tensors,
        meta_info={"reward_extra_keys": ["score"]},
    )


def test_select_best_turn_tie_uses_earliest():
    feedbacks = [
        _feedback(0.5, 0.5, 0.5, 1),
        _feedback(0.2, 0.2, 0.2, 2),
        _feedback(0.5, 0.5, 0.5, 3),
    ]

    selection, reason = select_best_turn(feedbacks)

    assert reason == "split"
    assert selection.turn == 1


def test_select_best_turn_all_zero_no_split():
    feedbacks = [_feedback(0.0, 0.0, 0.0, 1), _feedback(0.0, 0.0, 0.0, 2)]

    selection, reason = select_best_turn(feedbacks)

    assert selection is None
    assert reason == "skip_all_zero"


def test_select_best_turn_monotonic_improvement_no_split():
    feedbacks = [_feedback(0.1, 0.1, 0.1, 1), _feedback(0.2, 0.2, 0.2, 2)]

    selection, reason = select_best_turn(feedbacks)

    assert selection is None
    assert reason == "skip_best_is_final"


def test_select_best_turn_missing_metric_fails_fast():
    feedbacks = [{"turn": 1, "mrr": 0.5, "recall@20": 0.5}]

    with pytest.raises(ValueError, match="ndcg@20"):
        select_best_turn(feedbacks)


def test_apply_best_turn_truncation_splits_prefix_and_invalidates_rm_scores():
    batch = _batch(
        [
            _feedback(0.5, 0.5, 0.5, 1),
            _feedback(0.2, 0.2, 0.2, 2),
            _feedback(0.5, 0.5, 0.5, 3),
        ]
    )

    truncated, stats = apply_best_turn_truncation(
        batch,
        config=_config(),
        pad_token_id=0,
        dp_size=1,
    )

    assert len(truncated) == 2
    assert stats["best_turn/split_count"] == 1.0
    assert "rm_scores" not in truncated.batch
    assert (
        truncated.non_tensor_batch["trajectory_variant"][0]
        == "full_original_after_best"
    )
    assert truncated.non_tensor_batch["trajectory_variant"][1] == "best_prefix"
    assert truncated.non_tensor_batch["uid"][0] == truncated.non_tensor_batch["uid"][1]
    assert truncated.batch["response_mask"][1].tolist() == [1, 0, 0, 0, 0, 0, 0, 0]
    assert truncated.batch["attention_mask"][1, 4:].tolist() == [
        1,
        1,
        0,
        0,
        0,
        0,
        0,
        0,
    ]
    assert truncated.non_tensor_batch["performance_metrics"][1]["mrr"] == 0.5
    assert truncated.non_tensor_batch["__num_turns__"][1] == 3
    assert stats["best_turn/group_count"] == 1.0
    assert stats["best_turn/groups_with_truncation_count"] == 1.0
    assert stats["best_turn/groups_with_truncation_rate"] == 1.0
    assert stats["best_turn/avg_truncations_per_group"] == 1.0
    assert stats["best_turn/max_truncations_per_group"] == 1.0


def test_apply_best_turn_truncation_group_metrics_no_samples_truncate():
    batch = _batch(
        [
            [_feedback(0.1, 0.1, 0.1, 1), _feedback(0.2, 0.2, 0.2, 2)],
            [_feedback(0.1, 0.1, 0.1, 1), _feedback(0.2, 0.2, 0.2, 2)],
            [_feedback(0.1, 0.1, 0.1, 1), _feedback(0.2, 0.2, 0.2, 2)],
            [_feedback(0.1, 0.1, 0.1, 1), _feedback(0.2, 0.2, 0.2, 2)],
        ]
    )

    truncated, stats = apply_best_turn_truncation(
        batch,
        config=_config(rollout_n=2),
        pad_token_id=0,
        dp_size=1,
    )

    assert len(truncated) == 4
    assert stats["best_turn/group_count"] == 2.0
    assert stats["best_turn/split_count"] == 0.0
    assert stats["best_turn/groups_with_truncation_count"] == 0.0
    assert stats["best_turn/groups_with_truncation_rate"] == 0.0
    assert stats["best_turn/avg_truncations_per_group"] == 0.0
    assert stats["best_turn/max_truncations_per_group"] == 0.0


def test_apply_best_turn_truncation_group_metrics_track_partial_and_multiple_truncations():
    split_feedbacks = [
        _feedback(0.5, 0.5, 0.5, 1),
        _feedback(0.2, 0.2, 0.2, 2),
        _feedback(0.5, 0.5, 0.5, 3),
    ]
    no_split_feedbacks = [
        _feedback(0.1, 0.1, 0.1, 1),
        _feedback(0.2, 0.2, 0.2, 2),
    ]
    batch = _batch(
        [
            split_feedbacks,
            no_split_feedbacks,
            split_feedbacks,
            no_split_feedbacks,
            split_feedbacks,
            no_split_feedbacks,
        ]
    )

    truncated, stats = apply_best_turn_truncation(
        batch,
        config=_config(rollout_n=3),
        pad_token_id=0,
        dp_size=1,
    )

    assert len(truncated) == 9
    assert stats["best_turn/group_count"] == 2.0
    assert stats["best_turn/split_count"] == 3.0
    assert stats["best_turn/groups_with_truncation_count"] == 2.0
    assert stats["best_turn/groups_with_truncation_rate"] == 1.0
    assert stats["best_turn/avg_truncations_per_group"] == 1.5
    assert stats["best_turn/max_truncations_per_group"] == 2.0


def test_apply_best_turn_truncation_group_metrics_respect_dp_drop():
    split_feedbacks = [
        _feedback(0.5, 0.5, 0.5, 1),
        _feedback(0.2, 0.2, 0.2, 2),
        _feedback(0.5, 0.5, 0.5, 3),
    ]
    batch = _batch(
        [
            split_feedbacks,
            split_feedbacks,
            split_feedbacks,
            split_feedbacks,
        ]
    )

    truncated, stats = apply_best_turn_truncation(
        batch,
        config=_config(rollout_n=2),
        pad_token_id=0,
        dp_size=3,
    )

    assert len(truncated) == 6
    assert stats["best_turn/drop_for_dp_divisor_count"] == 2.0
    assert stats["best_turn/group_count"] == 2.0
    assert stats["best_turn/split_count"] == 2.0
    assert stats["best_turn/groups_with_truncation_count"] == 1.0
    assert stats["best_turn/groups_with_truncation_rate"] == 0.5
    assert stats["best_turn/avg_truncations_per_group"] == 1.0
    assert stats["best_turn/max_truncations_per_group"] == 2.0


def test_best_turn_epoch_metrics_aggregate_group_stats():
    epoch_stats = {}
    step_one = {
        "best_turn/raw_rollout_count": 6.0,
        "best_turn/group_count": 2.0,
        "best_turn/split_count": 3.0,
        "best_turn/split_rate": 0.5,
        "best_turn/groups_with_truncation_count": 2.0,
        "best_turn/groups_with_truncation_rate": 1.0,
        "best_turn/avg_truncations_per_group": 1.5,
        "best_turn/max_truncations_per_group": 2.0,
        "best_turn/drop_for_dp_divisor_count": 1.0,
        "best_turn/skip_all_zero": 1.0,
        "best_turn/mean_cut_turn": 1.5,
    }
    step_two = {
        "best_turn/raw_rollout_count": 3.0,
        "best_turn/group_count": 1.0,
        "best_turn/split_count": 1.0,
        "best_turn/split_rate": 1.0 / 3.0,
        "best_turn/groups_with_truncation_count": 1.0,
        "best_turn/groups_with_truncation_rate": 1.0,
        "best_turn/avg_truncations_per_group": 1.0,
        "best_turn/max_truncations_per_group": 1.0,
        "best_turn/drop_for_dp_divisor_count": 0.0,
        "best_turn/skip_best_is_final": 2.0,
    }

    accumulate_best_turn_epoch_stats(epoch_stats, step_one)
    accumulate_best_turn_epoch_stats(epoch_stats, step_two)
    epoch_metrics = build_best_turn_epoch_metrics(epoch_stats)

    assert epoch_metrics["best_turn_epoch/raw_rollout_count"] == 9.0
    assert epoch_metrics["best_turn_epoch/group_count"] == 3.0
    assert epoch_metrics["best_turn_epoch/split_count"] == 4.0
    assert epoch_metrics["best_turn_epoch/split_rate"] == pytest.approx(4.0 / 9.0)
    assert epoch_metrics["best_turn_epoch/groups_with_truncation_count"] == 3.0
    assert epoch_metrics["best_turn_epoch/groups_with_truncation_rate"] == 1.0
    assert epoch_metrics["best_turn_epoch/avg_truncations_per_group"] == pytest.approx(
        4.0 / 3.0
    )
    assert epoch_metrics["best_turn_epoch/max_truncations_per_group"] == 2.0
    assert epoch_metrics["best_turn_epoch/drop_for_dp_divisor_count"] == 1.0
    assert epoch_metrics["best_turn_epoch/skip_all_zero"] == 1.0
    assert epoch_metrics["best_turn_epoch/skip_best_is_final"] == 2.0


def test_final_reward_uses_recall20_not_recall10():
    score = FinalRewardFunction(
        data_source="qid-1",
        solution_str="",
        ground_truth={},
        extra_info={
            "performance_metrics": {
                "recall@10": 1.0,
                "recall@20": 0.25,
                "ndcg@20": 0.4,
                "mrr": 0.75,
            },
            "format_penalties": {
                "num_queries": 2,
                "dag_depth": 1,
                "num_refines": 0,
                "refine_actions_by_subquery": {},
                "breadth": 2,
                "hard_gate_failed": False,
            },
            "assistant_turns": 1,
            "num_turns": 3,
        },
        recall_weight=0.25,
        mrr_weight=0.75,
    )

    assert score["reward/performance"] == pytest.approx(0.25 * 0.25 + 0.75 * 0.75)
    assert score["perf/recall@10"] == 1.0
    assert score["perf/recall@20"] == 0.25
    assert score["perf/ndcg@20"] == 0.4


def test_final_reward_gap_decay_scores_prefix_and_full_variants():
    common_extra_info = {
        "performance_metrics": {
            "recall@10": 0.9,
            "recall@20": 0.25,
            "ndcg@20": 0.4,
            "mrr": 0.75,
        },
        "format_penalties": {
            "num_queries": 2,
            "dag_depth": 1,
            "num_refines": 0,
            "refine_actions_by_subquery": {},
            "breadth": 2,
            "hard_gate_failed": False,
        },
        "num_turns": 5,
        "best_turn_truncation": {
            "applied": True,
            "best_turn": 1,
        },
    }
    expected_reward = 0.25 * 0.25 + 0.75 * 0.75

    prefix_score = FinalRewardFunction(
        data_source="qid-1",
        solution_str="",
        ground_truth={},
        extra_info={
            **common_extra_info,
            "trajectory_variant": "best_prefix",
            "assistant_turns": 1,
        },
        use_gap_decay_score=True,
        gap_decay_gamma=0.9,
    )
    full_score = FinalRewardFunction(
        data_source="qid-1",
        solution_str="",
        ground_truth={},
        extra_info={
            **common_extra_info,
            "trajectory_variant": "full_original_after_best",
            "assistant_turns": 3,
        },
        use_gap_decay_score=True,
        gap_decay_gamma=0.9,
    )

    assert prefix_score["score"] == pytest.approx(expected_reward)
    assert prefix_score["reward/total"] == pytest.approx(expected_reward)
    assert prefix_score["base_reward"] == pytest.approx(expected_reward)
    assert prefix_score["reward/schema_applied"] == pytest.approx(1.0)
    assert prefix_score["reward/schema_gap"] == pytest.approx(0.0)
    assert prefix_score["reward/schema_gamma"] == pytest.approx(0.9)
    assert prefix_score["reward/schema_multiplier"] == pytest.approx(1.0)
    assert prefix_score["reward/schema_score"] == pytest.approx(expected_reward)

    assert full_score["score"] == pytest.approx(expected_reward * (0.9**2))
    assert full_score["reward/total"] == pytest.approx(expected_reward * (0.9**2))
    assert full_score["base_reward"] == pytest.approx(expected_reward)
    assert full_score["reward/schema_applied"] == pytest.approx(1.0)
    assert full_score["reward/schema_gap"] == pytest.approx(2.0)
    assert full_score["reward/schema_gamma"] == pytest.approx(0.9)
    assert full_score["reward/schema_multiplier"] == pytest.approx(0.9**2)
    assert full_score["reward/schema_score"] == pytest.approx(
        expected_reward * (0.9**2)
    )


def test_final_reward_gap_decay_disabled_uses_performance_reward():
    score = FinalRewardFunction(
        data_source="qid-1",
        solution_str="",
        ground_truth={},
        extra_info={
            "performance_metrics": {
                "recall@10": 1.0,
                "recall@20": 0.25,
                "ndcg@20": 0.4,
                "mrr": 0.75,
            },
            "format_penalties": {
                "num_queries": 2,
                "dag_depth": 1,
                "num_refines": 0,
                "refine_actions_by_subquery": {},
                "breadth": 2,
                "hard_gate_failed": False,
            },
            "assistant_turns": 3,
            "num_turns": 5,
            "trajectory_variant": "full_original_after_best",
            "best_turn_truncation": {
                "applied": True,
                "best_turn": 1,
            },
        },
        recall_weight=0.25,
        mrr_weight=0.75,
        use_gap_decay_score=False,
        gap_decay_gamma=0.9,
    )

    expected_reward = 0.25 * 0.25 + 0.75 * 0.75
    assert score["score"] == pytest.approx(expected_reward)
    assert score["reward/total"] == pytest.approx(expected_reward)
    assert score["base_reward"] == pytest.approx(expected_reward)
    assert score["reward/schema_applied"] == pytest.approx(0.0)
    assert score["reward/schema_multiplier"] == pytest.approx(1.0)


def test_final_reward_without_truncation_metadata_uses_performance_reward():
    score = FinalRewardFunction(
        data_source="qid-1",
        solution_str="",
        ground_truth={},
        extra_info={
            "performance_metrics": {
                "recall@10": 0.8,
                "recall@20": 0.4,
                "ndcg@20": 0.35,
                "mrr": 0.6,
            },
            "format_penalties": {
                "num_queries": 2,
                "dag_depth": 1,
                "num_refines": 0,
                "refine_actions_by_subquery": {},
                "breadth": 2,
                "hard_gate_failed": False,
            },
            "assistant_turns": 3,
            "num_turns": 5,
        },
        recall_weight=0.25,
        mrr_weight=0.75,
        use_gap_decay_score=True,
        gap_decay_gamma=0.9,
    )

    expected_reward = 0.4 * 0.25 + 0.6 * 0.75
    assert score["score"] == pytest.approx(expected_reward)
    assert score["reward/total"] == pytest.approx(expected_reward)
    assert score["base_reward"] == pytest.approx(expected_reward)
    assert score["reward/schema_applied"] == pytest.approx(0.0)
    assert score["reward/schema_gap"] == pytest.approx(0.0)
    assert score["reward/schema_multiplier"] == pytest.approx(1.0)
