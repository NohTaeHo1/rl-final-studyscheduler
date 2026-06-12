from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Any

import numpy as np


MIN_HALF_LIFE_DAYS = 15.0 / (24.0 * 60.0)
MAX_HALF_LIFE_DAYS = 274.0


@dataclass
class MemoryParams:
    """Small, project-designed half-life memory model parameters."""

    bias: float = -0.90
    w_right: float = 0.55
    w_wrong: float = -0.35
    subject_biases: tuple[float, float, float] = (0.10, 0.00, -0.10)


@dataclass
class Item:
    subject_id: int
    introduced_day: int
    last_review_day: int
    right: int = 0
    wrong: int = 0


class StudySchedulerEnv:
    """Minimal study scheduling environment for smoke-tested MDP experiments."""

    def __init__(
        self,
        subject_names: tuple[str, str, str] = ("SubjectA", "SubjectB", "SubjectC"),
        subject_weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
        memory_params: MemoryParams | None = None,
        horizon_days: int = 365,
        blocks_per_day: int = 4,
        block_minutes: int = 15,
        items_per_new_block: int = 8,
        items_per_review_block: int = 24,
        curriculum_size_per_subject: int = 160,
        target_recall: float = 0.70,
        backlog_penalty: float = 0.08,
        coverage_gap_penalty: float = 0.02,
        seed: int = 0,
    ) -> None:
        if len(subject_names) != 3:
            raise ValueError("This project scope uses exactly 3 subjects.")
        if len(subject_weights) != 3:
            raise ValueError("subject_weights must contain exactly 3 values.")

        self.subject_names = subject_names
        self.n_subjects = 3
        weights = np.asarray(subject_weights, dtype=np.float32)
        self.subject_weights = weights / max(float(weights.sum()), 1e-8)
        self.memory_params = memory_params or MemoryParams()
        self.horizon_days = horizon_days
        self.blocks_per_day = blocks_per_day
        self.block_minutes = block_minutes
        self.items_per_new_block = items_per_new_block
        self.items_per_review_block = items_per_review_block
        self.curriculum_size_per_subject = curriculum_size_per_subject
        self.target_recall = target_recall
        self.backlog_penalty = backlog_penalty
        self.coverage_gap_penalty = coverage_gap_penalty
        self.rng = np.random.default_rng(seed)

        self.actions = self._enumerate_actions(total_blocks=blocks_per_day, n_bins=2 * self.n_subjects)
        self.obs_dim = self.n_subjects * 6 + 1
        self.reset()

    @staticmethod
    def _enumerate_actions(total_blocks: int, n_bins: int) -> list[np.ndarray]:
        actions: list[np.ndarray] = []
        for bars in itertools.combinations(range(total_blocks + n_bins - 1), n_bins - 1):
            counts = []
            previous = -1
            for bar in bars:
                counts.append(bar - previous - 1)
                previous = bar
            counts.append(total_blocks + n_bins - 1 - previous - 1)
            actions.append(np.asarray(counts, dtype=np.int64))
        return actions

    def reset(self) -> np.ndarray:
        self.day = 0
        self.items: list[Item] = []
        self.items_by_subject: list[list[Item]] = [[] for _ in range(self.n_subjects)]
        self.history: list[dict[str, float]] = []
        return self._get_obs()

    def decode_action(self, action_index: int) -> np.ndarray:
        if action_index < 0 or action_index >= len(self.actions):
            raise IndexError(f"action_index {action_index} outside [0, {len(self.actions)})")
        return self.actions[action_index].copy()

    def item_lag_days(self, item: Item, day: int | None = None) -> int:
        current_day = self.day if day is None else day
        return max(0, current_day - item.last_review_day)

    def item_half_life(self, item: Item) -> float:
        params = self.memory_params
        log2_h = (
            params.bias
            + params.w_right * math.sqrt(1.0 + item.right)
            + params.w_wrong * math.sqrt(1.0 + item.wrong)
            + params.subject_biases[item.subject_id]
        )
        return float(np.clip(2.0**log2_h, MIN_HALF_LIFE_DAYS, MAX_HALF_LIFE_DAYS))

    def item_recall_prob(self, item: Item, day: int | None = None) -> float:
        lag = self.item_lag_days(item, day=day)
        half_life = self.item_half_life(item)
        p_recall = 2.0 ** (-lag / half_life)
        return float(np.clip(p_recall, 1e-4, 0.9999))

    def _subject_items(self, subject_id: int) -> list[Item]:
        return self.items_by_subject[subject_id]

    def _subject_summary(self, subject_id: int) -> dict[str, float]:
        subject_items = self._subject_items(subject_id)
        if not subject_items:
            base_log2_h = self.memory_params.bias + self.memory_params.subject_biases[subject_id]
            return {
                "n_items": 0.0,
                "mean_recall": 0.0,
                "due_ratio": 0.0,
                "mean_log2_half_life": base_log2_h,
                "mean_lag": 0.0,
            }

        probs = np.asarray([self.item_recall_prob(item) for item in subject_items], dtype=np.float32)
        half_lives = np.asarray([self.item_half_life(item) for item in subject_items], dtype=np.float32)
        lags = np.asarray([self.item_lag_days(item) for item in subject_items], dtype=np.float32)
        due = probs < self.target_recall
        return {
            "n_items": float(len(subject_items)),
            "mean_recall": float(probs.mean()),
            "due_ratio": float(due.mean()),
            "mean_log2_half_life": float(np.log2(half_lives).mean()),
            "mean_lag": float(lags.mean()),
        }

    def _get_obs(self) -> np.ndarray:
        features: list[float] = []
        for subject_id in range(self.n_subjects):
            summary = self._subject_summary(subject_id)
            features.extend(
                [
                    summary["n_items"] / self.curriculum_size_per_subject,
                    summary["mean_recall"],
                    summary["due_ratio"],
                    summary["mean_log2_half_life"] / 8.0,
                    summary["mean_lag"] / 30.0,
                    float(self.subject_weights[subject_id]),
                ]
            )
        features.append(self.day / self.horizon_days)
        return np.asarray(features, dtype=np.float32)

    def retained_knowledge(self) -> float:
        total = 0.0
        for subject_id in range(self.n_subjects):
            subject_items = self._subject_items(subject_id)
            if not subject_items:
                continue
            probs = [self.item_recall_prob(item) for item in subject_items]
            total += float(self.subject_weights[subject_id]) * float(np.sum(probs))
        return total

    def backlog(self) -> float:
        total = 0.0
        for subject_id in range(self.n_subjects):
            subject_items = self._subject_items(subject_id)
            if not subject_items:
                continue
            due_count = sum(self.item_recall_prob(item) < self.target_recall for item in subject_items)
            total += float(self.subject_weights[subject_id]) * float(due_count)
        return total

    def coverage_gap(self) -> float:
        total = 0.0
        for subject_id in range(self.n_subjects):
            introduced = len(self._subject_items(subject_id))
            remaining = max(0, self.curriculum_size_per_subject - introduced)
            total += float(self.subject_weights[subject_id]) * float(remaining)
        return total

    def coverage_ratio(self) -> float:
        total_capacity = self.n_subjects * self.curriculum_size_per_subject
        if total_capacity <= 0:
            return 1.0
        return float(len(self.items) / total_capacity)

    def _add_new_items(self, subject_id: int, n_blocks: int) -> int:
        remaining = self.curriculum_size_per_subject - len(self._subject_items(subject_id))
        n_items = min(n_blocks * self.items_per_new_block, max(0, remaining))
        for _ in range(n_items):
            item = Item(subject_id=subject_id, introduced_day=self.day, last_review_day=self.day)
            self.items.append(item)
            self.items_by_subject[subject_id].append(item)
        return n_items

    def _review_items(self, subject_id: int, n_blocks: int) -> tuple[int, int, int]:
        n_slots = n_blocks * self.items_per_review_block
        if n_slots <= 0:
            return 0, 0, 0
        candidates = self._subject_items(subject_id)
        if not candidates:
            return 0, 0, 0

        candidates.sort(key=self.item_recall_prob)
        review_set = candidates[: min(n_slots, len(candidates))]
        successes = 0
        failures = 0
        for item in review_set:
            if self.rng.random() < self.item_recall_prob(item):
                item.right += 1
                successes += 1
            else:
                item.wrong += 1
                failures += 1
            item.last_review_day = self.day
        return len(review_set), successes, failures

    def step(self, action_index: int) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        action = self.decode_action(action_index)
        knowledge_before = self.retained_knowledge()
        backlog_before = self.backlog()
        coverage_gap_before = self.coverage_gap()

        new_items = 0
        reviewed_items = 0
        review_successes = 0
        review_failures = 0
        for subject_id in range(self.n_subjects):
            new_blocks = int(action[2 * subject_id])
            review_blocks = int(action[2 * subject_id + 1])
            new_items += self._add_new_items(subject_id, new_blocks)
            reviewed, successes, failures = self._review_items(subject_id, review_blocks)
            reviewed_items += reviewed
            review_successes += successes
            review_failures += failures

        self.day += 1
        knowledge_after = self.retained_knowledge()
        backlog_after = self.backlog()
        coverage_gap_after = self.coverage_gap()
        reward = (
            (knowledge_after - knowledge_before)
            - self.backlog_penalty * backlog_after
            - self.coverage_gap_penalty * coverage_gap_after
        )
        done = self.day >= self.horizon_days

        info = {
            "day": float(self.day),
            "knowledge_before": knowledge_before,
            "knowledge_after": knowledge_after,
            "backlog_before": backlog_before,
            "backlog_after": backlog_after,
            "coverage_gap_before": coverage_gap_before,
            "coverage_gap_after": coverage_gap_after,
            "coverage_ratio": self.coverage_ratio(),
            "new_items": float(new_items),
            "reviewed_items": float(reviewed_items),
            "review_successes": float(review_successes),
            "review_failures": float(review_failures),
        }
        self.history.append(info)
        return self._get_obs(), float(reward), done, info

    def metrics(self) -> dict[str, float]:
        metrics = {
            "final_retained_knowledge": self.retained_knowledge(),
            "final_backlog": self.backlog(),
            "final_coverage_gap": self.coverage_gap(),
            "final_coverage_ratio": self.coverage_ratio(),
            "retained_knowledge_auc": float(sum(row["knowledge_after"] for row in self.history)),
            "coverage_ratio_auc": float(sum(row["coverage_ratio"] for row in self.history)),
        }
        for subject_id, subject_name in enumerate(self.subject_names):
            summary = self._subject_summary(subject_id)
            metrics[f"{subject_name}_n_items"] = summary["n_items"]
            metrics[f"{subject_name}_mean_recall"] = summary["mean_recall"]
            metrics[f"{subject_name}_due_ratio"] = summary["due_ratio"]
        return metrics
