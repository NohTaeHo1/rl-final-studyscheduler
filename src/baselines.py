from __future__ import annotations

from collections.abc import Callable

import numpy as np

from src.study_scheduler_env import StudySchedulerEnv


Policy = Callable[[StudySchedulerEnv, np.random.Generator], int]


def action_index(env: StudySchedulerEnv, counts: list[int]) -> int:
    for index, action in enumerate(env.actions):
        if action.tolist() == counts:
            return index
    raise ValueError(f"Invalid action counts: {counts}")


def random_policy(env: StudySchedulerEnv, rng: np.random.Generator) -> int:
    return int(rng.integers(0, len(env.actions)))


def equal_split_policy(env: StudySchedulerEnv, rng: np.random.Generator) -> int:
    del rng
    subject_offset = env.day % env.n_subjects
    counts = [0] * (2 * env.n_subjects)

    for block in range(env.blocks_per_day):
        subject_id = (subject_offset + block) % env.n_subjects
        is_review_block = block % 2 == 1
        counts[2 * subject_id + int(is_review_block)] += 1
    return action_index(env, counts)


def fixed_ratio_policy(env: StudySchedulerEnv, rng: np.random.Generator) -> int:
    del rng
    summaries = [env._subject_summary(subject_id) for subject_id in range(env.n_subjects)]
    least_covered = np.argsort([summary["n_items"] for summary in summaries])
    most_due = np.argsort(
        [summary["due_ratio"] * max(summary["n_items"], 1.0) for summary in summaries]
    )[::-1]

    counts = [0] * (2 * env.n_subjects)
    counts[2 * int(least_covered[0])] += 1
    counts[2 * int(least_covered[1])] += 1
    counts[2 * int(most_due[0]) + 1] += 1
    counts[2 * int(most_due[1]) + 1] += 1
    return action_index(env, counts)


def review_first_policy(env: StudySchedulerEnv, rng: np.random.Generator) -> int:
    del rng
    summaries = [env._subject_summary(subject_id) for subject_id in range(env.n_subjects)]
    due_pressure = [summary["due_ratio"] * max(summary["n_items"], 1.0) for summary in summaries]
    coverage = [summary["n_items"] for summary in summaries]

    review_order = np.argsort(due_pressure)[::-1]
    new_subject = int(np.argmin(coverage))
    counts = [0] * (2 * env.n_subjects)
    counts[2 * int(review_order[0]) + 1] += 2
    counts[2 * int(review_order[1]) + 1] += 1
    counts[2 * new_subject] += 1
    return action_index(env, counts)


def greedy_immediate_gain_policy(env: StudySchedulerEnv, rng: np.random.Generator) -> int:
    del rng
    summaries = [env._subject_summary(subject_id) for subject_id in range(env.n_subjects)]
    best_index = 0
    best_score = -float("inf")

    for index, action in enumerate(env.actions):
        score = 0.0
        for subject_id in range(env.n_subjects):
            new_blocks = int(action[2 * subject_id])
            review_blocks = int(action[2 * subject_id + 1])
            summary = summaries[subject_id]
            weight = float(env.subject_weights[subject_id])
            due_pressure = summary["due_ratio"] * max(summary["n_items"], 1.0)

            score += weight * new_blocks * env.items_per_new_block * 0.8
            score += weight * review_blocks * due_pressure * 1.5
            score += weight * review_blocks * (1.0 - summary["mean_recall"]) * env.items_per_review_block * 0.2
            score -= weight * max(0, new_blocks - review_blocks) * max(summary["n_items"] / 50.0, 0.0)

        if score > best_score:
            best_score = score
            best_index = index

    return best_index


BASELINE_POLICIES: dict[str, Policy] = {
    "random": random_policy,
    "equal_split": equal_split_policy,
    "fixed_ratio": fixed_ratio_policy,
    "review_first": review_first_policy,
    "greedy_immediate_gain": greedy_immediate_gain_policy,
}
