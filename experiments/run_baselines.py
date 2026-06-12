from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.baselines import BASELINE_POLICIES
from src.study_scheduler_env import StudySchedulerEnv


def run_policy(policy_name: str, seed: int, horizon_days: int) -> dict[str, float | int | str]:
    rng = np.random.default_rng(seed)
    env = StudySchedulerEnv(horizon_days=horizon_days, seed=seed)
    policy = BASELINE_POLICIES[policy_name]

    env.reset()
    done = False
    rewards: list[float] = []
    while not done:
        action = policy(env, rng)
        _, reward, done, _ = env.step(action)
        rewards.append(reward)

    row: dict[str, float | int | str] = env.metrics()
    row["episode_return"] = float(np.sum(rewards))
    row["policy"] = policy_name
    row["seed"] = seed
    return row


def aggregate(rows: list[dict[str, float | int | str]]) -> list[dict[str, float | str]]:
    policies = sorted({str(row["policy"]) for row in rows})
    metric_keys = [
        key
        for key, value in rows[0].items()
        if key not in {"policy", "seed"} and isinstance(value, (int, float))
    ]

    out: list[dict[str, float | str]] = []
    for policy in policies:
        policy_rows = [row for row in rows if row["policy"] == policy]
        agg_row: dict[str, float | str] = {"policy": policy}
        for key in metric_keys:
            values = np.asarray([float(row[key]) for row in policy_rows], dtype=np.float64)
            agg_row[key] = float(values.mean())
            agg_row[f"{key}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        out.append(agg_row)
    return out


def write_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_aggregates(outdir: Path, aggregate_rows: list[dict[str, float | str]]) -> None:
    metrics = [
        "final_retained_knowledge",
        "final_backlog",
        "retained_knowledge_auc",
        "episode_return",
    ]
    labels = [str(row["policy"]) for row in aggregate_rows]
    for metric in metrics:
        means = [float(row[metric]) for row in aggregate_rows]
        stds = [float(row[f"{metric}_std"]) for row in aggregate_rows]

        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.bar(labels, means, yerr=stds, capsize=4)
        ax.set_title(metric)
        ax.set_ylabel(metric)
        ax.tick_params(axis="x", rotation=20)
        fig.tight_layout()
        fig.savefig(outdir / f"{metric}.png", dpi=160)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run baseline policies for Study Scheduler RL.")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--horizon", type=int, default=365)
    parser.add_argument("--outdir", default="results/baselines")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, float | int | str]] = []
    for policy_name in BASELINE_POLICIES:
        for seed in range(args.episodes):
            rows.append(run_policy(policy_name=policy_name, seed=seed, horizon_days=args.horizon))

    aggregate_rows = aggregate(rows)
    write_csv(outdir / "baseline_metrics_per_run.csv", rows)
    write_csv(outdir / "baseline_metrics_aggregated.csv", aggregate_rows)
    plot_aggregates(outdir, aggregate_rows)

    print(f"saved {len(rows)} raw rows to {outdir / 'baseline_metrics_per_run.csv'}")
    print(f"saved {len(aggregate_rows)} aggregate rows to {outdir / 'baseline_metrics_aggregated.csv'}")


if __name__ == "__main__":
    main()
