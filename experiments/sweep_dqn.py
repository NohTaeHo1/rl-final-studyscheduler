from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))

import numpy as np
import pandas as pd
import torch

from experiments.train_dqn import aggregate_evaluation, evaluate, plot_training, train, write_csv


DEFAULT_SWEEP_GRID: dict[str, list[float | int]] = {
    "lr": [1e-3, 5e-4, 1e-4],
    "gamma": [0.95, 0.99],
    "hidden_dim": [128, 256],
    "epsilon_end": [0.05, 0.10],
}


def build_grid(grid: dict[str, list[float | int]] | None = None) -> list[dict[str, float | int]]:
    grid = grid or DEFAULT_SWEEP_GRID
    keys = list(grid)
    configs: list[dict[str, float | int]] = []
    for values in itertools.product(*(grid[key] for key in keys)):
        config = dict(zip(keys, values, strict=True))
        configs.append(config)
    return configs


def args_for_config(
    config: dict[str, float | int],
    seed: int,
    episodes: int,
    horizon: int,
    eval_episodes: int,
    batch_size: int,
    buffer_capacity: int,
    target_update_steps: int,
    epsilon_start: float,
    device: str,
) -> argparse.Namespace:
    return argparse.Namespace(
        episodes=episodes,
        horizon=horizon,
        eval_episodes=eval_episodes,
        seed=seed,
        hidden_dim=int(config["hidden_dim"]),
        lr=float(config["lr"]),
        gamma=float(config["gamma"]),
        batch_size=batch_size,
        buffer_capacity=buffer_capacity,
        target_update_steps=target_update_steps,
        epsilon_start=epsilon_start,
        epsilon_end=float(config["epsilon_end"]),
        device=device,
    )


def add_config_fields(
    row: dict[str, Any],
    config_id: int,
    config: dict[str, float | int],
    train_seed: int,
) -> dict[str, Any]:
    out = dict(row)
    out["config_id"] = config_id
    out["train_seed"] = train_seed
    for key, value in config.items():
        out[key] = value
    return out


def run_config_seed(
    config_id: int,
    config: dict[str, float | int],
    train_seed: int,
    outdir: Path,
    episodes: int,
    horizon: int,
    eval_episodes: int,
    batch_size: int,
    buffer_capacity: int,
    target_update_steps: int,
    epsilon_start: float,
    device: str,
    save_model: bool,
) -> dict[str, Any]:
    run_dir = outdir / "runs" / f"config_{config_id:03d}_seed_{train_seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    args = args_for_config(
        config=config,
        seed=train_seed,
        episodes=episodes,
        horizon=horizon,
        eval_episodes=eval_episodes,
        batch_size=batch_size,
        buffer_capacity=buffer_capacity,
        target_update_steps=target_update_steps,
        epsilon_start=epsilon_start,
        device=device,
    )
    agent, train_rows = train(args)
    eval_rows = evaluate(
        agent=agent,
        horizon_days=horizon,
        episodes=eval_episodes,
        seed=train_seed + 10_000,
    )
    summary = aggregate_evaluation(eval_rows)[0]
    summary = add_config_fields(summary, config_id=config_id, config=config, train_seed=train_seed)

    write_csv(run_dir / "train_log.csv", train_rows)
    write_csv(run_dir / "evaluation_per_run.csv", eval_rows)
    write_csv(run_dir / "evaluation_summary.csv", [summary])
    plot_training(run_dir, train_rows)
    if save_model:
        torch.save(agent.q_net.state_dict(), run_dir / "dqn_policy.pt")
    return summary


def read_single_row_csv(path: Path) -> dict[str, Any]:
    with path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 1:
        raise ValueError(f"Expected exactly one row in {path}, found {len(rows)}.")
    return rows[0]


def coerce_numeric_fields(row: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in row.items():
        if value == "":
            out[key] = value
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            out[key] = value
            continue
        out[key] = int(number) if number.is_integer() else number
    return out


def completed_run_summary(run_dir: Path) -> dict[str, Any] | None:
    summary_path = run_dir / "evaluation_summary.csv"
    train_log_path = run_dir / "train_log.csv"
    eval_path = run_dir / "evaluation_per_run.csv"
    if not (summary_path.exists() and train_log_path.exists() and eval_path.exists()):
        return None
    return coerce_numeric_fields(read_single_row_csv(summary_path))


def select_best_config(summary_df: pd.DataFrame) -> dict[str, Any]:
    grouped = summarize_sweep_results(summary_df)
    return grouped.iloc[0].to_dict()


def summarize_sweep_results(results: pd.DataFrame) -> pd.DataFrame:
    aggregations: dict[str, tuple[str, str]] = {
        "final_retained_knowledge_mean": ("final_retained_knowledge", "mean"),
        "final_retained_knowledge_std": ("final_retained_knowledge", "std"),
        "final_backlog_mean": ("final_backlog", "mean"),
        "final_backlog_std": ("final_backlog", "std"),
        "episode_return_mean": ("episode_return", "mean"),
        "episode_return_std": ("episode_return", "std"),
        "lr": ("lr", "first"),
        "gamma": ("gamma", "first"),
        "hidden_dim": ("hidden_dim", "first"),
        "epsilon_end": ("epsilon_end", "first"),
    }
    if "final_coverage_ratio" in results.columns:
        aggregations["final_coverage_ratio_mean"] = ("final_coverage_ratio", "mean")
        aggregations["final_coverage_ratio_std"] = ("final_coverage_ratio", "std")
    if "final_coverage_gap" in results.columns:
        aggregations["final_coverage_gap_mean"] = ("final_coverage_gap", "mean")
        aggregations["final_coverage_gap_std"] = ("final_coverage_gap", "std")

    grouped = results.groupby("config_id", as_index=False).agg(**aggregations).fillna(0.0)
    return grouped.sort_values(
        by=["episode_return_mean", "final_retained_knowledge_mean", "final_backlog_mean"],
        ascending=[False, False, True],
    ).reset_index(drop=True)


def run_sweep(
    outdir: Path,
    configs: list[dict[str, float | int]],
    train_seeds: list[int],
    episodes: int,
    horizon: int,
    eval_episodes: int,
    batch_size: int = 64,
    buffer_capacity: int = 20_000,
    target_update_steps: int = 250,
    epsilon_start: float = 1.0,
    device: str = "cpu",
    save_models: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    outdir.mkdir(parents=True, exist_ok=True)
    with (outdir / "sweep_configs.json").open("w", encoding="utf-8") as f:
        json.dump(configs, f, indent=2, sort_keys=True)

    rows: list[dict[str, Any]] = []
    total = len(configs) * len(train_seeds)
    completed = 0
    for config_id, config in enumerate(configs):
        for train_seed in train_seeds:
            completed += 1
            run_dir = outdir / "runs" / f"config_{config_id:03d}_seed_{train_seed}"
            existing_summary = completed_run_summary(run_dir)
            if existing_summary is not None:
                print(
                    f"[sweep {completed}/{total}] skip completed config={config_id} seed={train_seed}",
                    flush=True,
                )
                rows.append(existing_summary)
                pd.DataFrame(rows).to_csv(outdir / "sweep_results_partial.csv", index=False)
                continue

            print(f"[sweep {completed}/{total}] config={config_id} seed={train_seed} {config}", flush=True)
            row = run_config_seed(
                config_id=config_id,
                config=config,
                train_seed=train_seed,
                outdir=outdir,
                episodes=episodes,
                horizon=horizon,
                eval_episodes=eval_episodes,
                batch_size=batch_size,
                buffer_capacity=buffer_capacity,
                target_update_steps=target_update_steps,
                epsilon_start=epsilon_start,
                device=device,
                save_model=save_models,
            )
            rows.append(row)
            pd.DataFrame(rows).to_csv(outdir / "sweep_results_partial.csv", index=False)

    results = pd.DataFrame(rows)
    results.to_csv(outdir / "sweep_results.csv", index=False)

    grouped = summarize_sweep_results(results)
    grouped.to_csv(outdir / "sweep_grouped.csv", index=False)

    best = grouped.iloc[0].to_dict()
    with (outdir / "best_config.json").open("w", encoding="utf-8") as f:
        json.dump(best, f, indent=2, sort_keys=True)
    return results, best


def write_final_summary(outdir: Path, results: pd.DataFrame) -> pd.DataFrame:
    summary_row = {
        "policy": "dqn_best",
        "n_train_seeds": int(len(results)),
        "final_retained_knowledge": float(results["final_retained_knowledge"].mean()),
        "final_retained_knowledge_std": float(results["final_retained_knowledge"].std(ddof=1))
        if len(results) > 1
        else 0.0,
        "final_backlog": float(results["final_backlog"].mean()),
        "final_backlog_std": float(results["final_backlog"].std(ddof=1)) if len(results) > 1 else 0.0,
        "episode_return": float(results["episode_return"].mean()),
        "episode_return_std": float(results["episode_return"].std(ddof=1)) if len(results) > 1 else 0.0,
        "retained_knowledge_auc": float(results["retained_knowledge_auc"].mean()),
        "retained_knowledge_auc_std": float(results["retained_knowledge_auc"].std(ddof=1))
        if len(results) > 1
        else 0.0,
    }
    if "final_coverage_ratio" in results.columns:
        summary_row["final_coverage_ratio"] = float(results["final_coverage_ratio"].mean())
        summary_row["final_coverage_ratio_std"] = (
            float(results["final_coverage_ratio"].std(ddof=1)) if len(results) > 1 else 0.0
        )
    if "final_coverage_gap" in results.columns:
        summary_row["final_coverage_gap"] = float(results["final_coverage_gap"].mean())
        summary_row["final_coverage_gap_std"] = (
            float(results["final_coverage_gap"].std(ddof=1)) if len(results) > 1 else 0.0
        )
    for key in ["lr", "gamma", "hidden_dim", "epsilon_end"]:
        if key in results.columns:
            summary_row[key] = results[key].iloc[0]

    summary = pd.DataFrame([summary_row])
    summary.to_csv(outdir / "final_dqn_summary.csv", index=False)
    return summary


def run_final_evaluation(
    outdir: Path,
    best_config: dict[str, Any],
    train_seeds: list[int],
    episodes: int,
    horizon: int,
    eval_episodes: int,
    batch_size: int = 64,
    buffer_capacity: int = 20_000,
    target_update_steps: int = 250,
    epsilon_start: float = 1.0,
    device: str = "cpu",
) -> pd.DataFrame:
    outdir.mkdir(parents=True, exist_ok=True)
    config = {
        "lr": float(best_config["lr"]),
        "gamma": float(best_config["gamma"]),
        "hidden_dim": int(best_config["hidden_dim"]),
        "epsilon_end": float(best_config["epsilon_end"]),
    }
    rows: list[dict[str, Any]] = []
    for train_seed in train_seeds:
        run_dir = outdir / "runs" / f"config_{int(best_config.get('config_id', 0)):03d}_seed_{train_seed}"
        existing_summary = completed_run_summary(run_dir)
        if existing_summary is not None:
            print(f"[final] skip completed seed={train_seed}", flush=True)
            rows.append(existing_summary)
            pd.DataFrame(rows).to_csv(outdir / "final_dqn_results_partial.csv", index=False)
            continue

        print(f"[final] seed={train_seed} {config}", flush=True)
        row = run_config_seed(
            config_id=int(best_config.get("config_id", 0)),
            config=config,
            train_seed=train_seed,
            outdir=outdir,
            episodes=episodes,
            horizon=horizon,
            eval_episodes=eval_episodes,
            batch_size=batch_size,
            buffer_capacity=buffer_capacity,
            target_update_steps=target_update_steps,
            epsilon_start=epsilon_start,
            device=device,
            save_model=True,
        )
        rows.append(row)
        pd.DataFrame(rows).to_csv(outdir / "final_dqn_results_partial.csv", index=False)

    results = pd.DataFrame(rows)
    results.to_csv(outdir / "final_dqn_results.csv", index=False)
    write_final_summary(outdir, results)
    with (outdir / "best_config_used.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, sort_keys=True)
    return results


def create_comparison_summary(
    baseline_csv: Path,
    final_dqn_results_csv: Path,
    outdir: Path,
) -> pd.DataFrame:
    baseline = pd.read_csv(baseline_csv)
    dqn = pd.read_csv(final_dqn_results_csv)
    dqn_summary = write_final_summary(final_dqn_results_csv.parent, dqn).iloc[0].to_dict()

    comparison_cols = [
        "policy",
        "final_retained_knowledge",
        "final_retained_knowledge_std",
        "final_backlog",
        "final_backlog_std",
        "episode_return",
        "episode_return_std",
        "retained_knowledge_auc",
        "retained_knowledge_auc_std",
    ]
    optional_cols = [
        "final_coverage_ratio",
        "final_coverage_ratio_std",
        "final_coverage_gap",
        "final_coverage_gap_std",
    ]
    for col in optional_cols:
        if col in baseline.columns and col in dqn_summary:
            comparison_cols.append(col)
    comparison = pd.concat(
        [baseline[comparison_cols], pd.DataFrame([dqn_summary])[comparison_cols]],
        ignore_index=True,
    )
    comparison = comparison.sort_values(
        by=["final_retained_knowledge", "final_backlog"],
        ascending=[False, True],
    )
    outdir.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(outdir / "baseline_vs_dqn_summary.csv", index=False)
    return comparison


def write_run_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)


def make_results_zip(results_root: Path, zip_path: Path) -> Path:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    if zip_path.suffix == ".zip":
        archive_base = zip_path.with_suffix("")
    else:
        archive_base = zip_path
    archive = shutil.make_archive(str(archive_base), "zip", root_dir=results_root)
    return Path(archive)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DQN hyperparameter sweep and final evaluation.")
    parser.add_argument("--outdir", default="results/dqn_sweep")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--horizon", type=int, default=365)
    parser.add_argument("--eval-episodes", type=int, default=3)
    parser.add_argument("--train-seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--final-train-seeds", type=int, nargs="+", default=[100, 101, 102, 103, 104])
    parser.add_argument("--final-episodes", type=int, default=150)
    parser.add_argument("--final-eval-episodes", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--buffer-capacity", type=int, default=20_000)
    parser.add_argument("--target-update-steps", type=int, default=250)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-configs", type=int, default=0, help="0 means use the full 24-config grid.")
    args = parser.parse_args()

    configs = build_grid()
    if args.max_configs > 0:
        configs = configs[: args.max_configs]

    outdir = Path(args.outdir)
    write_run_manifest(
        outdir / "run_manifest.json",
        {
            "horizon": args.horizon,
            "experiment_version": EXPERIMENT_VERSION,
            "selection_metric": "episode_return_mean",
            "reward_fix": "coverage_gap_penalty prevents no-learning policies from receiving zero penalty.",
            "tuning": {
                "configs": len(configs),
                "episodes": args.episodes,
                "eval_episodes": args.eval_episodes,
                "train_seeds": args.train_seeds,
            },
            "final": {
                "episodes": args.final_episodes,
                "eval_episodes": args.final_eval_episodes,
                "train_seeds": args.final_train_seeds,
            },
            "batch_size": args.batch_size,
            "buffer_capacity": args.buffer_capacity,
            "target_update_steps": args.target_update_steps,
            "epsilon_start": args.epsilon_start,
            "device": args.device,
        },
    )
    print(f"device={args.device}")
    print(f"configs={len(configs)} train_seeds={args.train_seeds}")
    _, best = run_sweep(
        outdir=outdir / "tuning",
        configs=configs,
        train_seeds=args.train_seeds,
        episodes=args.episodes,
        horizon=args.horizon,
        eval_episodes=args.eval_episodes,
        batch_size=args.batch_size,
        buffer_capacity=args.buffer_capacity,
        target_update_steps=args.target_update_steps,
        epsilon_start=args.epsilon_start,
        device=args.device,
        save_models=False,
    )
    print(f"best_config={best}")
    run_final_evaluation(
        outdir=outdir / "final",
        best_config=best,
        train_seeds=args.final_train_seeds,
        episodes=args.final_episodes,
        horizon=args.horizon,
        eval_episodes=args.final_eval_episodes,
        batch_size=args.batch_size,
        buffer_capacity=args.buffer_capacity,
        target_update_steps=args.target_update_steps,
        epsilon_start=args.epsilon_start,
        device=args.device,
    )
    archive = make_results_zip(outdir, outdir.with_suffix(".zip"))
    print(f"zipped results: {archive}")


if __name__ == "__main__":
    main()
