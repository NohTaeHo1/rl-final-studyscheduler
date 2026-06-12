from pathlib import Path

import pandas as pd

from experiments.sweep_dqn import build_grid, create_comparison_summary, run_final_evaluation, run_sweep
from experiments.run_baselines import aggregate, write_csv


def test_tiny_sweep_and_final_eval_create_outputs(tmp_path: Path):
    configs = build_grid(
        {
            "lr": [1e-3],
            "gamma": [0.95],
            "hidden_dim": [32],
            "epsilon_end": [0.10],
        }
    )
    sweep_dir = tmp_path / "sweep"
    _, best = run_sweep(
        outdir=sweep_dir,
        configs=configs,
        train_seeds=[0],
        episodes=1,
        horizon=2,
        eval_episodes=1,
        batch_size=2,
        buffer_capacity=20,
        target_update_steps=2,
        device="cpu",
    )
    resumed_results, resumed_best = run_sweep(
        outdir=sweep_dir,
        configs=configs,
        train_seeds=[0],
        episodes=1,
        horizon=2,
        eval_episodes=1,
        batch_size=2,
        buffer_capacity=20,
        target_update_steps=2,
        device="cpu",
    )

    final_dir = tmp_path / "final"
    final_results = run_final_evaluation(
        outdir=final_dir,
        best_config=best,
        train_seeds=[10],
        episodes=1,
        horizon=2,
        eval_episodes=1,
        batch_size=2,
        buffer_capacity=20,
        target_update_steps=2,
        device="cpu",
    )
    resumed_final_results = run_final_evaluation(
        outdir=final_dir,
        best_config=best,
        train_seeds=[10],
        episodes=1,
        horizon=2,
        eval_episodes=1,
        batch_size=2,
        buffer_capacity=20,
        target_update_steps=2,
        device="cpu",
    )

    baseline_rows = [
        {
            "policy": "baseline",
            "seed": 0,
            "final_retained_knowledge": 1.0,
            "final_backlog": 2.0,
            "episode_return": 3.0,
            "retained_knowledge_auc": 4.0,
        }
    ]
    baseline_csv = tmp_path / "baseline_metrics_aggregated.csv"
    write_csv(baseline_csv, aggregate(baseline_rows))
    comparison = create_comparison_summary(
        baseline_csv=baseline_csv,
        final_dqn_results_csv=final_dir / "final_dqn_results.csv",
        outdir=tmp_path / "comparison",
    )

    assert (sweep_dir / "sweep_results.csv").exists()
    assert (sweep_dir / "sweep_grouped.csv").exists()
    assert (sweep_dir / "best_config.json").exists()
    assert (final_dir / "final_dqn_results.csv").exists()
    assert (final_dir / "final_dqn_summary.csv").exists()
    assert (tmp_path / "comparison" / "baseline_vs_dqn_summary.csv").exists()
    assert len(resumed_results) == 1
    assert resumed_best["config_id"] == best["config_id"]
    assert len(final_results) == 1
    assert len(resumed_final_results) == 1
    assert len(comparison) == 2
    assert len(pd.read_csv(final_dir / "final_dqn_summary.csv")) == 1
