from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.dqn_agent import DQNAgent
from src.study_scheduler_env import StudySchedulerEnv


def linear_epsilon(
    episode: int,
    total_episodes: int,
    epsilon_start: float,
    epsilon_end: float,
) -> float:
    if total_episodes <= 1:
        return epsilon_end
    progress = min(1.0, episode / float(total_episodes - 1))
    return epsilon_start + progress * (epsilon_end - epsilon_start)


def write_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def train(args: argparse.Namespace) -> tuple[DQNAgent, list[dict[str, float | int | str]]]:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    probe_env = StudySchedulerEnv(horizon_days=args.horizon, seed=args.seed)
    agent = DQNAgent(
        obs_dim=probe_env.obs_dim,
        n_actions=len(probe_env.actions),
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        gamma=args.gamma,
        batch_size=args.batch_size,
        buffer_capacity=args.buffer_capacity,
        seed=args.seed,
        device=args.device,
    )

    rows: list[dict[str, float | int | str]] = []
    global_step = 0
    for episode in range(args.episodes):
        epsilon = linear_epsilon(
            episode=episode,
            total_episodes=args.episodes,
            epsilon_start=args.epsilon_start,
            epsilon_end=args.epsilon_end,
        )
        env = StudySchedulerEnv(horizon_days=args.horizon, seed=args.seed + episode)
        obs = env.reset()
        done = False
        episode_return = 0.0
        losses: list[float] = []

        while not done:
            action = agent.select_action(obs, epsilon=epsilon)
            next_obs, reward, done, _ = env.step(action)
            agent.replay.add(obs, action, reward, next_obs, done)
            loss = agent.train_step()
            if loss is not None:
                losses.append(loss)

            obs = next_obs
            episode_return += reward
            global_step += 1
            if global_step % args.target_update_steps == 0:
                agent.update_target()

        row: dict[str, float | int | str] = env.metrics()
        row["episode"] = episode
        row["episode_return"] = float(episode_return)
        row["epsilon"] = float(epsilon)
        row["mean_loss"] = float(np.mean(losses)) if losses else 0.0
        rows.append(row)

    agent.update_target()
    return agent, rows


def evaluate(
    agent: DQNAgent,
    horizon_days: int,
    episodes: int,
    seed: int,
) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for episode in range(episodes):
        env = StudySchedulerEnv(horizon_days=horizon_days, seed=seed + episode)
        obs = env.reset()
        done = False
        episode_return = 0.0
        while not done:
            action = agent.select_action(obs, epsilon=0.0)
            obs, reward, done, _ = env.step(action)
            episode_return += reward

        row: dict[str, float | int | str] = env.metrics()
        row["episode"] = episode
        row["episode_return"] = float(episode_return)
        rows.append(row)
    return rows


def aggregate_evaluation(rows: list[dict[str, float | int | str]]) -> list[dict[str, float | str]]:
    metric_keys = [key for key, value in rows[0].items() if isinstance(value, (int, float))]
    out: dict[str, float | str] = {"policy": "dqn"}
    for key in metric_keys:
        values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
        out[key] = float(values.mean())
        out[f"{key}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    return [out]


def plot_training(outdir: Path, rows: list[dict[str, float | int | str]]) -> None:
    episodes = [int(row["episode"]) for row in rows]
    returns = [float(row["episode_return"]) for row in rows]
    losses = [float(row["mean_loss"]) for row in rows]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(episodes, returns)
    ax.set_title("DQN episode return")
    ax.set_xlabel("episode")
    ax.set_ylabel("return")
    fig.tight_layout()
    fig.savefig(outdir / "training_return.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(episodes, losses)
    ax.set_title("DQN mean loss")
    ax.set_xlabel("episode")
    ax.set_ylabel("mean loss")
    fig.tight_layout()
    fig.savefig(outdir / "training_loss.png", dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a small DQN on the study scheduler MDP.")
    parser.add_argument("--episodes", type=int, default=80)
    parser.add_argument("--horizon", type=int, default=365)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--buffer-capacity", type=int, default=20_000)
    parser.add_argument("--target-update-steps", type=int, default=250)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--outdir", default="results/dqn")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    agent, train_rows = train(args)
    eval_rows = evaluate(
        agent=agent,
        horizon_days=args.horizon,
        episodes=args.eval_episodes,
        seed=args.seed + 10_000,
    )
    eval_summary = aggregate_evaluation(eval_rows)

    write_csv(outdir / "train_log.csv", train_rows)
    write_csv(outdir / "evaluation_per_run.csv", eval_rows)
    write_csv(outdir / "evaluation_summary.csv", eval_summary)
    torch.save(agent.q_net.state_dict(), outdir / "dqn_policy.pt")
    plot_training(outdir, train_rows)

    print(f"saved train log to {outdir / 'train_log.csv'}")
    print(f"saved evaluation summary to {outdir / 'evaluation_summary.csv'}")
    print(f"saved model checkpoint to {outdir / 'dqn_policy.pt'}")


if __name__ == "__main__":
    main()
