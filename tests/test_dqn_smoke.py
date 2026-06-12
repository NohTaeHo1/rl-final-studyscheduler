import argparse

import numpy as np

from experiments.train_dqn import evaluate, train
from src.dqn_agent import DQNAgent, ReplayBuffer
from src.study_scheduler_env import StudySchedulerEnv


def test_replay_buffer_samples_transitions():
    buffer = ReplayBuffer(capacity=10, seed=0)
    obs = np.zeros(3, dtype=np.float32)
    next_obs = np.ones(3, dtype=np.float32)

    for action in range(4):
        buffer.add(obs, action, reward=1.0, next_obs=next_obs, done=False)

    batch = buffer.sample(batch_size=2)
    assert len(batch) == 2
    assert all(transition.obs.shape == (3,) for transition in batch)


def test_dqn_agent_selects_valid_action_and_trains():
    env = StudySchedulerEnv(horizon_days=2, seed=0)
    agent = DQNAgent(
        obs_dim=env.obs_dim,
        n_actions=len(env.actions),
        batch_size=2,
        seed=0,
    )
    obs = env.reset()
    action = agent.select_action(obs, epsilon=0.0)
    assert 0 <= action < len(env.actions)

    for _ in range(2):
        next_obs, reward, done, _ = env.step(action)
        agent.replay.add(obs, action, reward, next_obs, done)
        obs = next_obs

    loss = agent.train_step()
    assert loss is not None
    assert loss >= 0.0


def test_short_dqn_train_and_eval_functions_run():
    args = argparse.Namespace(
        episodes=2,
        horizon=3,
        eval_episodes=1,
        seed=0,
        hidden_dim=32,
        lr=1e-3,
        gamma=0.95,
        batch_size=2,
        buffer_capacity=100,
        target_update_steps=2,
        epsilon_start=0.5,
        epsilon_end=0.1,
        device="cpu",
    )

    agent, train_rows = train(args)
    eval_rows = evaluate(agent=agent, horizon_days=3, episodes=1, seed=100)

    assert len(train_rows) == 2
    assert len(eval_rows) == 1
    assert "episode_return" in train_rows[0]
    assert "final_retained_knowledge" in eval_rows[0]
