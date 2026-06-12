import numpy as np

from src.baselines import BASELINE_POLICIES
from src.study_scheduler_env import StudySchedulerEnv


def test_all_baselines_return_valid_actions():
    env = StudySchedulerEnv(seed=0)
    rng = np.random.default_rng(0)

    for policy_name, policy in BASELINE_POLICIES.items():
        env.reset()
        action = policy(env, rng)
        assert isinstance(action, int), policy_name
        assert 0 <= action < len(env.actions), policy_name
        assert env.decode_action(action).sum() == env.blocks_per_day


def test_baselines_can_step_environment():
    rng = np.random.default_rng(0)

    for policy_name, policy in BASELINE_POLICIES.items():
        env = StudySchedulerEnv(horizon_days=3, seed=0)
        env.reset()
        done = False
        while not done:
            action = policy(env, rng)
            _, _, done, info = env.step(action)

        assert env.day == 3, policy_name
        assert "knowledge_after" in info

