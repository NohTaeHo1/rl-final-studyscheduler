import numpy as np

from src.study_scheduler_env import StudySchedulerEnv


def action_index(env: StudySchedulerEnv, counts: list[int]) -> int:
    for index, action in enumerate(env.actions):
        if action.tolist() == counts:
            return index
    raise AssertionError(f"action not found: {counts}")


def test_action_and_observation_shapes():
    env = StudySchedulerEnv(seed=0)
    obs = env.reset()

    assert len(env.actions) == 126
    assert env.obs_dim == 19
    assert obs.shape == (19,)
    assert obs.dtype == np.float32


def test_valid_step_returns_expected_types():
    env = StudySchedulerEnv(seed=0)
    obs = env.reset()
    next_obs, reward, done, info = env.step(0)

    assert obs.shape == next_obs.shape
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert "knowledge_after" in info
    assert "backlog_after" in info
    assert "coverage_gap_after" in info


def test_new_learning_creates_items():
    env = StudySchedulerEnv(seed=0)
    env.reset()
    action_with_four_new_blocks_for_subject0 = action_index(env, [4, 0, 0, 0, 0, 0])

    _, _, _, info = env.step(action_with_four_new_blocks_for_subject0)

    assert info["new_items"] == 32.0
    assert len(env.items) == 32
    assert all(item.subject_id == 0 for item in env.items)


def test_new_learning_stops_at_curriculum_size():
    env = StudySchedulerEnv(curriculum_size_per_subject=40, seed=0)
    env.reset()
    action_with_four_new_blocks_for_subject0 = action_index(env, [4, 0, 0, 0, 0, 0])

    env.step(action_with_four_new_blocks_for_subject0)
    _, _, _, info = env.step(action_with_four_new_blocks_for_subject0)

    assert info["new_items"] == 8.0
    assert len(env._subject_items(0)) == 40


def test_no_new_learning_has_coverage_gap_penalty():
    env = StudySchedulerEnv(seed=0)
    env.reset()
    review_only_action = action_index(env, [0, 4, 0, 0, 0, 0])

    _, reward, _, info = env.step(review_only_action)

    assert info["new_items"] == 0.0
    assert info["coverage_gap_after"] > 0.0
    assert reward < 0.0


def test_review_updates_item_history():
    env = StudySchedulerEnv(seed=0)
    env.reset()
    new_action = action_index(env, [4, 0, 0, 0, 0, 0])
    review_action = action_index(env, [0, 4, 0, 0, 0, 0])

    env.step(new_action)
    _, _, _, info = env.step(review_action)

    assert info["reviewed_items"] > 0
    assert sum(item.right + item.wrong for item in env.items) == info["reviewed_items"]


def test_short_rollout_terminates():
    env = StudySchedulerEnv(horizon_days=3, seed=0)
    env.reset()
    done = False
    steps = 0
    while not done:
        _, _, done, _ = env.step(0)
        steps += 1

    assert steps == 3
    assert env.day == 3
    metrics = env.metrics()
    assert "final_retained_knowledge" in metrics
    assert "final_coverage_ratio" in metrics
