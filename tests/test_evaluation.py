import pytest
from tensordict import TensorDict

from src.algo.actor import SquashedGaussianSACActor
from src.algo.evaluation import evaluate_policy
from src.envs import LeapCubeReorientEnv


def test_evaluate_policy_runs_deterministic():
    env = LeapCubeReorientEnv(max_steps=5)
    actor = SquashedGaussianSACActor(55, 4)

    metrics = evaluate_policy(actor, env, num_episodes=2, max_steps=5)

    expected_keys = {
        "success_rate",
        "mean_consecutive_successes",
        "mean_geodesic_error",
        "mean_return",
        "difficulty",
    }
    assert expected_keys.issubset(metrics.keys())
    assert 0.0 <= metrics["success_rate"] <= 1.0
    assert metrics["mean_consecutive_successes"] >= 0.0
    assert metrics["mean_geodesic_error"] >= 0.0
    assert isinstance(metrics["mean_return"], float)
    assert 0.0 <= metrics["difficulty"] <= 1.0


def test_evaluate_policy_tracks_curriculum_difficulty():
    env = LeapCubeReorientEnv(enable_domain_randomization=True, max_steps=5)
    assert env.curriculum_controller is not None
    env.curriculum_controller.set_difficulty(0.35)
    actor = SquashedGaussianSACActor(55, 4)

    metrics = evaluate_policy(actor, env, num_episodes=1, max_steps=3)
    assert metrics["difficulty"] == pytest.approx(0.35)


def test_evaluate_policy_detects_consecutive_success(monkeypatch):
    env = LeapCubeReorientEnv(max_steps=4)
    actor = SquashedGaussianSACActor(55, 4)

    call_count = {"resets": 0}
    orig_reset = env.reset
    orig_step = env.step

    def mock_reset(*args, **kwargs):
        call_count["resets"] += 1
        return orig_reset(*args, **kwargs)

    def mock_step(td: TensorDict):
        res = orig_step(td)
        if call_count["resets"] == 1:
            env.consecutive_successes = 2
        else:
            env.consecutive_successes = 0
        return res

    monkeypatch.setattr(env, "reset", mock_reset)
    monkeypatch.setattr(env, "step", mock_step)

    metrics = evaluate_policy(actor, env, num_episodes=2, max_steps=2)
    assert metrics["success_rate"] == pytest.approx(0.5)
    assert metrics["mean_consecutive_successes"] == pytest.approx(1.0)
