import sys
from pathlib import Path

import numpy as np
import torch

# Ensure repository root is on sys.path for direct pytest invocation
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.algo import (
    PolicyModule,
    SquashedGaussianSACActor,
    create_her_replay_buffer,
    create_trajectory_collector,
)
from src.envs import LeapCubeReorientEnv


def test_default_max_steps_is_400():
    env = LeapCubeReorientEnv()
    assert env.max_steps == 400


def test_extended_horizon_400_steps():
    env = LeapCubeReorientEnv(max_steps=400, seed=42)
    td = env.reset()

    assert env.current_step == 0
    assert env.max_steps == 400

    current_td = td
    for step_idx in range(1, 401):
        current_td["action"] = torch.zeros(16)
        step_out = env.step(current_td)
        next_td = step_out["next"]

        # Check termination & truncation flags
        terminated = bool(next_td["terminated"].item())
        truncated = bool(next_td["truncated"].item())
        done = bool(next_td["done"].item())

        assert not terminated, f"Step {step_idx}: terminated should be False (no suicide drop)"

        if step_idx < 400:
            assert not truncated, f"Step {step_idx}: truncated should be False before step 400"
            assert not done, f"Step {step_idx}: done should be False before step 400"
        else:
            assert truncated, "Step 400: truncated must be True at max_steps horizon"
            assert done, "Step 400: done must be True at max_steps horizon"

        # Verify no NaNs or Infs across all observation and reward components
        assert torch.isfinite(next_td["obs"]).all(), f"NaN/Inf detected in obs at step {step_idx}"
        assert torch.isfinite(next_td["cube_z"]).all(), f"NaN/Inf detected in cube_z at step {step_idx}"
        assert torch.isfinite(next_td["reward"]).all(), f"NaN/Inf detected in reward at step {step_idx}"
        assert torch.isfinite(next_td["desired_goal"]).all(), f"NaN/Inf in desired_goal at step {step_idx}"
        assert torch.isfinite(next_td["achieved_goal"]).all(), f"NaN/Inf in achieved_goal at step {step_idx}"

        current_td = next_td

    assert env.current_step == 400


def test_consecutive_goal_chaining_multi_target():
    env = LeapCubeReorientEnv(
        max_steps=400,
        consecutive_steps_for_success=5,
        target_mode="so3",
        reward_mode="sparse",
        seed=101,
    )
    td = env.reset()

    assert env.consecutive_successes == 0
    goal_0 = env.desired_goal.clone()

    orig_step_delta = env.pd_controller.step_delta

    # Oracle controller to simulate in-hand reorientation to desired_goal
    def oracle_step_delta(data, delta, **kwargs):
        orig_step_delta(data, delta, **kwargs)
        data.qpos[19:23] = env.desired_goal.cpu().numpy()
        data.qpos[18] = 0.18
        data.qvel[:] = 0.0

    env.pd_controller.step_delta = oracle_step_delta

    current_td = td

    # Phase 1: Hold goal_0 for 5 steps -> trigger chain to goal_1
    for step_i in range(1, 6):
        cube_pos_before = env.data.qpos[16:19].copy()
        hand_qpos_before = env.data.qpos[:16].copy()

        current_td["action"] = torch.zeros(16)
        out = env.step(current_td)
        current_td = out["next"]

        if step_i < 5:
            assert env.consecutive_successes == 0
            assert torch.allclose(env.desired_goal, goal_0, atol=1e-5)
        else:
            assert env.consecutive_successes == 1
            assert not torch.allclose(env.desired_goal, goal_0, atol=1e-3)
            # Hand and cube state preserved across goal chain
            np.testing.assert_allclose(env.data.qpos[16:19], cube_pos_before, atol=0.05)
            np.testing.assert_allclose(env.data.qpos[:16], hand_qpos_before, atol=0.05)

    goal_1 = env.desired_goal.clone()
    assert env.consecutive_successes == 1

    # Phase 2: Hold goal_1 for another 5 steps -> trigger chain to goal_2
    for step_i in range(1, 6):
        cube_pos_before = env.data.qpos[16:19].copy()
        hand_qpos_before = env.data.qpos[:16].copy()

        current_td["action"] = torch.zeros(16)
        out = env.step(current_td)
        current_td = out["next"]

        if step_i < 5:
            assert env.consecutive_successes == 1
            assert torch.allclose(env.desired_goal, goal_1, atol=1e-5)
        else:
            assert env.consecutive_successes == 2
            assert not torch.allclose(env.desired_goal, goal_1, atol=1e-3)
            assert not torch.allclose(env.desired_goal, goal_0, atol=1e-3)
            # Hand and cube state preserved
            np.testing.assert_allclose(env.data.qpos[16:19], cube_pos_before, atol=0.05)
            np.testing.assert_allclose(env.data.qpos[:16], hand_qpos_before, atol=0.05)

    goal_2 = env.desired_goal.clone()
    assert env.consecutive_successes == 2

    # Phase 3: Streak interruption and recovery
    # Step for 3 steps on goal_2 (streak = 3)
    for _ in range(3):
        current_td["action"] = torch.zeros(16)
        out = env.step(current_td)
        current_td = out["next"]
    assert env.consecutive_successes == 2
    assert env._success_streak == 3

    # Interrupt streak by corrupting orientation away from goal_2
    def mismatch_step_delta(data, delta, **kwargs):
        orig_step_delta(data, delta, **kwargs)
        # Invert orientation or set perpendicular
        data.qpos[19:23] = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
        data.qpos[18] = 0.18

    env.pd_controller.step_delta = mismatch_step_delta
    current_td["action"] = torch.zeros(16)
    out = env.step(current_td)
    current_td = out["next"]

    assert env.consecutive_successes == 2
    assert env._success_streak == 0
    assert torch.allclose(env.desired_goal, goal_2, atol=1e-5)

    # Resume oracle tracking for 5 consecutive steps -> chain to goal_3
    env.pd_controller.step_delta = oracle_step_delta
    for step_i in range(1, 6):
        current_td["action"] = torch.zeros(16)
        out = env.step(current_td)
        current_td = out["next"]

        if step_i < 5:
            assert env.consecutive_successes == 2
        else:
            assert env.consecutive_successes == 3
            assert not torch.allclose(env.desired_goal, goal_2, atol=1e-3)


def test_extended_horizon_cube_drop_no_suicide():
    env = LeapCubeReorientEnv(max_steps=400, seed=77)
    td = env.reset()

    # Force cube drop below drop_z_threshold (0.04)
    env.data.qpos[18] = 0.02
    current_td = td

    for step_idx in range(1, 401):
        current_td["action"] = torch.zeros(16)
        step_out = env.step(current_td)
        next_td = step_out["next"]

        terminated = bool(next_td["terminated"].item())
        truncated = bool(next_td["truncated"].item())

        assert not terminated, f"Step {step_idx}: terminated should be False even when cube dropped"

        if step_idx < 400:
            assert not truncated
        else:
            assert truncated

        assert float(next_td["reward"].item()) == -1.0
        current_td = next_td

    assert env.current_step == 400


def test_trajectory_collector_400_steps_her_buffer():
    env_fn = lambda: LeapCubeReorientEnv(
        max_steps=400,
        target_mode="so3",
        reward_mode="sparse",
        enable_domain_randomization=True,
    )
    actor = SquashedGaussianSACActor(obs_dim=55, goal_dim=4, action_dim=16)
    policy = PolicyModule(actor)
    replay_buffer = create_her_replay_buffer(
        storage_size=2000, slice_len=40, batch_size=200
    )

    collector = create_trajectory_collector(
        env_fn=env_fn,
        policy=policy,
        replay_buffer=replay_buffer,
        frames_per_batch=400,
        total_frames=400,
        enable_domain_randomization=True,
    )

    try:
        batches = list(collector)
        assert len(batches) == 1
    finally:
        collector.shutdown()

    assert len(replay_buffer) == 400

    sampled = replay_buffer.sample()
    assert sampled.shape[0] == 200
    assert "obs" in sampled
    assert "desired_goal" in sampled
    assert "action" in sampled
    assert "next" in sampled
    assert "reward" in sampled["next"]
    assert "achieved_goal" in sampled["next"]

    assert torch.isfinite(sampled["obs"]).all()
    assert torch.isfinite(sampled["action"]).all()
    assert torch.isfinite(sampled["desired_goal"]).all()
    assert torch.isfinite(sampled["next", "reward"]).all()
    assert torch.isfinite(sampled["next", "achieved_goal"]).all()
