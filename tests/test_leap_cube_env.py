import math
import sys
from pathlib import Path
from unittest.mock import patch

import mujoco
import numpy as np
import pytest
import torch
from torchrl.envs.utils import check_env_specs

# Ensure repository root is on sys.path for direct pytest invocation
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.envs import (
    DEFAULT_CUBE_POS,
    DEFAULT_CUBE_QUAT,
    DomainRandomizer,
    LeapCubeReorientEnv,
    build_leap_cube_spec,
)


def test_envs_package_exports():
    """Verify DomainRandomizer and LeapCubeReorientEnv are exported from src.envs."""
    from src.envs.domain_randomization import DomainRandomizer as DRDirect
    from src.envs.leap_cube_env import LeapCubeReorientEnv as EnvDirect

    assert DomainRandomizer is DRDirect
    assert LeapCubeReorientEnv is EnvDirect


def test_domain_randomizer_parameter_ranges_and_mj_set_const():
    """Verify DomainRandomizer mutates mass, friction, damping within specification intervals and calls mj_setConst."""
    spec = build_leap_cube_spec()
    model = spec.compile()
    data = mujoco.MjData(model)

    randomizer = DomainRandomizer(model, data)
    nominal_mass = randomizer.nominal_mass
    nominal_friction = randomizer.nominal_friction.copy()
    nominal_damping = randomizer.nominal_damping.copy()

    # Track mj_setConst calls
    with patch("mujoco.mj_setConst", wraps=mujoco.mj_setConst) as mock_set_const:
        rng = np.random.default_rng(42)
        for _ in range(20):
            randomizer.randomize(rng)
            mock_set_const.assert_called()

            # Cube mass in [0.5x, 1.5x]
            cube_body_id = randomizer.cube_body_id
            current_mass = model.body_mass[cube_body_id]
            assert 0.5 * nominal_mass - 1e-6 <= current_mass <= 1.5 * nominal_mass + 1e-6

            # Cube friction in [0.6x, 1.4x]
            cube_geom_id = randomizer.cube_geom_id
            current_friction = model.geom_friction[cube_geom_id]
            for i in range(3):
                assert 0.6 * nominal_friction[i] - 1e-6 <= current_friction[i] <= 1.4 * nominal_friction[i] + 1e-6

            # Joint damping in [0.7x, 1.3x]
            current_damping = model.dof_damping[:16]
            for i in range(16):
                assert 0.7 * nominal_damping[i] - 1e-6 <= current_damping[i] <= 1.3 * nominal_damping[i] + 1e-6

            # Latency in [0, 1, 2]
            assert randomizer.current_latency_steps in (0, 1, 2)

    # Verify reset_to_nominal restores exact original parameters
    randomizer.reset_to_nominal()
    assert math.isclose(model.body_mass[cube_body_id], nominal_mass, rel_tol=1e-6)
    np.testing.assert_allclose(model.geom_friction[cube_geom_id], nominal_friction, rtol=1e-5)
    np.testing.assert_allclose(model.dof_damping[:16], nominal_damping, rtol=1e-5)
    assert randomizer.current_latency_steps == 0


def test_domain_randomizer_latency_buffer():
    """Verify latency FIFO buffer delays observations by exactly 0, 1, or 2 steps."""
    spec = build_leap_cube_spec()
    model = spec.compile()
    data = mujoco.MjData(model)
    randomizer = DomainRandomizer(model, data)

    # Test latency = 0
    randomizer.current_latency_steps = 0
    obs0 = np.ones(55, dtype=np.float32) * 0.0
    obs1 = np.ones(55, dtype=np.float32) * 1.0
    assert np.array_equal(randomizer.get_delayed_observation(obs0), obs0)
    assert np.array_equal(randomizer.get_delayed_observation(obs1), obs1)

    # Test latency = 1
    randomizer.reset_latency()
    randomizer.current_latency_steps = 1
    # Step 0: queue empty, padded with oldest
    out0 = randomizer.get_delayed_observation(obs0)
    assert np.array_equal(out0, obs0)
    # Step 1: queue has [obs0, obs1], delayed by 1 returns obs0
    out1 = randomizer.get_delayed_observation(obs1)
    assert np.array_equal(out1, obs0)
    # Step 2: queue has [obs0, obs1, obs2], delayed by 1 returns obs1
    obs2 = np.ones(55, dtype=np.float32) * 2.0
    out2 = randomizer.get_delayed_observation(obs2)
    assert np.array_equal(out2, obs1)

    # Test latency = 2
    randomizer.reset_latency()
    randomizer.current_latency_steps = 2
    obs_seq = [np.ones(55, dtype=np.float32) * float(i) for i in range(5)]
    delayed_outs = [randomizer.get_delayed_observation(o) for o in obs_seq]
    # For latency=2:
    # step 0: returns obs0
    # step 1: returns obs0 (queue has [0, 1], index max(0, 2-1-2) = 0)
    # step 2: returns obs0 (queue has [0, 1, 2], index max(0, 3-1-2) = 0)
    # step 3: returns obs1 (queue has [0, 1, 2, 3], index 4-1-2 = 1)
    # step 4: returns obs2 (queue has [..., 2, 3, 4], index 5-1-2 = 2)
    assert np.array_equal(delayed_outs[0], obs_seq[0])
    assert np.array_equal(delayed_outs[1], obs_seq[0])
    assert np.array_equal(delayed_outs[2], obs_seq[0])
    assert np.array_equal(delayed_outs[3], obs_seq[1])
    assert np.array_equal(delayed_outs[4], obs_seq[2])


def test_leap_cube_env_specs_and_reset():
    """Verify LeapCubeReorientEnv.reset() returns valid TensorDict with expected keys and finite values."""
    env = LeapCubeReorientEnv(max_steps=100, seed=123)

    # Verify TorchRL specs pass check_env_specs
    check_env_specs(env)

    # Verify action spec is bounded [-1, 1] with shape (16,)
    assert env.action_spec.shape == (16,)
    assert (env.action_spec.space.low == -1.0).all()
    assert (env.action_spec.space.high == 1.0).all()

    # Reset environment
    td = env.reset()

    # Check required keys in TensorDict
    required_keys = {"obs", "desired_goal", "achieved_goal", "prev_action", "cube_z"}
    assert required_keys.issubset(set(td.keys()))

    # Check shapes
    assert td["obs"].shape == (55,)
    assert td["desired_goal"].shape == (4,)
    assert td["achieved_goal"].shape == (4,)
    assert td["prev_action"].shape == (16,)
    assert td["cube_z"].shape == (1,)

    # Check all values are finite
    for key in required_keys:
        assert torch.isfinite(td[key]).all(), f"Non-finite values found in key '{key}'"

    # Prev action on reset must be all zeros
    assert torch.all(td["prev_action"] == 0.0)

    # Desired goal must be unit quaternion
    dg_norm = torch.norm(td["desired_goal"], p=2).item()
    assert math.isclose(dg_norm, 1.0, rel_tol=1e-5)

    # Achieved goal must be unit quaternion
    ag_norm = torch.norm(td["achieved_goal"], p=2).item()
    assert math.isclose(ag_norm, 1.0, rel_tol=1e-5)

    # Verify cube initial position XY randomization (+- 1.5 cm) across multiple resets
    xy_positions = []
    for _ in range(30):
        env.reset()
        cube_pos = env.data.qpos[16:19].copy()
        xy_positions.append(cube_pos[:2])
        # Verify within +- 1.5 cm of nominal
        assert abs(cube_pos[0] - DEFAULT_CUBE_POS[0]) <= 0.015 + 1e-5
        assert abs(cube_pos[1] - DEFAULT_CUBE_POS[1]) <= 0.015 + 1e-5

    xy_arr = np.array(xy_positions)
    # Assert variance across resets is positive (not static)
    assert np.var(xy_arr[:, 0]) > 1e-6
    assert np.var(xy_arr[:, 1]) > 1e-6


def test_leap_cube_env_step_and_no_alive_penalty_suicide():
    """Verify step applies action delta, returns updated observations, sparse reward, and no alive penalty suicide."""
    env = LeapCubeReorientEnv(max_steps=50, domain_randomization=False, seed=42)
    td = env.reset()

    # Apply continuous action in [-1.0, 1.0]
    action = torch.ones(16, dtype=torch.float32) * 0.5
    td["action"] = action
    td_step = env.step(td)

    # Output structure must contain "next"
    assert "next" in td_step
    next_td = td_step["next"]

    assert next_td["obs"].shape == (55,)
    assert next_td["prev_action"].shape == (16,)
    # prev_action must match the applied action
    np.testing.assert_allclose(next_td["prev_action"].numpy(), action.numpy(), rtol=1e-5)

    # Sparse reward is either 0.0 or -1.0
    reward = next_td["reward"].item()
    assert reward in (-1.0, 0.0)

    # Verify TorchRL rollout runs smoothly for 10 steps
    rollout = env.rollout(max_steps=10)
    assert rollout.shape[0] == 10
    assert not rollout["next", "done"][:-1].any()

    # CRITICAL: Verify NO ALIVE-PENALTY SUICIDE
    # Simulate cube falling off palm (cube_z dropped below 0.04m)
    env.reset()
    env.data.qpos[18] = 0.0  # drop cube to z=0.0
    step_td = env.reset()
    step_td["action"] = torch.zeros(16)
    env.data.qpos[18] = 0.0  # keep cube dropped

    out = env.step(step_td)
    # Dropped cube must yield -1.0 reward, but must NOT terminate or truncate early
    assert out["next", "reward"].item() == -1.0
    assert not out["next", "done"].item(), "Cube drop must NOT terminate episode early (alive-penalty suicide prevention)"
    assert not out["next", "terminated"].item()
    assert not out["next", "truncated"].item()

    # Advance until max_steps to confirm truncation occurs ONLY at fixed horizon
    current_td = out["next"]
    for step_i in range(1, 50):
        current_td["action"] = torch.zeros(16)
        step_out = env.step(current_td)
        current_td = step_out["next"]
        if step_i < 49:
            assert not current_td["done"].item()
        else:
            assert current_td["truncated"].item()
            assert current_td["done"].item()


def test_consecutive_goal_chaining_in_place():
    """Verify consecutive goal chaining samples new goal in-place after 5 consecutive successful steps."""
    env = LeapCubeReorientEnv(max_steps=100, consecutive_steps_for_success=5, domain_randomization=False, seed=99)
    td = env.reset()

    initial_goal = env.desired_goal.clone()
    assert env.consecutive_successes == 0

    # Simulate maintaining target orientation during physics steps
    orig_step_delta = env.pd_controller.step_delta

    def holding_step_delta(data, delta, **kwargs):
        orig_step_delta(data, delta, **kwargs)
        # Hold cube at current desired_goal with cube resting in palm
        data.qpos[19:23] = env.desired_goal.cpu().numpy()
        data.qpos[18] = 0.18

    env.pd_controller.step_delta = holding_step_delta

    current_td = td
    # Simulate reaching the desired orientation for 5 consecutive control steps
    for step_count in range(1, 6):
        cube_pos_before = env.data.qpos[16:19].copy()
        hand_qpos_before = env.data.qpos[:16].copy()

        current_td["action"] = torch.zeros(16)
        step_out = env.step(current_td)
        current_td = step_out["next"]

        if step_count < 5:
            # Goal has not chained yet
            assert env.consecutive_successes == 0
            assert torch.allclose(env.desired_goal, initial_goal, atol=1e-5)
        else:
            # On 5th consecutive step, goal chaining MUST trigger!
            assert env.consecutive_successes == 1, "consecutive_successes should increment to 1"
            # New target quaternion is sampled in-place
            assert not torch.allclose(env.desired_goal, initial_goal, atol=1e-3), (
                "New desired goal should differ from initial goal"
            )
            # Cube position and hand posture must NOT be reset to initial state
            np.testing.assert_allclose(env.data.qpos[16:19], cube_pos_before, atol=0.05)
            np.testing.assert_allclose(env.data.qpos[:16], hand_qpos_before, atol=0.05)
            # New desired goal in next TensorDict matches env.desired_goal
            assert torch.allclose(current_td["desired_goal"], env.desired_goal)


def test_leap_cube_env_clipped_prev_action():
    """Verify out-of-bounds actions are clipped to [-1, 1] when stored into prev_action."""
    env = LeapCubeReorientEnv(max_steps=50, domain_randomization=False, seed=42)
    td = env.reset()

    # Provide unclipped action values outside [-1.0, 1.0]
    unclipped_action = torch.tensor([5.0, -10.0, 2.5, -3.0] + [0.0] * 12, dtype=torch.float32)
    td["action"] = unclipped_action
    step_out = env.step(td)

    next_td = step_out["next"]
    # Returned prev_action must be strictly clipped to [-1.0, 1.0]
    expected_clipped = torch.tensor([1.0, -1.0, 1.0, -1.0] + [0.0] * 12, dtype=torch.float32)
    np.testing.assert_allclose(next_td["prev_action"].numpy(), expected_clipped.numpy(), atol=1e-6)
    np.testing.assert_allclose(env.prev_action, expected_clipped.numpy(), atol=1e-6)


def test_leap_cube_env_configurable_drop_z_threshold():
    """Verify drop_z_threshold is configurable and respected during step."""
    custom_threshold = 0.08
    env = LeapCubeReorientEnv(
        max_steps=50,
        drop_z_threshold=custom_threshold,
        randomize_initial_orientation=False,
        domain_randomization=False,
        seed=42,
    )
    assert env.drop_z_threshold == custom_threshold

    td = env.reset()
    # Align achieved goal with desired goal
    env.data.qpos[19:23] = env.desired_goal.cpu().numpy()

    # Case 1: Cube z is 0.06 (above default 0.04, but below custom 0.08) -> dropped (-1.0)
    env.data.qpos[18] = 0.06
    td["action"] = torch.zeros(16)
    step_out = env.step(td)
    assert step_out["next", "reward"].item() == -1.0

    # Case 2: Cube z is 0.10 (above custom 0.08) and orientation matches -> success (0.0)
    next_td = step_out["next"]
    env.data.qpos[19:23] = env.desired_goal.cpu().numpy()
    env.data.qpos[18] = 0.10
    next_td["action"] = torch.zeros(16)
    step_out_2 = env.step(next_td)
    assert step_out_2["next", "reward"].item() == 0.0


def test_leap_cube_env_yaw_target_and_dense_reward():
    """Verify LeapCubeReorientEnv with target_mode='yaw' and reward_mode='dense'."""
    from src.utils.quaternion import quat_geodesic_distance

    env = LeapCubeReorientEnv(
        max_steps=50,
        target_mode="yaw",
        reward_mode="dense",
        action_penalty_weight=0.01,
        domain_randomization=False,
        seed=42,
    )
    assert env.target_mode == "yaw"
    assert env.reward_mode == "dense"

    td = env.reset()
    desired_goal = td["desired_goal"]
    # Single-axis Z-yaw target must have x=0 and y=0
    assert torch.isclose(desired_goal[1], torch.tensor(0.0), atol=1e-6)
    assert torch.isclose(desired_goal[2], torch.tensor(0.0), atol=1e-6)
    assert math.isclose(torch.norm(desired_goal, p=2).item(), 1.0, rel_tol=1e-5)

    action = torch.ones(16, dtype=torch.float32) * 0.5
    td["action"] = action
    step_out = env.step(td)
    next_td = step_out["next"]

    reward = next_td["reward"].item()
    achieved = next_td["achieved_goal"]
    desired = next_td["desired_goal"]
    theta = quat_geodesic_distance(achieved, desired).item()
    penalty = 0.01 * (0.5**2 * 16)
    expected_reward = -theta - penalty
    assert math.isclose(reward, expected_reward, rel_tol=1e-4)
    assert reward < 0.0


def test_leap_cube_env_invalid_target_and_reward_modes():
    """Verify ValueError is raised for invalid target_mode or reward_mode."""
    import pytest

    with pytest.raises(ValueError, match="target_mode"):
        LeapCubeReorientEnv(target_mode="invalid")

    with pytest.raises(ValueError, match="reward_mode"):
        LeapCubeReorientEnv(reward_mode="invalid")


def test_reset_orientation_distribution():
    """Verify reset randomizes cube initial orientation on S^3 (or yaw) and respects flags."""
    env_default = LeapCubeReorientEnv(max_steps=50, seed=42)
    assert env_default.randomize_initial_orientation is True
    assert env_default.xy_offset_range == 0.015

    orientations = []
    for _ in range(20):
        td = env_default.reset()
        quat = env_default.data.qpos[19:23].copy()
        pos = env_default.data.qpos[16:19].copy()
        orientations.append(quat)

        assert math.isclose(np.linalg.norm(quat), 1.0, rel_tol=1e-5)
        assert abs(pos[0] - DEFAULT_CUBE_POS[0]) <= env_default.xy_offset_range + 1e-6
        assert abs(pos[1] - DEFAULT_CUBE_POS[1]) <= env_default.xy_offset_range + 1e-6
        np.testing.assert_allclose(td["achieved_goal"].cpu().numpy(), quat, atol=1e-5)

    quat_arr = np.array(orientations)
    assert np.var(quat_arr) > 1e-4

    env_fixed = LeapCubeReorientEnv(randomize_initial_orientation=False, seed=42)
    assert env_fixed.randomize_initial_orientation is False
    for _ in range(5):
        td = env_fixed.reset()
        np.testing.assert_allclose(env_fixed.data.qpos[19:23], DEFAULT_CUBE_QUAT, atol=1e-6)
        np.testing.assert_allclose(td["achieved_goal"].cpu().numpy(), DEFAULT_CUBE_QUAT, atol=1e-6)

    env_yaw = LeapCubeReorientEnv(target_mode="yaw", randomize_initial_orientation=True, seed=42)
    for _ in range(10):
        td = env_yaw.reset()
        quat = env_yaw.data.qpos[19:23]
        assert abs(quat[1]) < 1e-6
        assert abs(quat[2]) < 1e-6
        assert math.isclose(np.linalg.norm(quat), 1.0, rel_tol=1e-5)


def test_leap_cube_env_set_target_mode():
    """Verify dynamic target_mode switching between 'so3' and 'yaw', input validation, and goal sampling."""
    env = LeapCubeReorientEnv(target_mode="so3", seed=42)
    assert env.target_mode == "so3"

    # Switch to yaw
    env.set_target_mode("yaw")
    assert env.target_mode == "yaw"

    # Verify sampled goals under yaw mode have x=0, y=0 and norm 1
    for _ in range(10):
        goal = env._sample_goal()
        # quaternion is [w, x, y, z]
        assert abs(float(goal[1])) < 1e-6
        assert abs(float(goal[2])) < 1e-6
        assert math.isclose(float(torch.norm(goal)), 1.0, rel_tol=1e-5)

    # Reset also uses new target mode
    td = env.reset()
    desired_goal = td["desired_goal"]
    assert abs(float(desired_goal[1])) < 1e-6
    assert abs(float(desired_goal[2])) < 1e-6

    # Switch back to so3
    env.set_target_mode("so3")
    assert env.target_mode == "so3"

    # Sample goals under so3 mode - should have non-zero x or y
    has_nonzero_xy = False
    for _ in range(10):
        goal = env._sample_goal()
        assert math.isclose(float(torch.norm(goal)), 1.0, rel_tol=1e-5)
        if abs(float(goal[1])) > 1e-4 or abs(float(goal[2])) > 1e-4:
            has_nonzero_xy = True
    assert has_nonzero_xy

    # Test invalid target mode raises ValueError
    with pytest.raises(ValueError, match="Invalid target_mode"):
        env.set_target_mode("invalid_mode")
    with pytest.raises(ValueError, match="Invalid target_mode"):
        env.set_target_mode("pitch")



