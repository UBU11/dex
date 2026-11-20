import math
import sys
from pathlib import Path

import mujoco
import numpy as np

# Ensure repository root is on sys.path for direct pytest invocation
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.envs import (
    DEFAULT_HAND_QPOS,
    PDController,
    build_leap_cube_spec,
)
from src.envs.pd_controller import PDController as PDControllerDirect
from src.envs.spec import build_leap_cube_spec as build_spec_direct


def test_envs_package_exports():
    """Verify clean public exports from src.envs package namespace."""
    assert build_spec_direct is build_leap_cube_spec
    assert PDControllerDirect is PDController
    assert isinstance(DEFAULT_HAND_QPOS, np.ndarray)
    assert DEFAULT_HAND_QPOS.shape == (16,)


def test_build_leap_cube_spec_compilation_and_actuators():
    """Verify build_leap_cube_spec compiles into valid MjModel with 16 actuators and cube freejoint."""
    spec = build_leap_cube_spec()
    model = spec.compile()

    # 16 actuators for 16-DoF LEAP hand
    assert model.nu == 16, f"Expected 16 actuators, got {model.nu}"

    # Cube body with a freejoint
    cube_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
    assert cube_body_id != -1, "Body 'cube' not found in model"
    assert model.body_jntnum[cube_body_id] == 1, "Cube should have exactly 1 joint"

    cube_jnt_id = model.body_jntadr[cube_body_id]
    assert model.jnt_type[cube_jnt_id] == mujoco.mjtJoint.mjJNT_FREE, "Cube joint must be a freejoint"

    # nq = 16 (hand) + 7 (freejoint) = 23
    assert model.nq == 23, f"Expected nq=23, got {model.nq}"
    # nv = 16 (hand) + 6 (freejoint) = 22
    assert model.nv == 22, f"Expected nv=22, got {model.nv}"

    # Cube geom dimensions: 7 cm cube has half-size [0.035, 0.035, 0.035]
    cube_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "cube")
    assert cube_geom_id != -1, "Geom 'cube' not found in model"
    np.testing.assert_allclose(
        model.geom_size[cube_geom_id],
        np.array([0.035, 0.035, 0.035]),
        rtol=1e-5,
        err_msg="Cube geom half-size should be 0.035m (7 cm full width)",
    )

    # Cube mass ~ 0.1 kg nominal
    assert math.isclose(model.body_mass[cube_body_id], 0.1, rel_tol=1e-3), (
        f"Cube mass should be ~0.1 kg, got {model.body_mass[cube_body_id]}"
    )

    # Simulation timestep should be 0.005s (200 Hz)
    assert math.isclose(model.opt.timestep, 0.005, rel_tol=1e-5), (
        f"Timestep should be 0.005s (200 Hz), got {model.opt.timestep}"
    )


def test_palm_body_20_deg_downward_tilt():
    """Verify the palm body is mounted with a 20° downward tilt relative to horizontal pose."""
    spec = build_leap_cube_spec()
    model = spec.compile()

    palm_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "palm")
    assert palm_id != -1, "Body 'palm' not found in model"

    palm_quat = model.body_quat[palm_id]
    # Base horizontal palm pose in MuJoCo Menagerie leap_hand is [0, 1, 0, 0]
    base_quat = np.array([0.0, 1.0, 0.0, 0.0])

    # Geodesic rotation angle: 2 * arccos(|<q1, q2>|)
    dot = np.clip(np.abs(np.dot(base_quat, palm_quat)), 0.0, 1.0)
    angle_rad = 2.0 * np.arccos(dot)
    angle_deg = np.rad2deg(angle_rad)

    assert math.isclose(angle_deg, 20.0, abs_tol=0.2), (
        f"Expected palm tilt angle to be ~20.0 deg, got {angle_deg:.2f} deg"
    )

    # Verify downward pitch: rotation matrix rotates forward vector (+x) downward (-z)
    mat = np.zeros(9)
    mujoco.mju_quat2Mat(mat, palm_quat)
    rot_matrix = mat.reshape(3, 3)
    # Forward vector along hand fingers in local palm frame (local +x)
    forward_world = rot_matrix @ np.array([1.0, 0.0, 0.0])
    assert forward_world[2] < 0.0, f"Expected downward tilt (z < 0), got forward vector {forward_world}"


def test_zero_action_rollout_physics_sanity():
    """Zero-action rollout over 50 control steps (500 physics substeps at 200 Hz).

    Verifies:
    - No NaNs in qpos or qvel.
    - Cube settles into palm; does not fall through hand mesh onto floor.
    - No contact explosions.
    """
    spec = build_leap_cube_spec()
    model = spec.compile()
    data = mujoco.MjData(model)

    # Reset to default keyframe ('home')
    home_key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if home_key_id != -1:
        mujoco.mj_resetDataKeyframe(model, data, home_key_id)
    else:
        # Fallback to DEFAULT_HAND_QPOS and default cube pos
        data.qpos[:16] = DEFAULT_HAND_QPOS
        data.qpos[16:19] = np.array([-0.06, 0.05, 0.18])
        data.qpos[19:23] = np.array([1.0, 0.0, 0.0, 0.0])

    mujoco.mj_forward(model, data)

    cube_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
    initial_cube_z = data.xpos[cube_id][2]
    assert initial_cube_z > 0.15, f"Initial cube z should be > 0.15m, got {initial_cube_z}"

    # Initialize PD controller with nominal gains kp=3.0, kd=0.1
    controller = PDController(model=model, kp=3.0, kd=0.1)

    # Hold initial hand posture (zero action) for 50 control steps (500 substeps)
    target_qpos = data.qpos[:16].copy()
    n_control_steps = 50
    n_substeps = 10

    for _ in range(n_control_steps):
        controller.step(data, target_qpos=target_qpos, n_substeps=n_substeps)

    # 1. No NaNs in state
    assert not np.isnan(data.qpos).any(), "qpos contains NaNs"
    assert not np.isnan(data.qvel).any(), "qvel contains NaNs"
    assert not np.isinf(data.qpos).any(), "qpos contains Infs"
    assert not np.isinf(data.qvel).any(), "qvel contains Infs"

    # 2. Cube settles into palm: z must remain well above the floor (floor is z=0, cube half-size is 0.035)
    # Palm surface is at z ~ 0.12 - 0.14 m, so resting cube z ~ 0.16 m
    final_cube_z = data.xpos[cube_id][2]
    assert final_cube_z > 0.12, (
        f"Cube fell through palm mesh! Final cube z = {final_cube_z:.4f} m (floor level ~ 0.035 m)"
    )
    assert final_cube_z < 0.22, (
        f"Cube exploded or launched upward! Final cube z = {final_cube_z:.4f} m"
    )

    # 3. Cube velocities should be stable / settled
    cube_vel = data.qvel[16:19]
    cube_speed = np.linalg.norm(cube_vel)
    assert cube_speed < 0.2, f"Cube did not settle; speed = {cube_speed:.4f} m/s"


def test_pd_controller_action_clipping_and_torque():
    """Verify PDController correctly clips target qpos and calculates PD torque."""
    spec = build_leap_cube_spec()
    model = spec.compile()
    controller = PDController(model=model, kp=3.0, kd=0.1)

    # Test torque computation
    qpos = np.zeros(model.nq)
    qvel = np.zeros(model.nv)
    target = np.full(16, 0.5)

    tau = controller.compute_torques(qpos, qvel, target)
    assert tau.shape == (16,)
    # tau = kp * (target - q) - kd * qvel = 3.0 * 0.5 - 0.1 * 0 = 1.5
    np.testing.assert_allclose(tau, np.full(16, 1.5), atol=1e-5)

    # Target exceeding joint limits must be clipped
    extreme_target = np.full(16, 10.0)
    tau_clipped = controller.compute_torques(qpos, qvel, extreme_target)
    upper_limits = model.actuator_ctrlrange[:16, 1]
    expected_tau = 3.0 * upper_limits
    np.testing.assert_allclose(tau_clipped, expected_tau, atol=1e-5)
