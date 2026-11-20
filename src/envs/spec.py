"""LEAP Hand + Cube MjSpec assembly."""

import mujoco
import mujoco_menagerie as mm
import numpy as np

# Nominal initial hand posture with gentle finger and thumb flexion
# forming a stable cradle for the 7 cm cube.
DEFAULT_HAND_QPOS = np.zeros(16, dtype=np.float64)
# Index finger MCP, PIP, DIP
DEFAULT_HAND_QPOS[0] = 0.35
DEFAULT_HAND_QPOS[2] = 0.35
DEFAULT_HAND_QPOS[3] = 0.35
# Middle finger MCP, PIP, DIP
DEFAULT_HAND_QPOS[4] = 0.35
DEFAULT_HAND_QPOS[6] = 0.35
DEFAULT_HAND_QPOS[7] = 0.35
# Ring finger MCP, PIP, DIP
DEFAULT_HAND_QPOS[8] = 0.35
DEFAULT_HAND_QPOS[10] = 0.35
DEFAULT_HAND_QPOS[11] = 0.35
# Thumb CMC, MCP
DEFAULT_HAND_QPOS[12] = 0.35
DEFAULT_HAND_QPOS[14] = 0.35

DEFAULT_CUBE_POS = np.array([-0.06, 0.05, 0.18], dtype=np.float64)
DEFAULT_CUBE_QUAT = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)


def build_leap_cube_spec(
    tilt_degrees: float = 20.0,
    cube_half_size: float = 0.035,
    cube_mass: float = 0.1,
    cube_pos: tuple[float, float, float] = (-0.06, 0.05, 0.18),
    timestep: float = 0.005,
    kp: float = 3.0,
    kd: float = 0.1,
) -> mujoco.MjSpec:
    """Assembles tilted 16-DoF LEAP Hand with a 7 cm cube freejoint in MuJoCo MjSpec.

    Args:
        tilt_degrees: Downward palm pitch tilt angle in degrees (nominal 20.0).
        cube_half_size: Half-size of cube geom in meters (0.035 = 7 cm full width).
        cube_mass: Nominal mass of cube body in kg (nominal 0.1 kg).
        cube_pos: Initial position of cube body (x, y, z) slightly above palm.
        timestep: Physics simulation timestep (0.005s = 200 Hz).
        kp: Position gain for LEAP actuators.
        kd: Damping gain for LEAP actuators.

    Returns:
        mujoco.MjSpec: Editable MuJoCo specification ready for compilation.
    """
    # ponytail: assemble MjSpec directly from mm.get('leap_hand').spec() without raw XML cloning
    spec = mm.get("leap_hand").spec()

    # Set physics timestep (200 Hz default)
    spec.option.timestep = timestep

    # ponytail: apply downward pitch tilt directly to palm body quaternion
    palm = spec.body("palm")
    theta = np.deg2rad(tilt_degrees)
    # Base palm quat is [0, 1, 0, 0] in leap_hand.
    # Pitching downward around Y rotates forward vector (+x) towards (-z).
    palm.quat = [
        0.0,
        float(np.cos(theta / 2.0)),
        0.0,
        float(-np.sin(theta / 2.0)),
    ]

    # ponytail: tune actuator gain and affine bias parameters for low-level PD tracking
    for act in spec.actuators:
        act.gainprm[0] = kp
        act.biasprm[1] = -kp
        act.biasprm[2] = -kd

    # ponytail: single cube freejoint body with box collision geometry
    worldbody = spec.worldbody
    cube_body = worldbody.add_body()
    cube_body.name = "cube"
    cube_body.pos = list(cube_pos)
    cube_body.add_freejoint()

    cube_geom = cube_body.add_geom()
    cube_geom.name = "cube"
    cube_geom.type = mujoco.mjtGeom.mjGEOM_BOX
    cube_geom.size = [cube_half_size, cube_half_size, cube_half_size]
    cube_geom.mass = cube_mass
    cube_geom.rgba = [0.85, 0.25, 0.25, 1.0]
    # Stable contact parameters to prevent contact explosions at 200 Hz
    cube_geom.solref = [0.01, 1.0]
    cube_geom.solimp = [0.9, 0.95, 0.001, 0.5, 2]
    cube_geom.friction = [1.0, 0.005, 0.0001]

    # ponytail: define 'home' keyframe for one-call data reset
    key = spec.add_key()
    key.name = "home"
    key_qpos = np.zeros(23, dtype=np.float64)
    key_qpos[:16] = DEFAULT_HAND_QPOS
    key_qpos[16:19] = list(cube_pos)
    key_qpos[19] = 1.0
    key.qpos = key_qpos

    return spec
