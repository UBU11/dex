"""TorchRL 0.15 EnvBase implementation for 16-DoF LEAP Hand cube reorientation."""

import mujoco
import numpy as np
import torch
from tensordict import TensorDict
from torchrl.data import Bounded, Categorical, Composite, Unbounded
from torchrl.envs import EnvBase

from src.envs.domain_randomization import DomainRandomizer
from src.envs.pd_controller import PDController
from src.envs.spec import (
    DEFAULT_CUBE_POS,
    DEFAULT_CUBE_QUAT,
    DEFAULT_HAND_QPOS,
    build_leap_cube_spec,
)
from src.utils.quaternion import compute_sparse_her_reward, sample_uniform_quat


class LeapCubeReorientEnv(EnvBase):
    """TorchRL environment for 16-DoF LEAP Hand in-hand cube reorientation.

    Features:
    - 20 Hz control via 10 substeps of low-level PD tracking at 200 Hz.
    - Continuous bounded action [-1.0, 1.0] mapped to relative delta targets [-0.15, 0.15] rad.
    - Sparse binary HER reward (0.0 on success, -1.0 otherwise).
    - No alive-penalty suicide: Fixed episode horizon without premature drop termination.
    - Consecutive goal chaining: In-place new target sampling after 5 consecutive successful steps.
    - Domain randomization for cube mass, surface friction, joint damping, and latency.
    """

    def __init__(
        self,
        model: mujoco.MjModel | None = None,
        data: mujoco.MjData | None = None,
        spec: mujoco.MjSpec | None = None,
        tilt_degrees: float = 20.0,
        cube_half_size: float = 0.035,
        cube_mass: float = 0.1,
        max_steps: int = 400,
        n_substeps: int = 10,
        max_delta: float = 0.15,
        angle_tol: float = 0.15,
        drop_z_threshold: float = 0.04,
        consecutive_steps_for_success: int = 5,
        xy_offset_range: float = 0.015,
        domain_randomization: bool = True,
        seed: int | None = None,
        device: torch.device | None = None,
    ) -> None:
        super().__init__(device=device, batch_size=torch.Size([]))

        self.max_steps = max_steps
        self.n_substeps = n_substeps
        self.max_delta = max_delta
        self.angle_tol = angle_tol
        self.drop_z_threshold = drop_z_threshold
        self.consecutive_steps_for_success = consecutive_steps_for_success
        self.xy_offset_range = xy_offset_range
        self.domain_randomization = domain_randomization

        # ponytail: compile MjSpec on demand if raw model is not supplied
        if model is None:
            if spec is None:
                spec = build_leap_cube_spec(
                    tilt_degrees=tilt_degrees,
                    cube_half_size=cube_half_size,
                    cube_mass=cube_mass,
                )
            self.model = spec.compile()
        else:
            self.model = model

        self.data = mujoco.MjData(self.model) if data is None else data

        # Low-level PD controller and domain randomizer
        self.pd_controller = PDController(self.model)
        self.domain_randomizer = DomainRandomizer(self.model, self.data)

        # Episode state variables
        self.current_step = 0
        self.consecutive_successes = 0
        self._success_streak = 0
        self.prev_action = np.zeros(16, dtype=np.float32)
        self.desired_goal = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)

        self._set_seed(seed)
        self._make_specs()

    def _make_specs(self) -> None:
        """Configures TorchRL action, observation, reward, and done specs."""
        self.action_spec = Bounded(
            low=-1.0,
            high=1.0,
            shape=(16,),
            dtype=torch.float32,
            device=self.device,
        )

        # Observation spec: concatenated delayed obs (55,), goals (4,), prev_action (16,), cube_z (1,)
        self.observation_spec = Composite(
            obs=Unbounded(shape=(55,), dtype=torch.float32, device=self.device),
            desired_goal=Unbounded(shape=(4,), dtype=torch.float32, device=self.device),
            achieved_goal=Unbounded(shape=(4,), dtype=torch.float32, device=self.device),
            prev_action=Bounded(
                low=-1.0,
                high=1.0,
                shape=(16,),
                dtype=torch.float32,
                device=self.device,
            ),
            cube_z=Unbounded(shape=(1,), dtype=torch.float32, device=self.device),
            shape=torch.Size([]),
        )

        self.reward_spec = Unbounded(
            shape=(1,),
            dtype=torch.float32,
            device=self.device,
        )

        self.done_spec = Composite(
            done=Categorical(2, shape=(1,), dtype=torch.bool, device=self.device),
            terminated=Categorical(2, shape=(1,), dtype=torch.bool, device=self.device),
            truncated=Categorical(2, shape=(1,), dtype=torch.bool, device=self.device),
            shape=torch.Size([]),
        )

    def _set_seed(self, seed: int | None) -> None:
        """Sets internal random number generators."""
        self.np_rng = np.random.default_rng(seed)
        if seed is not None:
            torch.manual_seed(seed)

    def _reset(self, tensordict: TensorDict | None = None, **kwargs) -> TensorDict:
        """Resets hand posture, cube with XY offset, desired goal, and latency queue."""
        # ponytail: randomize or restore nominal domain parameters on reset
        if self.domain_randomization:
            self.domain_randomizer.randomize(self.np_rng)
        else:
            self.domain_randomizer.reset_to_nominal()

        # Reset hand joint angles and actuator controls to cradle posture
        self.data.qpos[:16] = DEFAULT_HAND_QPOS
        self.data.ctrl[:16] = DEFAULT_HAND_QPOS

        # ponytail: randomize cube XY position by +- 1.5 cm to avoid distribution shift
        dx = float(self.np_rng.uniform(-self.xy_offset_range, self.xy_offset_range))
        dy = float(self.np_rng.uniform(-self.xy_offset_range, self.xy_offset_range))
        initial_cube_pos = DEFAULT_CUBE_POS.copy()
        initial_cube_pos[0] += dx
        initial_cube_pos[1] += dy
        self.data.qpos[16:19] = initial_cube_pos
        self.data.qpos[19:23] = DEFAULT_CUBE_QUAT

        # Zero out velocities
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

        # Reset episode trackers
        self.current_step = 0
        self.consecutive_successes = 0
        self._success_streak = 0
        self.prev_action = np.zeros(16, dtype=np.float32)

        # Sample initial uniform SO(3) target quaternion
        self.desired_goal = sample_uniform_quat(1, device=self.device)[0]

        # Initial raw observation vector
        qpos_hand = self.data.qpos[:16].astype(np.float32)
        qvel_hand = self.data.qvel[:16].astype(np.float32)
        cube_p = self.data.qpos[16:19].astype(np.float32)
        cube_q = self.data.qpos[19:23].astype(np.float32)
        raw_obs = np.concatenate([qpos_hand, qvel_hand, cube_p, cube_q, self.prev_action])

        # Reset latency FIFO queue
        self.domain_randomizer.reset_latency()
        delayed_obs = self.domain_randomizer.get_delayed_observation(raw_obs)

        return TensorDict(
            {
                "obs": torch.as_tensor(delayed_obs, device=self.device, dtype=torch.float32),
                "desired_goal": self.desired_goal.clone(),
                "achieved_goal": torch.as_tensor(cube_q, device=self.device, dtype=torch.float32),
                "prev_action": torch.as_tensor(self.prev_action, device=self.device, dtype=torch.float32),
                "cube_z": torch.as_tensor([cube_p[2]], device=self.device, dtype=torch.float32),
            },
            batch_size=self.batch_size,
        )

    def _step(self, tensordict: TensorDict) -> TensorDict:
        """Applies relative action delta, evaluates sparse reward, and handles goal chaining."""
        action = tensordict["action"]
        action_np = action.detach().cpu().numpy().astype(np.float64).reshape(-1)

        # ponytail: scale normalized [-1, 1] action directly to joint delta [-0.15, 0.15] rad
        delta_q = np.clip(action_np, -1.0, 1.0) * self.max_delta
        self.pd_controller.step_delta(
            self.data,
            delta_q,
            n_substeps=self.n_substeps,
            max_delta=self.max_delta,
        )

        # Extract current state from MuJoCo data
        qpos_hand = self.data.qpos[:16].astype(np.float32)
        qvel_hand = self.data.qvel[:16].astype(np.float32)
        cube_p = self.data.qpos[16:19].astype(np.float32)
        cube_q = self.data.qpos[19:23].astype(np.float32)
        current_act = np.clip(action_np, -1.0, 1.0).astype(np.float32)
        self.prev_action = current_act

        raw_obs = np.concatenate([qpos_hand, qvel_hand, cube_p, cube_q, current_act])
        delayed_obs = self.domain_randomizer.get_delayed_observation(raw_obs)

        achieved_goal = torch.as_tensor(cube_q, device=self.device, dtype=torch.float32)
        cube_z = torch.as_tensor([cube_p[2]], device=self.device, dtype=torch.float32)

        # Compute sparse binary HER reward and success flag
        reward, is_success = compute_sparse_her_reward(
            achieved_goal,
            self.desired_goal,
            cube_z,
            angle_tol=self.angle_tol,
            drop_z_threshold=self.drop_z_threshold,
        )

        # ponytail: consecutive goal chaining without resetting hand or cube
        if bool(is_success.squeeze()):
            self._success_streak += 1
            if self._success_streak >= self.consecutive_steps_for_success:
                self.consecutive_successes += 1
                self._success_streak = 0
                self.desired_goal = sample_uniform_quat(1, device=self.device)[0]
        else:
            self._success_streak = 0

        self.current_step += 1
        truncated = self.current_step >= self.max_steps
        # ponytail: no alive-penalty suicide: episode continues emitting -1.0 even if cube drops
        terminated = False
        done = truncated or terminated

        return TensorDict(
            {
                "obs": torch.as_tensor(delayed_obs, device=self.device, dtype=torch.float32),
                "desired_goal": self.desired_goal.clone(),
                "achieved_goal": achieved_goal,
                "prev_action": torch.as_tensor(current_act, device=self.device, dtype=torch.float32),
                "cube_z": cube_z,
                "reward": reward.to(self.device).reshape(1),
                "done": torch.tensor([done], device=self.device, dtype=torch.bool),
                "terminated": torch.tensor([terminated], device=self.device, dtype=torch.bool),
                "truncated": torch.tensor([truncated], device=self.device, dtype=torch.bool),
            },
            batch_size=self.batch_size,
        )
