"""TorchRL 0.15 EnvBase implementation for 16-DoF LEAP Hand cube reorientation."""

import mujoco
import numpy as np
import torch
from tensordict import TensorDict
from torchrl.data import Bounded, Categorical, Composite, Unbounded
from torchrl.envs import EnvBase

from src.envs.domain_randomization import ADRCurriculumController, DomainRandomizer
from src.envs.pd_controller import PDController
from src.envs.spec import (
    DEFAULT_CUBE_POS,
    DEFAULT_CUBE_QUAT,
    DEFAULT_HAND_QPOS,
    build_leap_cube_spec,
)
from src.utils.quaternion import (
    compute_dense_orientation_reward,
    compute_sparse_her_reward,
    sample_uniform_quat,
    sample_yaw_quat,
)


class LeapCubeReorientEnv(EnvBase):
    """TorchRL environment for 16-DoF LEAP Hand in-hand cube reorientation.

    Features:
    - 20 Hz control via 10 substeps of low-level PD tracking at 200 Hz.
    - Continuous bounded action [-1.0, 1.0] mapped to relative delta targets [-0.15, 0.15] rad.
    - Sparse binary HER reward (0.0 on success, -1.0 otherwise) or dense negative reward.
    - SO(3) or single-axis Z-yaw target modes.
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
        target_mode: str = "so3",
        reward_mode: str = "sparse",
        action_penalty_weight: float = 0.01,
        enable_domain_randomization: bool = False,
        randomizer: DomainRandomizer | None = None,
        domain_randomization: bool | None = None,
        seed: int | None = None,
        device: torch.device | None = None,
    ) -> None:
        super().__init__(device=device, batch_size=torch.Size([]))

        if target_mode not in ("so3", "yaw"):
            raise ValueError(f"Invalid target_mode '{target_mode}'. Expected 'so3' or 'yaw'.")
        if reward_mode not in ("sparse", "dense"):
            raise ValueError(f"Invalid reward_mode '{reward_mode}'. Expected 'sparse' or 'dense'.")

        self.max_steps = max_steps
        self.n_substeps = n_substeps
        self.max_delta = max_delta
        self.angle_tol = angle_tol
        self.drop_z_threshold = drop_z_threshold
        self.consecutive_steps_for_success = consecutive_steps_for_success
        self.xy_offset_range = xy_offset_range
        self.target_mode = target_mode
        self.reward_mode = reward_mode
        self.action_penalty_weight = action_penalty_weight

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

        # Low-level PD controller
        self.pd_controller = PDController(self.model)

        self._enable_domain_randomization = False
        self.randomizer: DomainRandomizer | None = randomizer
        self.curriculum_controller: ADRCurriculumController | None = None

        if domain_randomization is not None:
            enable_domain_randomization = domain_randomization
        self.enable_domain_randomization = enable_domain_randomization

        # Episode state variables
        self.current_step = 0
        self.consecutive_successes = 0
        self._success_streak = 0
        self.prev_action = np.zeros(16, dtype=np.float32)
        self.desired_goal = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)

        self._set_seed(seed)
        self._make_specs()

    @property
    def enable_domain_randomization(self) -> bool:
        return self._enable_domain_randomization

    @enable_domain_randomization.setter
    def enable_domain_randomization(self, value: bool) -> None:
        self._enable_domain_randomization = bool(value)
        if self._enable_domain_randomization:
            if self.randomizer is None:
                self.randomizer = DomainRandomizer(self.model, self.data)
            if self.curriculum_controller is None:
                self.curriculum_controller = ADRCurriculumController(self.randomizer)

    @property
    def domain_randomization(self) -> bool:
        return self.enable_domain_randomization

    @domain_randomization.setter
    def domain_randomization(self, value: bool) -> None:
        self.enable_domain_randomization = value

    @property
    def domain_randomizer(self) -> DomainRandomizer | None:
        return self.randomizer

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

    def _get_obs(self) -> np.ndarray:
        qpos_hand = self.data.qpos[:16].astype(np.float32)
        qvel_hand = self.data.qvel[:16].astype(np.float32)
        cube_p = self.data.qpos[16:19].astype(np.float32)
        cube_q = self.data.qpos[19:23].astype(np.float32)
        raw_obs = np.concatenate([qpos_hand, qvel_hand, cube_p, cube_q, self.prev_action])
        if self.enable_domain_randomization and self.randomizer is not None:
            return self.randomizer.get_delayed_observation(raw_obs, rng=self.np_rng)
        return raw_obs

    def _sample_goal(self) -> torch.Tensor:
        if self.target_mode == "yaw":
            return sample_yaw_quat(1, device=self.device)[0]
        return sample_uniform_quat(1, device=self.device)[0]

    def _reset(self, tensordict: TensorDict | None = None, **kwargs) -> TensorDict:
        """Resets hand posture, cube with XY offset, desired goal, and latency queue."""
        if self.enable_domain_randomization and self.randomizer is not None:
            self.randomizer.randomize(self.np_rng)

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

        # Sample initial target quaternion based on target_mode
        self.desired_goal = self._sample_goal()

        if self.enable_domain_randomization and self.randomizer is not None:
            self.randomizer.reset_latency()

        obs = self._get_obs()
        cube_p = self.data.qpos[16:19].astype(np.float32)
        cube_q = self.data.qpos[19:23].astype(np.float32)

        return TensorDict(
            {
                "obs": torch.as_tensor(obs, device=self.device, dtype=torch.float32),
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

        current_act = np.clip(action_np, -1.0, 1.0).astype(np.float32)
        self.prev_action = current_act

        obs = self._get_obs()

        cube_p = self.data.qpos[16:19].astype(np.float32)
        cube_q = self.data.qpos[19:23].astype(np.float32)
        achieved_goal = torch.as_tensor(cube_q, device=self.device, dtype=torch.float32)
        cube_z = torch.as_tensor([cube_p[2]], device=self.device, dtype=torch.float32)

        # Compute sparse binary HER reward and success flag
        sparse_reward, is_success = compute_sparse_her_reward(
            achieved_goal,
            self.desired_goal,
            cube_z,
            angle_tol=self.angle_tol,
            drop_z_threshold=self.drop_z_threshold,
        )

        if self.reward_mode == "dense":
            reward = compute_dense_orientation_reward(
                achieved_goal,
                self.desired_goal,
                action,
                action_penalty_weight=self.action_penalty_weight,
            )
        else:
            reward = sparse_reward

        # ponytail: consecutive goal chaining without resetting hand or cube
        if bool(is_success.squeeze()):
            self._success_streak += 1
            if self._success_streak >= self.consecutive_steps_for_success:
                self.consecutive_successes += 1
                self._success_streak = 0
                self.desired_goal = self._sample_goal()
        else:
            self._success_streak = 0

        self.current_step += 1
        truncated = self.current_step >= self.max_steps
        # ponytail: no alive-penalty suicide: episode continues emitting -1.0 even if cube drops
        terminated = False
        done = truncated or terminated

        return TensorDict(
            {
                "obs": torch.as_tensor(obs, device=self.device, dtype=torch.float32),
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
