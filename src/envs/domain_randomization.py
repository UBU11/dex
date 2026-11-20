"""Domain randomization and observation latency buffer for LEAP Hand cube manipulation."""

from collections import deque

import mujoco
import numpy as np
import torch


class DomainRandomizer:
    """Domain randomizer for LEAP Hand and cube manipulation physics."""

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        cube_body_name: str = "cube",
        cube_geom_name: str = "cube",
        max_latency: int = 2,
        mass_range: tuple[float, float] = (0.5, 1.5),
        friction_range: tuple[float, float] = (0.6, 1.4),
        damping_range: tuple[float, float] = (0.7, 1.3),
        sensor_noise_std: float = 0.005,
    ) -> None:
        self.model = model
        self.data = data
        self.max_latency = max_latency
        self.mass_range = mass_range
        self.friction_range = friction_range
        self.damping_range = damping_range
        self.sensor_noise_std = sensor_noise_std
        self.difficulty = 0.0
        self.current_sensor_noise_std = 0.0

        # Locate cube body and geom IDs
        self.cube_body_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, cube_body_name
        )
        if self.cube_body_id == -1:
            raise ValueError(f"Body '{cube_body_name}' not found in model.")

        self.cube_geom_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, cube_geom_name
        )
        if self.cube_geom_id == -1:
            raise ValueError(f"Geom '{cube_geom_name}' not found in model.")

        # ponytail: cache nominal physical parameters directly from compiled model
        self.nominal_mass = float(self.model.body_mass[self.cube_body_id])
        self.nominal_inertia = self.model.body_inertia[self.cube_body_id].copy()
        self.nominal_friction = self.model.geom_friction[self.cube_geom_id].copy()
        self.nominal_damping = self.model.dof_damping[:16].copy()

        # Observation delay FIFO queue (0 to max_latency steps)
        self.current_latency_steps = 0
        self.obs_queue: deque[np.ndarray] = deque(maxlen=max(max_latency + 5, 10))

    def set_difficulty(self, level: float) -> None:
        """Sets ADR difficulty level clamped to [0.0, 1.0], scaling dynamic ranges."""
        clamped = float(np.clip(level, 0.0, 1.0))
        self.difficulty = clamped
        self.mass_range = (1.0 - 0.5 * clamped, 1.0 + 0.5 * clamped)
        self.friction_range = (1.0 - 0.4 * clamped, 1.0 + 0.4 * clamped)
        self.damping_range = (1.0 - 0.3 * clamped, 1.0 + 0.3 * clamped)
        self.max_latency = round(2 * clamped)
        self.current_sensor_noise_std = self.sensor_noise_std * clamped

    def randomize(self, rng: np.random.Generator | None = None) -> None:
        """Mutates model parameters and latency, calling mj_setConst."""
        if rng is None:
            rng = np.random.default_rng()

        mass_scale = float(rng.uniform(self.mass_range[0], self.mass_range[1]))
        self.model.body_mass[self.cube_body_id] = self.nominal_mass * mass_scale
        self.model.body_inertia[self.cube_body_id] = self.nominal_inertia * mass_scale

        friction_scale = rng.uniform(
            self.friction_range[0], self.friction_range[1], size=3
        )
        self.model.geom_friction[self.cube_geom_id] = (
            self.nominal_friction * friction_scale
        )

        damping_scale = rng.uniform(
            self.damping_range[0], self.damping_range[1], size=16
        )
        self.model.dof_damping[:16] = self.nominal_damping * damping_scale

        self.current_latency_steps = int(rng.integers(0, self.max_latency + 1))
        self.obs_queue.clear()

        # ponytail: update derived model constants in MuJoCo after in-place mutations
        mujoco.mj_setConst(self.model, self.data)

    def reset_to_nominal(self) -> None:
        """Restores model parameters to nominal values and resets latency queue."""
        self.model.body_mass[self.cube_body_id] = self.nominal_mass
        self.model.body_inertia[self.cube_body_id] = self.nominal_inertia.copy()
        self.model.geom_friction[self.cube_geom_id] = self.nominal_friction.copy()
        self.model.dof_damping[:16] = self.nominal_damping.copy()
        self.current_latency_steps = 0
        self.obs_queue.clear()
        mujoco.mj_setConst(self.model, self.data)

    def reset_latency(
        self, initial_obs: np.ndarray | torch.Tensor | None = None
    ) -> None:
        """Clears observation delay queue and optionally stores initial observation."""
        self.obs_queue.clear()
        if initial_obs is not None:
            if isinstance(initial_obs, torch.Tensor):
                arr = initial_obs.detach().cpu().numpy().copy()
            else:
                arr = np.asarray(initial_obs).copy()
            self.obs_queue.append(arr)

    def get_delayed_observation(
        self,
        raw_obs: np.ndarray | torch.Tensor,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray | torch.Tensor:
        """Appends raw observation to buffer and returns delayed observation with sensor noise."""
        is_torch = isinstance(raw_obs, torch.Tensor)
        if is_torch:
            arr = raw_obs.detach().cpu().numpy().copy()
        else:
            arr = np.asarray(raw_obs).copy()

        self.obs_queue.append(arr)
        # ponytail: clamp index to return oldest available observation during initial warm-up
        idx = max(0, len(self.obs_queue) - 1 - self.current_latency_steps)
        delayed_arr = self.obs_queue[idx].copy()

        if self.current_sensor_noise_std > 0:
            if rng is None:
                rng = np.random.default_rng()
            noise = rng.normal(
                0.0, self.current_sensor_noise_std, size=delayed_arr.shape
            )
            delayed_arr = (delayed_arr + noise).astype(delayed_arr.dtype)

        if is_torch:
            return torch.as_tensor(
                delayed_arr, device=raw_obs.device, dtype=raw_obs.dtype
            )
        return delayed_arr


class ADRCurriculumController:
    """Automatic Domain Randomization (ADR) curriculum controller.

    Ramps randomization difficulty from 0.0 to 1.0 as evaluation success rate
    meets or exceeds `start_threshold`, targeting benchmark success at `target_threshold`.
    """

    def __init__(
        self,
        randomizer: DomainRandomizer,
        start_threshold: float = 0.50,
        target_threshold: float = 0.85,
        step_size: float = 0.05,
        initial_difficulty: float = 0.0,
    ) -> None:
        """Initializes ADR controller.

        Args:
            randomizer: DomainRandomizer instance to modulate.
            start_threshold: Minimum success rate to trigger difficulty progression.
            target_threshold: Upper benchmark success rate goal targeted at difficulty 1.0.
            step_size: Step increment for difficulty adjustments.
            initial_difficulty: Initial difficulty in [0.0, 1.0].
        """
        self.randomizer = randomizer
        self.start_threshold = start_threshold
        self.target_threshold = target_threshold
        self.step_size = step_size
        self._difficulty = 0.0
        self.set_difficulty(initial_difficulty)

    @property
    def difficulty(self) -> float:
        return self._difficulty

    def set_difficulty(self, level: float) -> None:
        self._difficulty = float(np.clip(level, 0.0, 1.0))
        self.randomizer.set_difficulty(self._difficulty)

    def update(self, success_rate: float) -> float:
        """Updates ADR difficulty based on evaluation success rate."""
        if success_rate >= self.start_threshold:
            new_diff = min(1.0, self.difficulty + self.step_size)
            self.set_difficulty(new_diff)
        elif success_rate < self.start_threshold * 0.5:
            new_diff = max(0.0, self.difficulty - self.step_size * 0.5)
            self.set_difficulty(new_diff)
        return self._difficulty
