"""Domain randomization and observation latency buffer for LEAP Hand cube manipulation."""

from collections import deque

import mujoco
import numpy as np
import torch


class DomainRandomizer:
    """Domain randomizer for LEAP Hand and cube manipulation physics.

    Randomizes:
    - Cube mass: [0.5x, 1.5x] of nominal (~70g to 210g)
    - Cube surface friction (sliding, torsional, rolling): [0.6x, 1.4x] of nominal
    - Hand joint damping: [0.7x, 1.3x] of nominal
    - Observation latency queue: 0, 1, or 2 steps of latency
    - Always synchronizes model via mujoco.mj_setConst(model, data).
    """

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
    ) -> None:
        self.model = model
        self.data = data
        self.max_latency = max_latency
        self.mass_range = mass_range
        self.friction_range = friction_range
        self.damping_range = damping_range

        # Locate cube body and geom IDs
        self.cube_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, cube_body_name)
        if self.cube_body_id == -1:
            raise ValueError(f"Body '{cube_body_name}' not found in model.")

        self.cube_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, cube_geom_name)
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

    def randomize(self, rng: np.random.Generator | None = None) -> None:
        """Mutates model parameters and latency, calling mj_setConst."""
        if rng is None:
            rng = np.random.default_rng()

        # 1. Cube Mass: [0.5x, 1.5x] of nominal
        mass_scale = float(rng.uniform(self.mass_range[0], self.mass_range[1]))
        self.model.body_mass[self.cube_body_id] = self.nominal_mass * mass_scale
        self.model.body_inertia[self.cube_body_id] = self.nominal_inertia * mass_scale

        # 2. Cube Surface Friction (sliding, torsional, rolling): [0.6x, 1.4x] of nominal
        friction_scale = rng.uniform(self.friction_range[0], self.friction_range[1], size=3)
        self.model.geom_friction[self.cube_geom_id] = self.nominal_friction * friction_scale

        # 3. Hand Joint Damping: [0.7x, 1.3x] of nominal
        damping_scale = rng.uniform(self.damping_range[0], self.damping_range[1], size=16)
        self.model.dof_damping[:16] = self.nominal_damping * damping_scale

        # 4. Latency steps: 0, 1, or 2 steps
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

    def reset_latency(self, initial_obs: np.ndarray | torch.Tensor | None = None) -> None:
        """Clears observation delay queue and optionally stores initial observation."""
        self.obs_queue.clear()
        if initial_obs is not None:
            if isinstance(initial_obs, torch.Tensor):
                arr = initial_obs.detach().cpu().numpy().copy()
            else:
                arr = np.asarray(initial_obs).copy()
            self.obs_queue.append(arr)

    def get_delayed_observation(
        self, raw_obs: np.ndarray | torch.Tensor
    ) -> np.ndarray | torch.Tensor:
        """Appends raw observation to buffer and returns delayed observation."""
        is_torch = isinstance(raw_obs, torch.Tensor)
        if is_torch:
            arr = raw_obs.detach().cpu().numpy().copy()
        else:
            arr = np.asarray(raw_obs).copy()

        self.obs_queue.append(arr)
        # ponytail: clamp index to return oldest available observation during initial warm-up
        idx = max(0, len(self.obs_queue) - 1 - self.current_latency_steps)
        delayed_arr = self.obs_queue[idx]

        if is_torch:
            return torch.as_tensor(delayed_arr, device=raw_obs.device, dtype=raw_obs.dtype)
        return delayed_arr
