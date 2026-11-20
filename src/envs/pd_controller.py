"""Internal PD controller for 16-DoF LEAP Hand relative joint targets."""

import mujoco
import numpy as np


class PDController:
    """Internal PD controller for 16-DoF LEAP Hand relative joint targets.

    Runs low-level tracking at 200 Hz (10 substeps) while RL policy queries at 20 Hz.
    Tracks tau = Kp * (q_target - q) - Kd * qvel clipped to joint limits / ctrlrange.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        kp: float = 3.0,
        kd: float = 0.1,
    ) -> None:
        self.model = model
        self.kp = float(kp)
        self.kd = float(kd)
        self.nu = 16  # 16 actuated joints in LEAP Hand
        self.ctrlrange = model.actuator_ctrlrange[: self.nu].copy()

        # ponytail: synchronize model actuator parameters with requested kp and kd gains
        self.model.actuator_gainprm[: self.nu, 0] = self.kp
        self.model.actuator_biasprm[: self.nu, 1] = -self.kp
        self.model.actuator_biasprm[: self.nu, 2] = -self.kd

    def compute_torques(
        self,
        qpos: np.ndarray,
        qvel: np.ndarray,
        target_qpos: np.ndarray,
    ) -> np.ndarray:
        """Computes PD torques tau = Kp * (target_qpos - q) - Kd * qvel.

        Target qpos is clipped to actuator joint limits prior to error calculation.
        """
        clipped_target = np.clip(
            target_qpos[: self.nu],
            self.ctrlrange[:, 0],
            self.ctrlrange[:, 1],
        )
        current_q = qpos[: self.nu]
        current_qvel = qvel[: self.nu]
        return self.kp * (clipped_target - current_q) - self.kd * current_qvel

    def step(
        self,
        data: mujoco.MjData,
        target_qpos: np.ndarray,
        n_substeps: int = 10,
    ) -> None:
        """Tracks target_qpos across n_substeps physics steps.

        Args:
            data: MuJoCo MjData state container.
            target_qpos: Target joint angles for the 16 LEAP actuators.
            n_substeps: Number of physics simulation substeps (default 10 at 200 Hz).
        """
        clipped_target = np.clip(
            target_qpos[: self.nu],
            self.ctrlrange[:, 0],
            self.ctrlrange[:, 1],
        )
        data.ctrl[: self.nu] = clipped_target

        # ponytail: leverage implicitfast solver for unconditionally stable servo tracking
        for _ in range(n_substeps):
            mujoco.mj_step(self.model, data)

    def step_delta(
        self,
        data: mujoco.MjData,
        delta_qpos: np.ndarray,
        n_substeps: int = 10,
        max_delta: float = 0.15,
    ) -> None:
        """Applies relative action delta_q clipped to [-max_delta, max_delta].

        Args:
            data: MuJoCo MjData state container.
            delta_qpos: Relative joint angle deltas.
            n_substeps: Number of physics substeps.
            max_delta: Maximum delta per control step (nominal 0.15 rad).
        """
        delta = np.clip(delta_qpos[: self.nu], -max_delta, max_delta)
        target_qpos = data.qpos[: self.nu] + delta
        self.step(data, target_qpos, n_substeps=n_substeps)
