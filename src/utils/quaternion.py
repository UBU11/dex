import math

import torch
import torch.nn.functional as F


def quat_geodesic_distance(
    q_achieved: torch.Tensor, q_desired: torch.Tensor
) -> torch.Tensor:
    """
    Computes geodesic angular distance (in radians) between unit quaternions in SO(3).
    Accounts for double-cover (q == -q) and clamps to prevent NaN gradients in acos.
    Shape: (..., 4) -> (..., 1)
    """
    q_a = F.normalize(q_achieved, p=2, dim=-1)
    q_d = F.normalize(q_desired, p=2, dim=-1)
    # ponytail: absolute inner product for SO(3) double cover clamped strictly to [0.0, 1.0 - 1e-7]
    dot = torch.abs(torch.sum(q_a * q_d, dim=-1, keepdim=True))
    dot_clamped = torch.clamp(dot, min=0.0, max=1.0 - 1e-7)
    return 2.0 * torch.acos(dot_clamped)


def compute_sparse_her_reward(
    q_achieved: torch.Tensor,
    q_desired: torch.Tensor,
    cube_z: torch.Tensor,
    angle_tol: float = 0.15,
    drop_z_threshold: float = 0.04,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Computes sparse binary reward and success flag for HER relabeling and env stepping.
    Returns:
        reward: 0.0 if (angle < angle_tol and cube_z > drop_z_threshold) else -1.0
        is_success: boolean tensor indicating success
    """
    ang_err = quat_geodesic_distance(q_achieved, q_desired)
    # ponytail: align cube_z shape with distance tensor to prevent unintended broadcast
    z = cube_z.reshape_as(ang_err) if cube_z.numel() == ang_err.numel() else cube_z
    not_dropped = z > drop_z_threshold
    is_success = (ang_err < angle_tol) & not_dropped
    # ponytail: sparse reward: 0.0 on goal achievement, -1.0 everywhere else
    reward = torch.where(
        is_success,
        torch.zeros_like(ang_err),
        torch.full_like(ang_err, -1.0),
    )
    return reward, is_success


def compute_dense_orientation_reward(
    q_achieved: torch.Tensor,
    q_desired: torch.Tensor,
    action: torch.Tensor,
    action_penalty_weight: float = 0.01,
) -> torch.Tensor:
    """
    Computes dense negative orientation tracking reward with action L2 penalty:
    r_t = -theta(q_achieved, q_desired) - action_penalty_weight * ||a_t||^2
    """
    theta = quat_geodesic_distance(q_achieved, q_desired)
    action_t = (
        action
        if isinstance(action, torch.Tensor)
        else torch.as_tensor(action, device=theta.device, dtype=theta.dtype)
    )
    action_norm_sq = torch.sum(action_t**2, dim=-1, keepdim=True)
    return -theta - action_penalty_weight * action_norm_sq


def sample_uniform_quat(
    batch_size: int, device: torch.device | None = None
) -> torch.Tensor:
    """
    Samples uniform random quaternions in SO(3) using Shoemake's subgroup algorithm.
    Returns:
        Tensor of shape (batch_size, 4) in [w, x, y, z] convention.
    """
    if batch_size <= 0:
        return torch.empty((0, 4), device=device)

    # ponytail: Shoemake (1992) uniform SO(3) rotation sampling in [w, x, y, z] format
    u = torch.rand((batch_size, 3), device=device)
    u1, u2, u3 = u[:, 0], u[:, 1], u[:, 2]
    w = torch.sqrt(1.0 - u1) * torch.sin(2.0 * math.pi * u2)
    x = torch.sqrt(1.0 - u1) * torch.cos(2.0 * math.pi * u2)
    y = torch.sqrt(u1) * torch.sin(2.0 * math.pi * u3)
    z = torch.sqrt(u1) * torch.cos(2.0 * math.pi * u3)
    return torch.stack([w, x, y, z], dim=-1)


def sample_yaw_quat(
    batch_size: int, device: torch.device | None = None
) -> torch.Tensor:
    """
    Samples single-axis Z-yaw random quaternions [cos(psi/2), 0, 0, sin(psi/2)] for psi in [-pi, pi].
    Returns:
        Tensor of shape (batch_size, 4) in [w, x, y, z] convention.
    """
    if batch_size <= 0:
        return torch.empty((0, 4), device=device)

    psi = (torch.rand((batch_size,), device=device) * 2.0 - 1.0) * math.pi
    half_psi = psi * 0.5
    w = torch.cos(half_psi)
    x = torch.zeros_like(w)
    y = torch.zeros_like(w)
    z = torch.sin(half_psi)
    return torch.stack([w, x, y, z], dim=-1)

