import math
import sys
from pathlib import Path

import torch

# Ensure repository root is on sys.path for direct pytest invocation
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.quaternion import (
    compute_sparse_her_reward,
    quat_geodesic_distance,
    sample_uniform_quat,
)


def test_quaternion_double_cover():
    """Geodesic distance d(q, -q) must be ~0.0 due to SO(3) double cover."""
    q = torch.tensor([[0.5, 0.5, 0.5, 0.5]], dtype=torch.float32)
    q_neg = -q

    dist_identical = quat_geodesic_distance(q, q)
    dist_neg = quat_geodesic_distance(q, q_neg)

    assert torch.isclose(dist_identical, torch.zeros_like(dist_identical), atol=2e-3)
    assert torch.isclose(dist_neg, torch.zeros_like(dist_neg), atol=2e-3)
    assert torch.allclose(dist_identical, dist_neg, atol=1e-6)


def test_quaternion_identical_gradient_finite():
    """Geodesic distance between identical quaternions has finite gradients (no NaN)."""
    q1 = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32, requires_grad=True)
    q2 = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)

    dist = quat_geodesic_distance(q1, q2)
    assert torch.isclose(dist, torch.zeros_like(dist), atol=2e-3)

    dist.backward()
    assert q1.grad is not None
    assert torch.isfinite(q1.grad).all()
    assert not torch.isnan(q1.grad).any()


def test_quaternion_orthogonal_rotation():
    """Orthogonal 90-degree (pi/2 rad) rotation yields pi/2 geodesic distance."""
    # Identity rotation vs 90-degree rotation around x-axis
    q_id = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
    # 90 deg = pi/2 -> half-angle = pi/4
    q_rot90 = torch.tensor(
        [[math.cos(math.pi / 4), math.sin(math.pi / 4), 0.0, 0.0]], dtype=torch.float32
    )

    dist = quat_geodesic_distance(q_id, q_rot90)
    expected = math.pi / 2
    assert math.isclose(dist.item(), expected, rel_tol=1e-4)


def test_quaternion_clamping_prevents_nan_edge_cases():
    """Clamping ensures that dot products >= 1.0 (drift/precision) do not produce NaNs."""
    # Unnormalized inputs with dot product > 1.0 before or after clamp
    q1 = torch.tensor(
        [[1.00001, 0.0, 0.0, 0.0]], dtype=torch.float32, requires_grad=True
    )
    q2 = torch.tensor([[1.00001, 0.0, 0.0, 0.0]], dtype=torch.float32)

    dist = quat_geodesic_distance(q1, q2)
    assert torch.isfinite(dist).all()
    assert not torch.isnan(dist).any()

    dist.backward()
    assert q1.grad is not None
    assert torch.isfinite(q1.grad).all()
    assert not torch.isnan(q1.grad).any()


def test_sample_uniform_quat_shape_and_norm():
    """Shoemake uniform sampling returns unit quaternions with shape (B, 4)."""
    batch_size = 512
    device = torch.device("cpu")
    quats = sample_uniform_quat(batch_size, device=device)

    assert quats.shape == (batch_size, 4)
    assert quats.device == device

    norms = torch.linalg.norm(quats, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_sample_uniform_quat_empty_batch():
    """Sampling with batch_size 0 returns an empty tensor with shape (0, 4)."""
    quats = sample_uniform_quat(0)
    assert quats.shape == (0, 4)


def test_compute_sparse_her_reward_cases():
    """
    Reward is 0.0 when angle < angle_tol and cube_z > drop_z_threshold;
    Reward is -1.0 everywhere else.
    """
    angle_tol = 0.15
    drop_z_threshold = 0.04

    # Case 1: Achieved == Desired (angle ~ 0 < 0.15), cube_z = 0.05 > 0.04 -> (0.0, True)
    q_achieved_1 = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    q_desired_1 = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    z_1 = torch.tensor([[0.05]])
    r1, s1 = compute_sparse_her_reward(
        q_achieved_1, q_desired_1, z_1, angle_tol, drop_z_threshold
    )
    assert r1.item() == 0.0
    assert s1.item() is True

    # Case 2: Orthogonal rotation (angle = pi/2 > 0.15), cube_z = 0.05 > 0.04 -> (-1.0, False)
    q_desired_2 = torch.tensor(
        [[math.cos(math.pi / 4), math.sin(math.pi / 4), 0.0, 0.0]]
    )
    r2, s2 = compute_sparse_her_reward(
        q_achieved_1, q_desired_2, z_1, angle_tol, drop_z_threshold
    )
    assert r2.item() == -1.0
    assert s2.item() is False

    # Case 3: Achieved == Desired, but cube dropped (cube_z = 0.02 < 0.04) -> (-1.0, False)
    z_dropped = torch.tensor([[0.02]])
    r3, s3 = compute_sparse_her_reward(
        q_achieved_1, q_desired_1, z_dropped, angle_tol, drop_z_threshold
    )
    assert r3.item() == -1.0
    assert s3.item() is False

    # Case 4: Orthogonal rotation and cube dropped -> (-1.0, False)
    r4, s4 = compute_sparse_her_reward(
        q_achieved_1, q_desired_2, z_dropped, angle_tol, drop_z_threshold
    )
    assert r4.item() == -1.0
    assert s4.item() is False


def test_compute_sparse_her_reward_batched():
    """Batched evaluation correctly evaluates elements independently."""
    angle_tol = 0.15
    drop_z_threshold = 0.04

    q_achieved = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
        ]
    )
    q_desired = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],  # match
            [math.cos(math.pi / 4), math.sin(math.pi / 4), 0.0, 0.0],  # mismatch
            [1.0, 0.0, 0.0, 0.0],  # match
            [math.cos(math.pi / 4), math.sin(math.pi / 4), 0.0, 0.0],  # mismatch
        ]
    )
    cube_z = torch.tensor([0.05, 0.05, 0.02, 0.02])  # (4,) 1D tensor

    rewards, successes = compute_sparse_her_reward(
        q_achieved, q_desired, cube_z, angle_tol, drop_z_threshold
    )

    assert rewards.shape == (4, 1)
    assert rewards.dtype == q_achieved.dtype
    assert successes.shape == (4, 1)
    assert torch.equal(rewards, torch.tensor([[0.0], [-1.0], [-1.0], [-1.0]]))
    assert torch.equal(successes, torch.tensor([[True], [False], [False], [False]]))


def test_package_exports():
    """Verify package-level imports from src.utils."""
    from src import utils

    assert hasattr(utils, "quat_geodesic_distance")
    assert hasattr(utils, "compute_sparse_her_reward")
    assert hasattr(utils, "sample_uniform_quat")


def test_random_quaternion_double_cover_and_batch_dims():
    """Double cover holds across arbitrary random quaternions and multi-dimensional batches."""
    q = sample_uniform_quat(64).reshape(8, 8, 4)
    dist_self = quat_geodesic_distance(q, q)
    dist_neg = quat_geodesic_distance(q, -q)

    assert dist_self.shape == (8, 8, 1)
    assert dist_neg.shape == (8, 8, 1)
    assert torch.allclose(dist_self, dist_neg, atol=1e-6)
    assert (dist_self < 2e-3).all()
