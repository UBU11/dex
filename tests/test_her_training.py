import math
import sys
from pathlib import Path

import numpy as np
import torch
from tensordict import TensorDict

# Ensure repository root is on sys.path for direct pytest invocation
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.algo import (
    PolicyModule,
    SACTrainer,
    SquashedGaussianSACActor,
    TwinContinuousQCritic,
    create_her_replay_buffer,
    create_trajectory_collector,
)
from src.envs import LeapCubeReorientEnv


def test_create_trajectory_collector_wiring():
    """Verify create_trajectory_collector initializes collector with replay_write_mode='trajectory'."""
    env_fn = lambda: LeapCubeReorientEnv(max_steps=10)
    actor = SquashedGaussianSACActor(obs_dim=55, goal_dim=4, action_dim=16)
    policy = PolicyModule(actor)
    rb = create_her_replay_buffer(storage_size=200, slice_len=10, batch_size=20)

    collector = create_trajectory_collector(
        env_fn=env_fn,
        policy=policy,
        replay_buffer=rb,
        frames_per_batch=20,
        total_frames=20,
    )

    try:
        assert getattr(collector, "replay_write_mode", None) == "trajectory"
    finally:
        collector.shutdown()


def test_policy_module_forward():
    """Verify PolicyModule forward pass produces expected actions in [-1, 1] for TorchRL collectors."""
    actor = SquashedGaussianSACActor(obs_dim=55, goal_dim=4, action_dim=16)
    policy = PolicyModule(actor)

    td = TensorDict(
        {
            "obs": torch.randn(8, 55),
            "desired_goal": torch.randn(8, 4),
        },
        batch_size=[8],
    )

    out = policy(td)
    assert "action" in out
    assert out["action"].shape == (8, 16)
    assert torch.all(out["action"] >= -1.0) and torch.all(out["action"] <= 1.0)
    assert torch.isfinite(out["action"]).all()


def test_full_so3_sparse_her_training_verification():
    """
    Verify full SO(3) sparse environment trajectory collection into HER buffer,
    sampling batches with relabeled positive reward ratio in [0.15, 0.38],
    and executing SACTrainer optimization steps with finite losses and alpha.
    """
    torch.manual_seed(42)
    np.random.seed(42)

    env_fn = lambda: LeapCubeReorientEnv(
        max_steps=20,
        target_mode="so3",
        reward_mode="sparse",
        seed=42,
    )
    actor = SquashedGaussianSACActor(obs_dim=55, goal_dim=4, action_dim=16)
    critic = TwinContinuousQCritic(obs_dim=55, goal_dim=4, action_dim=16)
    trainer = SACTrainer(actor=actor, critic=critic)
    policy = PolicyModule(actor)

    rb = create_her_replay_buffer(storage_size=2000, slice_len=20, batch_size=200)

    collector = create_trajectory_collector(
        env_fn=env_fn,
        policy=policy,
        replay_buffer=rb,
        frames_per_batch=100,
        total_frames=200,
    )

    try:
        for _ in collector:
            pass
    finally:
        collector.shutdown()

    assert len(rb) >= 200

    metrics, pos_ratio = trainer.train_step_from_buffer(rb)
    assert 0.15 <= pos_ratio <= 0.38, (
        f"Expected pos_ratio in [0.15, 0.38], got {pos_ratio:.4f}"
    )
    for k, v in metrics.items():
        assert math.isfinite(v), f"Metric {k} is not finite: {v}"
    assert metrics["alpha"] > 0.0
