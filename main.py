"""Dexterous In-Hand Cube Reorientation: Policy Rollout & Sparse HER Training Demonstration."""

import torch

from src.algo import (
    PolicyModule,
    SACTrainer,
    SquashedGaussianSACActor,
    TwinContinuousQCritic,
    create_her_replay_buffer,
    create_trajectory_collector,
)
from src.envs import LeapCubeReorientEnv


def main() -> None:
    """Runs full SO(3) sparse reorientation trajectory collection, HER relabeling, and SAC training."""
    print("=" * 80)
    print("Dexterous In-Hand Cube Reorientation: Full SO(3) Sparse HER Training Demonstration")
    print("=" * 80)

    # 1. Instantiate environment, policy, critic, and trainer
    print("Initializing full SO(3) sparse LeapCubeReorientEnv and SAC models...")
    env_fn = lambda: LeapCubeReorientEnv(
        max_steps=20,
        target_mode="so3",
        reward_mode="sparse",
    )
    actor = SquashedGaussianSACActor(obs_dim=55, goal_dim=4, action_dim=16)
    critic = TwinContinuousQCritic(obs_dim=55, goal_dim=4, action_dim=16)
    policy = PolicyModule(actor)
    trainer = SACTrainer(actor=actor, critic=critic)

    # 2. Configure HER replay buffer
    print("Configuring HER replay buffer (storage_size=2000, slice_len=20)...")
    replay_buffer = create_her_replay_buffer(storage_size=2000, slice_len=20)

    # 3. Create trajectory collector
    print("Creating SyncDataCollector with trajectory write mode...")
    collector = create_trajectory_collector(
        env_fn=env_fn,
        policy=policy,
        replay_buffer=replay_buffer,
        frames_per_batch=100,
        total_frames=200,
    )

    print("Collecting trajectories into HER replay buffer...")
    try:
        for _ in collector:
            pass
    finally:
        collector.shutdown()

    print(f"Total transitions stored in replay buffer: {len(replay_buffer)}")

    # 4. Perform SAC training steps from HER replay buffer
    print("\nExecuting SAC training steps from HER replay buffer:")
    header = (
        f"{'Step':>4} | {'Loss Critic':>12} | {'Loss Actor':>12} | "
        f"{'Loss Alpha':>12} | {'Alpha':>8} | {'Pos Reward %':>12}"
    )
    print(header)
    print("-" * len(header))

    for step in range(5):
        metrics, pos_ratio = trainer.train_step_from_buffer(replay_buffer, batch_size=200)

        # Assert no NaNs
        assert all(torch.isfinite(torch.tensor(v)) for v in metrics.values())
        assert torch.isfinite(torch.tensor(pos_ratio))

        print(
            f"{step + 1:>4} | {metrics['loss_critic']:>12.4f} | {metrics['loss_actor']:>12.4f} | "
            f"{metrics['loss_alpha']:>12.4f} | {metrics['alpha']:>8.4f} | {pos_ratio * 100:>11.1f}%"
        )

    print("-" * len(header))
    print("HER training demonstration completed successfully without errors or NaNs.")
    print("=" * 80)


if __name__ == "__main__":
    main()
