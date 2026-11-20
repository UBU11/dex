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
    print("Initializing full SO(3) sparse LeapCubeReorientEnv with Domain Randomization & SAC models...")
    env_fn = lambda: LeapCubeReorientEnv(
        max_steps=20,
        target_mode="so3",
        reward_mode="sparse",
        enable_domain_randomization=True,
    )

    demo_env = env_fn()
    assert demo_env.curriculum_controller is not None
    assert demo_env.randomizer is not None
    print(f"Initial ADR difficulty: {demo_env.curriculum_controller.difficulty:.2f}")
    print(f"Nominal cube mass: {demo_env.randomizer.nominal_mass:.4f} kg")
    print(f"Initial mass scale range: {demo_env.randomizer.mass_range}")

    # Demonstrate curriculum difficulty ramping
    demo_env.curriculum_controller.update(success_rate=0.80)
    print(f"Ramped ADR difficulty (after 80% success rate): {demo_env.curriculum_controller.difficulty:.2f}")
    print(f"Scaled mass range: {demo_env.randomizer.mass_range}")
    print(f"Scaled max latency steps: {demo_env.randomizer.max_latency}")

    actor = SquashedGaussianSACActor(obs_dim=55, goal_dim=4, action_dim=16)
    critic = TwinContinuousQCritic(obs_dim=55, goal_dim=4, action_dim=16)
    policy = PolicyModule(actor)
    trainer = SACTrainer(actor=actor, critic=critic)

    # 2. Configure HER replay buffer
    print("\nConfiguring HER replay buffer (storage_size=2000, slice_len=20, batch_size=200)...")
    replay_buffer = create_her_replay_buffer(
        storage_size=2000, slice_len=20, batch_size=200
    )

    # 3. Create trajectory collector
    print("Creating SyncDataCollector with trajectory write mode and domain randomization...")
    collector = create_trajectory_collector(
        env_fn=env_fn,
        policy=policy,
        replay_buffer=replay_buffer,
        frames_per_batch=100,
        total_frames=200,
        enable_domain_randomization=True,
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
        metrics, pos_ratio = trainer.train_step_from_buffer(replay_buffer)

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
