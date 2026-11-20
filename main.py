"""Dexterous In-Hand Cube Reorientation: Extended Horizon (T=400) & Multi-Target Demo."""

import torch

from src.algo import (
    PolicyModule,
    SACTrainer,
    SquashedGaussianSACActor,
    TwinContinuousQCritic,
    create_her_replay_buffer,
    create_trajectory_collector,
)
from src.envs import DEFAULT_CUBE_POS, LeapCubeReorientEnv


def showcase_reset_hardening(env: LeapCubeReorientEnv) -> None:
    print("\n--- 1. Reset Distribution Hardening Showcase ---")
    for reset_idx in range(3):
        td = env.reset()
        cube_pos = env.data.qpos[16:19].copy()
        cube_quat = env.data.qpos[19:23].copy()
        dx = cube_pos[0] - DEFAULT_CUBE_POS[0]
        dy = cube_pos[1] - DEFAULT_CUBE_POS[1]
        goal_quat = td["desired_goal"].cpu().numpy()
        print(
            f"Reset {reset_idx + 1}: Palm XY Offset = [{dx:+.4f}, {dy:+.4f}] m | "
            f"Init Quat = [{cube_quat[0]:.3f}, {cube_quat[1]:.3f}, {cube_quat[2]:.3f}, {cube_quat[3]:.3f}] | "
            f"Target Quat = [{goal_quat[0]:.3f}, {goal_quat[1]:.3f}, {goal_quat[2]:.3f}, {goal_quat[3]:.3f}]"
        )


def showcase_consecutive_goal_chaining(env: LeapCubeReorientEnv) -> None:
    print("\n--- 2. Multi-Target Consecutive Goal Chaining Showcase ---")
    td = env.reset()
    orig_step_delta = env.pd_controller.step_delta

    def oracle_step_delta(data, delta, **kwargs):
        orig_step_delta(data, delta, **kwargs)
        data.qpos[19:23] = env.desired_goal.cpu().numpy()
        data.qpos[18] = 0.18
        data.qvel[:] = 0.0

    env.pd_controller.step_delta = oracle_step_delta
    current_td = td

    try:
        for chain_target in range(1, 3):
            initial_goal = env.desired_goal.clone().cpu().numpy()
            print(
                f"Tracking Target {chain_target}: "
                f"[{initial_goal[0]:.3f}, {initial_goal[1]:.3f}, {initial_goal[2]:.3f}, {initial_goal[3]:.3f}]"
            )
            for _ in range(5):
                current_td["action"] = torch.zeros(16)
                out = env.step(current_td)
                current_td = out["next"]
            new_goal = env.desired_goal.clone().cpu().numpy()
            print(
                f"  Goal Achieved! Consecutive successes: {env.consecutive_successes} | "
                f"New Target Resampled: [{new_goal[0]:.3f}, {new_goal[1]:.3f}, {new_goal[2]:.3f}, {new_goal[3]:.3f}]"
            )
    finally:
        env.pd_controller.step_delta = orig_step_delta


def main() -> None:
    print("=" * 80)
    print("Dexterous In-Hand Cube Reorientation: Extended Horizon (T=400) & Multi-Target Demo")
    print("=" * 80)

    env_fn = lambda: LeapCubeReorientEnv(
        max_steps=400,
        target_mode="so3",
        reward_mode="sparse",
        enable_domain_randomization=True,
        randomize_initial_orientation=True,
        xy_offset_range=0.015,
    )

    demo_env = env_fn()
    assert demo_env.curriculum_controller is not None
    assert demo_env.randomizer is not None
    print(
        f"Initialized LeapCubeReorientEnv: max_steps={demo_env.max_steps}, "
        f"ADR difficulty={demo_env.curriculum_controller.difficulty:.2f}"
    )

    showcase_reset_hardening(demo_env)
    showcase_consecutive_goal_chaining(demo_env)

    print("\n--- 3. Trajectory Collection (T=400) & HER Replay Buffer ---")
    actor = SquashedGaussianSACActor(obs_dim=55, goal_dim=4, action_dim=16)
    critic = TwinContinuousQCritic(obs_dim=55, goal_dim=4, action_dim=16)
    policy = PolicyModule(actor)
    trainer = SACTrainer(actor=actor, critic=critic)

    replay_buffer = create_her_replay_buffer(
        storage_size=2000, slice_len=20, batch_size=200
    )

    collector = create_trajectory_collector(
        env_fn=env_fn,
        policy=policy,
        replay_buffer=replay_buffer,
        frames_per_batch=400,
        total_frames=400,
        enable_domain_randomization=True,
    )

    print("Collecting full T=400 step trajectory with domain randomization...")
    try:
        for _ in collector:
            pass
    finally:
        collector.shutdown()

    print(f"Trajectory collected successfully: {len(replay_buffer)} transitions in HER replay buffer.")

    print("\n--- 4. SAC Training Steps with HER Relabeling ---")
    header = (
        f"{'Step':>4} | {'Loss Critic':>12} | {'Loss Actor':>12} | "
        f"{'Loss Alpha':>12} | {'Alpha':>8} | {'Pos Reward %':>12}"
    )
    print(header)
    print("-" * len(header))

    for step in range(5):
        metrics, pos_ratio = trainer.train_step_from_buffer(replay_buffer)

        assert all(torch.isfinite(torch.tensor(v)) for v in metrics.values())
        assert torch.isfinite(torch.tensor(pos_ratio))

        print(
            f"{step + 1:>4} | {metrics['loss_critic']:>12.4f} | {metrics['loss_actor']:>12.4f} | "
            f"{metrics['loss_alpha']:>12.4f} | {metrics['alpha']:>8.4f} | {pos_ratio * 100:>11.1f}%"
        )

    print("-" * len(header))
    print("Extended horizon rollout, multi-target chaining, and SAC HER training complete.")
    print("=" * 80)


if __name__ == "__main__":
    main()
