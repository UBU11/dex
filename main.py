"""Dexterous In-Hand Cube Reorientation: Policy Rollout Demonstration."""

import torch

from src.algo import SquashedGaussianSACActor
from src.envs import LeapCubeReorientEnv
from src.utils import quat_geodesic_distance


def main() -> None:
    """Runs a 50-step rollout loop executing SquashedGaussianSACActor in LeapCubeReorientEnv."""
    print("=" * 80)
    print("Dexterous In-Hand Cube Reorientation: 50-Step Policy Rollout")
    print("=" * 80)

    # 1. Instantiate environment and actor policy
    print("Instantiating LeapCubeReorientEnv(max_steps=50)...")
    env = LeapCubeReorientEnv(max_steps=50)

    print("Instantiating SquashedGaussianSACActor(obs_dim=55, goal_dim=4, action_dim=16)...")
    actor = SquashedGaussianSACActor(obs_dim=55, goal_dim=4, action_dim=16)

    # 2. Reset environment to obtain initial observation TensorDict
    td = env.reset()

    print("\nStarting 50-step rollout loop:")
    header = (
        f"{'Step':>4} | {'Geodesic Error θ (rad)':>22} | "
        f"{'Reward':>8} | {'Streak':>6} | {'Consecutive Successes':>21}"
    )
    print(header)
    print("-" * len(header))

    # 3. Rollout loop
    for step in range(50):
        obs = td["obs"]
        desired_goal = td["desired_goal"]

        # Policy inference (deterministic action evaluation)
        with torch.no_grad():
            action, _ = actor(obs, desired_goal, deterministic=True)

        # Step environment
        td["action"] = action
        step_td = env.step(td)
        next_td = step_td["next"]

        # Compute geodesic angular distance θ ∈ [0, π]
        achieved_goal = next_td["achieved_goal"]
        current_desired = next_td["desired_goal"]
        angle_err = quat_geodesic_distance(achieved_goal, current_desired).item()
        reward = next_td["reward"].item()
        consecutive_successes = env.consecutive_successes
        streak = env._success_streak

        # Verify no NaNs or Infs
        assert torch.isfinite(next_td["obs"]).all(), f"NaN/Inf detected in obs at step {step + 1}"
        assert torch.isfinite(torch.tensor(angle_err)), (
            f"NaN/Inf detected in angle error at step {step + 1}"
        )
        assert torch.isfinite(next_td["reward"]).all(), (
            f"NaN/Inf detected in reward at step {step + 1}"
        )

        print(
            f"{step + 1:>4} | {angle_err:>22.4f} | {reward:>8.1f} | "
            f"{streak:>6} | {consecutive_successes:>21}"
        )

        td = next_td
        if next_td["done"].item():
            print(f"\nEpisode reached terminal boundary at step {step + 1} (done=True).")
            break

    print("-" * len(header))
    print("Rollout completed successfully without errors or NaNs.")
    print(f"Total chained target reorientations achieved: {env.consecutive_successes}")
    print("=" * 80)


if __name__ == "__main__":
    main()
