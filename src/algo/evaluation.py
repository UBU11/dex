import torch
from torch import nn
from torchrl.envs.utils import step_mdp

from src.envs.leap_cube_env import LeapCubeReorientEnv
from src.utils.quaternion import quat_geodesic_distance


def evaluate_policy(
    actor: nn.Module,
    env: LeapCubeReorientEnv,
    num_episodes: int = 10,
    max_steps: int | None = None,
    device: torch.device | None = None,
) -> dict[str, float]:
    if num_episodes <= 0:
        return {
            "success_rate": 0.0,
            "mean_consecutive_successes": 0.0,
            "mean_geodesic_error": 0.0,
            "mean_return": 0.0,
            "difficulty": (
                float(env.curriculum_controller.difficulty)
                if getattr(env, "curriculum_controller", None) is not None
                else 0.0
            ),
        }

    target_device = (
        device
        if device is not None
        else next(actor.parameters(), torch.tensor(0.0)).device
    )

    was_training = actor.training
    actor.eval()

    episode_consecutive_successes: list[float] = []
    episode_returns: list[float] = []
    all_geodesic_errors: list[float] = []

    limit_steps = max_steps if max_steps is not None else getattr(env, "max_steps", 400)

    try:
        with torch.no_grad():
            for _ in range(num_episodes):
                td = env.reset()
                ep_return = 0.0
                step_count = 0

                while step_count < limit_steps:
                    obs = td["obs"].to(target_device)
                    desired_goal = td["desired_goal"].to(target_device)

                    action, _ = actor(obs, desired_goal, deterministic=True)
                    td["action"] = action.to(td.device)

                    next_td = env.step(td)

                    if ("next", "reward") in next_td.keys(True):
                        reward = next_td["next", "reward"]
                    else:
                        reward = next_td["reward"]

                    if ("next", "achieved_goal") in next_td.keys(True):
                        achieved_goal = next_td["next", "achieved_goal"]
                    else:
                        achieved_goal = next_td["achieved_goal"]

                    geo_err = quat_geodesic_distance(
                        achieved_goal.unsqueeze(0) if achieved_goal.ndim == 1 else achieved_goal,
                        desired_goal.to(achieved_goal.device).unsqueeze(0)
                        if desired_goal.ndim == 1
                        else desired_goal.to(achieved_goal.device),
                    ).mean().item()
                    all_geodesic_errors.append(geo_err)

                    ep_return += float(reward.sum().item())
                    step_count += 1

                    if ("next", "done") in next_td.keys(True):
                        done = bool(next_td["next", "done"].any().item())
                    else:
                        done = bool(next_td["done"].any().item())

                    if done:
                        break

                    td = step_mdp(next_td)

                consecutive = float(getattr(env, "consecutive_successes", 0))
                episode_consecutive_successes.append(consecutive)
                episode_returns.append(ep_return)
    finally:
        if was_training:
            actor.train()

    successes = sum(1.0 for c in episode_consecutive_successes if c >= 1.0)
    success_rate = successes / num_episodes
    mean_consecutive = sum(episode_consecutive_successes) / num_episodes
    mean_geo_err = (
        sum(all_geodesic_errors) / len(all_geodesic_errors)
        if all_geodesic_errors
        else 0.0
    )
    mean_return = sum(episode_returns) / num_episodes

    difficulty = (
        float(env.curriculum_controller.difficulty)
        if getattr(env, "curriculum_controller", None) is not None
        else (
            float(env.randomizer.difficulty)
            if getattr(env, "randomizer", None) is not None
            else 0.0
        )
    )

    return {
        "success_rate": float(success_rate),
        "mean_consecutive_successes": float(mean_consecutive),
        "mean_geodesic_error": float(mean_geo_err),
        "mean_return": float(mean_return),
        "difficulty": float(difficulty),
    }
