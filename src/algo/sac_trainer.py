import copy
import math

import torch
import torch.nn.functional as F
from tensordict import TensorDict
from torch import nn

from src.algo.actor import SquashedGaussianSACActor
from src.algo.critic import TwinContinuousQCritic
from src.utils.quaternion import quat_geodesic_distance


class SACLoss(nn.Module):
    """Soft Actor-Critic loss with Polyak target network and auto-alpha temperature."""

    def __init__(
        self,
        actor: SquashedGaussianSACActor,
        critic: TwinContinuousQCritic,
        target_entropy: float = -16.0,
        alpha_init: float = 1.0,
        gamma: float = 0.99,
        tau: float = 0.005,
    ) -> None:
        super().__init__()
        self.actor = actor
        self.critic = critic
        self.target_critic = copy.deepcopy(critic)
        for p in self.target_critic.parameters():
            p.requires_grad = False

        self.target_entropy = target_entropy
        self.gamma = gamma
        self.tau = tau

        # ponytail: log_alpha Parameter ensures alpha > 0 via exp()
        self.log_alpha = nn.Parameter(
            torch.tensor(math.log(alpha_init), dtype=torch.float32)
        )

    @property
    def alpha(self) -> float:
        return self.log_alpha.exp().item()

    def update_target_network(self) -> None:
        """Polyak averaging: theta_target <- tau * theta + (1 - tau) * theta_target."""
        with torch.no_grad():
            for p, p_target in zip(
                self.critic.parameters(), self.target_critic.parameters(), strict=True
            ):
                p_target.data.copy_(self.tau * p.data + (1.0 - self.tau) * p_target.data)

    def _extract_batch(
        self, batch: TensorDict
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        obs = batch["obs"]
        desired_goal = batch["desired_goal"]
        action = batch["action"]

        if ("next", "obs") in batch.keys(True):
            next_obs = batch["next", "obs"]
        else:
            next_obs = batch["next"]["obs"]

        if ("next", "desired_goal") in batch.keys(True):
            next_desired_goal = batch["next", "desired_goal"]
        else:
            next_desired_goal = desired_goal

        if ("next", "reward") in batch.keys(True):
            reward = batch["next", "reward"]
        else:
            reward = batch["reward"]

        if ("next", "terminated") in batch.keys(True):
            done = batch["next", "terminated"]
        elif ("next", "done") in batch.keys(True):
            done = batch["next", "done"]
        elif "done" in batch:
            done = batch["done"]
        else:
            done = torch.zeros_like(reward)

        if reward.dim() == 1:
            reward = reward.unsqueeze(-1)
        if done.dim() == 1:
            done = done.unsqueeze(-1)

        return (
            obs,
            desired_goal,
            action,
            next_obs,
            next_desired_goal,
            reward,
            done.float(),
        )

    def critic_loss(self, batch: TensorDict) -> torch.Tensor:
        (
            obs,
            desired_goal,
            action,
            next_obs,
            next_desired_goal,
            reward,
            done,
        ) = self._extract_batch(batch)

        with torch.no_grad():
            next_action, next_log_prob = self.actor(next_obs, next_desired_goal)
            target_q1, target_q2 = self.target_critic(
                next_obs, next_desired_goal, next_action
            )
            target_q = (
                torch.min(target_q1, target_q2) - self.log_alpha.exp() * next_log_prob
            )
            target_value = reward + self.gamma * (1.0 - done) * target_q

        current_q1, current_q2 = self.critic(obs, desired_goal, action)
        return F.mse_loss(current_q1, target_value) + F.mse_loss(
            current_q2, target_value
        )

    def actor_and_alpha_loss(
        self, batch: TensorDict
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        obs = batch["obs"]
        desired_goal = batch["desired_goal"]

        action_pi, log_prob = self.actor(obs, desired_goal)
        q1_pi, q2_pi = self.critic(obs, desired_goal, action_pi)
        min_q_pi = torch.min(q1_pi, q2_pi)

        alpha = self.log_alpha.exp()
        loss_actor = (alpha.detach() * log_prob - min_q_pi).mean()
        loss_alpha = (
            -self.log_alpha * (log_prob.detach() + self.target_entropy)
        ).mean()
        entropy = -log_prob.mean().detach()

        return loss_actor, loss_alpha, entropy

    def forward(self, batch: TensorDict) -> dict[str, torch.Tensor]:
        loss_critic = self.critic_loss(batch)
        loss_actor, loss_alpha, entropy = self.actor_and_alpha_loss(batch)
        return {
            "loss_actor": loss_actor,
            "loss_critic": loss_critic,
            "loss_alpha": loss_alpha,
            "alpha": self.log_alpha.exp().detach(),
            "entropy": entropy,
        }


class SACTrainer:
    """Trainer managing SAC optimization, temperature auto-tuning, and target updates."""

    def __init__(
        self,
        actor: SquashedGaussianSACActor,
        critic: TwinContinuousQCritic,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        alpha_lr: float = 3e-4,
        target_entropy: float = -16.0,
        alpha_init: float = 1.0,
        gamma: float = 0.99,
        tau: float = 0.005,
        device: torch.device | None = None,
    ) -> None:
        self.device = (
            device
            if device is not None
            else next(actor.parameters(), torch.tensor(0.0)).device
        )
        self.actor = actor.to(self.device)
        self.critic = critic.to(self.device)
        self.loss = SACLoss(
            actor=self.actor,
            critic=self.critic,
            target_entropy=target_entropy,
            alpha_init=alpha_init,
            gamma=gamma,
            tau=tau,
        ).to(self.device)

        self.target_entropy = target_entropy
        self.gamma = gamma
        self.tau = tau
        self.log_alpha = self.loss.log_alpha

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)

    @property
    def alpha(self) -> float:
        return self.loss.alpha

    def update_target_network(self) -> None:
        self.loss.update_target_network()

    def step(self, batch: TensorDict) -> dict[str, float]:
        batch = batch.to(self.device)

        # 1. Update Critic
        self.critic_optimizer.zero_grad()
        loss_critic = self.loss.critic_loss(batch)
        loss_critic.backward()
        self.critic_optimizer.step()

        # 2. Update Actor and Alpha
        for p in self.critic.parameters():
            p.requires_grad = False
        try:
            self.actor_optimizer.zero_grad()
            self.alpha_optimizer.zero_grad()
            loss_actor, loss_alpha, entropy = self.loss.actor_and_alpha_loss(batch)
            loss_actor.backward()
            self.actor_optimizer.step()

            loss_alpha.backward()
            self.alpha_optimizer.step()
        finally:
            for p in self.critic.parameters():
                p.requires_grad = True

        # 3. Target network Polyak soft update
        self.update_target_network()

        return {
            "loss_actor": loss_actor.item(),
            "loss_critic": loss_critic.item(),
            "loss_alpha": loss_alpha.item(),
            "alpha": self.alpha,
            "entropy": entropy.item(),
        }

    def evaluate_rollout(self, env, num_steps: int = 50) -> dict[str, float]:
        """Runs a deterministic evaluation rollout in env and computes metrics."""
        was_training = self.actor.training
        self.actor.eval()

        td = env.reset()
        total_reward = 0.0
        geodesic_errors: list[float] = []

        with torch.no_grad():
            for _ in range(num_steps):
                obs = td["obs"]
                desired_goal = td["desired_goal"]

                obs_in = obs.unsqueeze(0) if obs.dim() == 1 else obs
                goal_in = (
                    desired_goal.unsqueeze(0)
                    if desired_goal.dim() == 1
                    else desired_goal
                )

                action, _ = self.actor(
                    obs_in.to(self.device),
                    goal_in.to(self.device),
                    deterministic=True,
                )

                act_to_env = action.squeeze(0).to(td.device)
                td["action"] = act_to_env
                step_td = env.step(td)

                next_td = (
                    step_td["next"] if "next" in step_td.keys(True) else step_td
                )
                if "achieved_goal" in next_td:
                    achieved_goal = next_td["achieved_goal"]
                    err = quat_geodesic_distance(
                        achieved_goal.unsqueeze(0)
                        if achieved_goal.dim() == 1
                        else achieved_goal,
                        desired_goal.unsqueeze(0)
                        if desired_goal.dim() == 1
                        else desired_goal,
                    )
                    geodesic_errors.append(err.mean().item())

                reward = next_td.get("reward", torch.tensor(0.0))
                total_reward += reward.mean().item()

                done = next_td.get("done", torch.tensor(False))
                if bool(done.any()):
                    td = env.reset()
                else:
                    td = next_td

        self.actor.train(was_training)

        mean_geodesic_err = float(
            sum(geodesic_errors) / len(geodesic_errors) if geodesic_errors else 0.0
        )
        return {
            "mean_reward": total_reward / max(num_steps, 1),
            "mean_geodesic_error": mean_geodesic_err,
            "geodesic_error": mean_geodesic_err,
            "alpha": self.alpha,
            "consecutive_successes": float(getattr(env, "consecutive_successes", 0)),
        }
