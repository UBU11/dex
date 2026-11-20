import math

import torch
import torch.nn.functional as F
from torch import nn

LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0


def stable_tanh_jacobian_correction(u: torch.Tensor) -> torch.Tensor:
    """Computes log(1 - tanh(u)^2) stably via softplus identity.

    Avoids catastrophic float underflow/infinities when |u| >= 4:
    log(1 - tanh(u)^2) = 2.0 * (ln(2) - u - softplus(-2u))
    """
    # ponytail: exact analytical identity replaces naive tanh^2 subtraction
    return 2.0 * (math.log(2.0) - u - F.softplus(-2.0 * u))


class SquashedGaussianSACActor(nn.Module):
    """Soft Actor-Critic Gaussian policy with Tanh squashing and stable log-prob.

    Outputs actions in (-1, 1)^action_dim for a 16-DoF robot hand.
    """

    def __init__(
        self,
        obs_dim: int,
        goal_dim: int,
        action_dim: int = 16,
        hidden_dim: int = 512,
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.goal_dim = goal_dim
        self.action_dim = action_dim

        # ponytail: standard two-layer MLP with LayerNorm and SiLU activations
        self.net = nn.Sequential(
            nn.Linear(obs_dim + goal_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        self.mu_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)

    def forward(
        self,
        obs: torch.Tensor,
        desired_goal: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute action and squashed log-probability.

        Args:
            obs: Observation tensor of shape (..., obs_dim)
            desired_goal: Goal tensor of shape (..., goal_dim)
            deterministic: If True, uses mean directly (eval mode)

        Returns:
            action: Squashed action tensor in (-1, 1) of shape (..., action_dim)
            log_prob: Corrected log-probability of shape (..., 1)
        """
        x = torch.cat([obs, desired_goal], dim=-1)
        h = self.net(x)
        mu = self.mu_head(h)
        log_std = torch.clamp(self.log_std_head(h), LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        if deterministic:
            u = mu
        else:
            # Reparameterization trick: u = mu + std * eps
            eps = torch.randn_like(mu)
            u = mu + std * eps

        # ponytail: clamp tanh to strictly open interval (-1, 1) preventing FP32 saturation
        action = torch.clamp(torch.tanh(u), -1.0 + 1e-7, 1.0 - 1e-7)

        # Unsquashed Gaussian log-prob
        log_prob_u = -0.5 * (
            ((u - mu) / (std + 1e-8)) ** 2 + 2.0 * log_std + math.log(2.0 * math.pi)
        )

        # Numerically stable Jacobian correction for Tanh squashing
        jacobian_correction = stable_tanh_jacobian_correction(u)

        # Sum over action dimensions
        log_prob = (log_prob_u - jacobian_correction).sum(dim=-1, keepdim=True)
        return action, log_prob
