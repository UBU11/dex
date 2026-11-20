import torch
from torch import nn


def _build_critic_mlp(in_dim: int, hidden_dim: int) -> nn.Sequential:
    """Helper to build a Q-network with LayerNorm and SiLU."""
    # ponytail: minimal 2-hidden-layer MLP with LayerNorm and SiLU
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.SiLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.SiLU(),
        nn.Linear(hidden_dim, 1),
    )


class TwinContinuousQCritic(nn.Module):
    """Twin clipped Q-critic for continuous Soft Actor-Critic (SAC).

    Evaluates Q1(s, g, a) and Q2(s, g, a) to prevent overestimation bias.
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

        in_dim = obs_dim + goal_dim + action_dim
        self.q1_net = _build_critic_mlp(in_dim, hidden_dim)
        self.q2_net = _build_critic_mlp(in_dim, hidden_dim)

    def forward(
        self,
        obs: torch.Tensor,
        desired_goal: torch.Tensor,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute twin Q-values.

        Args:
            obs: Observation tensor of shape (..., obs_dim)
            desired_goal: Goal tensor of shape (..., goal_dim)
            action: Action tensor of shape (..., action_dim)

        Returns:
            q1: First Q-value prediction of shape (..., 1)
            q2: Second Q-value prediction of shape (..., 1)
        """
        x = torch.cat([obs, desired_goal, action], dim=-1)
        return self.q1_net(x), self.q2_net(x)

    def q1(
        self,
        obs: torch.Tensor,
        desired_goal: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """Compute only Q1 value for actor policy loss optimization."""
        x = torch.cat([obs, desired_goal, action], dim=-1)
        return self.q1_net(x)
