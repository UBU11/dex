from collections.abc import Callable
from typing import Any

from tensordict import TensorDict
from torch import nn
from torchrl import collectors
from torchrl.collectors import Collector
from torchrl.data import TensorDictReplayBuffer
from torchrl.envs import EnvBase

from src.algo.actor import SquashedGaussianSACActor

SyncDataCollector = getattr(collectors, "SyncDataCollector", Collector)


class PolicyModule(nn.Module):
    """Wraps SquashedGaussianSACActor for TorchRL collectors."""

    def __init__(
        self, actor: SquashedGaussianSACActor, deterministic: bool = False
    ) -> None:
        super().__init__()
        self.actor = actor
        self.deterministic = deterministic

    def forward(self, td: TensorDict) -> TensorDict:
        obs = td["obs"]
        desired_goal = td["desired_goal"]
        action, _ = self.actor(obs, desired_goal, deterministic=self.deterministic)
        td["action"] = action
        return td


def create_trajectory_collector(
    env_fn: Callable[[], EnvBase],
    policy: nn.Module,
    replay_buffer: TensorDictReplayBuffer,
    frames_per_batch: int,
    total_frames: int = -1,
    **kwargs: Any,
) -> Any:
    """Creates a SyncDataCollector configured for trajectory-mode writing into replay buffer."""
    collector_kwargs = {
        "replay_write_mode": "trajectory",
        "auto_register_policy_transforms": False,
    }
    collector_kwargs.update(kwargs)

    return SyncDataCollector(
        create_env_fn=env_fn,
        policy=policy,
        replay_buffer=replay_buffer,
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
        **collector_kwargs,
    )
