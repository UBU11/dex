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
        actor_device = next(self.actor.parameters()).device
        obs = td["obs"]
        desired_goal = td["desired_goal"]
        out_device = td.device if td.device is not None else obs.device
        action, _ = self.actor(
            obs.to(actor_device),
            desired_goal.to(actor_device),
            deterministic=self.deterministic,
        )
        td["action"] = action.to(out_device)
        return td


def create_trajectory_collector(
    env_fn: Callable[[], EnvBase],
    policy: nn.Module,
    replay_buffer: TensorDictReplayBuffer,
    frames_per_batch: int,
    total_frames: int = -1,
    enable_domain_randomization: bool = False,
    **kwargs: Any,
) -> Collector:
    """Creates a SyncDataCollector configured for trajectory-mode writing into replay buffer."""
    if enable_domain_randomization:
        original_env_fn = env_fn

        def wrapped_env_fn() -> EnvBase:
            try:
                return original_env_fn(enable_domain_randomization=True)
            except TypeError:
                env = original_env_fn()
                if (
                    hasattr(env, "enable_domain_randomization")
                    and not env.enable_domain_randomization
                ):
                    env.enable_domain_randomization = True
                    if getattr(env, "randomizer", None) is None:
                        from src.envs.domain_randomization import (
                            ADRCurriculumController,
                            DomainRandomizer,
                        )

                        env.randomizer = DomainRandomizer(env.model, env.data)
                        env.curriculum_controller = ADRCurriculumController(
                            env.randomizer
                        )
                        env.domain_randomizer = env.randomizer
                return env

        env_fn = wrapped_env_fn

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


def update_collector_target_mode(collector: Any, target_mode: str) -> None:
    """Updates target orientation mode across environments in a collector.

    Supports single LeapCubeReorientEnv, ParallelEnv, SerialEnv, or custom batched environments.

    Args:
        collector: Collector instance or environment.
        target_mode: Target orientation mode ("so3" or "yaw").
    """
    if target_mode not in ("so3", "yaw"):
        raise ValueError(
            f"Invalid target_mode '{target_mode}'. Expected 'so3' or 'yaw'."
        )

    # If collector is already an environment, do not access .env as batched envs (e.g. ParallelEnv)
    # intercept __getattr__("env") and dispatch to workers or raise if closed.
    if isinstance(collector, EnvBase):
        env = collector
    elif (
        hasattr(type(collector), "env")
        or isinstance(
            collector,
            (Collector, getattr(collectors, "BaseCollector", Collector)),
        )
        or "env" in getattr(collector, "__dict__", {})
    ):
        env = getattr(collector, "env", collector)
    elif not isinstance(collector, EnvBase) and hasattr(collector, "env"):
        try:
            candidate = collector.env
            env = candidate if candidate is not None else collector
        except (AttributeError, RuntimeError):
            env = collector
    else:
        env = collector

    # Start closed/unstarted batched environments before attribute checks
    try:
        if (
            getattr(env, "is_closed", False)
            and hasattr(env, "start")
            and callable(getattr(env, "start", None))
        ):
            env.start()
    except RuntimeError:
        pass

    # Batched / ParallelEnv / single LeapCubeReorientEnv dispatching set_target_mode
    try:
        has_set_target_mode = hasattr(env, "set_target_mode") and callable(
            getattr(env, "set_target_mode", None)
        )
    except RuntimeError:
        has_set_target_mode = False

    if has_set_target_mode:
        env.set_target_mode(target_mode)
        return

    # Sub-environments list (e.g. SerialEnv or custom batched env)
    if "_envs" in getattr(env, "__dict__", {}) and isinstance(
        env._envs, (list, tuple)
    ):
        for sub_env in env._envs:
            update_collector_target_mode(sub_env, target_mode)
        return

    if "sub_envs" in getattr(env, "__dict__", {}) and isinstance(
        env.sub_envs, (list, tuple)
    ):
        for sub_env in env.sub_envs:
            update_collector_target_mode(sub_env, target_mode)
        return

    # Transformed / wrapped environment
    unwrapped = env
    while (
        hasattr(type(unwrapped), "base_env")
        or "base_env" in getattr(unwrapped, "__dict__", {})
        or (hasattr(unwrapped, "_modules") and "base_env" in unwrapped._modules)
    ):
        unwrapped = unwrapped.base_env
        try:
            if (
                getattr(unwrapped, "is_closed", False)
                and hasattr(unwrapped, "start")
                and callable(getattr(unwrapped, "start", None))
            ):
                unwrapped.start()
        except RuntimeError:
            pass
        try:
            if hasattr(unwrapped, "set_target_mode") and callable(
                getattr(unwrapped, "set_target_mode", None)
            ):
                unwrapped.set_target_mode(target_mode)
                return
        except RuntimeError:
            pass

    # Attribute fallback
    if hasattr(type(env), "target_mode") or "target_mode" in getattr(
        env, "__dict__", {}
    ):
        env.target_mode = target_mode

