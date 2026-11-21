import inspect
from collections.abc import Callable
from functools import partial
from typing import Any

import torch
from tensordict import TensorDict
from torch import nn
from torchrl import collectors
from torchrl.collectors import Collector
from torchrl.data import TensorDictReplayBuffer
from torchrl.envs import EnvBase, ParallelEnv

from src.algo.actor import SquashedGaussianSACActor

SyncDataCollector = getattr(collectors, "SyncDataCollector", Collector)


def _supports_kwarg(fn: Callable[..., Any], kwarg: str) -> bool:
    """Checks whether a callable accepts a given keyword argument or **kwargs."""
    try:
        sig = inspect.signature(fn)
        return any(
            p.kind == inspect.Parameter.VAR_KEYWORD or p.name == kwarg
            for p in sig.parameters.values()
        )
    except (ValueError, TypeError):
        return False


def _make_worker_env(
    base_env_fn: Callable[..., EnvBase],
    enable_adr: bool = False,
    seed: int | None = None,
) -> EnvBase:
    """Instantiates a worker environment with optional domain randomization and distinct seed."""
    kwargs: dict[str, Any] = {}
    if enable_adr and _supports_kwarg(base_env_fn, "enable_domain_randomization"):
        kwargs["enable_domain_randomization"] = True
    if seed is not None and _supports_kwarg(base_env_fn, "seed"):
        kwargs["seed"] = seed

    env = base_env_fn(**kwargs)

    if seed is not None and hasattr(env, "set_seed") and "seed" not in kwargs:
        try:
            env.set_seed(seed)
        except (AttributeError, RuntimeError, TypeError):
            pass

    if (
        enable_adr
        and hasattr(env, "enable_domain_randomization")
        and not env.enable_domain_randomization
    ):
        env.enable_domain_randomization = True
        if getattr(env, "randomizer", None) is None:
            from src.envs.domain_randomization import (
                ADRCurriculumController,
                DomainRandomizer,
            )

            env.randomizer = DomainRandomizer(env.model, env.data)
            env.curriculum_controller = ADRCurriculumController(env.randomizer)
            env.domain_randomizer = env.randomizer

    return env


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
    num_workers: int = 1,
    seed: int | None = None,
    **kwargs: Any,
) -> Collector:
    """Creates a SyncDataCollector configured for trajectory-mode writing into replay buffer.

    Args:
        env_fn: Factory function returning an environment instance.
        policy: Policy module used for action generation during rollouts.
        replay_buffer: Target replay buffer for trajectory storage.
        frames_per_batch: Number of frames to collect per iteration.
        total_frames: Total number of frames before collector termination (-1 for infinite).
        enable_domain_randomization: Whether to wrap environments with ADR domain randomization.
        num_workers: Number of parallel environment workers. When > 1, wraps in ParallelEnv.
        seed: Base random seed for environment workers.
        **kwargs: Additional keyword arguments forwarded to SyncDataCollector.
    """
    if num_workers < 1:
        raise ValueError(f"num_workers must be at least 1, got {num_workers}.")

    if num_workers > 1:
        base_seed = (
            seed
            if seed is not None
            else int(torch.randint(0, 2**31 - 1, (1,)).item())
        )
        workers = [
            partial(
                _make_worker_env,
                base_env_fn=env_fn,
                enable_adr=enable_domain_randomization,
                seed=base_seed + i * 1000,
            )
            for i in range(num_workers)
        ]
        penv = ParallelEnv(num_workers, workers)
        penv.set_seed(base_seed)
        env_fn = penv
    elif enable_domain_randomization or seed is not None:
        env_fn = partial(
            _make_worker_env,
            base_env_fn=env_fn,
            enable_adr=enable_domain_randomization,
            seed=seed,
        )

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

