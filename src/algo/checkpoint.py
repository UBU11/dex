from pathlib import Path
from typing import Any

import torch
from torch import nn

from src.algo.sac_trainer import SACTrainer
from src.envs.domain_randomization import ADRCurriculumController


class _DeterministicPolicyWrapper(nn.Module):
    def __init__(self, actor: nn.Module) -> None:
        super().__init__()
        self.actor = actor

    def forward(self, obs: torch.Tensor, desired_goal: torch.Tensor) -> torch.Tensor:
        action, _ = self.actor(obs, desired_goal, deterministic=True)
        return action


def save_checkpoint(
    checkpoint_path: str | Path,
    actor: nn.Module,
    critic: nn.Module,
    trainer: SACTrainer,
    curriculum_controller: ADRCurriculumController | None = None,
    step: int = 0,
    best_success_rate: float = 0.0,
    extra_info: dict[str, Any] | None = None,
) -> None:
    path = Path(checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    state: dict[str, Any] = {
        "step": step,
        "best_success_rate": best_success_rate,
        "actor_state_dict": actor.state_dict(),
        "critic_state_dict": critic.state_dict(),
        "extra_info": extra_info or {},
    }

    if trainer is not None:
        if hasattr(trainer.loss, "target_critic"):
            state["target_critic_state_dict"] = trainer.loss.target_critic.state_dict()
        if hasattr(trainer, "actor_optimizer"):
            state["actor_optimizer_state_dict"] = trainer.actor_optimizer.state_dict()
        if hasattr(trainer, "critic_optimizer"):
            state["critic_optimizer_state_dict"] = trainer.critic_optimizer.state_dict()
        if hasattr(trainer, "alpha_optimizer"):
            state["alpha_optimizer_state_dict"] = trainer.alpha_optimizer.state_dict()
        if hasattr(trainer.loss, "log_alpha"):
            state["log_alpha"] = trainer.loss.log_alpha.detach().cpu()

    if curriculum_controller is not None:
        state["curriculum_difficulty"] = float(curriculum_controller.difficulty)

    torch.save(state, path)


def load_checkpoint(
    checkpoint_path: str | Path | dict[str, Any],
    actor: nn.Module,
    critic: nn.Module,
    trainer: SACTrainer | None = None,
    curriculum_controller: ADRCurriculumController | None = None,
    device: torch.device | None = None,
) -> dict[str, Any]:
    if isinstance(checkpoint_path, dict):
        checkpoint = checkpoint_path
    else:
        path = Path(checkpoint_path)
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint file not found: {path}")
        checkpoint = torch.load(path, map_location=device, weights_only=False)

    actor.load_state_dict(checkpoint["actor_state_dict"])
    critic.load_state_dict(checkpoint["critic_state_dict"])

    if trainer is not None:
        if (
            "target_critic_state_dict" in checkpoint
            and checkpoint["target_critic_state_dict"] is not None
            and hasattr(trainer.loss, "target_critic")
        ):
            trainer.loss.target_critic.load_state_dict(checkpoint["target_critic_state_dict"])
        if (
            "actor_optimizer_state_dict" in checkpoint
            and checkpoint["actor_optimizer_state_dict"] is not None
            and hasattr(trainer, "actor_optimizer")
        ):
            trainer.actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
        if (
            "critic_optimizer_state_dict" in checkpoint
            and checkpoint["critic_optimizer_state_dict"] is not None
            and hasattr(trainer, "critic_optimizer")
        ):
            trainer.critic_optimizer.load_state_dict(checkpoint["critic_optimizer_state_dict"])
        if (
            "alpha_optimizer_state_dict" in checkpoint
            and checkpoint["alpha_optimizer_state_dict"] is not None
            and hasattr(trainer, "alpha_optimizer")
        ):
            trainer.alpha_optimizer.load_state_dict(checkpoint["alpha_optimizer_state_dict"])
        if "log_alpha" in checkpoint and checkpoint["log_alpha"] is not None:
            val = checkpoint["log_alpha"]
            if isinstance(val, torch.Tensor):
                trainer.loss.log_alpha.data.copy_(val.to(trainer.loss.log_alpha.device))
            else:
                trainer.loss.log_alpha.data.fill_(float(val))

    if curriculum_controller is not None and "curriculum_difficulty" in checkpoint:
        diff = checkpoint["curriculum_difficulty"]
        if diff is not None:
            curriculum_controller.set_difficulty(float(diff))

    return checkpoint


def export_policy_onnx(
    onnx_path: str | Path,
    actor: nn.Module,
    obs_dim: int = 55,
    goal_dim: int = 4,
    device: torch.device | None = None,
) -> None:
    path = Path(onnx_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    target_device = (
        device
        if device is not None
        else next(actor.parameters(), torch.tensor(0.0)).device
    )

    was_training = actor.training
    try:
        wrapper = _DeterministicPolicyWrapper(actor).to(target_device)
        wrapper.eval()

        dummy_obs = torch.randn(1, obs_dim, device=target_device)
        dummy_goal = torch.randn(1, goal_dim, device=target_device)

        torch.onnx.export(
            wrapper,
            (dummy_obs, dummy_goal),
            str(path),
            dynamo=False,
            input_names=["obs", "desired_goal"],
            output_names=["action"],
            dynamic_axes={
                "obs": {0: "batch_size"},
                "desired_goal": {0: "batch_size"},
                "action": {0: "batch_size"},
            },
            opset_version=17,
        )
    finally:
        actor.train(was_training)
