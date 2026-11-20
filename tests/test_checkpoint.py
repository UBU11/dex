import onnx
import pytest
import torch

from src.algo.actor import SquashedGaussianSACActor
from src.algo.checkpoint import export_policy_onnx, load_checkpoint, save_checkpoint
from src.algo.critic import TwinContinuousQCritic
from src.algo.sac_trainer import SACTrainer
from src.envs import ADRCurriculumController, DomainRandomizer, LeapCubeReorientEnv


def test_save_and_load_checkpoint(tmp_path):
    actor = SquashedGaussianSACActor(55, 4)
    critic = TwinContinuousQCritic(55, 4)
    trainer = SACTrainer(actor, critic)

    env = LeapCubeReorientEnv(enable_domain_randomization=True)
    assert env.curriculum_controller is not None
    curriculum = env.curriculum_controller
    curriculum.set_difficulty(0.45)

    ckpt_path = tmp_path / "checkpoints" / "model_step_100.pt"
    save_checkpoint(
        checkpoint_path=ckpt_path,
        actor=actor,
        critic=critic,
        trainer=trainer,
        curriculum_controller=curriculum,
        step=100,
        best_success_rate=0.75,
        extra_info={"experiment": "test_run"},
    )
    assert ckpt_path.is_file()

    actor2 = SquashedGaussianSACActor(55, 4)
    critic2 = TwinContinuousQCritic(55, 4)
    trainer2 = SACTrainer(actor2, critic2)
    curriculum2 = ADRCurriculumController(DomainRandomizer(env.model, env.data))
    curriculum2.set_difficulty(0.0)

    # Scramble weights to verify restoration
    with torch.no_grad():
        for p in actor2.parameters():
            p.add_(1.0)
        for p in critic2.parameters():
            p.add_(1.0)

    meta = load_checkpoint(
        checkpoint_path=ckpt_path,
        actor=actor2,
        critic=critic2,
        trainer=trainer2,
        curriculum_controller=curriculum2,
    )

    assert meta["step"] == 100
    assert meta["best_success_rate"] == 0.75
    assert meta["extra_info"]["experiment"] == "test_run"
    assert curriculum2.difficulty == pytest.approx(0.45)

    for p1, p2 in zip(actor.parameters(), actor2.parameters(), strict=True):
        assert torch.allclose(p1, p2)
    for p1, p2 in zip(critic.parameters(), critic2.parameters(), strict=True):
        assert torch.allclose(p1, p2)
    for p1, p2 in zip(
        trainer.loss.target_critic.parameters(),
        trainer2.loss.target_critic.parameters(),
        strict=True,
    ):
        assert torch.allclose(p1, p2)
    assert trainer.alpha == pytest.approx(trainer2.alpha)


def test_load_checkpoint_without_trainer_or_curriculum(tmp_path):
    actor = SquashedGaussianSACActor(55, 4)
    critic = TwinContinuousQCritic(55, 4)
    trainer = SACTrainer(actor, critic)

    ckpt_path = tmp_path / "ckpt.pt"
    save_checkpoint(
        checkpoint_path=ckpt_path,
        actor=actor,
        critic=critic,
        trainer=trainer,
        step=50,
        best_success_rate=0.3,
    )

    actor2 = SquashedGaussianSACActor(55, 4)
    critic2 = TwinContinuousQCritic(55, 4)
    meta = load_checkpoint(
        checkpoint_path=ckpt_path,
        actor=actor2,
        critic=critic2,
        trainer=None,
        curriculum_controller=None,
    )
    assert meta["step"] == 50
    for p1, p2 in zip(actor.parameters(), actor2.parameters(), strict=True):
        assert torch.allclose(p1, p2)


def test_export_policy_onnx(tmp_path):
    actor = SquashedGaussianSACActor(55, 4)
    onnx_path = tmp_path / "policy.onnx"
    export_policy_onnx(onnx_path, actor, obs_dim=55, goal_dim=4)

    assert onnx_path.is_file()
    assert onnx_path.stat().st_size > 0

    model = onnx.load(str(onnx_path))
    onnx.checker.check_model(model)
    input_names = [inp.name for inp in model.graph.input]
    output_names = [out.name for out in model.graph.output]
    assert "obs" in input_names
    assert "desired_goal" in input_names
    assert "action" in output_names
