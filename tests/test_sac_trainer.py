import math
import sys
from pathlib import Path

import torch
from tensordict import TensorDict

# Ensure repository root is on sys.path for direct pytest invocation
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.algo import (
    SACLoss,
    SACTrainer,
    SquashedGaussianSACActor,
    TwinContinuousQCritic,
)
from src.envs.leap_cube_env import LeapCubeReorientEnv


def make_dummy_batch(
    batch_size: int = 32,
    obs_dim: int = 55,
    goal_dim: int = 4,
    action_dim: int = 16,
    device: torch.device | None = None,
) -> TensorDict:
    """Creates a transition batch TensorDict for SAC loss and trainer."""
    return TensorDict(
        {
            "obs": torch.randn(batch_size, obs_dim, device=device),
            "desired_goal": torch.randn(batch_size, goal_dim, device=device),
            "action": torch.clamp(
                torch.randn(batch_size, action_dim, device=device), -1.0, 1.0
            ),
            "next": TensorDict(
                {
                    "obs": torch.randn(batch_size, obs_dim, device=device),
                    "desired_goal": torch.randn(batch_size, goal_dim, device=device),
                    "reward": torch.randn(batch_size, 1, device=device),
                    "done": torch.zeros(
                        batch_size, 1, dtype=torch.bool, device=device
                    ),
                    "terminated": torch.zeros(
                        batch_size, 1, dtype=torch.bool, device=device
                    ),
                },
                batch_size=[batch_size],
                device=device,
            ),
        },
        batch_size=[batch_size],
        device=device,
    )


def test_sac_exports():
    """Verify SACLoss and SACTrainer are exported in src.algo."""
    from src.algo.sac_trainer import SACLoss as DirectSACLoss
    from src.algo.sac_trainer import SACTrainer as DirectSACTrainer

    assert SACLoss is DirectSACLoss
    assert SACTrainer is DirectSACTrainer


def test_sac_loss_defaults_and_forward_outputs():
    """Verify SACLoss defaults target_entropy to -16.0 and forward outputs expected keys."""
    obs_dim, goal_dim, action_dim = 55, 4, 16
    actor = SquashedGaussianSACActor(obs_dim, goal_dim, action_dim)
    critic = TwinContinuousQCritic(obs_dim, goal_dim, action_dim)

    loss_fn = SACLoss(actor, critic)

    # Verify default target entropy is -16.0
    assert loss_fn.target_entropy == -16.0
    assert math.isclose(loss_fn.alpha, 1.0, rel_tol=1e-5)

    batch = make_dummy_batch(batch_size=16, obs_dim=obs_dim, goal_dim=goal_dim, action_dim=action_dim)
    loss_out = loss_fn(batch)

    # Verify required keys in output
    for key in ("loss_actor", "loss_critic", "loss_alpha", "alpha", "entropy"):
        assert key in loss_out, f"Missing expected key '{key}' in SACLoss forward output"

    assert torch.isfinite(loss_out["loss_actor"]).all()
    assert torch.isfinite(loss_out["loss_critic"]).all()
    assert torch.isfinite(loss_out["loss_alpha"]).all()
    assert torch.isfinite(loss_out["entropy"]).all()
    assert float(loss_out["alpha"]) > 0.0


def test_sac_loss_configurable_target_entropy():
    """Verify target_entropy can be customized."""
    obs_dim, goal_dim, action_dim = 55, 4, 16
    actor = SquashedGaussianSACActor(obs_dim, goal_dim, action_dim)
    critic = TwinContinuousQCritic(obs_dim, goal_dim, action_dim)

    loss_fn = SACLoss(actor, critic, target_entropy=-8.0, alpha_init=0.5)
    assert loss_fn.target_entropy == -8.0
    assert math.isclose(loss_fn.alpha, 0.5, rel_tol=1e-5)


def test_sac_trainer_step_updates_parameters():
    """Verify SACTrainer.step(batch) updates actor, critic, and log_alpha with finite gradients."""
    obs_dim, goal_dim, action_dim = 55, 4, 16
    actor = SquashedGaussianSACActor(obs_dim, goal_dim, action_dim)
    critic = TwinContinuousQCritic(obs_dim, goal_dim, action_dim)

    trainer = SACTrainer(actor, critic, target_entropy=-16.0, alpha_init=1.0)

    assert trainer.target_entropy == -16.0
    assert math.isclose(trainer.alpha, 1.0, rel_tol=1e-5)
    assert isinstance(trainer.log_alpha, torch.nn.Parameter)

    # Snapshot parameters before step
    actor_param_before = next(trainer.actor.parameters()).clone()
    critic_param_before = next(trainer.critic.parameters()).clone()
    log_alpha_before = trainer.log_alpha.clone()

    batch = make_dummy_batch(batch_size=16, obs_dim=obs_dim, goal_dim=goal_dim, action_dim=action_dim)
    step_metrics = trainer.step(batch)

    # Check return dictionary contains scalar floats
    for metric_name in ("loss_actor", "loss_critic", "loss_alpha", "alpha", "entropy"):
        assert metric_name in step_metrics
        assert isinstance(step_metrics[metric_name], float)
        assert torch.isfinite(torch.tensor(step_metrics[metric_name]))

    # Verify parameters updated
    actor_param_after = next(trainer.actor.parameters())
    critic_param_after = next(trainer.critic.parameters())
    log_alpha_after = trainer.log_alpha

    assert not torch.equal(actor_param_before, actor_param_after), "Actor parameter was not updated"
    assert not torch.equal(critic_param_before, critic_param_after), "Critic parameter was not updated"
    assert not torch.equal(log_alpha_before, log_alpha_after), "log_alpha parameter was not updated"

    # Verify gradients are finite
    assert trainer.log_alpha.grad is not None
    assert torch.isfinite(trainer.log_alpha.grad).all()


def test_target_network_polyak_update():
    """Verify target network weights update with Polyak coefficient tau = 0.005."""
    obs_dim, goal_dim, action_dim = 55, 4, 16
    actor = SquashedGaussianSACActor(obs_dim, goal_dim, action_dim)
    critic = TwinContinuousQCritic(obs_dim, goal_dim, action_dim)

    tau = 0.005
    trainer = SACTrainer(actor, critic, tau=tau)

    # Manually perturb critic weights to differ from target
    with torch.no_grad():
        for p in trainer.critic.parameters():
            p.add_(torch.randn_like(p) * 2.0)

    target_before = [p.clone() for p in trainer.loss.target_critic.parameters()]
    critic_before = [p.clone() for p in trainer.critic.parameters()]

    trainer.update_target_network()

    target_after = list(trainer.loss.target_critic.parameters())

    for w_target_0, w_critic, w_target_1 in zip(target_before, critic_before, target_after, strict=True):
        expected_w = (1.0 - tau) * w_target_0 + tau * w_critic
        assert torch.allclose(w_target_1, expected_w, atol=1e-6)


def test_sac_training_loop_stability_dense_yaw():
    """Verify short training loop on synthetic dense yaw rollout verifies alpha and losses remain finite and stable."""
    obs_dim, goal_dim, action_dim = 55, 4, 16
    actor = SquashedGaussianSACActor(obs_dim, goal_dim, action_dim)
    critic = TwinContinuousQCritic(obs_dim, goal_dim, action_dim)

    trainer = SACTrainer(
        actor,
        critic,
        actor_lr=1e-3,
        critic_lr=1e-3,
        alpha_lr=1e-3,
        target_entropy=-16.0,
        alpha_init=1.0,
    )

    batch = make_dummy_batch(batch_size=32, obs_dim=obs_dim, goal_dim=goal_dim, action_dim=action_dim)

    for step_idx in range(10):
        metrics = trainer.step(batch)
        assert torch.isfinite(torch.tensor(metrics["loss_critic"])), f"Critic loss diverged at step {step_idx}"
        assert torch.isfinite(torch.tensor(metrics["loss_actor"])), f"Actor loss diverged at step {step_idx}"
        assert torch.isfinite(torch.tensor(metrics["loss_alpha"])), f"Alpha loss diverged at step {step_idx}"
        assert metrics["alpha"] > 0.0, f"Alpha became non-positive at step {step_idx}"


def test_evaluate_rollout():
    """Verify evaluate_rollout executes deterministic rollout and returns alpha and geodesic error metrics."""
    env = LeapCubeReorientEnv(
        target_mode="yaw",
        reward_mode="dense",
        max_steps=50,
        domain_randomization=False,
        seed=123,
    )

    obs_dim = 55
    goal_dim = 4
    action_dim = 16
    actor = SquashedGaussianSACActor(obs_dim, goal_dim, action_dim)
    critic = TwinContinuousQCritic(obs_dim, goal_dim, action_dim)

    trainer = SACTrainer(actor, critic)

    eval_results = trainer.evaluate_rollout(env, num_steps=20)

    assert "alpha" in eval_results
    assert "mean_geodesic_error" in eval_results
    assert isinstance(eval_results["alpha"], float)
    assert isinstance(eval_results["mean_geodesic_error"], float)
    assert 0.0 <= eval_results["mean_geodesic_error"] <= math.pi + 1e-4
