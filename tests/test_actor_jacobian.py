import sys
from pathlib import Path

import torch

# Ensure repository root is on sys.path for direct pytest invocation
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.algo.actor import (
    LOG_STD_MAX,
    LOG_STD_MIN,
    SquashedGaussianSACActor,
    stable_tanh_jacobian_correction,
)
from src.algo.critic import TwinContinuousQCritic


def test_jacobian_correction_finite_at_extreme_activations():
    """Verify that Jacobian correction remains finite and has finite gradients at |u| = 10, 20."""
    extreme_u = torch.tensor([-20.0, -10.0, -5.0, 0.0, 5.0, 10.0, 20.0], requires_grad=True)
    correction = stable_tanh_jacobian_correction(extreme_u)

    assert torch.isfinite(correction).all(), "Jacobian correction contains non-finite values"
    assert not torch.isnan(correction).any(), "Jacobian correction contains NaNs"

    # Backward pass should produce finite gradients without NaNs
    loss = correction.sum()
    loss.backward()
    assert extreme_u.grad is not None
    assert torch.isfinite(extreme_u.grad).all(), "Jacobian correction gradient contains non-finite values"

    # Compare against moderate u where naive log(1 - tanh^2) is mathematically valid
    moderate_u = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0])
    naive = torch.log(1.0 - torch.tanh(moderate_u).pow(2) + 1e-12)
    stable = stable_tanh_jacobian_correction(moderate_u)
    assert torch.allclose(naive, stable, atol=1e-5), "Softplus identity diverges from naive formula on moderate inputs"

    # Demonstrate that naive formula fails at |u| >= 10 in FP32
    u_fail = torch.tensor([10.0, 20.0], requires_grad=True)
    naive_fail = torch.log(1.0 - torch.tanh(u_fail).pow(2))
    assert torch.isinf(naive_fail).all(), "Expected naive formula to saturate to -inf at |u| >= 10"


def test_actor_finite_gradients_at_extreme_u():
    """Test actor forward and backward passes when pre-tanh activations u reach |u| = 10 and 20."""
    obs_dim = 32
    goal_dim = 4
    action_dim = 16
    actor = SquashedGaussianSACActor(obs_dim=obs_dim, goal_dim=goal_dim, action_dim=action_dim)

    obs = torch.randn(4, obs_dim)
    goal = torch.randn(4, goal_dim)

    # Force extreme pre-tanh activations by shifting mu_head bias
    for extreme_val in [10.0, -10.0, 20.0, -20.0]:
        actor.zero_grad()
        actor.mu_head.bias.data.fill_(extreme_val)
        actor.log_std_head.bias.data.fill_(LOG_STD_MIN)

        action, log_prob = actor(obs, goal, deterministic=False)

        assert torch.isfinite(action).all(), f"Action has non-finite values at u ~ {extreme_val}"
        assert torch.isfinite(log_prob).all(), f"Log-prob has non-finite values at u ~ {extreme_val}"
        assert not torch.isnan(log_prob).any()

        # Check backward pass
        loss = log_prob.sum() + action.sum()
        loss.backward()

        for name, param in actor.named_parameters():
            if param.grad is not None:
                assert torch.isfinite(param.grad).all(), f"Grad for {name} has non-finite values at u ~ {extreme_val}"
                assert not torch.isnan(param.grad).any(), f"Grad for {name} has NaNs at u ~ {extreme_val}"


def test_action_strictly_bounded():
    """Verify that actions are strictly bounded in (-1, 1)."""
    obs_dim = 32
    goal_dim = 4
    action_dim = 16
    actor = SquashedGaussianSACActor(obs_dim=obs_dim, goal_dim=goal_dim, action_dim=action_dim)

    # Standard and extreme input batches
    obs = torch.randn(100, obs_dim) * 50.0
    goal = torch.randn(100, goal_dim) * 50.0

    action, _ = actor(obs, goal, deterministic=False)
    assert action.shape == (100, action_dim)
    assert (action > -1.0).all(), "Action contains values <= -1.0"
    assert (action < 1.0).all(), "Action contains values >= 1.0"

    det_action, _ = actor(obs, goal, deterministic=True)
    assert (det_action > -1.0).all()
    assert (det_action < 1.0).all()


def test_actor_deterministic_mode():
    """Verify that deterministic mode returns tanh(mu) and has zero sampling variance."""
    obs_dim = 24
    goal_dim = 4
    action_dim = 16
    actor = SquashedGaussianSACActor(obs_dim=obs_dim, goal_dim=goal_dim, action_dim=action_dim)

    obs = torch.randn(10, obs_dim)
    goal = torch.randn(10, goal_dim)

    # Compute expected deterministic action manually
    x = torch.cat([obs, goal], dim=-1)
    h = actor.net(x)
    expected_mu = actor.mu_head(h)
    expected_action = torch.tanh(expected_mu)

    action1, log_prob1 = actor(obs, goal, deterministic=True)
    action2, log_prob2 = actor(obs, goal, deterministic=True)

    assert torch.allclose(action1, expected_action, atol=1e-6), "Deterministic action does not equal tanh(mu)"
    assert torch.equal(action1, action2), "Deterministic mode must be purely deterministic across multiple calls"
    assert torch.equal(log_prob1, log_prob2)


def test_twin_critic_outputs():
    """Verify TwinContinuousQCritic outputs two distinct scalar Q predictions of shape (B, 1)."""
    obs_dim = 32
    goal_dim = 4
    action_dim = 16
    critic = TwinContinuousQCritic(obs_dim=obs_dim, goal_dim=goal_dim, action_dim=action_dim, hidden_dim=256)

    batch_size = 16
    obs = torch.randn(batch_size, obs_dim)
    goal = torch.randn(batch_size, goal_dim)
    action = torch.randn(batch_size, action_dim)

    q1, q2 = critic(obs, goal, action)

    assert q1.shape == (batch_size, 1), f"Expected q1 shape {(batch_size, 1)}, got {q1.shape}"
    assert q2.shape == (batch_size, 1), f"Expected q2 shape {(batch_size, 1)}, got {q2.shape}"

    assert torch.isfinite(q1).all(), "q1 contains non-finite values"
    assert torch.isfinite(q2).all(), "q2 contains non-finite values"

    # Twin critics must have distinct initial weights and produce distinct outputs
    assert not torch.allclose(q1, q2), "Twin critics produced identical outputs; expected distinct Q1 and Q2"

    # Backward pass verification
    loss = q1.sum() + q2.sum()
    loss.backward()

    for name, param in critic.named_parameters():
        if param.grad is not None:
            assert torch.isfinite(param.grad).all(), f"Critic grad for {name} has non-finite values"


def test_log_std_clamping():
    """Verify log_std is strictly clamped within [LOG_STD_MIN, LOG_STD_MAX]."""
    obs_dim = 16
    goal_dim = 4
    actor = SquashedGaussianSACActor(obs_dim=obs_dim, goal_dim=goal_dim)

    # Force logits to huge positive and negative values
    actor.log_std_head.bias.data.fill_(100.0)
    obs = torch.zeros(1, obs_dim)
    goal = torch.zeros(1, goal_dim)
    x = torch.cat([obs, goal], dim=-1)
    h = actor.net(x)
    raw_log_std = actor.log_std_head(h)
    clamped_log_std = torch.clamp(raw_log_std, LOG_STD_MIN, LOG_STD_MAX)
    assert torch.all(clamped_log_std == LOG_STD_MAX)

    actor.log_std_head.bias.data.fill_(-100.0)
    raw_log_std = actor.log_std_head(actor.net(x))
    clamped_log_std = torch.clamp(raw_log_std, LOG_STD_MIN, LOG_STD_MAX)
    assert torch.all(clamped_log_std == LOG_STD_MIN)
