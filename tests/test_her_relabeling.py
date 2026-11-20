import sys
from pathlib import Path

import pytest
import torch
from tensordict import TensorDict

# Ensure repository root is on sys.path for direct pytest invocation
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.algo import VectorizedFutureHER as VectorizedFutureHERFromPackage
from src.algo.her import VectorizedFutureHER
from src.utils.quaternion import sample_uniform_quat


def test_her_exposed_in_algo_package():
    """Verify VectorizedFutureHER is exposed in src.algo package namespace."""
    assert VectorizedFutureHERFromPackage is VectorizedFutureHER


def make_dummy_episodes(
    batch_size: int = 16,
    time_steps: int = 10,
    obs_dim: int = 32,
    cube_z_val: float = 0.05,
    device: torch.device | None = None,
) -> TensorDict:
    """Creates a dummy episodic TensorDict batch of shape [B, T]."""
    q_achieved = sample_uniform_quat(batch_size * time_steps, device=device).view(
        batch_size, time_steps, 4
    )
    # Initialize desired goals distinct from achieved
    q_desired = sample_uniform_quat(batch_size * time_steps, device=device).view(
        batch_size, time_steps, 4
    )

    episodes = TensorDict(
        {
            "obs": torch.randn(batch_size, time_steps, obs_dim, device=device),
            "desired_goal": q_desired.clone(),
            "action": torch.randn(batch_size, time_steps, 16, device=device),
            "next": TensorDict(
                {
                    "obs": torch.randn(batch_size, time_steps, obs_dim, device=device),
                    "achieved_goal": q_achieved.clone(),
                    "desired_goal": q_desired.clone(),
                    "cube_z": torch.full(
                        (batch_size, time_steps, 1), cube_z_val, device=device
                    ),
                    "reward": torch.full(
                        (batch_size, time_steps, 1), -1.0, device=device
                    ),
                    "done": torch.zeros(
                        (batch_size, time_steps, 1), dtype=torch.bool, device=device
                    ),
                },
                batch_size=[batch_size, time_steps],
                device=device,
            ),
        },
        batch_size=[batch_size, time_steps],
        device=device,
    )
    return episodes


def test_her_relabel_probability_one_sets_reward_zero():
    """
    When relabel_prob=1.0 and achieved orientation is reached (e.g. constant across time),
    the new desired goal equals the next achieved goal of the transition.
    With cube_z > 0.04 (not dropped), recomputed reward at next step must equal 0.0.
    """
    torch.manual_seed(42)
    B, T = 32, 4
    episodes = make_dummy_episodes(batch_size=B, time_steps=T, cube_z_val=0.05)
    # Set identical achieved orientation across all time steps within each episode
    const_achieved = episodes["next", "achieved_goal"][:, :1, :].expand(B, T, 4).clone()
    episodes["next", "achieved_goal"] = const_achieved

    her = VectorizedFutureHER(relabel_prob=1.0)
    transition = her(episodes)

    # All transitions should be relabeled with the achieved goal
    next_achieved = transition["next", "achieved_goal"]
    new_desired = transition["desired_goal"]
    next_desired = transition["next", "desired_goal"]

    assert torch.allclose(new_desired, next_achieved, atol=1e-6)
    assert torch.allclose(next_desired, next_achieved, atol=1e-6)

    # Recomputed reward must equal 0.0 for all items in batch
    expected_reward = torch.zeros((B, 1))
    assert torch.allclose(transition["next", "reward"], expected_reward)


def test_her_dropped_cube_recomputes_penalty_even_if_reoriented():
    """
    When cube_z <= 0.04 (dropped), recomputed reward must remain -1.0
    even if the desired goal matches the achieved goal.
    """
    torch.manual_seed(42)
    B, T = 16, 4
    episodes = make_dummy_episodes(batch_size=B, time_steps=T, cube_z_val=0.02)
    # Set identical achieved orientation across all time steps within each episode
    const_achieved = episodes["next", "achieved_goal"][:, :1, :].expand(B, T, 4).clone()
    episodes["next", "achieved_goal"] = const_achieved

    her = VectorizedFutureHER(relabel_prob=1.0)
    transition = her(episodes)

    expected_reward = torch.full((B, 1), -1.0)
    assert torch.allclose(transition["next", "reward"], expected_reward)


def test_her_relabel_probability_zero_preserves_originals():
    """
    When relabel_prob=0.0, no transitions should be relabeled.
    Desired goals and rewards must match the original values at the sampled transition.
    """
    torch.manual_seed(123)
    B, T = 32, 5
    episodes = make_dummy_episodes(batch_size=B, time_steps=T, cube_z_val=0.05)
    orig_desired = episodes["desired_goal"].clone()
    orig_next_desired = episodes["next", "desired_goal"].clone()

    her = VectorizedFutureHER(relabel_prob=0.0)
    transition = her(episodes)

    # Output reward should be -1.0 since random desired != random achieved
    assert torch.all(transition["next", "reward"] == -1.0)

    # Check desired goal equals one of the original time step slices in episode
    for b in range(B):
        match_orig = torch.any(
            torch.all(torch.isclose(orig_desired[b], transition["desired_goal"][b], atol=1e-6), dim=-1)
        )
        assert match_orig, f"Batch index {b} desired_goal does not match any original time step"

        match_next = torch.any(
            torch.all(torch.isclose(orig_next_desired[b], transition["next", "desired_goal"][b], atol=1e-6), dim=-1)
        )
        assert match_next, f"Batch index {b} next desired_goal does not match any original time step"

    # Specifically, desired_goal and next desired_goal must be identical
    assert torch.allclose(
        transition["desired_goal"], transition["next", "desired_goal"]
    )


def test_her_output_shape_batch_size_and_keys():
    """
    Assert output has batch_size [B] (sampled transitions from [B, T])
    and contains all expected keys.
    """
    B, T = 8, 12
    episodes = make_dummy_episodes(batch_size=B, time_steps=T)

    her = VectorizedFutureHER(relabel_prob=0.8)
    transition = her(episodes)

    # Batch size must be [B]
    assert transition.batch_size == torch.Size([B])
    assert transition["next"].batch_size == torch.Size([B])

    # Required top-level keys
    assert "obs" in transition
    assert "desired_goal" in transition
    assert "action" in transition
    assert "next" in transition

    # Required next keys
    next_td = transition["next"]
    assert "obs" in next_td
    assert "achieved_goal" in next_td
    assert "desired_goal" in next_td
    assert "cube_z" in next_td
    assert "reward" in next_td

    # Shapes
    assert transition["desired_goal"].shape == (B, 4)
    assert transition["next", "desired_goal"].shape == (B, 4)
    assert transition["next", "reward"].shape == (B, 1)


def test_her_statistical_relabel_rate():
    """
    On a large batch (B=1000), approximately 80% of transitions should be relabeled.
    Transitions with relabel_mask have desired_goal replaced with a future orientation
    different from original desired_goal.
    """
    torch.manual_seed(999)
    B, T = 1000, 4
    episodes = make_dummy_episodes(batch_size=B, time_steps=T)

    # Initialize original desired goals to a fixed distinct quaternion [1, 0, 0, 0]
    orig_goal = torch.tensor([1.0, 0.0, 0.0, 0.0]).expand(B, T, 4).clone()
    episodes["desired_goal"] = orig_goal
    episodes["next", "desired_goal"] = orig_goal.clone()

    # Initialize achieved goals to orthogonal quaternion [0, 1, 0, 0]
    future_achieved = torch.tensor([0.0, 1.0, 0.0, 0.0]).expand(B, T, 4).clone()
    episodes["next", "achieved_goal"] = future_achieved

    her = VectorizedFutureHER(relabel_prob=0.8)
    transition = her(episodes)

    # Relabeled transitions have desired_goal replaced with future orientation [0, 1, 0, 0]
    is_relabeled = torch.all(
        torch.isclose(transition["desired_goal"], torch.tensor([0.0, 1.0, 0.0, 0.0]), atol=1e-5), dim=-1
    )
    # Non-relabeled transitions retain original desired_goal [1, 0, 0, 0]
    is_original = torch.all(
        torch.isclose(transition["desired_goal"], torch.tensor([1.0, 0.0, 0.0, 0.0]), atol=1e-5), dim=-1
    )

    # Every transition is either relabeled with future achieved goal or keeps original goal
    assert torch.all(is_relabeled | is_original)
    assert not torch.any(is_relabeled & is_original)

    relabel_frac = is_relabeled.float().mean().item()
    # With B=1000, 0.8 relabeling prob should fall reliably in [0.75, 0.85]
    assert 0.75 <= relabel_frac <= 0.85, f"Observed relabel fraction: {relabel_frac}"


def test_her_future_idx_samples_terminal_step():
    """
    Verify that future_idx == T - 1 is actively sampled and reachable
    when sampling from episodic trajectory batch.
    """
    torch.manual_seed(42)
    B, T = 2000, 5
    episodes = make_dummy_episodes(batch_size=B, time_steps=T)

    # Set all steps 0..T-2 achieved goal to [1, 0, 0, 0]
    episodes["next", "achieved_goal"] = torch.tensor([1.0, 0.0, 0.0, 0.0]).expand(B, T, 4).clone()
    # Set terminal step T-1 achieved goal to distinct marker quaternion [0, 0, 0, 1]
    terminal_marker = torch.tensor([0.0, 0.0, 0.0, 1.0])
    episodes["next", "achieved_goal"][:, T - 1] = terminal_marker

    her = VectorizedFutureHER(relabel_prob=1.0)
    transition = her(episodes)

    # Transitions where future_idx == T - 1 will have desired_goal == terminal_marker
    is_terminal = torch.all(
        torch.isclose(transition["desired_goal"], terminal_marker, atol=1e-5), dim=-1
    )
    terminal_count = is_terminal.sum().item()

    # Terminal step must be sampled across the batch
    assert terminal_count > 0, "Terminal step T-1 was never sampled as future_idx"
    terminal_frac = terminal_count / B
    assert terminal_frac > 0.10, f"Terminal step fraction too low: {terminal_frac}"


def test_her_future_idx_greater_than_t_idx_unreached_reward():
    """
    Verify that when future_idx > t_idx, an orientation achieved at a future time step
    is not yet reached at t_idx and correctly yields reward -1.0 until reached.
    """
    torch.manual_seed(2026)
    B, T = 1000, 4
    episodes = make_dummy_episodes(batch_size=B, time_steps=T, cube_z_val=0.05)

    # 4 mutually orthogonal unit quaternions along time axis
    quats = torch.tensor([
        [1.0, 0.0, 0.0, 0.0],  # step 0
        [0.0, 1.0, 0.0, 0.0],  # step 1
        [0.0, 0.0, 1.0, 0.0],  # step 2
        [0.0, 0.0, 0.0, 1.0],  # step 3
    ])
    episodes["next", "achieved_goal"] = quats.unsqueeze(0).expand(B, T, 4).clone()

    her = VectorizedFutureHER(relabel_prob=1.0)
    transition = her(episodes)

    next_achieved = transition["next", "achieved_goal"]
    new_desired = transition["desired_goal"]
    rewards = transition["next", "reward"]

    # When future_idx == t_idx, new_desired == next_achieved -> reward must be 0.0
    same_step_mask = torch.all(torch.isclose(new_desired, next_achieved, atol=1e-5), dim=-1)
    # When future_idx > t_idx, new_desired != next_achieved -> reward must be -1.0
    future_step_mask = ~same_step_mask

    assert same_step_mask.sum().item() > 0, "No same-step transitions sampled"
    assert future_step_mask.sum().item() > 0, "No strictly future transitions sampled"

    # Verify rewards
    assert torch.all(rewards[same_step_mask] == 0.0)
    assert torch.all(rewards[future_step_mask] == -1.0)


def test_her_does_not_mutate_original_episode_batch():
    """
    Assert that the relabeler operates on cloned transitions and does not
    mutate the source episode batch in-place.
    """
    B, T = 16, 5
    episodes = make_dummy_episodes(batch_size=B, time_steps=T)
    orig_desired_clone = episodes["desired_goal"].clone()
    orig_reward_clone = episodes["next", "reward"].clone()

    her = VectorizedFutureHER(relabel_prob=1.0)
    _ = her(episodes)

    assert torch.allclose(episodes["desired_goal"], orig_desired_clone)
    assert torch.allclose(episodes["next", "reward"], orig_reward_clone)


def test_her_invalid_time_dimension():
    """
    Assert ValueError is raised when time dimension T < 2 or tensor is not 2D.
    """
    her = VectorizedFutureHER()

    # T=1
    episodes_t1 = TensorDict(
        {"desired_goal": torch.randn(4, 1, 4)}, batch_size=[4, 1]
    )
    with pytest.raises(ValueError, match="T >= 2"):
        her(episodes_t1)

    # 1D batch
    episodes_1d = TensorDict({"desired_goal": torch.randn(4, 4)}, batch_size=[4])
    with pytest.raises(ValueError, match="2D episodic"):
        her(episodes_1d)
