import math
import sys
from pathlib import Path
from unittest.mock import patch

import mujoco
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.algo import (
    PolicyModule,
    SACTrainer,
    SquashedGaussianSACActor,
    TwinContinuousQCritic,
    create_her_replay_buffer,
    create_trajectory_collector,
)
from src.envs import ADRCurriculumController, DomainRandomizer, LeapCubeReorientEnv


def test_env_domain_randomization_flag_and_controller():
    """Verify LeapCubeReorientEnv exposes randomizer and ADRCurriculumController only when enabled."""
    env_disabled = LeapCubeReorientEnv(enable_domain_randomization=False, seed=42)
    assert env_disabled.randomizer is None
    assert env_disabled.curriculum_controller is None

    env_enabled = LeapCubeReorientEnv(enable_domain_randomization=True, seed=42)
    assert isinstance(env_enabled.randomizer, DomainRandomizer)
    assert isinstance(env_enabled.curriculum_controller, ADRCurriculumController)
    assert env_enabled.curriculum_controller.randomizer is env_enabled.randomizer

    custom_randomizer = DomainRandomizer(env_enabled.model, env_enabled.data)
    env_custom = LeapCubeReorientEnv(
        enable_domain_randomization=True, randomizer=custom_randomizer, seed=42
    )
    assert env_custom.randomizer is custom_randomizer
    assert env_custom.curriculum_controller.randomizer is custom_randomizer


def test_env_randomization_on_reset_and_mj_set_const():
    """Verify mass/friction are randomized on reset and mj_setConst is invoked."""
    env = LeapCubeReorientEnv(enable_domain_randomization=True, seed=123)
    env.curriculum_controller.set_difficulty(1.0)

    nominal_mass = env.randomizer.nominal_mass
    nominal_friction = env.randomizer.nominal_friction.copy()

    masses = []
    frictions = []

    with patch.object(mujoco, "mj_setConst", wraps=mujoco.mj_setConst) as mock_set_const:
        for _ in range(15):
            env.reset()
            current_mass = float(env.model.body_mass[env.randomizer.cube_body_id])
            current_friction = env.model.geom_friction[env.randomizer.cube_geom_id].copy()
            masses.append(current_mass)
            frictions.append(current_friction)

        assert mock_set_const.call_count >= 15

    assert np.var(masses) > 1e-6
    assert np.var(np.array(frictions)[:, 0]) > 1e-6

    for m in masses:
        assert 0.5 * nominal_mass <= m <= 1.5 * nominal_mass
    for f in frictions:
        assert np.all(f >= 0.6 * nominal_friction)
        assert np.all(f <= 1.4 * nominal_friction)


def test_env_observation_latency_and_noise():
    """Verify observation latency and sensor noise are injected via randomizer in _get_obs()."""
    env = LeapCubeReorientEnv(enable_domain_randomization=True, seed=999)

    # 1. Test observation latency without noise (difficulty=0.0)
    env.curriculum_controller.set_difficulty(0.0)
    td0 = env.reset()
    env.randomizer.current_latency_steps = 1
    obs0 = td0["obs"].clone()
    assert torch.isfinite(obs0).all()
    assert obs0.shape == (55,)

    action = torch.zeros(16, dtype=torch.float32)
    step_td = td0.clone()
    step_td["action"] = action

    td1 = env.step(step_td)
    obs1 = td1["obs"].clone()
    # At step 1 with latency=1 and no noise, observation matches initial observation obs0
    assert torch.allclose(obs1, obs0, atol=1e-5)

    # 2. Test sensor noise injection when difficulty > 0
    env.curriculum_controller.set_difficulty(1.0)
    assert env.randomizer.current_sensor_noise_std > 0.0

    noises = []
    for _ in range(10):
        td = env.reset()
        raw_hand = env.data.qpos[:16].astype(np.float32)
        # obs hand is first 16 dimensions
        obs_hand = td["obs"][:16].numpy()
        noises.append(obs_hand - raw_hand)

    assert np.std(np.concatenate(noises)) > 0.001


def test_create_trajectory_collector_with_domain_randomization():
    """Verify create_trajectory_collector with enable_domain_randomization collects valid HER transitions."""
    env_fn = lambda: LeapCubeReorientEnv(
        max_steps=20,
        target_mode="so3",
        reward_mode="sparse",
    )
    actor = SquashedGaussianSACActor(obs_dim=55, goal_dim=4, action_dim=16)
    policy = PolicyModule(actor)
    replay_buffer = create_her_replay_buffer(
        storage_size=500, slice_len=10, batch_size=32
    )

    collector = create_trajectory_collector(
        env_fn=env_fn,
        policy=policy,
        replay_buffer=replay_buffer,
        frames_per_batch=40,
        total_frames=80,
        enable_domain_randomization=True,
    )

    try:
        for _ in collector:
            pass
    finally:
        collector.shutdown()

    assert len(replay_buffer) >= 80

    batch = replay_buffer.sample()
    assert "obs" in batch
    assert "action" in batch
    assert "next" in batch
    assert "reward" in batch["next"]
    assert torch.isfinite(batch["obs"]).all()
    assert torch.isfinite(batch["action"]).all()
    assert torch.isfinite(batch["next"]["reward"]).all()


def test_sac_trainer_optimization_with_domain_randomization():
    """Verify SAC optimization step runs stably without NaNs under randomized physics and HER replay."""
    actor = SquashedGaussianSACActor(obs_dim=55, goal_dim=4, action_dim=16)
    critic = TwinContinuousQCritic(obs_dim=55, goal_dim=4, action_dim=16)
    policy = PolicyModule(actor)
    trainer = SACTrainer(actor=actor, critic=critic)

    env_fn = lambda: LeapCubeReorientEnv(
        max_steps=20,
        enable_domain_randomization=True,
        target_mode="so3",
        reward_mode="sparse",
    )
    replay_buffer = create_her_replay_buffer(
        storage_size=500, slice_len=10, batch_size=32
    )

    collector = create_trajectory_collector(
        env_fn=env_fn,
        policy=policy,
        replay_buffer=replay_buffer,
        frames_per_batch=50,
        total_frames=100,
    )

    try:
        for _ in collector:
            pass
    finally:
        collector.shutdown()

    metrics, pos_ratio = trainer.train_step_from_buffer(replay_buffer)

    for k, v in metrics.items():
        assert math.isfinite(v), f"Metric {k} is non-finite: {v}"
    assert math.isfinite(pos_ratio)


def test_env_enable_domain_randomization_property_setter():
    """Verify enable_domain_randomization property setter dynamically toggles randomizer and curriculum."""
    env = LeapCubeReorientEnv(enable_domain_randomization=False, seed=42)
    assert not env.enable_domain_randomization
    assert env.randomizer is None
    assert env.curriculum_controller is None

    # Dynamically enable
    env.enable_domain_randomization = True
    assert env.enable_domain_randomization
    assert isinstance(env.randomizer, DomainRandomizer)
    assert isinstance(env.curriculum_controller, ADRCurriculumController)
    assert env.curriculum_controller.randomizer is env.randomizer

    # Dynamically disable
    env.enable_domain_randomization = False
    assert not env.enable_domain_randomization
    assert env.randomizer is not None
