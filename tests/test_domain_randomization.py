import math
import sys
from pathlib import Path
from unittest.mock import patch

import mujoco
import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.envs import ADRCurriculumController, DomainRandomizer, build_leap_cube_spec


@pytest.fixture
def mj_model_data():
    spec = build_leap_cube_spec()
    model = spec.compile()
    data = mujoco.MjData(model)
    return model, data


def test_domain_randomizer_difficulty_zero(mj_model_data):
    model, data = mj_model_data
    randomizer = DomainRandomizer(model, data, sensor_noise_std=0.005)
    randomizer.set_difficulty(0.0)

    nominal_mass = randomizer.nominal_mass
    nominal_friction = randomizer.nominal_friction.copy()
    nominal_damping = randomizer.nominal_damping.copy()

    rng = np.random.default_rng(123)
    for _ in range(10):
        randomizer.randomize(rng)
        assert math.isclose(
            float(model.body_mass[randomizer.cube_body_id]), nominal_mass, rel_tol=1e-6
        )
        assert np.allclose(
            model.geom_friction[randomizer.cube_geom_id], nominal_friction, atol=1e-6
        )
        assert np.allclose(model.dof_damping[:16], nominal_damping, atol=1e-6)
        assert randomizer.current_latency_steps == 0

    obs = np.ones(20, dtype=np.float32)
    delayed_obs = randomizer.get_delayed_observation(obs, rng=rng)
    assert np.allclose(delayed_obs, obs, atol=1e-7)


def test_domain_randomizer_difficulty_one(mj_model_data):
    model, data = mj_model_data
    randomizer = DomainRandomizer(model, data, sensor_noise_std=0.005)
    randomizer.set_difficulty(1.0)

    assert randomizer.mass_range == (0.5, 1.5)
    assert randomizer.friction_range == (0.6, 1.4)
    assert randomizer.damping_range == (0.7, 1.3)
    assert randomizer.max_latency == 2
    assert math.isclose(randomizer.current_sensor_noise_std, 0.005, abs_tol=1e-7)

    rng = np.random.default_rng(42)
    latencies = set()
    mass_scales = []
    for _ in range(50):
        randomizer.randomize(rng)
        mass_scale = (
            float(model.body_mass[randomizer.cube_body_id]) / randomizer.nominal_mass
        )
        mass_scales.append(mass_scale)
        latencies.add(randomizer.current_latency_steps)
        assert 0.5 <= mass_scale <= 1.5

    assert max(latencies) == 2
    assert min(mass_scales) < 0.7
    assert max(mass_scales) > 1.3

    obs = np.zeros(1000, dtype=np.float32)
    delayed_obs = randomizer.get_delayed_observation(obs, rng=rng)
    assert not np.allclose(delayed_obs, obs)
    assert 0.003 < np.std(delayed_obs) < 0.007


def test_domain_randomizer_sensor_noise_tensor(mj_model_data):
    model, data = mj_model_data
    randomizer = DomainRandomizer(model, data, sensor_noise_std=0.01)
    randomizer.set_difficulty(1.0)

    tensor_obs = torch.zeros(500, dtype=torch.float32)
    rng = np.random.default_rng(99)
    noisy_obs = randomizer.get_delayed_observation(tensor_obs, rng=rng)

    assert isinstance(noisy_obs, torch.Tensor)
    assert noisy_obs.shape == tensor_obs.shape
    assert noisy_obs.dtype == tensor_obs.dtype
    assert torch.std(noisy_obs).item() > 0.005


def test_domain_randomizer_mj_set_const_called(mj_model_data):
    model, data = mj_model_data
    randomizer = DomainRandomizer(model, data)
    randomizer.set_difficulty(0.5)

    with patch("mujoco.mj_setConst", wraps=mujoco.mj_setConst) as mock_set_const:
        randomizer.randomize()
        mock_set_const.assert_called_once_with(model, data)


def test_adr_curriculum_controller_progression(mj_model_data):
    model, data = mj_model_data
    randomizer = DomainRandomizer(model, data)
    controller = ADRCurriculumController(
        randomizer=randomizer,
        start_threshold=0.50,
        target_threshold=0.85,
        step_size=0.05,
    )

    assert controller.difficulty == 0.0
    assert randomizer.difficulty == 0.0

    # Below start_threshold: no progression
    new_diff = controller.update(0.40)
    assert new_diff == 0.0
    assert controller.difficulty == 0.0

    # At or above start_threshold: advances by step_size
    new_diff = controller.update(0.50)
    assert math.isclose(new_diff, 0.05, abs_tol=1e-6)
    assert math.isclose(controller.difficulty, 0.05, abs_tol=1e-6)
    assert math.isclose(randomizer.difficulty, 0.05, abs_tol=1e-6)

    # Clamping at 1.0
    controller.set_difficulty(0.98)
    controller.update(0.90)
    assert controller.difficulty == 1.0

    # Bounds clamping in set_difficulty
    controller.set_difficulty(-0.5)
    assert controller.difficulty == 0.0
    controller.set_difficulty(1.5)
    assert controller.difficulty == 1.0
