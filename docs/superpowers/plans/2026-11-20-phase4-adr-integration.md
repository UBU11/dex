# Phase 4: Automatic Domain Randomization (ADR) Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement Phase 4 Automatic Domain Randomization (ADR) pipeline by creating an ADR curriculum controller, adding sensor noise, and wiring domain randomization natively into `LeapCubeReorientEnv` and `create_trajectory_collector`.

**Architecture:** Extend `DomainRandomizer` to support parametrized difficulty levels $\lambda \in [0.0, 1.0]$ and Gaussian sensor noise. Create `ADRCurriculumController` that modulates parameter scaling once policy success exceeds 50%. Wire domain randomization into `LeapCubeReorientEnv` reset and observation generation.

**Tech Stack:** Python 3.14 / 3.13, PyTorch 2.15+, TorchRL 0.15, TensorDict 0.15, MuJoCo 3.4+, pytest, ruff.

## Global Constraints

- Do not write unnecessary bloated comments in code files; only write comments if strictly necessary.
- Verification commands run via `.venv/bin/pytest tests/ -q` and `.venv/bin/ruff check src/ tests/` (with `BypassSandbox: true` when running python/.venv binaries if required by environment pathing).
- Full ADR bounds at $\lambda = 1.0$:
  - Cube Mass: $[0.5\times, 1.5\times]$
  - Surface Friction: $[0.6\times, 1.4\times]$
  - Hand Joint Damping: $[0.7\times, 1.3\times]$
  - Observation Latency: $0$ to $2$ steps FIFO delay
  - Sensor Noise: Gaussian noise $\sigma = 0.005 \times \lambda$ on cube pose and joint velocities
- Curriculum activation begins only when policy reaches $\ge 50\%$ nominal success rate.
- Maintain existing 59 unit tests intact and passing.

---

### Task 1: ADR Curriculum Controller & Parametrized Randomization

**Files:**
- Modify: `src/envs/domain_randomization.py`
- Modify: `src/envs/__init__.py`
- Create: `tests/test_domain_randomization.py`

**Interfaces:**
- Produces:
  - `ADRCurriculumController(randomizer: DomainRandomizer, start_threshold: float = 0.50, target_threshold: float = 0.85, step_size: float = 0.05)`
    - `difficulty: float` (property $\lambda \in [0.0, 1.0]$)
    - `set_difficulty(level: float)`
    - `update(success_rate: float) -> float`
  - `DomainRandomizer(..., sensor_noise_std: float = 0.005)`
    - `set_difficulty(level: float)`
    - `randomize(rng: np.random.Generator | None = None)`
    - `get_delayed_observation(raw_obs: np.ndarray | torch.Tensor, rng: np.random.Generator | None = None) -> np.ndarray | torch.Tensor`

- [ ] **Step 1: Write failing unit tests for ADR curriculum and noise**
  In `tests/test_domain_randomization.py`:
  - Test difficulty $\lambda = 0.0$ preserves nominal mass, friction, damping, 0 latency, and 0 noise.
  - Test difficulty $\lambda = 1.0$ spans $[0.5, 1.5]$ mass, $[0.6, 1.4]$ friction, $[0.7, 1.3]$ damping, up to 2 latency steps, and $\sigma=0.005$ sensor noise.
  - Test `ADRCurriculumController.update` stays at $\lambda = 0.0$ when success $< 0.50$, and increments $\lambda$ when success $\ge 0.50$.
  - Test `mujoco.mj_setConst` is always called on `randomize()`.

- [ ] **Step 2: Run test to confirm failure**
  Run: `.venv/bin/pytest tests/test_domain_randomization.py -q`

- [ ] **Step 3: Implement `ADRCurriculumController` and dynamic ranges in `DomainRandomizer`**
  In `src/envs/domain_randomization.py`:
  - Add `set_difficulty(level: float)` to `DomainRandomizer` scaling ranges:
    - mass: $[1.0 - 0.5\lambda, 1.0 + 0.5\lambda]$
    - friction: $[1.0 - 0.4\lambda, 1.0 + 0.4\lambda]$
    - damping: $[1.0 - 0.3\lambda, 1.0 + 0.3\lambda]$
    - max latency: $\text{int}(\text{round}(2 \lambda))$
    - sensor noise: $0.005 \lambda$
  - Add sensor noise injection inside `get_delayed_observation`.
  - Implement `ADRCurriculumController`.
  In `src/envs/__init__.py`:
  - Export `DomainRandomizer` and `ADRCurriculumController`.

- [ ] **Step 4: Run tests to confirm pass**
  Run: `.venv/bin/pytest tests/test_domain_randomization.py -q`

- [ ] **Step 5: Run linter and commit changes**
  Run: `.venv/bin/ruff check src/ tests/ && git add src/envs/ tests/test_domain_randomization.py && git commit -m "feat(envs): implement ADRCurriculumController and parametrized domain randomization"`

---

### Task 2: Wire Domain Randomization into `LeapCubeReorientEnv` & Collector

**Files:**
- Modify: `src/envs/leap_cube_env.py`
- Modify: `src/algo/collector.py`
- Create: `tests/test_adr_integration.py`
- Modify: `main.py`

**Interfaces:**
- Produces:
  - `LeapCubeReorientEnv(..., enable_domain_randomization: bool = False, randomizer: DomainRandomizer | None = None)`
  - `create_trajectory_collector(..., enable_domain_randomization: bool = False)`
  - `env.curriculum_controller: ADRCurriculumController | None`

- [ ] **Step 1: Write failing integration test for environment domain randomization**
  In `tests/test_adr_integration.py`:
  - Test `LeapCubeReorientEnv(enable_domain_randomization=True)` mutates mass/friction on reset and applies observation latency.
  - Test `create_trajectory_collector` with domain-randomized env collects valid trajectories into HER replay buffer.
  - Test SACTrainer optimization remains stable under randomized physics.

- [ ] **Step 2: Run test to confirm failure**
  Run: `.venv/bin/pytest tests/test_adr_integration.py -q`

- [ ] **Step 3: Wire domain randomization in `LeapCubeReorientEnv` and `collector.py`**
  In `src/envs/leap_cube_env.py`:
  - Initialize `DomainRandomizer` if `enable_domain_randomization=True` (or use provided instance).
  - Call `randomizer.randomize(self.np_rng)` during `reset()`.
  - Pass observations through `randomizer.get_delayed_observation(...)` in `_get_obs()`.
  In `src/algo/collector.py`:
  - Support `enable_domain_randomization` flag in environment factory helper.
  In `main.py`:
  - Add demonstration of domain randomization and ADR difficulty ramping.

- [ ] **Step 4: Run tests to confirm pass**
  Run: `.venv/bin/pytest tests/test_adr_integration.py -q`

- [ ] **Step 5: Run full test suite & linter, and commit changes**
  Run: `.venv/bin/pytest tests/ -q && .venv/bin/ruff check src/ tests/ main.py && git add src/ tests/ main.py && git commit -m "feat(envs): wire domain randomization into LeapCubeReorientEnv and collector"`
