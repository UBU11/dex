# Reset Distribution Hardening & Extended Horizon Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement palm offset and random orientation distribution during `env.reset()`, and verify extended horizon ($T=400$) consecutive goal chaining without hand reset.

**Architecture:** Extend `LeapCubeReorientEnv` with `randomize_initial_orientation: bool = True` and configurable `xy_offset_range: float = 0.015`. During `reset()`, sample uniform $SO(3)$ rotation (or yaw rotation) and $\pm 1.5$ cm XY position offsets. Support full $T=400$ horizon where in-place consecutive goal transitions accumulate multiple successes per episode.

**Tech Stack:** Python 3.14 / 3.13, PyTorch 2.15+, TorchRL 0.15, TensorDict 0.15, MuJoCo 3.4+, pytest, ruff.

## Global Constraints

- Do not write unnecessary bloated comments in code files; only write comments if strictly necessary.
- Verification commands run via `.venv/bin/pytest tests/ -q` and `.venv/bin/ruff check src/ tests/` (with `BypassSandbox: true` when running python/.venv binaries if required by environment pathing).
- XY offset distribution: $x, y \sim \mathcal{U}[-0.015, 0.015]$ meters around default cube position.
- Orientation distribution: sample unit quaternion via `sample_uniform_quat` (or `sample_yaw_quat` for yaw mode) on reset when `randomize_initial_orientation=True`.
- Support `randomize_initial_orientation: bool = True` with backward-compatible option `False` for deterministic initial pose tests.
- Maintain existing 70 unit tests intact and passing.

---

### Task 1: Palm Offset & Random Orientation Distribution on Reset

**Files:**
- Modify: `src/envs/leap_cube_env.py`
- Modify: `tests/test_leap_cube_env.py`

**Interfaces:**
- Produces:
  - `LeapCubeReorientEnv(..., xy_offset_range: float = 0.015, randomize_initial_orientation: bool = True)`
  - Reset state: `cube_p[:2]` randomized within $\pm \text{xy\_offset\_range}$, `cube_q` randomized on $SO(3)$ when enabled.

- [ ] **Step 1: Write failing unit tests for reset orientation and offset distribution**
  In `tests/test_leap_cube_env.py`:
  - Test `env.reset()` produces varied cube orientations across multiple resets when `randomize_initial_orientation=True`.
  - Test `env.reset()` produces `DEFAULT_CUBE_QUAT` when `randomize_initial_orientation=False`.
  - Test `cube_p[:2]` offsets stay strictly within $[-0.015, 0.015]$ relative to `DEFAULT_CUBE_POS[:2]`.

- [ ] **Step 2: Run test to confirm failure**
  Run: `.venv/bin/pytest tests/test_leap_cube_env.py -k "test_reset_orientation_distribution" -q`

- [ ] **Step 3: Implement random orientation on reset in `src/envs/leap_cube_env.py`**
  In `src/envs/leap_cube_env.py`:
  - Accept `randomize_initial_orientation: bool = True` and `xy_offset_range: float = 0.015` in `__init__`.
  - In `_reset()`:
    - Apply XY offset to cube position using `self.xy_offset_range`.
    - If `self.randomize_initial_orientation`: sample random unit quaternion using `sample_yaw_quat` if `target_mode == "yaw"` else `sample_uniform_quat`, and assign to `self.data.qpos[19:23]`.
    - Else: assign `DEFAULT_CUBE_QUAT`.
    - Call `mujoco.mj_forward(self.model, self.data)`.

- [ ] **Step 4: Run tests to confirm pass**
  Run: `.venv/bin/pytest tests/test_leap_cube_env.py -q`

- [ ] **Step 5: Run linter and commit changes**
  Run: `.venv/bin/ruff check src/ tests/ && git add src/envs/leap_cube_env.py tests/test_leap_cube_env.py && git commit -m "feat(envs): add random initial orientation and offset distribution on reset"`

---

### Task 2: Extended Horizon ($T=400$) & Multi-Target Consecutive Goal Verification

**Files:**
- Create: `tests/test_consecutive_goals.py`
- Modify: `main.py`

**Interfaces:**
- Produces:
  - Long horizon trajectory rollout up to $T=400$ steps without early drop termination.
  - Consecutive goal chaining: increments `consecutive_successes` and resamples `desired_goal` without hand or cube reset.

- [ ] **Step 1: Write integration tests for $T=400$ horizon and consecutive goal chaining**
  In `tests/test_consecutive_goals.py`:
  - Test episode stepping runs for full 400 steps without premature truncation.
  - Test mock/oracle orientation tracking that holds target orientation for 5 steps triggers goal chaining:
    - `consecutive_successes` increments to 1, then 2.
    - `desired_goal` updates to a new target quaternion.
    - Hand posture and cube positions are preserved continuously across the chain.
  - Test collector collects full 400-step trajectory cleanly into `TensorDictReplayBuffer`.

- [ ] **Step 2: Run test to confirm failure**
  Run: `.venv/bin/pytest tests/test_consecutive_goals.py -q`

- [ ] **Step 3: Verify and refine consecutive goal chaining and update `main.py`**
  In `src/envs/leap_cube_env.py` (if any adjustments needed) and `main.py`:
  - Demonstrate a multi-step rollout showcasing consecutive goal chaining and reset hardening.
  - Verify no NaNs or Infs occur across long horizons.

- [ ] **Step 4: Run tests to confirm pass**
  Run: `.venv/bin/pytest tests/test_consecutive_goals.py -q`

- [ ] **Step 5: Run full test suite & linter, and commit changes**
  Run: `.venv/bin/pytest tests/ -q && .venv/bin/ruff check src/ tests/ main.py && git add tests/test_consecutive_goals.py main.py && git commit -m "feat(envs): verify extended horizon T=400 and consecutive goal chaining"`
