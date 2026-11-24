# Parallel Trajectory Collection & Curriculum Warmup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement parallel trajectory collection (`num_workers > 1` via TorchRL `ParallelEnv`) and curriculum warmup (Phase 2 Z-yaw to full $SO(3)$ targets) to scale sample collection up to 10M frames from existing checkpoints.

**Architecture:** (1) Add explicit dynamic target mode switching (`set_target_mode`) on `LeapCubeReorientEnv` and collector wrapper handling both single and batched/parallel environments; (2) Extend `create_trajectory_collector` with `num_workers` to instantiate `ParallelEnv` for multi-worker simulation; (3) Update `train.py` with `--num-workers`, `--warmup-yaw-frames`, and automated transition from yaw warmup to full $SO(3)$ target sampling during training.

**Tech Stack:** Python 3.14 / 3.13, PyTorch 2.15+, TorchRL 0.15, TensorDict 0.15, MuJoCo 3.4+, pytest, ruff.

## Global Constraints

- Do not write unnecessary bloated comments in code files; only write comments if strictly necessary.
- Verification commands run via `.venv/bin/pytest tests/ -q` and `.venv/bin/ruff check src/ tests/ train.py` (with `BypassSandbox: true` when running python/.venv binaries if required by environment pathing).
- Benchmark Deliverable Target Metric: > 85% consecutive target-orientation success under randomized physics and latency.
- No direct access to `mjData` from `src/algo/`; all observations flow through `TensorDict`.
- Ensure numerical stability: no un-softplused log Jacobian and no unclamped quaternion geodesic acos.
- Trajectory writing into `TensorDictReplayBuffer` must remain deterministic and preserve `SliceSampler` requirements.
- Maintain existing 92 unit tests intact and passing.

---

### Task 1: Dynamic Target Mode Switching in `LeapCubeReorientEnv` and Collector

**Files:**
- Modify: `src/envs/leap_cube_env.py`
- Modify: `src/algo/collector.py`
- Modify: `src/algo/__init__.py`
- Test: `tests/test_leap_cube_env.py`
- Test: `tests/test_train_script.py`

**Interfaces:**
- `LeapCubeReorientEnv.set_target_mode(self, mode: str) -> None`:
  - Validates `mode in ("so3", "yaw")`.
  - Updates `self.target_mode = mode`.
- `update_collector_target_mode(collector: Collector, target_mode: str) -> None`:
  - Updates target mode on `collector.env` or calls dispatch on `ParallelEnv` sub-environments.

- [ ] **Step 1: Write failing tests for `set_target_mode` and `update_collector_target_mode`**
  In `tests/test_leap_cube_env.py`:
  - Test `env.set_target_mode("yaw")` and `env.set_target_mode("so3")`.
  - Test invalid target mode raises `ValueError`.
  - Verify sampled goals match the updated mode.
  In `tests/test_train_script.py`:
  - Test `update_collector_target_mode` updates target mode on single and parallel environments.

- [ ] **Step 2: Run tests to confirm failure**
  Run: `.venv/bin/pytest tests/test_leap_cube_env.py tests/test_train_script.py -k "target_mode" -q`

- [ ] **Step 3: Implement `set_target_mode` and `update_collector_target_mode`**
  In `src/envs/leap_cube_env.py`:
  - Implement `set_target_mode(self, mode: str) -> None`.
  In `src/algo/collector.py`:
  - Implement `update_collector_target_mode(collector: Collector, target_mode: str) -> None`.
  In `src/algo/__init__.py`:
  - Export `update_collector_target_mode`.

- [ ] **Step 4: Run tests to confirm pass**
  Run: `.venv/bin/pytest tests/test_leap_cube_env.py tests/test_train_script.py -k "target_mode" -q`

- [ ] **Step 5: Run linter and commit changes**
  Run: `.venv/bin/ruff check src/ tests/ train.py && git add src/ tests/ && git commit -m "feat(envs): add dynamic target mode switching for LeapCubeReorientEnv and collector"`

---

### Task 2: Multi-Worker Parallel Trajectory Collection

**Files:**
- Modify: `src/algo/collector.py`
- Modify: `tests/test_train_script.py`

**Interfaces:**
- `create_trajectory_collector(env_fn, policy, replay_buffer, frames_per_batch, total_frames, enable_domain_randomization, num_workers=1, **kwargs)`:
  - When `num_workers > 1`, instantiates `ParallelEnv(num_workers, [env_fn for _ in range(num_workers)])`.
  - Automatically handles `ParallelEnv` reset and trajectory buffering into `TensorDictReplayBuffer`.

- [ ] **Step 1: Write failing tests for multi-worker parallel trajectory collector**
  In `tests/test_train_script.py`:
  - Test `create_trajectory_collector` with `num_workers=2`.
  - Verify batches are collected without errors and stored properly into `replay_buffer`.
  - Verify `update_collector_target_mode` works seamlessly on multi-worker collectors.

- [ ] **Step 2: Run tests to confirm failure**
  Run: `.venv/bin/pytest tests/test_train_script.py -k "parallel or num_workers" -q`

- [ ] **Step 3: Implement `num_workers` support in `create_trajectory_collector`**
  In `src/algo/collector.py`:
  - Add `num_workers: int = 1` parameter.
  - If `num_workers > 1`, construct `ParallelEnv` using `torchrl.envs.ParallelEnv`.
  - Ensure ADR and domain randomization controllers wrap workers properly.

- [ ] **Step 4: Run tests to confirm pass**
  Run: `.venv/bin/pytest tests/test_train_script.py -q`

- [ ] **Step 5: Run linter and commit changes**
  Run: `.venv/bin/ruff check src/ tests/ train.py && git add src/algo/collector.py tests/test_train_script.py && git commit -m "feat(algo): support multi-worker parallel trajectory collection with ParallelEnv"`

---

### Task 3: Curriculum Warmup & Scale Sample Collection in `train.py`

**Files:**
- Modify: `train.py`
- Modify: `tests/test_train_script.py`

**Interfaces:**
- CLI arguments in `train.py`:
  - `--num-workers`: int (default 1)
  - `--warmup-yaw-frames`: int (default 0)
  - `--target-mode`: str (default "so3", choices: ["so3", "yaw"])
- Behavior in `train(args)`:
  - If `args.warmup_yaw_frames > 0` and `start_step < args.warmup_yaw_frames`:
    - Begins in `"yaw"` mode.
    - When `current_step >= args.warmup_yaw_frames`, switches collector and eval env to `"so3"`.
  - Pass `num_workers=args.num_workers` to `create_trajectory_collector`.
  - Support resuming from `--resume checkpoints/best_model.pt` with `--total-frames 10000000`.

- [ ] **Step 1: Write tests for curriculum warmup and parallel collector CLI**
  In `tests/test_train_script.py`:
  - Test parser recognizes `--num-workers` and `--warmup-yaw-frames`.
  - Test `train(args)` switches target mode dynamically when `warmup_yaw_frames` threshold is crossed.
  - Test resuming from checkpoint preserves/updates step counts and handles extended frame budgets.

- [ ] **Step 2: Run tests to confirm failure**
  Run: `.venv/bin/pytest tests/test_train_script.py -k "warmup or workers" -q`

- [ ] **Step 3: Implement curriculum warmup and parallel workers in `train.py`**
  In `train.py`:
  - Update `build_parser` with `--num-workers` and `--warmup-yaw-frames`.
  - Wire `create_trajectory_collector` with `num_workers=args.num_workers`.
  - Add warmup phase tracking and dynamic transition via `update_collector_target_mode`.

- [ ] **Step 4: Run full test suite and verify all tests pass**
  Run: `.venv/bin/pytest tests/ -q`

- [ ] **Step 5: Run linter and commit changes**
  Run: `.venv/bin/ruff check src/ tests/ train.py && git add train.py tests/test_train_script.py && git commit -m "feat(train): add curriculum warmup and multi-worker training support"`
