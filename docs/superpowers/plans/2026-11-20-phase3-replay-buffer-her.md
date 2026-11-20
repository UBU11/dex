# Phase 3: Replay Buffer + HER Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement Phase 3 Replay Buffer and Hindsight Experience Replay (HER) training pipeline with TorchRL collector trajectory writing, buffer transform integration, and full SO(3) sparse reorientation verification.

**Architecture:** Extend `VectorizedFutureHER` to operate as a callable buffer transform and sampler for `TensorDictReplayBuffer` with `SliceSampler`. Wire TorchRL's `SyncDataCollector` with `replay_write_mode="trajectory"` and verify 15%–35% relabeled positive reward ratio during full SO(3) sparse reorientation training.

**Tech Stack:** Python 3.14 / 3.13, PyTorch 2.15+, TorchRL 0.15, TensorDict 0.15, MuJoCo 3.4+, pytest, ruff.

## Global Constraints

- Do not write unnecessary bloated comments in code files; only write comments if strictly necessary.
- Verification commands run via `.venv/bin/pytest tests/ -q` and `.venv/bin/ruff check src/ tests/` (with `BypassSandbox: true` when running python/.venv binaries if required by environment pathing).
- Support TorchRL 0.15+ collector aliasing: `SyncDataCollector = getattr(torchrl.collectors, "SyncDataCollector", Collector)`.
- Trajectory collection must use `replay_write_mode="trajectory"` into `TensorDictReplayBuffer`.
- HER relabeling must produce 15%–35% positive rewards ($r=0.0$) on trajectories with unreached desired goals.
- Maintain existing 50 unit tests intact and passing.

---

### Task 1: VectorizedFutureHER Replay Buffer Transform & Helper Integration

**Files:**
- Modify: `src/algo/her.py`
- Modify: `src/algo/__init__.py`
- Modify: `tests/test_her_relabeling.py`

**Interfaces:**
- Produces:
  - `VectorizedFutureHER(relabel_prob: float = 0.8, angle_tol: float = 0.15, drop_z_threshold: float = 0.04, slice_len: int | None = None)`
    - `__call__(batch: TensorDict) -> TensorDict` (supporting both 2D episodic `[B, T]` and 1D sliced `[B * T]` via `slice_len` or trajectory reshaping)
    - `as_transform(slice_len: int) -> Callable[[TensorDict], TensorDict]`
  - `create_her_replay_buffer(storage_size: int, slice_len: int, relabel_prob: float = 0.8, batch_size: int = 256) -> TensorDictReplayBuffer`

- [ ] **Step 1: Write failing unit tests for buffer transform integration**
  In `tests/test_her_relabeling.py`:
  - Test `VectorizedFutureHER` with `slice_len` processes 1D TensorDicts of shape `[B * slice_len]`.
  - Test `create_her_replay_buffer` returns a `TensorDictReplayBuffer` with `SliceSampler` and HER transform.
  - Test sampling from `create_her_replay_buffer` returns relabeled transitions matching batch size with valid keys.

- [ ] **Step 2: Run test to confirm failure**
  Run: `.venv/bin/pytest tests/test_her_relabeling.py -k "test_her_replay_buffer" -q`

- [ ] **Step 3: Implement buffer transform and `create_her_replay_buffer`**
  In `src/algo/her.py`:
  - Enhance `VectorizedFutureHER` to accept `slice_len: int | None = None`. If input `batch` is 1D and `slice_len` is provided, reshape to `[B, slice_len]`, apply vectorized future relabeling along trajectory slices, and restore shape.
  - Implement `create_her_replay_buffer(storage_size, slice_len, relabel_prob=0.8, batch_size=256) -> TensorDictReplayBuffer`.
  In `src/algo/__init__.py`:
  - Export `create_her_replay_buffer`.

- [ ] **Step 4: Run tests to confirm pass**
  Run: `.venv/bin/pytest tests/test_her_relabeling.py -q`

- [ ] **Step 5: Run linter and commit changes**
  Run: `.venv/bin/ruff check src/ tests/ && git add src/algo/her.py src/algo/__init__.py tests/test_her_relabeling.py && git commit -m "feat(algo): integrate VectorizedFutureHER buffer transform and create_her_replay_buffer"`

---

### Task 2: SyncDataCollector Trajectory Wiring & Full SO(3) Sparse Training Verification

**Files:**
- Create: `src/algo/collector.py`
- Modify: `src/algo/sac_trainer.py`
- Modify: `src/algo/__init__.py`
- Create: `tests/test_her_training.py`
- Modify: `main.py`

**Interfaces:**
- Produces:
  - `create_trajectory_collector(env_fn: Callable[[], EnvBase], policy: nn.Module, replay_buffer: TensorDictReplayBuffer, frames_per_batch: int, total_frames: int = -1) -> BaseCollector`
  - `PolicyModule(actor: SquashedGaussianSACActor) -> nn.Module`
  - `SACTrainer.train_step_from_buffer(replay_buffer: TensorDictReplayBuffer, batch_size: int | None = None) -> tuple[dict[str, float], float]` (returning metrics and relabeled positive reward ratio)

- [ ] **Step 1: Write failing integration test for collector trajectory wiring and positive reward ratio**
  In `tests/test_her_training.py`:
  - Test `create_trajectory_collector` wires with `replay_write_mode="trajectory"`.
  - Test short training rollout on full SO(3) sparse environment (`target_mode="so3"`, `reward_mode="sparse"`).
  - Verify relabeled positive reward ratio in sampled buffer batches sits within 15%–35% (`0.15 <= pos_ratio <= 0.35`).
  - Verify `SACTrainer` updates without NaNs.

- [ ] **Step 2: Run test to confirm failure**
  Run: `.venv/bin/pytest tests/test_her_training.py -q`

- [ ] **Step 3: Implement `src/algo/collector.py` and `SACTrainer.train_step_from_buffer`**
  In `src/algo/collector.py`:
  - Implement `PolicyModule` wrapping `SquashedGaussianSACActor` for TorchRL collectors.
  - Implement `create_trajectory_collector` using `getattr(torchrl.collectors, "SyncDataCollector", Collector)` with `replay_write_mode="trajectory"`.
  In `src/algo/sac_trainer.py`:
  - Add `train_step_from_buffer(replay_buffer, batch_size=None)` that samples from replay buffer, computes `pos_ratio = (batch["next", "reward"] == 0.0).float().mean().item()`, steps optimizer, and returns `(metrics, pos_ratio)`.
  In `src/algo/__init__.py`:
  - Export `PolicyModule`, `create_trajectory_collector`.
  In `main.py`:
  - Demonstrate collector trajectory collection, replay buffer sampling, and HER relabeled positive reward verification.

- [ ] **Step 4: Run tests to confirm pass**
  Run: `.venv/bin/pytest tests/test_her_training.py -q`

- [ ] **Step 5: Run full test suite & linter, and commit changes**
  Run: `.venv/bin/pytest tests/ -q && .venv/bin/ruff check src/ tests/ && git add src/ tests/ main.py && git commit -m "feat(algo): wire SyncDataCollector trajectory writing and verify full SO(3) sparse HER training"`
