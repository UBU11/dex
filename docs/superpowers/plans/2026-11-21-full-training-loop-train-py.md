# Full Training Loop & Convergence Script (train.py) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a dedicated, robust training and convergence script (`train.py`) with modular evaluation, checkpointing, ADR curriculum progression, and logging targeting >85% consecutive target-orientation success under randomized physics and latency.

**Architecture:** Decompose into modular components: (1) `src/algo/checkpoint.py` for PyTorch checkpoint save/load/resume and ONNX policy export, (2) `src/algo/evaluation.py` for multi-episode deterministic evaluation computing consecutive target orientation success rate and geodesic error under domain randomization, and (3) `train.py` as the end-to-end executable orchestrating collection, SAC updates, HER relabeling, ADR progression, checkpointing, and convergence reporting.

**Tech Stack:** Python 3.14 / 3.13, PyTorch 2.15+, TorchRL 0.15, TensorDict 0.15, MuJoCo 3.4+, pytest, ruff.

## Global Constraints

- Do not write unnecessary bloated comments in code files; only write comments if strictly necessary.
- Verification commands run via `.venv/bin/pytest tests/ -q` and `.venv/bin/ruff check src/ tests/ train.py` (with `BypassSandbox: true` when running python/.venv binaries if required by environment pathing).
- Benchmark Deliverable Target Metric: > 85% consecutive target-orientation success under randomized physics and latency.
- No direct access to `mjData` from `src/algo/`; all observations flow through `TensorDict`.
- Ensure numerical stability: no un-softplused log Jacobian and no unclamped quaternion geodesic acos.
- Maintain existing 76 unit tests intact and passing.

---

### Task 1: Evaluation Metrics & Checkpointing Infrastructure

**Files:**
- Create: `src/algo/checkpoint.py`
- Create: `src/algo/evaluation.py`
- Modify: `src/algo/__init__.py`
- Create: `tests/test_checkpoint.py`
- Create: `tests/test_evaluation.py`

**Interfaces:**
- Produces:
  - `save_checkpoint(checkpoint_path: str | Path, actor: nn.Module, critic: nn.Module, trainer: SACTrainer, curriculum_controller: ADRCurriculumController | None = None, step: int = 0, best_success_rate: float = 0.0, extra_info: dict | None = None) -> None`
  - `load_checkpoint(checkpoint_path: str | Path, actor: nn.Module, critic: nn.Module, trainer: SACTrainer | None = None, curriculum_controller: ADRCurriculumController | None = None, device: torch.device | None = None) -> dict[str, Any]`
  - `export_policy_onnx(onnx_path: str | Path, actor: nn.Module, obs_dim: int = 55, goal_dim: int = 4, device: torch.device | None = None) -> None`
  - `evaluate_policy(actor: nn.Module, env: LeapCubeReorientEnv, num_episodes: int = 10, max_steps: int | None = None, device: torch.device | None = None) -> dict[str, float]`
    - Output keys: `success_rate`, `mean_consecutive_successes`, `mean_geodesic_error`, `mean_return`, `difficulty`

- [ ] **Step 1: Write failing tests for checkpointing and evaluation**
  In `tests/test_checkpoint.py`:
  - Test `save_checkpoint` saves weights, optimizer states, alpha, step count, and curriculum difficulty.
  - Test `load_checkpoint` restores actor/critic/trainer parameters and curriculum difficulty.
  - Test `export_policy_onnx` exports valid ONNX model file.
  In `tests/test_evaluation.py`:
  - Test `evaluate_policy` runs deterministic evaluation over specified episodes.
  - Test `evaluate_policy` returns expected metric keys and valid numeric bounds.
  - Test `evaluate_policy` detects consecutive target success when achieved.

- [ ] **Step 2: Run test to confirm failure**
  Run: `.venv/bin/pytest tests/test_checkpoint.py tests/test_evaluation.py -q`

- [ ] **Step 3: Implement `checkpoint.py` and `evaluation.py`**
  In `src/algo/checkpoint.py`:
  - Implement `save_checkpoint`, `load_checkpoint`, and `export_policy_onnx`.
  In `src/algo/evaluation.py`:
  - Implement `evaluate_policy` tracking consecutive successes, returns, and geodesic errors.
  In `src/algo/__init__.py`:
  - Export new symbols.

- [ ] **Step 4: Run tests to confirm pass**
  Run: `.venv/bin/pytest tests/test_checkpoint.py tests/test_evaluation.py -q`

- [ ] **Step 5: Run linter and commit changes**
  Run: `.venv/bin/ruff check src/ tests/ && git add src/algo/ tests/test_checkpoint.py tests/test_evaluation.py && git commit -m "feat(algo): add checkpointing, ONNX export, and multi-episode evaluation"`

---

### Task 2: End-to-End Training Loop & Convergence Script (`train.py`)

**Files:**
- Create: `train.py`
- Create: `tests/test_train_script.py`

**Interfaces:**
- Produces:
  - `train.py` executable with CLI arguments:
    - `--seed`, `--total-frames`, `--frames-per-batch`, `--storage-size`, `--batch-size`, `--utd-ratio`
    - `--eval-interval`, `--eval-episodes`, `--save-interval`, `--checkpoint-dir`, `--resume`
    - `--enable-adr`, `--target-success-rate` (default: 0.85)
  - `train(args: argparse.Namespace) -> dict[str, Any]` function for programmatic execution and test verification.

- [ ] **Step 1: Write integration test for `train.py`**
  In `tests/test_train_script.py`:
  - Test argument parsing defaults and overrides.
  - Test short programmatic training run (e.g. 200 frames, 1 eval step, checkpoint write).
  - Verify checkpoint directory contains saved checkpoint and metrics summary.

- [ ] **Step 2: Run test to confirm failure**
  Run: `.venv/bin/pytest tests/test_train_script.py -q`

- [ ] **Step 3: Implement `train.py`**
  In `train.py`:
  - Implement CLI argument parser with comprehensive training options.
  - Build environments (training with collection, evaluation with deterministic rollouts).
  - Initialize Actor, Twin Critic, SACTrainer, ReplayBuffer with VectorizedFutureHER, and Collector.
  - Implement training loop:
    - Interleave data collection batches and gradient update steps (UTD ratio).
    - Periodic evaluation via `evaluate_policy`.
    - Update `ADRCurriculumController` with evaluation success rate.
    - Checkpoint saving on eval interval and when a new `best_success_rate` is achieved.
    - Export ONNX policy when benchmark deliverable threshold (> 85%) is met.
    - Output clear, concise training logs.

- [ ] **Step 4: Run tests to confirm pass**
  Run: `.venv/bin/pytest tests/test_train_script.py -q`

- [ ] **Step 5: Run full test suite & linter, and commit changes**
  Run: `.venv/bin/pytest tests/ -q && .venv/bin/ruff check src/ tests/ train.py && git add train.py tests/test_train_script.py && git commit -m "feat(train): implement full training loop and convergence script"`
