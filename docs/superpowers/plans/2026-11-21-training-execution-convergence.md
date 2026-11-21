# Training Execution & Convergence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Execute the full reinforcement learning training run via `train.py` on CUDA/CPU, ensuring seamless device interoperability, checkpointing, and ADR curriculum progression towards >85% consecutive target-orientation success.

**Architecture:** (1) Fix tensor device mismatch in `PolicyModule` so the actor on CUDA can interact with CPU simulation environments in TorchRL `SyncDataCollector`, (2) Add early stopping and convergence tracking when the target success rate is achieved, and (3) Execute the end-to-end training pipeline via `train.py` with evaluation logging and ONNX export.

**Tech Stack:** Python 3.14 / 3.13, PyTorch 2.15 (CUDA), TorchRL 0.15, TensorDict 0.15, MuJoCo 4.0+, pytest, ruff.

## Global Constraints

- Do not write unnecessary bloated comments in code files; only write comments if strictly necessary.
- Verification commands run via `.venv/bin/pytest tests/ -q` and `.venv/bin/ruff check src/ tests/ train.py`.
- Benchmark Deliverable Target Metric: > 85% consecutive target-orientation success under randomized physics and latency.
- No direct access to `mjData` from `src/algo/`; all observations flow through `TensorDict`.
- Ensure numerical stability: no un-softplused log Jacobian and no unclamped quaternion geodesic acos.
- Maintain existing 87 unit tests intact and passing.

---

### Task 1: Device-Agnostic Tensor Handling in `PolicyModule` & CUDA Execution Support

**Files:**
- Modify: `src/algo/collector.py`
- Modify: `tests/test_train_script.py`

**Interfaces:**
- `PolicyModule`:
  - `forward(td: TensorDict) -> TensorDict`
  - Transfers `obs` and `desired_goal` to the actor's device, queries `actor`, and places `action` back onto `td.device`.

- [ ] **Step 1: Write test for device-agnostic PolicyModule and CUDA training**
  In `tests/test_train_script.py`:
  - Add test verifying `PolicyModule` correctly handles mismatched devices (e.g. CUDA/CPU or device tensors).
  - Test `train(args)` with `--device cuda` when CUDA is available.

- [ ] **Step 2: Run test to confirm failure**
  Run: `.venv/bin/pytest tests/test_train_script.py -k "device or cuda" -q`

- [ ] **Step 3: Implement device transfer in `PolicyModule`**
  In `src/algo/collector.py`:
  - Update `PolicyModule.forward` to move inputs to `next(self.actor.parameters()).device` and put `action` on `td.device`.

- [ ] **Step 4: Run tests to confirm pass**
  Run: `.venv/bin/pytest tests/test_train_script.py -q`

- [ ] **Step 5: Run linter and commit changes**
  Run: `.venv/bin/ruff check src/ tests/ train.py && git add src/algo/collector.py tests/test_train_script.py && git commit -m "fix(algo): support device-agnostic tensor transfer in PolicyModule for CUDA training"`

---

### Task 2: Training Pipeline Convergence Enhancements & Early Stopping

**Files:**
- Modify: `train.py`
- Modify: `tests/test_train_script.py`

**Interfaces:**
- `train(args: argparse.Namespace) -> dict[str, Any]`
  - Supports `--early-stop-on-target` (or stops when `target_success_rate` achieved under max difficulty).
  - Returns updated summary metrics including convergence status and steps.

- [ ] **Step 1: Write unit test for convergence stopping and metric tracking**
  In `tests/test_train_script.py`:
  - Test early stopping condition when target success rate is achieved at target difficulty.

- [ ] **Step 2: Run test to confirm failure**
  Run: `.venv/bin/pytest tests/test_train_script.py -k "early_stop" -q`

- [ ] **Step 3: Update `train.py` with convergence check and early stopping**
  In `train.py`:
  - Add `--early-stop-on-target` argument (default True).
  - When evaluation success meets or exceeds `args.target_success_rate` and difficulty is 1.0 (or max), trigger clean early exit and record `converged: True`.

- [ ] **Step 4: Run tests to confirm pass**
  Run: `.venv/bin/pytest tests/test_train_script.py -q`

- [ ] **Step 5: Run linter and commit changes**
  Run: `.venv/bin/ruff check src/ tests/ train.py && git add train.py tests/test_train_script.py && git commit -m "feat(train): add convergence early stopping and benchmark policy export"`

---

### Task 3: Execute Training Run & Verify Convergence

**Files:**
- Run: `train.py` via command line with CUDA
- Artifacts: `checkpoints/best_model.pt`, `checkpoints/best_policy.onnx`, `checkpoints/metrics_summary.json`

**Interfaces:**
- Command: `.venv/bin/python train.py --total-frames 1000000 --device cuda --checkpoint-dir checkpoints`

- [ ] **Step 1: Launch training run on CUDA**
  Execute `train.py` with progress tracking.

- [ ] **Step 2: Monitor and verify evaluation milestones, checkpoint generation, and summary**
  Check `checkpoints/metrics_summary.json` and ensure test suite passes.

- [ ] **Step 3: Commit trained summary or state report**
  Run: `git status && git add checkpoints/metrics_summary.json 2>/dev/null || true`
