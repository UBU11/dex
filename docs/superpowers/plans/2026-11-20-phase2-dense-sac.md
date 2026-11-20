# Phase 2: Dense SAC Training Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement Phase 2 Dense SAC training pipeline with single-axis Z-yaw orientation mode, dense reward option, `SACLoss`, target entropy $\mathcal{H}_{\text{target}} = -16.0$, and log-$\alpha$ temperature optimization.

**Architecture:** Extend environment with single-axis Z-yaw target sampling and dense orientation reward ($-\theta - 0.01\|a\|^2$). Build `SACTrainer` and `SACLoss` supporting automatic $\alpha$ temperature tuning, twin critic evaluation, target network Polyak averaging, and stable loss optimization.

**Tech Stack:** Python 3.14 / 3.13, PyTorch 2.15+, TorchRL 0.15, TensorDict 0.15, MuJoCo 3.4+, pytest, ruff.

## Global Constraints

- Do not write unnecessary bloated comments in code files; only write comments if strictly necessary.
- Policy outputs relative joint position targets $\Delta q \in [-0.15, 0.15]$ rad at 20 Hz, tracked by PD controller at 200 Hz.
- Target entropy fixed to $\mathcal{H}_{\text{target}} = -16.0$ ($-\text{dim}(\mathcal{A})$).
- Single-axis Z-yaw targets sampled via $[w, x, y, z] = [\cos(\psi/2), 0, 0, \sin(\psi/2)]$ for $\psi \sim \mathcal{U}[-\pi, \pi]$.
- Dense reward: $r_t = -\theta(q_{\text{achieved}}, q_{\text{desired}}) - 0.01 \|a_t\|^2$.
- Maintain strict numeric stability in $SO(3)$ geodesic distance and SAC actor log-prob calculations.
- Verification commands: `.venv/bin/pytest tests/ -q` and `.venv/bin/ruff check src/ tests/`.

---

### Task 1: Single-Axis Z-Yaw Target Mode & Dense Reward Option

**Files:**
- Modify: `src/utils/quaternion.py`
- Modify: `src/envs/leap_cube_env.py`
- Modify: `tests/test_quaternion_math.py`
- Modify: `tests/test_leap_cube_env.py`

**Interfaces:**
- Produces:
  - `sample_yaw_quat(batch_size: int, device: torch.device | None = None) -> torch.Tensor`
  - `compute_dense_orientation_reward(q_achieved: torch.Tensor, q_desired: torch.Tensor, action: torch.Tensor, action_penalty_weight: float = 0.01) -> torch.Tensor`
  - `LeapCubeReorientEnv(..., target_mode: str = "so3", reward_mode: str = "sparse", action_penalty_weight: float = 0.01)`

- [ ] **Step 1: Write failing unit tests for yaw sampling and dense reward**
  In `tests/test_quaternion_math.py`:
  - Test `sample_yaw_quat` outputs unit quaternions with $x=0, y=0$ and shape `(B, 4)`.
  - Test `compute_dense_orientation_reward` computes $-\theta - 0.01\|a\|^2$ accurately.
  In `tests/test_leap_cube_env.py`:
  - Test `LeapCubeReorientEnv(target_mode="yaw", reward_mode="dense")` produces yaw targets and returns dense negative scalar rewards.

- [ ] **Step 2: Run test to confirm failure**
  Run: `.venv/bin/pytest tests/test_quaternion_math.py tests/test_leap_cube_env.py -q`

- [ ] **Step 3: Implement yaw sampling and dense reward**
  In `src/utils/quaternion.py`:
  - Implement `sample_yaw_quat(batch_size, device)`.
  - Implement `compute_dense_orientation_reward(q_achieved, q_desired, action, action_penalty_weight=0.01)`.
  In `src/envs/leap_cube_env.py`:
  - Accept `target_mode: str = "so3"` (options: `"so3"`, `"yaw"`).
  - Accept `reward_mode: str = "sparse"` (options: `"sparse"`, `"dense"`).
  - Use `sample_yaw_quat` when `target_mode == "yaw"`.
  - Compute dense reward using `compute_dense_orientation_reward` when `reward_mode == "dense"`.

- [ ] **Step 4: Run tests to confirm pass**
  Run: `.venv/bin/pytest tests/test_quaternion_math.py tests/test_leap_cube_env.py -q`

- [ ] **Step 5: Run linter and commit changes**
  Run: `.venv/bin/ruff check src/ tests/ && git add src/utils/quaternion.py src/envs/leap_cube_env.py tests/test_quaternion_math.py tests/test_leap_cube_env.py && git commit -m "feat(envs): add single-axis Z-yaw target mode and dense reward option"`

---

### Task 2: SAC Loss, Auto-$\alpha$ Optimization & SACTrainer Pipeline

**Files:**
- Create: `src/algo/sac_trainer.py`
- Modify: `src/algo/__init__.py`
- Create: `tests/test_sac_trainer.py`

**Interfaces:**
- Consumes:
  - `SquashedGaussianSACActor` from `src.algo.actor`
  - `TwinContinuousQCritic` from `src.algo.critic`
- Produces:
  - `SACLoss(actor: SquashedGaussianSACActor, critic: TwinContinuousQCritic, target_entropy: float = -16.0, alpha_init: float = 1.0, gamma: float = 0.99, tau: float = 0.005)`
  - `SACTrainer(actor: SquashedGaussianSACActor, critic: TwinContinuousQCritic, actor_lr: float = 3e-4, critic_lr: float = 3e-4, alpha_lr: float = 3e-4, target_entropy: float = -16.0, alpha_init: float = 1.0, gamma: float = 0.99, tau: float = 0.005, device: torch.device | None = None)`
    - `step(batch: TensorDict) -> dict[str, float]`
    - `update_target_network()`
    - `log_alpha: torch.nn.Parameter`
    - `alpha: float`
    - `evaluate_rollout(env, num_steps: int = 50) -> dict[str, float]`

- [ ] **Step 1: Write failing unit test for SACLoss and SACTrainer**
  In `tests/test_sac_trainer.py`:
  - Test `SACLoss` forward outputs `loss_actor`, `loss_critic`, `loss_alpha`, `alpha`, `entropy`.
  - Test `target_entropy` defaults to `-16.0`.
  - Test `SACTrainer.step(batch)` updates actor, critic, and `log_alpha` parameters (verifying gradients are finite and `log_alpha` changes).
  - Test `target_network` weights update with Polyak coefficient $\tau = 0.005$.
  - Test short training loop on synthetic/dense yaw rollout verifies $\alpha$ and critic losses remain finite and stable.

- [ ] **Step 2: Run test to confirm failure**
  Run: `.venv/bin/pytest tests/test_sac_trainer.py -q`

- [ ] **Step 3: Implement `src/algo/sac_trainer.py` and export in `src/algo/__init__.py`**
  - Implement `SACLoss` computing twin critic loss against Polyak target critic, actor loss with stable Jacobian log-prob, and temperature loss $-\log \alpha (\log \pi(a|s) + \mathcal{H}_{\text{target}})$.
  - Implement `SACTrainer` managing optimizers (Adam for actor, critic, log_alpha), Polyak target updates, and `step(batch)` optimization method.
  - Implement rollout evaluation helper to monitor $\alpha$ and geodesic error.

- [ ] **Step 4: Run tests to confirm pass**
  Run: `.venv/bin/pytest tests/test_sac_trainer.py -q`

- [ ] **Step 5: Run full test suite & linter, and commit changes**
  Run: `.venv/bin/pytest tests/ -q && .venv/bin/ruff check src/ tests/ && git add src/algo/ tests/test_sac_trainer.py && git commit -m "feat(algo): implement SACLoss, auto-alpha optimization and SACTrainer pipeline"`
