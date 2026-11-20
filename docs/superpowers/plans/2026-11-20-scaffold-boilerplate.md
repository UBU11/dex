# Dexterous In-Hand Cube Reorientation Scaffold Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a robust, minimalist scaffold boilerplate for dexterous in-hand 7 cm cube reorientation with the 16-DoF LEAP Hand in MuJoCo, Soft Actor-Critic (SAC), Hindsight Experience Replay (HER), and Domain Randomization following `docs/AGENTS.md` and `docs/project.md`.

**Architecture:** Decoupled subsystems communicating through TorchRL `TensorDict` structures: (1) MuJoCo MjSpec assembly of tilted LEAP hand + 7 cm cube with low-level PD controller and domain randomization; (2) Quaternion $SO(3)$ math and Shoemake uniform sampling; (3) Stable squashed Gaussian SAC actor/twin critics with softplus Jacobian correction; (4) Vectorized Future HER trajectory relabeling.

**Tech Stack:** Python 3.12+, PyTorch 2.15, TorchRL 0.15, TensorDict 0.15, MuJoCo 4.0, MuJoCo Menagerie, pytest, ruff.

## Global Constraints

- Never use raw torque actions; policy outputs relative joint position targets $\Delta q \in [-0.15, 0.15]$ rad at 20 Hz, clipped to joint limits and tracked by PD controller at 200 Hz.
- Palm mounted with 20° downward tilt; cube is 7 cm free-joint box (half-size 0.035 m).
- Always normalize quaternions, take `torch.abs(dot)` for $SO(3)$ double-cover, and clamp strictly to `[0.0, 1.0 - 1e-7]` before `2.0 * torch.acos(dot)`.
- Never compute `torch.log(1 - action.pow(2))`; use softplus Jacobian identity `2.0 * (math.log(2.0) - u - F.softplus(-2.0 * u))`.
- Keep physics/simulator state inside `src/envs/`; policy modules in `src/algo/` only read observation keys from `TensorDict`—never access `mjData` directly.
- After mutating `mjModel` fields (`body_mass`, `body_inertia`, `geom_friction`, `dof_damping`) on reset, ALWAYS call `mujoco.mj_setConst(model, data)`.
- Under sparse rewards (0.0 on success, -1.0 otherwise), never terminate early on cube drop without terminal penalty.
- Ponytail mode: Build the minimum that works; no unrequested abstractions, no avoidable dependencies, no boilerplate. Mark intentional simplifications with `# ponytail:` comments.

---

### Task 1: Quaternion $SO(3)$ Math & Sparse Reward Utilities

**Files:**
- Create: `src/utils/__init__.py`
- Create: `src/utils/quaternion.py`
- Create: `tests/test_quaternion_math.py`

**Interfaces:**
- Produces:
  - `quat_geodesic_distance(q_achieved: torch.Tensor, q_desired: torch.Tensor) -> torch.Tensor`
  - `compute_sparse_her_reward(q_achieved: torch.Tensor, q_desired: torch.Tensor, cube_z: torch.Tensor, angle_tol: float = 0.15, drop_z_threshold: float = 0.04) -> tuple[torch.Tensor, torch.Tensor]`
  - `sample_uniform_quat(batch_size: int, device: torch.device | None = None) -> torch.Tensor`

- [ ] **Step 1: Write failing unit tests for quaternion utilities**
  In `tests/test_quaternion_math.py`, test:
  - Geodesic distance $d(q, -q) == 0.0$ (double cover).
  - Geodesic distance between identical quaternions is 0.0 with finite gradient (no NaN).
  - Orthogonal rotations yield expected radian angle ($\pi/2$).
  - Clamping prevents NaNs on edge cases (dot slightly $> 1$).
  - Shoemake uniform quats have unit norm and valid shape `(B, 4)`.
  - `compute_sparse_her_reward` returns `(0.0, True)` when angle $< 0.15$ and $z > 0.04$, and `(-1.0, False)` otherwise.

- [ ] **Step 2: Run test to confirm failure**
  Run: `pytest tests/test_quaternion_math.py -q`

- [ ] **Step 3: Implement `src/utils/quaternion.py`**
  Implement minimal, vectorized quaternion operations using PyTorch standard operations. Mark clean one-liners with `# ponytail:`.

- [ ] **Step 4: Run test to confirm pass**
  Run: `pytest tests/test_quaternion_math.py -q`

- [ ] **Step 5: Commit changes**
  Run: `git add src/utils/ tests/test_quaternion_math.py && git commit -m "feat(utils): implement quaternion SO(3) distance and sparse HER reward"`

---

### Task 2: Stable Squashed SAC Actor & Twin Critics

**Files:**
- Create: `src/algo/__init__.py`
- Create: `src/algo/actor.py`
- Create: `src/algo/critic.py`
- Create: `tests/test_actor_jacobian.py`

**Interfaces:**
- Consumes: None (pure PyTorch nn.Modules)
- Produces:
  - `SquashedGaussianSACActor(obs_dim: int, goal_dim: int, action_dim: int = 16, hidden_dim: int = 512)`
    - `forward(obs: torch.Tensor, desired_goal: torch.Tensor, deterministic: bool = False) -> tuple[torch.Tensor, torch.Tensor]`
  - `TwinContinuousQCritic(obs_dim: int, goal_dim: int, action_dim: int = 16, hidden_dim: int = 512)`
    - `forward(obs: torch.Tensor, desired_goal: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]`

- [ ] **Step 1: Write failing unit test for actor Jacobian & critic**
  In `tests/test_actor_jacobian.py`, test:
  - Finite forward and backward gradients at extreme pre-tanh activations $|u| = 10.0$ and $|u| = 20.0$ (no $-\infty$ or NaN).
  - Action outputs strictly bounded in $(-1, 1)$.
  - Deterministic evaluation mode returns $\tanh(\mu)$.
  - Twin critic outputs two distinct scalar Q predictions of shape `(B, 1)`.

- [ ] **Step 2: Run test to confirm failure**
  Run: `pytest tests/test_actor_jacobian.py -q`

- [ ] **Step 3: Implement `src/algo/actor.py` and `src/algo/critic.py`**
  Implement the squashed Gaussian policy using the softplus identity `2.0 * (math.log(2.0) - u - F.softplus(-2.0 * u))` for stable log-prob computation. Implement twin critics with LayerNorm and SiLU.

- [ ] **Step 4: Run test to confirm pass**
  Run: `pytest tests/test_actor_jacobian.py -q`

- [ ] **Step 5: Commit changes**
  Run: `git add src/algo/ tests/test_actor_jacobian.py && git commit -m "feat(algo): implement squashed SAC actor with stable Jacobian and twin critics"`

---

### Task 3: Vectorized Future Hindsight Experience Replay (HER)

**Files:**
- Create: `src/algo/her.py`
- Create: `tests/test_her_relabeling.py`

**Interfaces:**
- Consumes: `compute_sparse_her_reward` from `src.utils.quaternion`
- Produces:
  - `VectorizedFutureHER(relabel_prob: float = 0.8)`
    - `__call__(episode_batch: TensorDict) -> TensorDict`

- [ ] **Step 1: Write failing unit test for HER relabeling**
  In `tests/test_her_relabeling.py`, test:
  - Input `TensorDict` batch of episodes `[B, T]`.
  - Assert that when a transition is relabeled with its future achieved goal at `future_idx`, the recomputed reward at `next` step equals `0.0`.
  - Assert non-relabeled transitions preserve original `desired_goal` and `reward`.
  - Assert output has identical batch size and valid keys.

- [ ] **Step 2: Run test to confirm failure**
  Run: `pytest tests/test_her_relabeling.py -q`

- [ ] **Step 3: Implement `src/algo/her.py`**
  Vectorized future HER sampler sampling random future index $t_{\text{future}} \in [t, T-1]$, swapping desired goal with future achieved goal with probability 0.8, and recalculating sparse reward.

- [ ] **Step 4: Run test to confirm pass**
  Run: `pytest tests/test_her_relabeling.py -q`

- [ ] **Step 5: Commit changes**
  Run: `git add src/algo/her.py tests/test_her_relabeling.py && git commit -m "feat(algo): implement vectorized future HER relabeler"`

---

### Task 4: LEAP Hand + Cube MjSpec Assembly & PD Control (Phase 1 Sanity)

**Files:**
- Create: `src/envs/__init__.py`
- Create: `src/envs/spec.py`
- Create: `src/envs/pd_controller.py`
- Create: `tests/test_env_physics.py`

**Interfaces:**
- Produces:
  - `build_leap_cube_spec() -> mujoco.MjSpec` (assembles tilted LEAP hand and 7 cm cube)
  - `PDController(model: mujoco.MjModel, kp: float = 3.0, kd: float = 0.1)`
    - `step(data: mujoco.MjData, target_qpos: np.ndarray, n_substeps: int = 10)`

- [ ] **Step 1: Write failing test for spec assembly and physics sanity**
  In `tests/test_env_physics.py`, test:
  - `spec = build_leap_cube_spec()` compiles into valid `MjModel` with 16 actuators and cube free joint.
  - Palm body has 20° downward tilt.
  - Zero-action rollout (PD holding initial hand posture) over 50 control steps (500 physics substeps at 200 Hz):
    - No NaNs in `qpos` or `qvel`.
    - Cube settles into palm; does not fall through hand mesh.

- [ ] **Step 2: Run test to confirm failure**
  Run: `pytest tests/test_env_physics.py -q`

- [ ] **Step 3: Implement `src/envs/spec.py` and `src/envs/pd_controller.py`**
  Build clean MjSpec assembly from `mm.get('leap_hand').spec()`, add 7 cm box geom (size `[0.035, 0.035, 0.035]`), set downward palm tilt, and implement the internal PD controller clipping relative delta targets.

- [ ] **Step 4: Run test to confirm pass**
  Run: `pytest tests/test_env_physics.py -q`

- [ ] **Step 5: Commit changes**
  Run: `git add src/envs/ tests/test_env_physics.py && git commit -m "feat(envs): assemble LEAP hand + cube MjSpec and PD controller"`

---

### Task 5: Domain Randomization, Latency Buffer & TorchRL Environment

**Files:**
- Create: `src/envs/domain_randomization.py`
- Create: `src/envs/leap_cube_env.py`
- Create: `tests/test_leap_cube_env.py`

**Interfaces:**
- Consumes:
  - `build_leap_cube_spec`, `PDController` from `src.envs`
  - `quat_geodesic_distance`, `compute_sparse_her_reward`, `sample_uniform_quat` from `src.utils.quaternion`
- Produces:
  - `DomainRandomizer(model: mujoco.MjModel, data: mujoco.MjData)`
  - `LeapCubeReorientEnv(EnvBase)` with observation keys: `obs`, `desired_goal`, `achieved_goal`, `prev_action`, `cube_z`.

- [ ] **Step 1: Write failing unit test for domain randomizer and TorchRL environment**
  In `tests/test_leap_cube_env.py`, test:
  - `DomainRandomizer.randomize(...)` updates mass, friction, damping within specification intervals and calls `mj_setConst`.
  - `LeapCubeReorientEnv.reset()` returns valid `TensorDict` with expected keys and finite values.
  - `LeapCubeReorientEnv.step()` applies action delta, returns updated observations, sparse reward (-1.0 or 0.0), and done flag without alive-penalty suicide.
  - Consecutive goal chaining: when target orientation is reached for 5 consecutive steps, new desired goal is sampled in-place without resetting cube position.

- [ ] **Step 2: Run test to confirm failure**
  Run: `pytest tests/test_leap_cube_env.py -q`

- [ ] **Step 3: Implement `src/envs/domain_randomization.py` and `src/envs/leap_cube_env.py`**
  Implement minimal TorchRL `EnvBase` following TorchRL 0.15 semantics, latency FIFO queue (0-2 steps), and consecutive target chaining.

- [ ] **Step 4: Run test to confirm pass**
  Run: `pytest tests/test_leap_cube_env.py -q`

- [ ] **Step 5: Commit changes**
  Run: `git add src/envs/ tests/test_leap_cube_env.py && git commit -m "feat(envs): implement domain randomization and TorchRL LeapCubeReorientEnv"`

---

### Task 6: Baseline Integration, Linting & Verification Suite

**Files:**
- Modify: `main.py`
- Modify: `pyproject.toml`
- Test: Full test suite (`pytest tests/ -q` and `ruff check src/ tests/`)

**Interfaces:**
- Produces:
  - Runnable `main.py` demonstrating policy execution loop in `LeapCubeReorientEnv`.
  - Clean test suite meeting all requirements of `docs/AGENTS.md` Definition of Done.

- [ ] **Step 1: Update `main.py` with an executable rollout script**
  Instantiate `LeapCubeReorientEnv`, `SquashedGaussianSACActor`, and run a 50-step rollout printing observation keys, rewards, and geodesic angle error.

- [ ] **Step 2: Run ruff check and fix any styling issues**
  Run: `ruff check src/ tests/ main.py`

- [ ] **Step 3: Run the full test suite**
  Run: `pytest tests/ -v`

- [ ] **Step 4: Commit changes**
  Run: `git add main.py pyproject.toml && git commit -m "chore: integrate main rollout script and verify full test suite"`
