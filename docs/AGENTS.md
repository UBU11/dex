
# AGENTS.md — Dexterous In-Hand Cube Reorientation (SAC + HER)

## 1. Project Goal & Deliverable
Train a 16-DoF LEAP Hand in MuJoCo to reorient a 7 cm cube to arbitrary $SO(3)$ target orientations using Soft Actor-Critic (SAC) with Hindsight Experience Replay (HER) and Domain Randomization.
- **Target Metric:** >85% consecutive target-orientation success under randomized cube mass, surface friction, joint damping, and observation latency.

## 2. Canonical Stack (November 2026 Pins)
Use `uv` and Python 3.12 or 3.13. Do not substitute older major versions.
- **PyTorch:** `torch>=2.11,<2.14` (required for compatibility between TorchRL 0.15 and MuJoCo Playground 1.0)
- **TorchRL:** `torchrl==0.15.0` and `tensordict>=0.15.2,<0.16.0` (must match release line)
- **Physics:** `mujoco>=3.4`, `mujoco-menagerie==2026.9.0`
- **Optional GPU Sim:** `playground>=1.0` (MuJoCo Playground 1.0 with MuJoCo Warp / `warp-lang>=1.11`, NOT JAX/MJX)

### Setup & Verification Commands
Run these verbatim before claiming any task is complete:
```bash
uv venv --python 3.13 && source .venv/bin/activate
uv pip install "torchrl==0.15.0" "tensordict>=0.15.2,<0.16.0" "mujoco>=3.4" "mujoco-menagerie==2026.9.0" pytest ruff
ruff check src/ tests/
pytest tests/ -q

```

* On macOS, interactive MuJoCo viewer scripts must use `mjpython`, not `python`.
* On headless Linux, set `MUJOCO_GL=egl` (or `osmesa`) before running render tests.

## 3. Repository Boundaries

* `src/envs/`: MuJoCo Menagerie LEAP Hand + cube `MjSpec` assembly, PD control, domain randomization, latency FIFO, and TorchRL `EnvBase` / `MenagerieEnv` wrappers.
* `src/algo/`: SAC actor/critics, automatic $\alpha$ tuning, `SACLoss` wiring, and vectorized HER relabeling.
* `src/utils/`: Quaternion $SO(3)$ math and Shoemake uniform rotation sampling.
* `tests/`: Deterministic unit tests for physics sanity, quaternion math, Jacobian log-prob stability, and HER relabeling.
* **Rule:** Keep physics/simulator state inside `src/envs/`. Policy modules in `src/algo/` must only read observation keys from `TensorDict`—never access `mjData` directly.

## 4. Strict API Rules (Prevent Deprecated Syntax)

* **DO NOT** use `gym` (use `torchrl.envs` natively; `gymnasium` only if wrapping external tools).
* **DO NOT** use JAX/MJX or Playground 0.2 (`rss2025`) APIs for new code. Playground 1.0 uses PyTorch + MuJoCo Warp natively.
* **DO NOT** hardcode raw GitHub XML URLs or clone `mujoco_menagerie` manually. Load via:
```python
import mujoco_menagerie as mm
spec = mm.get("leap_hand").spec()  # editable mujoco.MjSpec

```


Or use TorchRL 0.15's native `MenagerieEnv` with custom reward transforms.
* **TorchRL 0.15 Breaking Change:** `EnvBase.reset()` and `EnvBase.rollout()` now default to `set_state=False`. Explicitly pass `set_state=True` if resetting to a specific state dict.
* **Action History:** Use TorchRL 0.15's built-in `LastAction` transform instead of writing custom action-history wrappers.
* **Episodic Buffer for HER:** Collect full episodes using `Collector(..., replay_write_mode="trajectory")` and store in `TensorDictReplayBuffer(storage=LazyTensorStorage(capacity, ndim=2), sampler=SliceSampler(output_layout="batch_time"))` (or `Sequence` sample units).
* **Deployment Export:** Use TorchRL 0.15's `export_onnx_policy` for policy export.

## 5. Non-Negotiable Physics & RL Invariants

Violating any of these causes silent divergence or NaNs. Enforce via unit tests:

1. **Control Interface:** Policy outputs relative joint position deltas $\Delta q \in [-0.15, 0.15]$ rad at 20 Hz, clipped to joint limits and tracked by an internal PD controller at 200–500 Hz. Never use raw torque actions.
2. **LEAP Hand Geometry:** Mount the LEAP Hand with a 20° downward palm tilt and a 7 cm free-joint cube. Include `LastAction` ($a_{t-1}$) in observations.
3. **Quaternion Geodesic Error (No L2):** Always normalize quaternions, take `torch.abs(dot)` for $SO(3)$ double-cover ($q \equiv -q$), and clamp strictly below 1.0 before `acos`:
```python
dot = torch.clamp(torch.abs((q_a * q_d).sum(dim=-1, keepdim=True)), 0.0, 1.0 - 1e-7)
angle = 2.0 * torch.acos(dot)

```


4. **TanhNormal Log-Prob Stability:** Never compute `torch.log(1 - action.pow(2))`. Either use TorchRL's `TanhNormal(..., safe_tanh=True)` or the softplus identity:
```python
jac = 2.0 * (math.log(2.0) - u - F.softplus(-2.0 * u))

```


5. **No Alive-Penalty Suicide:** Under sparse rewards (`0.0` on success, `-1.0` otherwise), NEVER terminate an episode early on cube drop without penalty. Either run to fixed horizon $T$ (emitting `-1.0` after drop) or apply a terminal drop penalty of `-(T - t)`.
6. **Domain Randomization Sync:** After mutating `mjModel` fields (`body_mass`, `body_inertia`, `geom_friction`, `dof_damping`) on reset, ALWAYS call `mujoco.mj_setConst(model, data)`.
7. **Consecutive Goal Chaining:** When `angle < 0.15` rad for 5 consecutive control steps (and cube is not dropped), sample a new uniform $SO(3)$ target quaternion in-place without resetting the hand or cube.

## 6. Phased Development Policy

Implement and verify strictly in order—do not enable Phase $N+1$ until Phase $N$ passes tests:

1. **Phase 1 (Physics Sanity):** Zero-action rollout settles the 7 cm cube into the 20°-tilted LEAP palm with zero NaNs and no mesh explosion.
2. **Phase 2 (Dense SAC):** Single-axis (Z-yaw) dense rotation with SAC + auto-$\alpha$ ($\mathcal{H}_{\text{target}} = -16.0$). Verify $\alpha$ decays smoothly and Twin Critics stay finite.
3. **Phase 3 (Sparse + HER):** Full $SO(3)$ sparse reward (`0/-1`) + `future` HER (80% relabeling). Assert relabeled positive-reward ratio in training batches is 15%–35% early in training.
4. **Phase 4 (ADR + Consecutive Goals):** Ramp mass (`0.5x–1.5x`), friction (`0.6x–1.4x`), damping (`0.7x–1.3x`), and observation latency (`0–2` steps) once nominal success exceeds 50%.

## 7. Definition of Done (Agent Checklist)

Before finishing any PR or task:

* Run `ruff check` and `pytest tests/ -q` and paste exact results.
* Verify unit tests cover: `quat_geodesic_distance(q, -q) == 0`, finite gradients on `TanhNormal` at $|u| = 10.0$, and HER relabeling producing `reward == 0.0` at `future_idx`.
* Report any command or GPU check that could not be executed in the current environment.

```
