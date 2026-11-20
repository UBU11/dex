Dexterous in-hand cube reorientation is a classic rite of passage in robot learning because contact-rich manipulation destroys naive reinforcement learning pipelines. Unlike locomotion—where a robot can stumble and recover on the next stride—an in-hand manipulation policy faces an irreversible absorbing state the millisecond the cube slips past the fingertips.

As your senior engineer on this project, I will break down the ecosystem updates, the architectural mental model, the mathematical traps that silently kill training runs, and a concrete implementation blueprint using **TorchRL 0.15** and **MuJoCo**.

---

## 1. The Stack & Research Landscape

Before writing a line of code, align your architecture with the latest tooling releases so you do not waste weeks fighting deprecated bindings or CPU-GPU bottlenecks:

* **TorchRL 0.15:** This release introduced native **MuJoCo Menagerie** integration (`mujoco-menagerie==2026.9.0`), low-level controller composition, and `replay_write_mode="trajectory"` inside data collectors. That trajectory write mode is essential for Hindsight Experience Replay (HER) because HER requires complete episode boundaries to relabel future achieved goals accurately. TorchRL 0.15 also supports direct `TensorDict` policy export to ONNX for hardware deployment.
* **MuJoCo Playground 1.0 & MuJoCo Warp:** DeepMind upgraded MuJoCo Playground from its earlier JAX-only (MJX) preview into a full PyTorch and **MuJoCo Warp** GPU-accelerated toolchain. While MJX in JAX had 1–3 minute JIT compilation bottlenecks, MuJoCo Warp allows zero-copy GPU tensor sharing directly with PyTorch across thousands of parallel environments (`nworld`).
* **Hardware Target Selection (LEAP Hand vs. Shadow Hand):** Use the **16-DoF LEAP Hand** rather than the 24-DoF Shadow Hand. In current manipulation benchmarks, LEAP Hand is the standard open-source baseline: its direct-drive/servo kinematics avoid the fragile tendon coupling of the Shadow Hand, its 16-DoF action space converges significantly faster under off-policy SAC, and DeepMind's reference in-hand reorientation tasks in MuJoCo Playground are specifically benchmarked on a 7 cm cube inside the LEAP Hand with a 20-degree downward palm tilt.

---

## 2. System Architecture & The "Big Picture"

Think of this system as four decoupled subsystems communicating via **TorchRL `TensorDict`s**:

```
+-------------------------------------------------------------------------+
| 1. Physics & Domain Randomization Layer (MuJoCo / MJX / Warp)           |
|    - 16-DoF LEAP Hand + 6-DoF Free-Joint Cube                           |
|    - Randomized: Mass, Sliding/Torsional Friction, Joint Damping        |
|    - Action Interface: Relative Joint Position Targets (PD Control)     |
+------------------------------------+------------------------------------+
                                     |
              Delayed Observation    |    Action u_t in [-1, 1]^16
              Queue (Latency Buffer) v
+-------------------------------------------------------------------------+
| 2. Goal-Conditioned TensorDict Rollout Collector                        |
|    - State s_t: [qpos_hand, qvel_hand, cube_pos, cube_quat, prev_act]   |
|    - Achieved Goal ag_t: Current Cube Quaternion (4D)                   |
|    - Desired Goal dg_t: Target SO(3) Quaternion (4D)                    |
+------------------------------------+------------------------------------+
                                     |
              Full Episode Write     |    (replay_write_mode="trajectory")
                                     v
+-------------------------------------------------------------------------+
| 3. Hindsight Experience Replay (HER) Buffer                             |
|    - Samples batch of transitions (s_t, a_t, s_{t+1}, dg)               |
|    - 80% probability: Replaces dg with ag_{t+k} from future step        |
|    - Recomputes sparse reward r' in-place on GPU                        |
+------------------------------------+------------------------------------+
                                     |
              Relabeled Batch        v
+-------------------------------------------------------------------------+
| 4. Soft Actor-Critic (SAC) Learner                                      |
|    - Squashed TanhNormal Actor (with exact Jacobian log-prob fix)       |
|    - Twin Clipped Critics Q1, Q2 (preventing Q-overestimation on edges) |
|    - Automatic Alpha Temperature Tuning (Target Entropy = -16.0)        |
+-------------------------------------------------------------------------+

```

### Critical Engineering Decisions

1. **Do Not Use Direct Torque Control:** Raw torque control in a 16-DoF hand causes violent finger chattering that launches the cube out of the palm on frame 2. Use **relative joint position targets** ($\Delta q$) fed into a low-level PD controller running at 200–500 Hz, while your RL policy queries at 20–50 Hz (decimation of 10).
2. **Why SAC + HER Instead of PPO?** On-policy PPO works well if you run 8,192 environments with heavily engineered dense rewards. However, dense orientation rewards often teach the robot "reward hacking" behaviors (e.g., trapping the cube between two fingers and vibrating to farm supervivencia bonuses). **SAC + HER** allows you to train on **purely sparse binary rewards** ($0$ if within tolerance, $-1$ otherwise). When the hand accidentally rotates the cube to orientation $q_{\text{accidental}}$ before dropping it, HER rewrites history and tells the critic: *"Pretend $q_{\text{accidental}}$ was the target all along—that trajectory was a success."*

---

## 3. Worked Mathematical & Code Examples

### A. $SO(3)$ Quaternion Geodesic Distance & Sparse HER Reward

Never use Euclidean distance ($L_2$ norm) on quaternions because $q$ and $-q$ represent the exact same 3D rotation in $SO(3)$. Your reward function and success metric must compute the geodesic angular distance $\theta \in [0, \pi]$:

$$\theta(q_{\text{achieved}}, q_{\text{desired}}) = 2 \arccos\left( \min\left(1.0, \, \left\vert{} q_{\text{achieved}} \cdot q_{\text{desired}} \right\vert{} \right) \right)$$

Notice the clamping to $1.0$: floating-point inaccuracy in dot products frequently yields $1.0000001$, which causes `torch.acos` to emit `NaN` and corrupt your entire Q-network in a single backward pass.

```python
import torch

def quat_geodesic_distance(q_achieved: torch.Tensor, q_desired: torch.Tensor) -> torch.Tensor:
    """
    Computes geodesic angular distance (in radians) between unit quaternions.
    Shape: (..., 4) -> (..., 1)
    """
    # Normalize to guard against simulator drift
    q_a = torch.nn.functional.normalize(q_achieved, p=2, dim=-1)
    q_d = torch.nn.functional.normalize(q_desired, p=2, dim=-1)
    
    # Inner product: |q_a . q_d| accounts for double-cover (q == -q)
    dot = torch.abs(torch.sum(q_a * q_d, dim=-1, keepdim=True))
    
    # Clamp strictly below 1.0 - eps to prevent NaN gradients in acos
    dot_clamped = torch.clamp(dot, min=0.0, max=1.0 - 1e-7)
    return 2.0 * torch.acos(dot_clamped)

def compute_sparse_her_reward(
    q_achieved: torch.Tensor, 
    q_desired: torch.Tensor, 
    cube_z: torch.Tensor,
    angle_tol: float = 0.15,  # ~8.6 degrees tolerance
    drop_z_threshold: float = 0.04
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns (reward, is_success) for HER relabeling and env stepping.
    """
    ang_err = quat_geodesic_distance(q_achieved, q_desired)
    not_dropped = (cube_z > drop_z_threshold)
    is_success = (ang_err < angle_tol) & not_dropped
    
    # Sparse reward: 0.0 on goal achievement, -1.0 everywhere else
    reward = torch.where(is_success, 0.0, -1.0)
    return reward, is_success

```

---

### B. Squashed TanhNormal Policy & Exact Log-Probability Jacobian

In SAC, the actor outputs unbounded Gaussian parameters $(\mu, \log \sigma)$ for all $D=16$ joints. We sample $u \sim \mathcal{N}(\mu, \sigma^2)$ and squash it via $a = \tanh(u) \in (-1, 1)^D$.

By the change-of-variables theorem, the log-probability of action $a$ requires subtracting the log-determinant of the Jacobian of $\tanh(u)$:

$$\log \pi(a \mid s) = \sum_{i=1}^{D} \left[ \log \mathcal{N}(u_i \mid \mu_i, \sigma_i^2) - \log(1 - \tanh^2(u_i)) \right]$$

**The Numerical Trap:** When policy certainty increases, $\vert{}u_i\vert{} > 4$, making $\tanh^2(u_i) \approx 1.0$ in FP32, so $\log(1 - 1) = -\infty$. Instead, use the algebraically equivalent softplus identity:

$$\log(1 - \tanh^2(u_i)) = 2 \left( \log(2) - u_i - \text{softplus}(-2 u_i) \right)$$

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict.nn import TensorDictModule

LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0

class SquashedGaussianSACActor(nn.Module):
    def __init__(self, obs_dim: int, goal_dim: int, action_dim: int = 16, hidden_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + goal_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        self.mu_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, obs: torch.Tensor, desired_goal: torch.Tensor, deterministic: bool = False):
        x = torch.cat([obs, desired_goal], dim=-1)
        h = self.net(x)
        mu = self.mu_head(h)
        log_std = torch.clamp(self.log_std_head(h), LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        if deterministic:
            u = mu
        else:
            # Reparameterization trick: u = mu + std * eps
            eps = torch.randn_like(mu)
            u = mu + std * eps

        action = torch.tanh(u)

        # Unsquashed Gaussian log-prob
        log_prob_u = -0.5 * (((u - mu) / (std + 1e-8)) ** 2 + 2.0 * log_std + math.log(2.0 * math.pi))
        
        # Numerically stable Jacobian correction for Tanh squashing
        jacobian_correction = 2.0 * (math.log(2.0) - u - F.softplus(-2.0 * u))
        
        # Sum over the 16 action dimensions
        log_prob = (log_prob_u - jacobian_correction).sum(dim=-1, keepdim=True)
        return action, log_prob

```

*(Note: In TorchRL 0.15, you can also wrap your network in `ProbabilisticActor` using `TanhNormal` with `safe_tanh=True`, which uses this exact identity under the hood, but implementing and verifying it yourself ensures you understand the entropy scale.)*

---

### C. Hindsight Experience Replay (HER) Relabeling in TorchRL

To make HER work with a **consecutive target-orientation** task, you must separate **Episode Termination** (dropping the cube) from **Goal Achievement** (hitting the orientation). Here is how a vectorized HER relabeler operates on trajectory batches in PyTorch:

```python
from tensordict import TensorDict

class VectorizedFutureHER:
    """
    Implements the 'future' HER strategy (k=4 -> 80% relabeling probability)
    on episodic TensorDict batches collected with replay_write_mode='trajectory'.
    """
    def __init__(self, relabel_prob: float = 0.8):
        self.relabel_prob = relabel_prob

    def __call__(self, episode_batch: TensorDict) -> TensorDict:
        # Assume episode_batch has shape [Batch_Episodes, Max_Steps]
        B, T = episode_batch.shape
        device = episode_batch.device

        # Sample random time indices t for training transitions
        t_idx = torch.randint(0, T - 1, (B,), device=device)
        
        # Sample future offsets delta_t in [t_idx, T - 1]
        rand_floats = torch.rand(B, device=device)
        future_idx = t_idx + (rand_floats * (T - 1 - t_idx)).long()

        # Extract the transition at t_idx
        batch_arange = torch.arange(B, device=device)
        transition = episode_batch[batch_arange, t_idx].clone()

        # Extract the achieved goal at future_idx
        future_achieved_quat = episode_batch["next", "achieved_goal"][batch_arange, future_idx]

        # Decide which transitions in the batch get relabeled (80%)
        relabel_mask = (torch.rand(B, 1, device=device) < self.relabel_prob)
        
        new_desired_goal = torch.where(
            relabel_mask, 
            future_achieved_quat, 
            transition["desired_goal"]
        )
        transition["desired_goal"] = new_desired_goal
        transition["next", "desired_goal"] = new_desired_goal

        # Recompute sparse reward under the new desired goal
        next_achieved_quat = transition["next", "achieved_goal"]
        next_cube_z = transition["next", "cube_z"]
        
        new_reward, _ = compute_sparse_her_reward(
            next_achieved_quat, new_desired_goal, next_cube_z
        )
        transition["next", "reward"] = new_reward
        return transition

```

---

### D. Domain Randomization & Observation Latency Buffer

Contact-rich policies overfit rapidly to the exact friction coefficient and mass of the MuJoCo XML model. If you evaluate under randomized physics without training under them, your policy will drop the cube immediately.

Implement two mechanisms in your environment wrapper:

1. **Physics Randomization at Reset (`env.reset()`):** Mutate `mjModel` fields directly (or via `vmap` model overrides in MJX/Warp) before calling `mj_forward`.
2. **Observation Latency FIFO Buffer:** Physical cameras (like the Intel RealSense D415 used on LEAP Hand setups) introduce 1–3 frames of pose estimation lag.

```python
import numpy as np
import mujoco
from collections import deque

class DomainRandomizedHandWrapper:
    def __init__(self, mj_model: mujoco.MjModel, mj_data: mujoco.MjData, cube_body_id: int, cube_geom_id: int):
        self.model = mj_model
        self.data = mj_data
        self.cube_body_id = cube_body_id
        self.cube_geom_id = cube_geom_id
        
        # Store nominal physical parameters
        self.nominal_mass = self.model.body_mass[cube_body_id].copy()
        self.nominal_friction = self.model.geom_friction[cube_geom_id].copy()
        self.nominal_damping = self.model.dof_damping[:16].copy()
        
        # Observation delay queue (simulating 0 to 3 control steps of latency)
        self.obs_queue = deque(maxlen=4)
        self.current_latency_steps = 1

    def randomize_domain(self, rng: np.random.Generator):
        # 1. Cube Mass: [0.5x, 1.5x] of nominal (~70g to 210g)
        mass_scale = rng.uniform(0.5, 1.5)
        self.model.body_mass[self.cube_body_id] = self.nominal_mass * mass_scale
        self.model.body_inertia[self.cube_body_id] *= mass_scale

        # 2. Cube Surface Friction (sliding, torsional, rolling)
        friction_scale = rng.uniform(0.6, 1.4, size=3)
        self.model.geom_friction[self.cube_geom_id] = self.nominal_friction * friction_scale

        # 3. Hand Joint Damping: [0.7x, 1.3x]
        damping_scale = rng.uniform(0.7, 1.3, size=16)
        self.model.dof_damping[:16] = self.nominal_damping * damping_scale

        # 4. Randomize observation latency for the episode (0, 1, or 2 steps)
        self.current_latency_steps = int(rng.integers(0, 3))

        # Recompute derived constants in MuJoCo
        mujoco.mj_setConst(self.model, self.data)

    def get_delayed_observation(self, raw_obs: np.ndarray) -> np.ndarray:
        # Add slight Gaussian sensor noise to cube pose and joint velocities
        noisy_obs = raw_obs + np.random.normal(0, 0.005, size=raw_obs.shape)
        self.obs_queue.append(noisy_obs)
        
        # Return delayed frame (if queue just reset, pad with oldest available)
        idx = max(0, len(self.obs_queue) - 1 - self.current_latency_steps)
        return self.obs_queue[idx]

```

---

## 4. Senior Engineer's Execution Roadmap (How to Approach This)

Do not build the entire stack at once. If you turn on Sparse Rewards, HER, SAC, and full Domain Randomization on Day 1, and the reward curve stays flat at `-1.0`, you will have no idea which component is broken. Follow this strict 4-phase validation schedule:

1. **Phase 1: Environment & PD Controller Sanity Check (Days 1–3):** Verify kinematics before touching RL.
Load the LEAP Hand from `mujoco_menagerie` (`mm.get('leap_hand').spec()`) and attach a free-joint 7 cm cube slightly above the palm. Configure your action space as **relative joint targets** ($\Delta q \in [-0.15, 0.15]$ rad) clipped to joint limits. Write a zero-action rollout script where the hand holds its default pose. Verify that the cube settles naturally into the palm without penetrating finger meshes or exploding due to overly stiff `solref`/`solimp` contact parameters.


2. **Phase 2: Single-Axis Planar Rotation with Dense SAC (Days 4–6):** Verify the SAC Learner and TanhNormal Jacobian.
Temporarily disable HER and Domain Randomization. Restrict target orientations to yaw-only rotations around the Z-axis, and use a dense reward: $r_t = -\theta(q_{\text{achieved}}, q_{\text{desired}}) - 0.01 \Vert{}a_t\Vert{}^2$. Train vanilla SAC with automatic $\alpha$ tuning ($\mathcal{H}_{\text{target}} = -\text{dim}(\mathcal{A}) = -16.0$). Confirm that $\alpha$ starts around $1.0$ and smoothly decays to $0.05\text{–}0.2$, and that Twin Critic $Q$-values do not diverge to $+\infty$.


3. **Phase 3: Full SO(3) Sparse Reward + HER Integration (Days 7–11):** Switch to sparse rewards and trajectory relabeling.
Switch to arbitrary $SO(3)$ target quaternions (sampled uniformly via Shoemake's method) and replace the dense reward with the binary sparse reward ($0$ / $-1$). Enable TorchRL's `replay_write_mode="trajectory"` and your `VectorizedFutureHER` sampler. **Crucial metric to watch:** Track the *relabeled* positive reward ratio in your training batches—it should sit between 15% and 35% early in training even when true evaluation success is 0%.


4. **Phase 4: Automatic Domain Randomization (ADR) & Consecutive Goals (Days 12–15):** Push to.
85% consecutive target success">
Instead of starting with maximum randomization (which prevents initial exploration), ramp randomization linearly once your policy hits 50% success on nominal physics. Implement **Consecutive Goal Chaining**: when `ang_err` is less than `0.15` rad for 5 consecutive control steps, do *not* reset the hand or cube—sample a new $q_{\text{desired}}$ in-place, increment `consecutive_successes`, and continue the episode up to $T=400$ steps.


---

## 5. Common Failure Modes & Debugging Checklist

| Symptom During Training | Root Cause | Concrete Fix |
| --- | --- | --- |
| **Critics $Q_1, Q_2$ explode to `NaN` around step 50k** | Unclamped dot product in `torch.acos` or un-softplused $\log(1 - \tanh^2(u))$ when $\vert{}u\vert{} > 5$. | Clamp quaternion inner product to `1.0 - 1e-7` and use `2*(log(2) - u - softplus(-2u))`. |
| **Hand violently throws the cube on step 1 to end the episode** | Alive-penalty suicide: since every step gives $-1$ reward, terminating early maximizes cumulative return. | Never terminate the episode without penalty on drop! Either fix episode length to $T$ (continuing to give $-1$ after drop) or apply a terminal drop penalty equal to $-(T - t)$. |
| **Policy rotates cube once, then wedges it between thumb and index finger** | Known LEAP Hand hardware/sim geometry trap + policy lacks past action context to overcome static friction. | Include the **previous action** $a_{t-1}$ and **fingertip contact binary flags** in the observation vector $s_t$. |
| **HER learns single reorientations (70%), but fails on consecutive targets (<30%)** | Initial state distribution mismatch: episode resets always start with the cube centered in the palm, whereas goal 2 starts near the edge. | During `env.reset()`, add random XY offsets ($\pm 1.5$ cm) and random initial orientations to the cube before dropping it into the palm. |
