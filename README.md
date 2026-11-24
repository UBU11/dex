# Dexterous In-Hand Cube Reorientation

Goal-conditioned reinforcement learning system for in-hand 3D cube reorientation using a 16-DoF LEAP Hand in MuJoCo. Built with TorchRL, PyTorch, and Soft Actor-Critic (SAC) combined with Hindsight Experience Replay (HER) and Automatic Domain Randomization (ADR).

![Simulation Environment](assets/simulation_environment.png)

---

## Overview

In-hand dexterous manipulation is characterized by contact-rich dynamics, unilateral force constraints, and irreversible absorbing failure states when an object slips past the fingertips. This repository implements an end-to-end framework where a 16-DoF anthropomorphic LEAP Hand rotates an unattached 7 cm cube to match arbitrary target orientations in $SO(3)$ using pure sparse binary rewards.

### System Characteristics

* **Robot Hardware:** 16-DoF LEAP Hand (four fingers, four actuated joints per digit).
* **Object:** 6-DoF free-joint cube (7 cm width, 0.1 kg nominal mass).
* **Physics Engine:** MuJoCo (`mujoco-menagerie` assembly via `MjSpec`).
* **Control Hierarchy:** Low-level joint PD controller at 200 Hz; high-level RL policy queries at 20 Hz (decimation factor of 10).
* **Algorithm:** Off-policy Soft Actor-Critic (SAC) with clipped twin critics and automatic entropy temperature tuning.
* **Goal Relabeling:** Trajectory-based Hindsight Experience Replay (HER) relabeling with future achieved goals ($p = 0.80$).
* **Curriculum and Robustness:** Warmup curriculum from 1D planar yaw rotation to full 3D $SO(3)$ reorientation; automatic domain randomization across physics parameters and initial pose distribution.
