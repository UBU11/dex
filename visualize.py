"""Interactive MuJoCo Viewer for Dexterous LEAP Hand Cube Reorientation."""

import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import torch

from src.algo.actor import SquashedGaussianSACActor
from src.envs.leap_cube_env import LeapCubeReorientEnv


def parse_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Launch MuJoCo interactive viewer for LEAP Hand cube simulation."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/best_model.pt",
        help="Path to trained policy checkpoint (.pt). Pass 'none' to run without policy.",
    )
    parser.add_argument(
        "--target-mode",
        type=str,
        default="so3",
        choices=["so3", "yaw"],
        help="Target orientation mode ('so3' or 'yaw').",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=400,
        help="Maximum steps per episode before auto-reset.",
    )
    parser.add_argument(
        "--export-xml",
        type=str,
        default=None,
        help="Optional path to export MJCF XML file for standalone MuJoCo viewer.",
    )
    return parser


def main() -> None:
    args = parse_args().parse_args()

    # Create environment
    env = LeapCubeReorientEnv(
        max_steps=args.max_steps,
        target_mode=args.target_mode,
        reward_mode="sparse",
        randomize_initial_orientation=True,
        xy_offset_range=0.015,
        enable_domain_randomization=False,
    )

    # Optional XML export
    if args.export_xml:
        xml_path = Path(args.export_xml)
        xml_path.parent.mkdir(parents=True, exist_ok=True)
        # Re-export spec to standalone XML
        from src.envs.spec import build_leap_cube_spec

        spec = build_leap_cube_spec()
        xml_str = spec.to_xml()
        with open(xml_path, "w") as f:
            f.write(xml_str)
        print(f"Exported standalone MJCF model to: {xml_path}")

    # Load policy if checkpoint exists
    actor = None
    ckpt_path = Path(args.checkpoint)
    if args.checkpoint.lower() != "none" and ckpt_path.is_file():
        print(f"Loading policy from checkpoint: {ckpt_path}")
        actor = SquashedGaussianSACActor(obs_dim=55, goal_dim=4, action_dim=16)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        actor.load_state_dict(ckpt["actor_state_dict"])
        actor.eval()
        print("Policy successfully loaded.")
    else:
        print("Running in passive/zero-action mode (no policy loaded).")

    print("\n" + "=" * 60)
    print("Launching MuJoCo Viewer...")
    print("Controls:")
    print("  - Space: Pause / Unpause")
    print("  - Backspace: Reset simulation")
    print("  - Right Mouse Drag: Orbit camera")
    print("  - Left Mouse Drag: Perturb / apply forces to bodies")
    print("  - Scroll: Zoom")
    print("=" * 60 + "\n")

    td = env.reset()

    # Launch passive viewer
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        step_dt = 1.0 / 20.0  # 20 Hz policy rate

        while viewer.is_running():
            step_start = time.time()

            if actor is not None:
                with torch.no_grad():
                    obs = td["obs"].unsqueeze(0)
                    goal = td["desired_goal"].unsqueeze(0)
                    action, _ = actor(obs, goal, deterministic=True)
                    td["action"] = action.squeeze(0)
            else:
                td["action"] = torch.zeros(16, dtype=torch.float32)

            next_td = env.step(td)

            # Sync viewer state with simulation
            viewer.sync()

            if bool(next_td["next", "done"].any().item()):
                td = env.reset()
            else:
                from torchrl.envs.utils import step_mdp

                td = step_mdp(next_td)

            # Maintain real-time 20 Hz stepping
            elapsed = time.time() - step_start
            time_until_next_step = step_dt - elapsed
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


if __name__ == "__main__":
    main()
