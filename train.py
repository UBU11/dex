import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.algo import (
    PolicyModule,
    SACTrainer,
    SquashedGaussianSACActor,
    TwinContinuousQCritic,
    create_her_replay_buffer,
    create_trajectory_collector,
    evaluate_policy,
    export_policy_onnx,
    load_checkpoint,
    save_checkpoint,
)
from src.envs import LeapCubeReorientEnv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="End-to-end training and evaluation script for LEAP Hand cube reorientation."
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--total-frames", type=int, default=1_000_000)
    parser.add_argument("--frames-per-batch", type=int, default=400)
    parser.add_argument("--storage-size", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--utd-ratio", type=float, default=1.0)
    parser.add_argument("--eval-interval", type=int, default=5_000)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--save-interval", type=int, default=10_000)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument(
        "--enable-adr", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--target-success-rate", type=float, default=0.85)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--slice-len", type=int, default=20)
    return parser


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(args)


def train(args: argparse.Namespace) -> dict[str, Any]:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    actor = SquashedGaussianSACActor(obs_dim=55, goal_dim=4, action_dim=16).to(device)
    critic = TwinContinuousQCritic(obs_dim=55, goal_dim=4, action_dim=16).to(device)
    trainer = SACTrainer(actor=actor, critic=critic, device=device)
    policy = PolicyModule(actor)

    eval_env = LeapCubeReorientEnv(
        max_steps=args.max_steps,
        enable_domain_randomization=args.enable_adr,
        seed=args.seed + 10_000,
    )
    eval_curriculum = getattr(eval_env, "curriculum_controller", None)

    slice_len = min(args.slice_len, args.max_steps, args.frames_per_batch)
    slice_len = max(2, slice_len)
    batch_size = max(slice_len, (args.batch_size // slice_len) * slice_len)
    replay_buffer = create_her_replay_buffer(
        storage_size=args.storage_size,
        slice_len=slice_len,
        batch_size=batch_size,
    )

    def train_env_fn() -> LeapCubeReorientEnv:
        return LeapCubeReorientEnv(
            max_steps=args.max_steps,
            enable_domain_randomization=args.enable_adr,
            seed=args.seed,
        )

    collector = create_trajectory_collector(
        env_fn=train_env_fn,
        policy=policy,
        replay_buffer=replay_buffer,
        frames_per_batch=args.frames_per_batch,
        total_frames=args.total_frames,
        enable_domain_randomization=args.enable_adr,
    )

    best_success_rate = 0.0
    start_step = 0
    if args.resume is not None:
        meta = load_checkpoint(
            checkpoint_path=args.resume,
            actor=actor,
            critic=critic,
            trainer=trainer,
            curriculum_controller=eval_curriculum,
            device=device,
        )
        best_success_rate = meta.get("best_success_rate", 0.0)
        start_step = meta.get("step", 0)
        if eval_curriculum is not None:
            train_ctrl = getattr(
                getattr(collector, "env", None), "curriculum_controller", None
            )
            if train_ctrl is not None:
                train_ctrl.set_difficulty(eval_curriculum.difficulty)

    total_frames = 0
    last_eval_frame = 0
    last_save_frame = 0
    saved_checkpoints: list[str] = []
    onnx_exported = False
    last_eval_metrics: dict[str, float] = {}
    current_step = start_step

    try:
        for batch in collector:
            batch_frames = (
                batch.numel()
                if hasattr(batch, "numel")
                else args.frames_per_batch
            )
            total_frames += batch_frames
            current_step = start_step + total_frames

            num_updates = (
                max(1, int(batch_frames * args.utd_ratio))
                if args.utd_ratio > 0
                else 0
            )
            current_buffer_len = len(replay_buffer)
            if current_buffer_len >= slice_len and num_updates > 0:
                target_batch_size = max(
                    slice_len, (args.batch_size // slice_len) * slice_len
                )
                effective_batch_size = (
                    target_batch_size
                    if current_buffer_len >= target_batch_size
                    else (current_buffer_len // slice_len) * slice_len
                )
                if effective_batch_size >= slice_len:
                    for _ in range(num_updates):
                        trainer.train_step_from_buffer(
                            replay_buffer, batch_size=effective_batch_size
                        )

            if (total_frames - last_eval_frame >= args.eval_interval) or (
                total_frames >= args.total_frames
            ):
                last_eval_frame = total_frames
                last_eval_metrics = evaluate_policy(
                    actor=actor,
                    env=eval_env,
                    num_episodes=args.eval_episodes,
                    max_steps=args.max_steps,
                    device=device,
                )

                if eval_curriculum is not None:
                    new_diff = eval_curriculum.update(
                        last_eval_metrics["success_rate"]
                    )
                    last_eval_metrics["difficulty"] = new_diff
                    train_ctrl = getattr(
                        getattr(collector, "env", None),
                        "curriculum_controller",
                        None,
                    )
                    if train_ctrl is not None:
                        train_ctrl.set_difficulty(new_diff)

                if last_eval_metrics["success_rate"] >= best_success_rate:
                    best_success_rate = last_eval_metrics["success_rate"]
                    best_ckpt = ckpt_dir / "best_model.pt"
                    save_checkpoint(
                        checkpoint_path=best_ckpt,
                        actor=actor,
                        critic=critic,
                        trainer=trainer,
                        curriculum_controller=eval_curriculum,
                        step=current_step,
                        best_success_rate=best_success_rate,
                        extra_info={"eval_metrics": last_eval_metrics},
                    )
                    saved_checkpoints.append(str(best_ckpt))

                    best_onnx = ckpt_dir / "best_policy.onnx"
                    export_policy_onnx(
                        onnx_path=best_onnx,
                        actor=actor,
                        obs_dim=55,
                        goal_dim=4,
                        device=device,
                    )
                    onnx_exported = True

                if last_eval_metrics["success_rate"] >= args.target_success_rate:
                    benchmark_onnx = ckpt_dir / "benchmark_policy.onnx"
                    export_policy_onnx(
                        onnx_path=benchmark_onnx,
                        actor=actor,
                        obs_dim=55,
                        goal_dim=4,
                        device=device,
                    )
                    onnx_exported = True

                print(
                    f"[Step {current_step:7d} / {start_step + args.total_frames:7d}] "
                    f"Success: {last_eval_metrics['success_rate'] * 100:5.1f}% | "
                    f"Consecutive: {last_eval_metrics['mean_consecutive_successes']:4.2f} | "
                    f"GeoErr: {last_eval_metrics['mean_geodesic_error']:5.3f} rad | "
                    f"Return: {last_eval_metrics['mean_return']:6.1f} | "
                    f"Diff: {last_eval_metrics.get('difficulty', 0.0):.2f} | "
                    f"Alpha: {trainer.alpha:.4f}"
                )

            if (total_frames - last_save_frame >= args.save_interval) or (
                total_frames >= args.total_frames
            ):
                last_save_frame = total_frames
                step_ckpt = ckpt_dir / f"checkpoint_step_{current_step}.pt"
                save_checkpoint(
                    checkpoint_path=step_ckpt,
                    actor=actor,
                    critic=critic,
                    trainer=trainer,
                    curriculum_controller=eval_curriculum,
                    step=current_step,
                    best_success_rate=best_success_rate,
                    extra_info={"eval_metrics": last_eval_metrics},
                )
                saved_checkpoints.append(str(step_ckpt))
    finally:
        collector.shutdown()
        eval_env.close()

    summary: dict[str, Any] = {
        "total_frames": total_frames,
        "final_step": current_step,
        "best_success_rate": float(best_success_rate),
        "last_eval_metrics": {
            k: float(v) for k, v in last_eval_metrics.items()
        },
        "saved_checkpoints": saved_checkpoints,
        "onnx_exported": onnx_exported,
        "checkpoint_dir": str(ckpt_dir),
    }

    summary_path = ckpt_dir / "metrics_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    return summary


if __name__ == "__main__":
    train(parse_args())
