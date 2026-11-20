import argparse
import json

import pytest

from train import build_parser, parse_args, train


def test_build_parser():
    parser = build_parser()
    assert isinstance(parser, argparse.ArgumentParser)


def test_parse_args_defaults():
    args = parse_args([])
    assert args.seed == 42
    assert args.total_frames == 1_000_000
    assert args.frames_per_batch == 400
    assert args.storage_size == 100_000
    assert args.batch_size == 256
    assert args.utd_ratio == pytest.approx(1.0)
    assert args.eval_interval == 5_000
    assert args.eval_episodes == 10
    assert args.save_interval == 10_000
    assert args.checkpoint_dir == "checkpoints"
    assert args.resume is None
    assert args.enable_adr is True
    assert args.target_success_rate == pytest.approx(0.85)
    assert args.device == "cpu"
    assert args.max_steps == 400


def test_parse_args_overrides():
    custom_argv = [
        "--seed",
        "123",
        "--total-frames",
        "5000",
        "--frames-per-batch",
        "50",
        "--storage-size",
        "2000",
        "--batch-size",
        "64",
        "--utd-ratio",
        "0.5",
        "--eval-interval",
        "250",
        "--eval-episodes",
        "3",
        "--save-interval",
        "500",
        "--checkpoint-dir",
        "/tmp/custom_ckpt",
        "--resume",
        "/tmp/custom_ckpt/model.pt",
        "--no-enable-adr",
        "--target-success-rate",
        "0.9",
        "--device",
        "cpu",
        "--max-steps",
        "50",
    ]
    args = parse_args(custom_argv)
    assert args.seed == 123
    assert args.total_frames == 5000
    assert args.frames_per_batch == 50
    assert args.storage_size == 2000
    assert args.batch_size == 64
    assert args.utd_ratio == pytest.approx(0.5)
    assert args.eval_interval == 250
    assert args.eval_episodes == 3
    assert args.save_interval == 500
    assert args.checkpoint_dir == "/tmp/custom_ckpt"
    assert args.resume == "/tmp/custom_ckpt/model.pt"
    assert args.enable_adr is False
    assert args.target_success_rate == pytest.approx(0.9)
    assert args.device == "cpu"
    assert args.max_steps == 50


def test_train_programmatic_short_run(tmp_path):
    ckpt_dir = tmp_path / "checkpoints"
    args = parse_args(
        [
            "--seed",
            "42",
            "--total-frames",
            "100",
            "--frames-per-batch",
            "50",
            "--storage-size",
            "500",
            "--batch-size",
            "30",
            "--utd-ratio",
            "0.1",
            "--eval-interval",
            "50",
            "--eval-episodes",
            "2",
            "--save-interval",
            "50",
            "--checkpoint-dir",
            str(ckpt_dir),
            "--max-steps",
            "10",
            "--device",
            "cpu",
        ]
    )

    summary = train(args)

    assert isinstance(summary, dict)
    assert summary["total_frames"] >= 100
    assert "best_success_rate" in summary
    assert "last_eval_metrics" in summary
    assert "saved_checkpoints" in summary
    assert "onnx_exported" in summary

    assert ckpt_dir.is_dir()
    ckpt_files = list(ckpt_dir.glob("checkpoint_step_*.pt"))
    assert len(ckpt_files) >= 1

    summary_file = ckpt_dir / "metrics_summary.json"
    assert summary_file.is_file()
    with open(summary_file) as f:
        data = json.load(f)
    assert data["total_frames"] == summary["total_frames"]
    assert "last_eval_metrics" in data


def test_train_resume(tmp_path):
    ckpt_dir = tmp_path / "checkpoints"
    args1 = parse_args(
        [
            "--seed",
            "42",
            "--total-frames",
            "50",
            "--frames-per-batch",
            "50",
            "--storage-size",
            "200",
            "--batch-size",
            "20",
            "--utd-ratio",
            "0.1",
            "--eval-interval",
            "50",
            "--eval-episodes",
            "1",
            "--save-interval",
            "50",
            "--checkpoint-dir",
            str(ckpt_dir),
            "--max-steps",
            "10",
        ]
    )
    summary1 = train(args1)
    saved_ckpts = summary1["saved_checkpoints"]
    assert len(saved_ckpts) > 0
    resume_ckpt = saved_ckpts[0]

    args2 = parse_args(
        [
            "--seed",
            "42",
            "--total-frames",
            "50",
            "--frames-per-batch",
            "50",
            "--storage-size",
            "200",
            "--batch-size",
            "20",
            "--utd-ratio",
            "0.1",
            "--eval-interval",
            "50",
            "--eval-episodes",
            "1",
            "--save-interval",
            "50",
            "--checkpoint-dir",
            str(ckpt_dir),
            "--resume",
            resume_ckpt,
            "--max-steps",
            "10",
        ]
    )
    summary2 = train(args2)
    assert summary2["total_frames"] >= 50
