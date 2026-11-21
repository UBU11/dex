import argparse
import json

import pytest
import torch
from tensordict import TensorDict

from src.algo import PolicyModule, SquashedGaussianSACActor
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
    assert args.early_stop_on_target is True
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
        "--no-early-stop-on-target",
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
    assert args.early_stop_on_target is False
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
    assert summary["converged"] is False
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_policy_module_device_mismatch():
    actor = SquashedGaussianSACActor(obs_dim=55, goal_dim=4, action_dim=16).to("cuda")
    policy = PolicyModule(actor)
    td = TensorDict(
        {"obs": torch.randn(2, 55), "desired_goal": torch.randn(2, 4)},
        batch_size=[2],
        device="cpu",
    )
    out_td = policy(td)
    assert "action" in out_td
    assert out_td["action"].device == torch.device("cpu")
    assert out_td["action"].shape == (2, 16)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_train_cuda_short_run(tmp_path):
    ckpt_dir = tmp_path / "checkpoints_cuda"
    args = parse_args(
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
            "--device",
            "cuda",
        ]
    )
    summary = train(args)
    assert isinstance(summary, dict)
    assert summary["total_frames"] >= 50
    assert "best_success_rate" in summary
    assert ckpt_dir.is_dir()
    assert (ckpt_dir / "metrics_summary.json").is_file()


def test_train_early_stopping_on_target_convergence(tmp_path, monkeypatch):
    ckpt_dir = tmp_path / "checkpoints_early_stop"
    args = parse_args(
        [
            "--seed",
            "42",
            "--total-frames",
            "10000",
            "--frames-per-batch",
            "50",
            "--storage-size",
            "500",
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
            "--no-enable-adr",
            "--early-stop-on-target",
            "--target-success-rate",
            "0.85",
        ]
    )

    mock_metrics = {
        "success_rate": 0.95,
        "mean_consecutive_successes": 5.0,
        "mean_geodesic_error": 0.04,
        "mean_return": 120.0,
    }
    monkeypatch.setattr("train.evaluate_policy", lambda **kwargs: mock_metrics)

    summary = train(args)

    assert summary["converged"] is True
    assert summary["total_frames"] == 50
    assert (ckpt_dir / "benchmark_policy.onnx").is_file()
    assert (ckpt_dir / "best_policy.onnx").is_file()


def test_train_early_stopping_adr_curriculum(tmp_path, monkeypatch):
    ckpt_dir = tmp_path / "checkpoints_adr_early_stop"
    args = parse_args(
        [
            "--seed",
            "42",
            "--total-frames",
            "10000",
            "--frames-per-batch",
            "50",
            "--storage-size",
            "500",
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
            "--enable-adr",
            "--early-stop-on-target",
            "--target-success-rate",
            "0.85",
        ]
    )

    mock_metrics = {
        "success_rate": 0.95,
        "mean_consecutive_successes": 5.0,
        "mean_geodesic_error": 0.04,
        "mean_return": 120.0,
    }
    monkeypatch.setattr("train.evaluate_policy", lambda **kwargs: mock_metrics)

    from src.envs.domain_randomization import ADRCurriculumController

    orig_init = ADRCurriculumController.__init__

    def patched_init(self, *a, **kw):
        orig_init(self, *a, **kw)
        self.set_difficulty(1.0)

    monkeypatch.setattr(ADRCurriculumController, "__init__", patched_init)

    summary = train(args)

    assert summary["converged"] is True
    assert summary["total_frames"] == 50
    assert (ckpt_dir / "benchmark_policy.onnx").is_file()


def test_train_no_early_stop_when_disabled(tmp_path, monkeypatch):
    ckpt_dir = tmp_path / "checkpoints_no_early_stop"
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
            "--no-enable-adr",
            "--no-early-stop-on-target",
            "--target-success-rate",
            "0.85",
        ]
    )

    mock_metrics = {
        "success_rate": 0.95,
        "mean_consecutive_successes": 5.0,
        "mean_geodesic_error": 0.04,
        "mean_return": 120.0,
    }
    monkeypatch.setattr("train.evaluate_policy", lambda **kwargs: mock_metrics)

    summary = train(args)

    assert summary["converged"] is True
    assert summary["total_frames"] == 100


def test_update_collector_target_mode_single_env():
    """Verify update_collector_target_mode updates target_mode on single LeapCubeReorientEnv collector."""
    from src.algo import update_collector_target_mode
    from src.algo.actor import SquashedGaussianSACActor
    from src.algo.collector import PolicyModule, create_trajectory_collector
    from src.algo.her import create_her_replay_buffer
    from src.envs import LeapCubeReorientEnv

    env_fn = lambda: LeapCubeReorientEnv(max_steps=10, target_mode="so3")
    actor = SquashedGaussianSACActor(obs_dim=55, goal_dim=4, action_dim=16)
    policy = PolicyModule(actor)
    rb = create_her_replay_buffer(storage_size=100, slice_len=10, batch_size=10)
    collector = create_trajectory_collector(
        env_fn=env_fn,
        policy=policy,
        replay_buffer=rb,
        frames_per_batch=10,
        total_frames=10,
    )
    try:
        assert collector.env.target_mode == "so3"
        update_collector_target_mode(collector, "yaw")
        assert collector.env.target_mode == "yaw"
        update_collector_target_mode(collector, "so3")
        assert collector.env.target_mode == "so3"
        with pytest.raises(ValueError, match="Invalid target_mode"):
            update_collector_target_mode(collector, "invalid")
    finally:
        collector.shutdown()


def test_update_collector_target_mode_parallel_env():
    """Verify update_collector_target_mode updates target_mode on batched / parallel env collector."""
    from functools import partial
    from unittest.mock import MagicMock

    from torchrl.envs.batched_envs import ParallelEnv, SerialEnv

    from src.algo import update_collector_target_mode
    from src.envs import LeapCubeReorientEnv

    # 1. Test with SerialEnv
    s_env = SerialEnv(
        2,
        [
            partial(LeapCubeReorientEnv, max_steps=10, target_mode="so3"),
            partial(LeapCubeReorientEnv, max_steps=10, target_mode="so3"),
        ],
    )
    s_env.start()
    mock_collector = MagicMock()
    mock_collector.env = s_env

    try:
        update_collector_target_mode(mock_collector, "yaw")
        for sub_env in s_env._envs:
            assert sub_env.target_mode == "yaw"

        update_collector_target_mode(mock_collector, "so3")
        for sub_env in s_env._envs:
            assert sub_env.target_mode == "so3"
    finally:
        s_env.close()

    # 2. Test with real ParallelEnv passed directly (unstarted) and via collector
    penv = ParallelEnv(
        2,
        [
            partial(LeapCubeReorientEnv, max_steps=20, target_mode="yaw"),
            partial(LeapCubeReorientEnv, max_steps=20, target_mode="yaw"),
        ],
    )
    try:
        # Directly passing unstarted ParallelEnv as collector
        update_collector_target_mode(penv, "so3")
        assert penv.target_mode() == ["so3", "so3"]

        update_collector_target_mode(penv, "yaw")
        assert penv.target_mode() == ["yaw", "yaw"]

        # Passing via collector wrapper with .env
        collector_wrapper = MagicMock()
        collector_wrapper.env = penv
        update_collector_target_mode(collector_wrapper, "so3")
        assert penv.target_mode() == ["so3", "so3"]

        update_collector_target_mode(collector_wrapper, "yaw")
        assert penv.target_mode() == ["yaw", "yaw"]
    finally:
        if not penv.is_closed:
            penv.close()


