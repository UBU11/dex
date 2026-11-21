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
    assert args.num_workers == 1
    assert args.warmup_yaw_frames == 0
    assert args.target_mode == "so3"


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
        "--num-workers",
        "4",
        "--warmup-yaw-frames",
        "20000",
        "--target-mode",
        "yaw",
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
    assert args.num_workers == 4
    assert args.warmup_yaw_frames == 20000
    assert args.target_mode == "yaw"


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
    from functools import partial

    from src.algo import update_collector_target_mode
    from src.algo.actor import SquashedGaussianSACActor
    from src.algo.collector import PolicyModule, create_trajectory_collector
    from src.algo.her import create_her_replay_buffer
    from src.envs import LeapCubeReorientEnv

    env_fn = partial(LeapCubeReorientEnv, max_steps=10, target_mode="so3")
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


def test_create_trajectory_collector_multi_worker_parallel():
    """Verify create_trajectory_collector supports num_workers > 1 with ParallelEnv."""
    from functools import partial

    from torchrl.envs.batched_envs import ParallelEnv

    from src.algo import update_collector_target_mode
    from src.algo.actor import SquashedGaussianSACActor
    from src.algo.collector import PolicyModule, create_trajectory_collector
    from src.algo.her import create_her_replay_buffer
    from src.envs import LeapCubeReorientEnv

    env_fn = partial(LeapCubeReorientEnv, max_steps=10, target_mode="so3")
    actor = SquashedGaussianSACActor(obs_dim=55, goal_dim=4, action_dim=16)
    policy = PolicyModule(actor)
    rb = create_her_replay_buffer(storage_size=100, slice_len=10, batch_size=10)

    collector = create_trajectory_collector(
        env_fn=env_fn,
        policy=policy,
        replay_buffer=rb,
        frames_per_batch=20,
        total_frames=20,
        num_workers=2,
    )
    try:
        assert isinstance(collector.env, ParallelEnv)
        assert collector.env.batch_size == torch.Size([2])
        # Test target mode update on multi-worker collector
        update_collector_target_mode(collector, "yaw")
        assert collector.env.target_mode() == ["yaw", "yaw"]
        update_collector_target_mode(collector, "so3")
        assert collector.env.target_mode() == ["so3", "so3"]

        # Collect transitions into replay buffer
        for _ in collector:
            pass
        assert len(rb) >= 20
    finally:
        collector.shutdown()


def test_create_trajectory_collector_multi_worker_with_domain_randomization():
    """Verify multi-worker collector properly wraps domain randomization on workers."""
    from functools import partial

    from torchrl.envs.batched_envs import ParallelEnv

    from src.algo.actor import SquashedGaussianSACActor
    from src.algo.collector import PolicyModule, create_trajectory_collector
    from src.algo.her import create_her_replay_buffer
    from src.envs import LeapCubeReorientEnv

    env_fn = partial(LeapCubeReorientEnv, max_steps=10, target_mode="so3")
    actor = SquashedGaussianSACActor(obs_dim=55, goal_dim=4, action_dim=16)
    policy = PolicyModule(actor)
    rb = create_her_replay_buffer(storage_size=100, slice_len=10, batch_size=10)

    collector = create_trajectory_collector(
        env_fn=env_fn,
        policy=policy,
        replay_buffer=rb,
        frames_per_batch=20,
        total_frames=20,
        enable_domain_randomization=True,
        num_workers=2,
    )
    try:
        assert isinstance(collector.env, ParallelEnv)
        for _ in collector:
            pass
        assert len(rb) >= 20
    finally:
        collector.shutdown()


def test_create_trajectory_collector_invalid_num_workers():
    """Verify create_trajectory_collector raises ValueError for invalid num_workers."""
    from functools import partial

    from src.algo.actor import SquashedGaussianSACActor
    from src.algo.collector import PolicyModule, create_trajectory_collector
    from src.algo.her import create_her_replay_buffer
    from src.envs import LeapCubeReorientEnv

    env_fn = partial(LeapCubeReorientEnv, max_steps=10)
    actor = SquashedGaussianSACActor(obs_dim=55, goal_dim=4, action_dim=16)
    policy = PolicyModule(actor)
    rb = create_her_replay_buffer(storage_size=100, slice_len=10, batch_size=10)

    with pytest.raises(ValueError, match="num_workers"):
        create_trajectory_collector(
            env_fn=env_fn,
            policy=policy,
            replay_buffer=rb,
            frames_per_batch=10,
            num_workers=0,
        )


def test_parse_args_invalid_target_mode():
    """Verify parse_args rejects invalid target modes."""
    with pytest.raises(SystemExit):
        parse_args(["--target-mode", "invalid_mode"])


def test_train_curriculum_warmup_transition(tmp_path, monkeypatch):
    """Verify train() dynamically switches from yaw to so3 mode at warmup_yaw_frames."""
    from src.algo import update_collector_target_mode

    ckpt_dir = tmp_path / "checkpoints_warmup"
    args = parse_args(
        [
            "--seed",
            "42",
            "--total-frames",
            "60",
            "--frames-per-batch",
            "30",
            "--storage-size",
            "500",
            "--batch-size",
            "30",
            "--utd-ratio",
            "0.1",
            "--eval-interval",
            "30",
            "--eval-episodes",
            "1",
            "--save-interval",
            "30",
            "--checkpoint-dir",
            str(ckpt_dir),
            "--max-steps",
            "10",
            "--warmup-yaw-frames",
            "30",
            "--target-mode",
            "so3",
        ]
    )

    mode_updates: list[str] = []
    orig_update = update_collector_target_mode

    def mock_update(collector, mode):
        mode_updates.append(mode)
        return orig_update(collector, mode)

    monkeypatch.setattr("train.update_collector_target_mode", mock_update)

    summary = train(args)

    assert summary["total_frames"] == 60
    assert mode_updates == ["so3"]
    assert summary.get("active_target_mode") == "so3"


def test_train_curriculum_warmup_resume_past_warmup(tmp_path, monkeypatch):
    """Verify resuming at a step >= warmup_yaw_frames does not trigger warmup mode switch."""
    ckpt_dir = tmp_path / "checkpoints_resume_warmup"
    args1 = parse_args(
        [
            "--seed",
            "42",
            "--total-frames",
            "30",
            "--frames-per-batch",
            "30",
            "--storage-size",
            "200",
            "--batch-size",
            "20",
            "--utd-ratio",
            "0.1",
            "--eval-interval",
            "30",
            "--eval-episodes",
            "1",
            "--save-interval",
            "30",
            "--checkpoint-dir",
            str(ckpt_dir),
            "--max-steps",
            "10",
        ]
    )
    summary1 = train(args1)
    resume_ckpt = summary1["saved_checkpoints"][0]

    args2 = parse_args(
        [
            "--seed",
            "42",
            "--total-frames",
            "30",
            "--frames-per-batch",
            "30",
            "--storage-size",
            "200",
            "--batch-size",
            "20",
            "--utd-ratio",
            "0.1",
            "--eval-interval",
            "30",
            "--eval-episodes",
            "1",
            "--save-interval",
            "30",
            "--checkpoint-dir",
            str(ckpt_dir),
            "--resume",
            resume_ckpt,
            "--max-steps",
            "10",
            "--warmup-yaw-frames",
            "20",
        ]
    )

    mode_updates: list[str] = []
    monkeypatch.setattr(
        "train.update_collector_target_mode",
        lambda col, mode: mode_updates.append(mode),
    )

    summary2 = train(args2)
    assert summary2["total_frames"] == 30
    assert summary2["final_step"] == 60
    assert mode_updates == []
    assert summary2.get("active_target_mode") == "so3"


def test_train_multi_worker_execution(tmp_path):
    """Verify train() runs successfully with multi-worker parallel collection."""
    ckpt_dir = tmp_path / "checkpoints_multi_worker"
    args = parse_args(
        [
            "--seed",
            "42",
            "--total-frames",
            "40",
            "--frames-per-batch",
            "20",
            "--storage-size",
            "500",
            "--batch-size",
            "20",
            "--utd-ratio",
            "0.1",
            "--eval-interval",
            "20",
            "--eval-episodes",
            "1",
            "--save-interval",
            "20",
            "--checkpoint-dir",
            str(ckpt_dir),
            "--max-steps",
            "10",
            "--num-workers",
            "2",
        ]
    )
    summary = train(args)
    assert summary["total_frames"] >= 40
    assert summary["final_step"] >= 40
    assert len(summary["saved_checkpoints"]) >= 1


def test_train_resume_extended_frames(tmp_path, monkeypatch):
    """Verify train() properly resumes with large frame budget and updates current_step = start_step + total_frames."""
    ckpt_dir = tmp_path / "checkpoints_resume_extended"
    args1 = parse_args(
        [
            "--seed",
            "42",
            "--total-frames",
            "30",
            "--frames-per-batch",
            "30",
            "--storage-size",
            "200",
            "--batch-size",
            "20",
            "--utd-ratio",
            "0.1",
            "--eval-interval",
            "30",
            "--eval-episodes",
            "1",
            "--save-interval",
            "30",
            "--checkpoint-dir",
            str(ckpt_dir),
            "--max-steps",
            "10",
        ]
    )
    summary1 = train(args1)
    resume_ckpt = summary1["saved_checkpoints"][0]

    args2 = parse_args(
        [
            "--seed",
            "42",
            "--total-frames",
            "3000000",
            "--frames-per-batch",
            "30",
            "--storage-size",
            "200",
            "--batch-size",
            "20",
            "--utd-ratio",
            "0.1",
            "--eval-interval",
            "30",
            "--eval-episodes",
            "1",
            "--save-interval",
            "30",
            "--checkpoint-dir",
            str(ckpt_dir),
            "--resume",
            resume_ckpt,
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

    summary2 = train(args2)
    assert summary2["converged"] is True
    assert summary2["total_frames"] == 30
    # start_step was 30, so current_step = start_step + total_frames = 60
    assert summary2["final_step"] == 60


def test_train_yaw_warmup_no_premature_early_stop(tmp_path, monkeypatch):
    """Verify that high success rate during yaw warmup does not trigger premature convergence or early stop."""
    ckpt_dir = tmp_path / "checkpoints_warmup_early_stop"
    args = parse_args(
        [
            "--seed",
            "42",
            "--total-frames",
            "90",
            "--frames-per-batch",
            "30",
            "--storage-size",
            "500",
            "--batch-size",
            "30",
            "--utd-ratio",
            "0.1",
            "--eval-interval",
            "30",
            "--eval-episodes",
            "1",
            "--save-interval",
            "30",
            "--checkpoint-dir",
            str(ckpt_dir),
            "--max-steps",
            "10",
            "--warmup-yaw-frames",
            "60",
            "--target-mode",
            "so3",
            "--early-stop-on-target",
            "--target-success-rate",
            "0.85",
            "--no-enable-adr",
        ]
    )

    eval_calls = 0

    def mock_eval(**kwargs):
        nonlocal eval_calls
        eval_calls += 1
        return {
            "success_rate": 1.0,
            "mean_consecutive_successes": 10.0,
            "mean_geodesic_error": 0.01,
            "mean_return": 150.0,
        }

    monkeypatch.setattr("train.evaluate_policy", mock_eval)

    summary = train(args)

    # Eval was called at frame 30 (during yaw warmup) and frame 60 (after warmup transition to so3)
    # At frame 30, warmup was active so early-stop did not trigger!
    # At frame 60, warmup completed, active_target_mode became "so3", so it converged and early-stopped.
    assert summary["converged"] is True
    assert summary["total_frames"] == 60
    assert summary["final_step"] == 60
    assert eval_calls == 2
    assert summary["active_target_mode"] == "so3"


def test_train_yaw_warmup_best_success_rate_reset(tmp_path, monkeypatch):
    """Verify best_success_rate resets to 0.0 upon transitioning from yaw warmup to full SO(3)."""
    ckpt_dir = tmp_path / "checkpoints_warmup_reset"
    args = parse_args(
        [
            "--seed",
            "42",
            "--total-frames",
            "60",
            "--frames-per-batch",
            "20",
            "--storage-size",
            "500",
            "--batch-size",
            "20",
            "--utd-ratio",
            "0.1",
            "--eval-interval",
            "20",
            "--eval-episodes",
            "1",
            "--save-interval",
            "20",
            "--checkpoint-dir",
            str(ckpt_dir),
            "--max-steps",
            "10",
            "--warmup-yaw-frames",
            "40",
            "--target-mode",
            "so3",
            "--no-early-stop-on-target",
            "--no-enable-adr",
        ]
    )

    eval_step = 0

    def mock_eval(**kwargs):
        nonlocal eval_step
        eval_step += 1
        if eval_step == 1:
            # During warmup eval at step 20 (warmup_active=True, yaw mode)
            return {
                "success_rate": 0.95,
                "mean_consecutive_successes": 8.0,
                "mean_geodesic_error": 0.02,
                "mean_return": 140.0,
            }
        elif eval_step == 2:
            # Post-warmup SO(3) eval at step 40 (warmup just transitioned to so3)
            return {
                "success_rate": 0.70,
                "mean_consecutive_successes": 3.0,
                "mean_geodesic_error": 0.12,
                "mean_return": 70.0,
            }
        else:
            # Post-warmup SO(3) eval at step 60
            return {
                "success_rate": 0.75,
                "mean_consecutive_successes": 4.0,
                "mean_geodesic_error": 0.10,
                "mean_return": 80.0,
            }

    monkeypatch.setattr("train.evaluate_policy", mock_eval)

    summary = train(args)

    # During warmup at step 20, best_success_rate reached 0.95.
    # At step 40, warmup finished and best_success_rate was reset to 0.0.
    # At step 40 and 60, success rates 0.70 and 0.75 were recorded.
    # If not reset, best_success_rate would remain stuck at 0.95.
    # With reset, the best SO(3) model achieved 0.75.
    assert summary["total_frames"] == 60
    assert summary["best_success_rate"] == pytest.approx(0.75)


def test_create_trajectory_collector_multi_worker_distinct_seeds():
    """Verify ParallelEnv workers sample distinct desired goals on reset (no duplicate RNG streams)."""
    from functools import partial

    from src.algo.actor import SquashedGaussianSACActor
    from src.algo.collector import PolicyModule, create_trajectory_collector
    from src.algo.her import create_her_replay_buffer
    from src.envs import LeapCubeReorientEnv

    env_fn = partial(LeapCubeReorientEnv, max_steps=10, target_mode="so3")
    actor = SquashedGaussianSACActor(obs_dim=55, goal_dim=4, action_dim=16)
    policy = PolicyModule(actor)
    rb = create_her_replay_buffer(storage_size=100, slice_len=10, batch_size=10)

    collector = create_trajectory_collector(
        env_fn=env_fn,
        policy=policy,
        replay_buffer=rb,
        frames_per_batch=20,
        total_frames=20,
        num_workers=2,
        seed=42,
    )
    try:
        penv = collector.env
        td = penv.reset()
        g0 = td[0]["desired_goal"]
        g1 = td[1]["desired_goal"]
        # Goals must differ across workers
        assert not torch.allclose(g0, g1), f"Worker goals should be distinct, got {g0} and {g1}"
    finally:
        collector.shutdown()




