from collections.abc import Callable

import torch
from tensordict import TensorDict
from torchrl.data import LazyTensorStorage, SliceSampler, TensorDictReplayBuffer

from src.utils.quaternion import compute_sparse_her_reward


class VectorizedFutureHER:
    """
    Implements the 'future' HER strategy (default relabel_prob=0.8).
    Supports 2D episodic trajectory batches [B, T] extracting [B] transitions,
    as well as 1D/2D sliced batches when slice_len is specified.
    """

    def __init__(
        self,
        relabel_prob: float = 0.8,
        angle_tol: float = 0.15,
        drop_z_threshold: float = 0.04,
        slice_len: int | None = None,
    ) -> None:
        if slice_len is not None and slice_len < 2:
            raise ValueError(f"slice_len must be >= 2, got {slice_len}")
        self.relabel_prob = relabel_prob
        self.angle_tol = angle_tol
        self.drop_z_threshold = drop_z_threshold
        self.slice_len = slice_len

    def as_transform(self, slice_len: int | None = None) -> Callable[[TensorDict], TensorDict]:
        eff_slice_len = slice_len if slice_len is not None else self.slice_len
        if eff_slice_len is None:
            raise ValueError("slice_len must be specified for buffer transform")
        return VectorizedFutureHER(
            relabel_prob=self.relabel_prob,
            angle_tol=self.angle_tol,
            drop_z_threshold=self.drop_z_threshold,
            slice_len=eff_slice_len,
        )

    def __call__(self, batch: TensorDict) -> TensorDict:
        if self.slice_len is not None:
            if len(batch.shape) == 1:
                n = batch.shape[0]
                if n % self.slice_len != 0:
                    raise ValueError(
                        f"Batch size {n} is not divisible by slice_len {self.slice_len}"
                    )
                b = n // self.slice_len
                t = self.slice_len
                reshaped = batch.clone().reshape(b, t)
                orig_shape = batch.shape
            elif len(batch.shape) == 2:
                b, t = batch.shape
                if t != self.slice_len:
                    raise ValueError(
                        f"Batch time dimension T={t} does not match slice_len={self.slice_len}"
                    )
                reshaped = batch.clone()
                orig_shape = batch.shape
            else:
                raise ValueError(
                    f"Expected 1D sliced or 2D batch with slice_len={self.slice_len}, got shape {batch.shape}"
                )

            device = reshaped.device
            if device is None:
                try:
                    target = reshaped.get("desired_goal", None)
                    device = target.device if target is not None else torch.device("cpu")
                except (AttributeError, KeyError):
                    device = torch.device("cpu")

            t_idx = torch.arange(t, device=device).unsqueeze(0).expand(b, t)
            remaining = t - t_idx
            rand_floats = torch.rand(b, t, device=device)
            offset = (rand_floats * remaining).long()
            future_idx = torch.clamp(t_idx + offset, max=t - 1)

            gather_idx = future_idx.unsqueeze(-1).expand(b, t, 4)
            future_achieved_quat = torch.gather(
                reshaped["next", "achieved_goal"], dim=1, index=gather_idx
            )

            relabel_mask = torch.rand(b, t, 1, device=device) < self.relabel_prob
            new_desired_goal = torch.where(
                relabel_mask,
                future_achieved_quat,
                reshaped["desired_goal"],
            )
            reshaped["desired_goal"] = new_desired_goal
            reshaped["next", "desired_goal"] = new_desired_goal

            next_achieved_quat = reshaped["next", "achieved_goal"]
            next_cube_z = reshaped["next", "cube_z"]

            new_reward, _ = compute_sparse_her_reward(
                next_achieved_quat,
                new_desired_goal,
                next_cube_z,
                angle_tol=self.angle_tol,
                drop_z_threshold=self.drop_z_threshold,
            )
            reshaped["next", "reward"] = new_reward
            return reshaped.reshape(orig_shape)

        if len(batch.shape) != 2:
            raise ValueError(
                f"Expected 2D episodic TensorDict batch of shape [B, T], got shape {batch.shape}"
            )

        b, t = batch.shape
        if t < 2:
            raise ValueError(
                f"Episodic batch must have time dimension T >= 2, got T={t}"
            )

        device = batch.device
        if device is None:
            try:
                target = batch.get("desired_goal", None)
                device = target.device if target is not None else torch.device("cpu")
            except (AttributeError, KeyError):
                device = torch.device("cpu")

        t_idx = torch.randint(0, t - 1, (b,), device=device)
        rand_floats = torch.rand(b, device=device)
        offset = (rand_floats * (t - t_idx)).long()
        future_idx = torch.clamp(t_idx + offset, max=t - 1)

        batch_arange = torch.arange(b, device=device)
        transition = batch[batch_arange, t_idx].clone()

        future_achieved_quat = batch["next", "achieved_goal"][
            batch_arange, future_idx
        ]

        relabel_mask = torch.rand(b, 1, device=device) < self.relabel_prob

        new_desired_goal = torch.where(
            relabel_mask,
            future_achieved_quat,
            transition["desired_goal"],
        )
        transition["desired_goal"] = new_desired_goal
        transition["next", "desired_goal"] = new_desired_goal

        next_achieved_quat = transition["next", "achieved_goal"]
        next_cube_z = transition["next", "cube_z"]

        new_reward, _ = compute_sparse_her_reward(
            next_achieved_quat,
            new_desired_goal,
            next_cube_z,
            angle_tol=self.angle_tol,
            drop_z_threshold=self.drop_z_threshold,
        )
        transition["next", "reward"] = new_reward
        return transition


def create_her_replay_buffer(
    storage_size: int,
    slice_len: int,
    relabel_prob: float = 0.8,
    batch_size: int = 256,
    angle_tol: float = 0.15,
    drop_z_threshold: float = 0.04,
) -> TensorDictReplayBuffer:
    """Creates a TensorDictReplayBuffer configured with SliceSampler and VectorizedFutureHER transform."""
    storage = LazyTensorStorage(max_size=storage_size)
    sampler = SliceSampler(slice_len=slice_len, traj_key=("collector", "traj_ids"))
    transform = VectorizedFutureHER(
        relabel_prob=relabel_prob,
        angle_tol=angle_tol,
        drop_z_threshold=drop_z_threshold,
        slice_len=slice_len,
    )
    return TensorDictReplayBuffer(
        storage=storage,
        sampler=sampler,
        transform=transform,
        batch_size=batch_size,
    )
