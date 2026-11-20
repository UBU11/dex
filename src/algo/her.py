import torch
from tensordict import TensorDict

from src.utils.quaternion import compute_sparse_her_reward


class VectorizedFutureHER:
    """
    Implements the 'future' HER strategy (default relabel_prob=0.8)
    on episodic TensorDict batches collected with replay_write_mode='trajectory'.
    """

    def __init__(
        self,
        relabel_prob: float = 0.8,
        angle_tol: float = 0.15,
        drop_z_threshold: float = 0.04,
    ) -> None:
        self.relabel_prob = relabel_prob
        self.angle_tol = angle_tol
        self.drop_z_threshold = drop_z_threshold

    def __call__(self, episode_batch: TensorDict) -> TensorDict:
        # ponytail: strictly validate 2D episodic batch structure [B, T]
        if len(episode_batch.shape) != 2:
            raise ValueError(
                f"Expected 2D episodic TensorDict batch of shape [B, T], got shape {episode_batch.shape}"
            )

        B, T = episode_batch.shape
        if T < 2:
            raise ValueError(
                f"Episodic batch must have time dimension T >= 2, got T={T}"
            )

        # ponytail: robust device resolution when TensorDict device is unspecified
        device = episode_batch.device
        if device is None:
            try:
                target = episode_batch.get("desired_goal", None)
                device = target.device if target is not None else torch.device("cpu")
            except (AttributeError, KeyError):
                device = torch.device("cpu")

        # ponytail: sample random time indices t in [0, T - 2]
        t_idx = torch.randint(0, T - 1, (B,), device=device)

        # ponytail: sample future offsets in [t_idx, T - 1]
        rand_floats = torch.rand(B, device=device)
        future_idx = t_idx + (rand_floats * (T - 1 - t_idx)).long()

        # ponytail: extract transition at t_idx via advanced indexing
        batch_arange = torch.arange(B, device=device)
        transition = episode_batch[batch_arange, t_idx].clone()

        # ponytail: extract achieved goal at future_idx
        future_achieved_quat = episode_batch["next", "achieved_goal"][
            batch_arange, future_idx
        ]

        # ponytail: stochastic relabeling with probability relabel_prob
        relabel_mask = torch.rand(B, 1, device=device) < self.relabel_prob

        new_desired_goal = torch.where(
            relabel_mask,
            future_achieved_quat,
            transition["desired_goal"],
        )
        transition["desired_goal"] = new_desired_goal
        transition["next", "desired_goal"] = new_desired_goal

        # ponytail: recompute sparse HER reward under relabeled goal
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
