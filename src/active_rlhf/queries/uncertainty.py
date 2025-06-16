from typing import List, Literal
import torch as th
from active_rlhf.data.buffers import ReplayBufferBatch

def estimate_uncertainties(
    rewards: th.Tensor, 
    first_indices: th.Tensor,
    second_indices: th.Tensor,
    method: Literal["reward_diff"]
    ) -> th.Tensor:
    """Estimate the uncertainties of the pairs of trajectories."""
    assert len(first_indices) == len(second_indices)

    print("first_indices_shape:", first_indices.shape)
    print("second_indices_shape:", second_indices.shape)
    print("rewards_shape:", rewards.shape)

    match (method):
        case "return_diff":
            first_returns = rewards[first_indices].sum(dim=1)
            second_returns = rewards[second_indices].sum(dim=1)
            print("first_returns_shape:", first_returns.shape)
            print("second_returns_shape:", second_returns.shape)
            return (first_returns - second_returns).var(dim=-1)
        case _:
            raise ValueError(f"Invalid method: {method}")


