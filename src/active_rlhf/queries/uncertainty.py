from typing import Literal

import torch as th

def estimate_uncertainties(
        rewards: th.Tensor,
        first_indices: th.Tensor,
        second_indices: th.Tensor,
        method: Literal["return_diff"]
) -> th.Tensor:
    """Estimate the uncertainties of the pairs of trajectories."""
    assert len(first_indices) == len(second_indices)

    match (method):
        case "return_diff":
            first_returns = rewards[first_indices].sum(dim=1)
            second_returns = rewards[second_indices].sum(dim=1)
            return (first_returns - second_returns).var(dim=-1, unbiased=False)
        case _:
            raise ValueError(f"Invalid method: {method}")


def preference_interval(theta: th.Tensor, alpha: float = 0.0, dim: int = 1):
    """
    Compute the α-quantile preference interval along `dim`.

    Args
    ----
    theta : Tensor (batch_size, ensemble_size)
      P(σ1 ≻ σ0) for every ensemble member.
    alpha : float in [0, 1)
      Width of the two tails to clip away. alpha = 0.0 reproduces min/max.
    dim   : int
      Dimension that indexes the ensemble.

    Returns
    -------
    lower, upper : Tensors (batch_size,)
      The two quantile bounds.
    """
    if not (0.0 <= alpha < 1.0):
        raise ValueError("alpha must be in [0, 1)")

    if alpha == 0.0:
        lower = theta.min(dim=dim).values
        upper = theta.max(dim=dim).values
    else:
        q_low = alpha / 2.0
        q_high = 1.0 - q_low
        # torch.quantile supports interpolation; default 'linear' is fine.
        lower = th.quantile(theta, q=q_low, dim=dim)
        upper = th.quantile(theta, q=q_high, dim=dim)

    return lower, upper


def estimate_epistemic_uncertainties(probs: th.Tensor, alpha: float = 0.0) -> th.Tensor:
    """Estimate the epistemic uncertainties of the pairs of trajectories from the rewards based on the following formula:

    max_i(θ_i) - min_i(θ_i) where θ_i = Pψi(σ_1 ≻ σ_0)

    Args:
      probs: The probabilities of the pairs of trajectories of shape (batch_size, ensemble_size, 2).
      alpha: The alpha quantile to use for the preference interval.

    Returns:
      The epistemic uncertainties of the pairs of trajectories.
    """
    # Safety check
    if probs.ndim != 3 or probs.shape[-1] != 2:
        raise ValueError("Expected shape (batch_size, ensemble_size, 2)")

    # θ_i – probability that σ1 beats σ0 for every ensemble member
    theta = probs[..., 1]  # (batch_size, ensemble_size)

    # Fast vectorised range = max − min along the ensemble dimension (dim=1)
    lower, upper = preference_interval(theta, alpha=alpha, dim=1)
    epistemic_uncertainty = upper - lower  # (batch_size,)

    return epistemic_uncertainty


def consensual_filter(probs: th.Tensor, alpha: float = 0.0, threshold: float = 0.5) -> th.Tensor:
    """Filter the pairs of trajectories based on the consensus probability
    
    Args:
      probs: The probabilities of the pairs of trajectories of shape (batch_size, ensemble_size, 2).
      alpha: The alpha quantile to use for the preference interval.
      threshold: The threshold for the consensus probability.

    Returns:
      The filtered pair indices of trajectories to keep (batch_size,).
    """
    # Safety check
    if probs.ndim != 3 or probs.shape[-1] != 2:
        raise ValueError("Expected shape (batch_size, ensemble_size, 2)")

    # Calculate the consensus probability
    theta = probs[..., 1]
    lower, upper = preference_interval(theta, alpha=alpha, dim=1)

    # We re interested in the pairs where min_p < threshold < max_p
    return (lower <= threshold) & (threshold <= upper)
