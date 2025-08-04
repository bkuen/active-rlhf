from active_rlhf.data.buffers import ReplayBufferBatch, TrajectoryPairBatch
from active_rlhf.queries.selector import RandomSelector, Selector
from active_rlhf.queries.uncertainty import estimate_epistemic_uncertainties, consensual_filter
from active_rlhf.rewards.reward_nets import PreferenceModel
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import torch as th
from torch.utils.tensorboard import SummaryWriter


class DUOSelector(Selector):
  def __init__(self,
               writer: SummaryWriter,
               preference_model: PreferenceModel,
               consensual_filter: bool = False,
               oversampling_factor: float = 10.0,
               random_state: int = 42,
               eps: float = 1e-8,
               ):
    self.writer = writer
    self.preference_model = preference_model
    self.consensual_filter = consensual_filter
    self.oversampling_factor = oversampling_factor
    self.eps = eps
    self.random_state = random_state

    self.random_selector = RandomSelector()

  def select_pairs(self, train_batch: ReplayBufferBatch, val_batch: ReplayBufferBatch, num_pairs: int, global_step: int) -> TrajectoryPairBatch:
    num_candidates = int(num_pairs * self.oversampling_factor)
    candidates = self.random_selector.select_pairs(train_batch, val_batch, num_pairs=num_candidates, global_step=global_step)
    origin_candidate_len = len(candidates)

    with th.no_grad():
      first_rews, second_rews, probs = self.preference_model(candidates.first_obs, candidates.first_acts, candidates.second_obs, candidates.second_acts)

    if self.consensual_filter:
        keep_indices = consensual_filter(probs)
        first_rews, second_rews, probs = first_rews[keep_indices], second_rews[keep_indices], probs[keep_indices]
        candidates = candidates[keep_indices]

        self.writer.add_scalar("duo/consensual_remaining_pairs", len(candidates), global_step)
        self.writer.add_scalar("duo/consensual_remaining_pairs_ratio", len(candidates) / origin_candidate_len, global_step)

    # Rank the pairs by the epistemic uncertainty
    ranked_indices = self._rank_by_epistemic_uncertainty(probs)
    top_indices = ranked_indices[:num_candidates//2]
    candidates = candidates[top_indices]
    first_rews, second_rews, probs = first_rews[top_indices], second_rews[top_indices], probs[top_indices]

    # Mean over the ensemble dimension of the remaining rewards and calculate difference vector
    first_rews_mean = first_rews.mean(dim=-1) # shape: (num_pairs, fragment_length)
    second_rews_mean = second_rews.mean(dim=-1) # shape: (num_pairs, fragment_length)
    rewards_diff = first_rews_mean - second_rews_mean # shape: (num_pairs, fragment_length)

    rewards_diff_standardized = (rewards_diff - rewards_diff.mean(dim=0)) / (rewards_diff.std(dim=0) + self.eps)
    
    # Select the most representative pairs
    selected_indices = self._select_from_clusters(rewards_diff_standardized, num_pairs=num_pairs, global_step=global_step)
    assert selected_indices.shape == (num_pairs,)
    
    return candidates[selected_indices]
  
  def _rank_by_epistemic_uncertainty(self, probs: th.Tensor) -> th.Tensor:
    """Rank the pairs by the epistemic uncertainty.
    
    Args:
      probs: The probabilities of the pairs of trajectories of shape (num_pairs, ensemble_size, 2).
    """
    epistemic_uncertainties = estimate_epistemic_uncertainties(probs)
    return th.argsort(epistemic_uncertainties, descending=True)

  def _select_from_clusters(self, rewards_diff: th.Tensor, num_pairs: int, global_step: int) -> th.Tensor:
    """Cluster the rewards difference and select the most representative pairs.

    Args:
      rewards_diff: The rewards difference of shape (num_pairs, fragment_length).
      num_pairs: The number of pairs to select.

    Returns:
      The indices of the pairs of shape (num_pairs,).
    """
    rewards_diff_np = rewards_diff.cpu().numpy()

    kmeans = KMeans(n_clusters=num_pairs, random_state=self.random_state, n_init=10, max_iter=300)
    kmeans.fit(rewards_diff_np)
    labels = kmeans.labels_  # array [N]
    centers = kmeans.cluster_centers_  # array [k, D]

    score = silhouette_score(rewards_diff_np, labels)
    self.writer.add_scalar("duo/silhouette", score, global_step)

    selected_indices = []
    for cluster_id in range(num_pairs):
        member_mask = (labels == cluster_id)
        if not np.any(member_mask):
            continue

        members = rewards_diff_np[member_mask]  # [M, D]
        center = centers[cluster_id]  # [D]
        # compute Euclidean distances from center
        dists = np.linalg.norm(members - center, axis=1)  # [M]
        # find the original index of the closest member
        member_indices = np.nonzero(member_mask)[0]  # [M]
        closest_member = member_indices[np.argmin(dists)]
        selected_indices.append(int(closest_member))

    return th.tensor(selected_indices)
    

