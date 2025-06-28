from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from active_rlhf.data.buffers import ReplayBufferBatch, TrajectoryPairBatch
from active_rlhf.queries.selector import RandomSelector, Selector
from active_rlhf.queries.uncertainty import estimate_epistemic_uncertainties, consensual_filter
from active_rlhf.rewards.reward_nets import PreferenceModel
from sklearn.cluster import KMeans
import torch as th
from torch.utils.tensorboard import SummaryWriter
import numpy as np

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

    print("Candidates before filtering:", origin_candidate_len)

    with th.no_grad():
      first_rews, second_rews, probs = self.preference_model(candidates.first_obs, candidates.first_acts, candidates.second_obs, candidates.second_acts)

    print("First rewards shape:", first_rews.shape)
    print("Second rewards shape:", second_rews.shape)

    if self.consensual_filter:
        keep_indices = consensual_filter(probs)
        first_rews, second_rews, probs = first_rews[keep_indices], second_rews[keep_indices], probs[keep_indices]
        candidates = candidates[keep_indices]

        print("Keep indices shape:", keep_indices.shape)
        print("Keep indices:", keep_indices)
        print("Consensual remaining pairs:", len(candidates))
        print("Consensual remaining pairs ratio:", len(candidates) / origin_candidate_len)
        print("First rewards shape 2:", first_rews.shape)
        print("Second rewards shape 2:", second_rews.shape)

        self.writer.add_scalar("duo/consensual_remaining_pairs", len(candidates), global_step)
        self.writer.add_scalar("duo/consensual_remaining_pairs_ratio", len(candidates) / origin_candidate_len, global_step)

    print("Candidates after filtering:", len(candidates))

    # Rank the pairs by the epistemic uncertainty
    ranked_indices = self._rank_by_epistemic_uncertainty(probs)
    top_indices = ranked_indices[:num_candidates//2]
    candidates = candidates[top_indices]
    first_rews, second_rews, probs = first_rews[top_indices], second_rews[top_indices], probs[top_indices]

    print("First rewards shape 3:", first_rews.shape)
    print("Second rewards shape 3:", second_rews.shape)

    # Mean over the ensemble dimension of the remaining rewards and calculate difference vector
    first_rews_mean = first_rews.mean(dim=-1) # shape: (num_pairs, fragment_length)
    second_rews_mean = second_rews.mean(dim=-1) # shape: (num_pairs, fragment_length)
    rewards_diff = first_rews_mean - second_rews_mean # shape: (num_pairs, fragment_length)

    print("Rewards first rew shape 4:", first_rews_mean.shape)
    print("Rewards second rew shape 4:", second_rews_mean.shape)
    print("Rewards diff shape 4:", rewards_diff.shape)

    rewards_diff_standardized = (rewards_diff - rewards_diff.mean(dim=0)) / (rewards_diff.std(dim=0) + self.eps)

    print("Rewards diff standardized shape:", rewards_diff_standardized.shape)
    
    # Select the most representative pairs
    selected_indices = self._select_from_clusters2(rewards_diff_standardized, num_pairs=num_pairs, global_step=global_step)
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
      # rewards_diff: (M, fragment_length) e.g. (80, 50)
      M, D = rewards_diff.shape
      # 1) CPU → numpy
      X = rewards_diff.cpu().numpy()  # shape (M, 50)

      # 2) PCA to 5 dims
      pca = PCA(n_components=min(5, D), random_state=self.random_state)
      Z = pca.fit_transform(X)  # shape (M, 5)

      # 3) KMeans on the 5-D embedding
      kmeans = KMeans(n_clusters=num_pairs, n_init=10, random_state=self.random_state)
      labels = kmeans.fit_predict(Z)  # (M,)

      score = silhouette_score(Z, labels)
      self.writer.add_scalar("duo/silhouette", score, global_step)

      # 4) pick one closest to each center
      centers = kmeans.cluster_centers_  # (num_pairs, 5)
      selected = []
      for c in range(num_pairs):
          mask = (labels == c)
          if not mask.any():
              # fallback: just take the top-uncertain if a cluster is empty
              selected.append(c % M)
              continue
          Zi = Z[mask]  # (m,5)
          idxs = np.nonzero(mask)[0]  # (m,)
          dists = np.linalg.norm(Zi - centers[c], axis=1)
          chosen = idxs[np.argmin(dists)]
          selected.append(int(chosen))
      return th.tensor(selected, device=rewards_diff.device)

  def _select_from_clusters2(self, rewards_diff: th.Tensor, num_pairs: int, global_step: int) -> th.Tensor:
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
    

