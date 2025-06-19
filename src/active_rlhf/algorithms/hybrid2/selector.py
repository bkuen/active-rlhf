from typing import List, Tuple
from active_rlhf.algorithms.variquery.vae import StateVAE, VAETrainer
from active_rlhf.algorithms.variquery.visualizer import VAEVisualizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from active_rlhf.queries.uncertainty import estimate_uncertainties, estimate_epistemic_uncertainties
from active_rlhf.rewards.reward_nets import RewardEnsemble, PreferenceModel
import numpy as np
import torch as th

from active_rlhf.data.buffers import TrajectoryPairBatch, ReplayBufferBatch
from active_rlhf.queries.selector import Selector, RandomSelector
from torch.utils.tensorboard import SummaryWriter


class HybridV2Selector(Selector):
    """Selector for the VariQuery algorithm, which selects pairs of trajectories based on a specific strategy."""

    def __init__(self,
                 writer: SummaryWriter,
                 preference_model: PreferenceModel,
                 fragment_length: int = 50,
                 vae_latent_dim: int = 16,
                 vae_hidden_dims: List[int] = [128, 64, 32],
                 vae_lr: float = 1e-3,
                 vae_weight_decay: float = 1e-4,
                 vae_dropout: float = 0.1,
                 vae_batch_size: int = 32,
                 vae_num_epochs: int = 25,
                 oversampling_factor: float = 10.0,
                 random_state: int = 42,
                 device: str = "cuda" if th.cuda.is_available() else "cpu"
                 ):
        self.preference_model = preference_model
        self.fragment_length = fragment_length
        self.vae_latent_dim = vae_latent_dim
        self.vae_hidden_dims = vae_hidden_dims
        self.vae_lr = vae_lr
        self.vae_weight_decay = vae_weight_decay
        self.vae_dropout = vae_dropout
        self.vae_batch_size = vae_batch_size
        self.vae_num_epochs = vae_num_epochs
        self.oversampling_factor = oversampling_factor
        self.random_state = random_state
        self.device = device

        self.visualizer = VAEVisualizer(writer=writer)
        self.random_selector = RandomSelector()

    def select_pairs(self, batch: ReplayBufferBatch, num_pairs: int, global_step: int) -> TrajectoryPairBatch:
        batch_size = batch.obs.shape[0]
        vae = StateVAE(
            state_dim=batch.obs.shape[2],
            latent_dim=self.vae_latent_dim,
            fragment_length=self.fragment_length,
            hidden_dims=self.vae_hidden_dims,
            dropout=self.vae_dropout,
        )

        vae_trainer = VAETrainer(
            vae=vae,
            lr=self.vae_lr,
            weight_decay=self.vae_weight_decay,
            batch_size=self.vae_batch_size,
            num_epochs=self.vae_num_epochs,
            device=self.device
        )

        # Train VAE and encode states
        metrics = vae_trainer.train(batch, global_step)

        num_candidates = int(num_pairs * self.oversampling_factor)
        candidates = self.random_selector.select_pairs(batch, num_pairs=num_candidates, global_step=global_step)

        with th.no_grad():
            first_rews, second_rews, probs = self.preference_model(candidates.first_obs, candidates.first_acts,
                                                                   candidates.second_obs, candidates.second_acts)

        epistemic_uncertainties = estimate_epistemic_uncertainties(probs)
        ranked_indices = th.argsort(epistemic_uncertainties, descending=True)

        # Keep a slightly smaller but still oversampled set of uncertain candidates
        top_indices = ranked_indices[:int(num_candidates / 2)]
        candidates = candidates[top_indices]
        first_rews, second_rews, probs = first_rews[top_indices], second_rews[top_indices], probs[top_indices]

        with th.no_grad():
            first_latents, _, _ = vae.encode(candidates.first_obs)
            second_latents, _, _ = vae.encode(candidates.second_obs)

        first_latents = (first_latents - first_latents.mean(dim=0)) / (first_latents.std(dim=0) + 1e-8)
        second_latents = (second_latents - second_latents.mean(dim=0)) / (second_latents.std(dim=0) + 1e-8)

        first_rews_mean = first_rews.mean(dim=-1)  # shape: (num_pairs, fragment_length)
        second_rews_mean = second_rews.mean(dim=-1)  # shape: (num_pairs, fragment_length)
        rewards_diff = first_rews_mean - second_rews_mean  # shape: (num_pairs, fragment_length)
        rewards_diff = (rewards_diff - rewards_diff.mean(dim=0)) / (rewards_diff.std(dim=0) + 1e-8)

        embedding = th.cat([first_latents, second_latents, rewards_diff], dim=1)

        # Select the most representative pairs
        selected_indices = self._select_from_clusters(embedding=embedding, num_pairs=num_pairs)
        assert selected_indices.shape == (num_pairs,)

        return candidates[selected_indices]

    def _select_from_clusters(self, embedding: th.Tensor, num_pairs: int) -> th.Tensor:
        """Cluster the embeeding and select the most representative pairs.

        Args:
          embedding: The embedding of shape (num_pairs, fragment_length).
          num_pairs: The number of pairs to select.

        Returns:
          The indices of the pairs of shape (num_pairs,).
        """
        embedding_np = embedding.cpu().numpy()

        kmeans = KMeans(n_clusters=num_pairs, random_state=self.random_state, n_init=10, max_iter=300)
        kmeans.fit(embedding_np)
        labels = kmeans.labels_  # array [N]
        centers = kmeans.cluster_centers_  # array [k, D]

        selected_indices = []
        for cluster_id in range(num_pairs):
            member_mask = (labels == cluster_id)
            if not np.any(member_mask):
                continue

            members = embedding[member_mask]  # [M, D]
            center = centers[cluster_id]  # [D]
            # compute Euclidean distances from center
            dists = np.linalg.norm(members - center, axis=1)  # [M]
            # find the original index of the closest member
            member_indices = np.nonzero(member_mask)[0]  # [M]
            closest_member = member_indices[np.argmin(dists)]
            selected_indices.append(int(closest_member))

        return th.tensor(selected_indices)

    def _sample_random_pair_indices(self, clusters: List[List[int]], num_pairs: int) -> Tuple[th.Tensor, th.Tensor]:
        first_indices = []
        second_indices = []
        for _ in range(num_pairs):
            # Sample two random indices from the clusters
            c1, c2 = np.random.choice(len(clusters), size=2, replace=False)
            idx1 = np.random.choice(len(clusters[c1]), replace=False)
            idx2 = np.random.choice(len(clusters[c2]), replace=False)

            index1 = clusters[c1][idx1]
            index2 = clusters[c2][idx2]

            first_indices.append(index1)
            second_indices.append(index2)

        return th.tensor(first_indices, device=self.device), th.tensor(second_indices, device=self.device)




