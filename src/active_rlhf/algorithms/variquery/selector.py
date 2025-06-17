from typing import List, Tuple
from active_rlhf.algorithms.variquery.vae import StateVAE, VAETrainer
from active_rlhf.algorithms.variquery.visualizer import VAEVisualizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from active_rlhf.queries.uncertainty import estimate_uncertainties
from active_rlhf.rewards.reward_nets import RewardEnsemble
import numpy as np
import torch as th

from active_rlhf.data.buffers import TrajectoryPairBatch, ReplayBufferBatch
from active_rlhf.queries.selector import Selector
from torch.utils.tensorboard import SummaryWriter

class VARIQuerySelector(Selector):
    """Selector for the VariQuery algorithm, which selects pairs of trajectories based on a specific strategy."""

    def __init__(self,
                 writer: SummaryWriter,
                 reward_ensemble: RewardEnsemble,
                 fragment_length: int = 50,
                 vae_latent_dim: int = 16,
                 vae_hidden_dims: List[int] = [128, 64, 32],
                 vae_lr: float = 1e-3,
                 vae_weight_decay: float = 1e-4,
                 vae_dropout: float = 0.1,
                 vae_batch_size: int = 32,
                 vae_num_epochs: int = 25,
                 device: str = "cuda" if th.cuda.is_available() else "cpu"
                 ):
        self.reward_ensemble = reward_ensemble
        self.fragment_length = fragment_length
        self.vae_latent_dim = vae_latent_dim
        self.vae_hidden_dims = vae_hidden_dims
        self.vae_lr = vae_lr
        self.vae_weight_decay = vae_weight_decay
        self.vae_dropout = vae_dropout
        self.vae_batch_size = vae_batch_size
        self.vae_num_epochs = vae_num_epochs
        self.device = device

        self.visualizer = VAEVisualizer(writer=writer)
        

    def select_pairs(self, batch: ReplayBufferBatch, num_pairs: int, global_step: int) -> TrajectoryPairBatch:
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

        # Step 2: Train VAE and encode states
        metrics = vae_trainer.train(batch, global_step)

        with th.no_grad():
            latent_states, _, _ = vae.encode(batch.obs)

        # Step 3: Cluster and sample pairs
        clusters = self._cluster_latent_space(latent_states=latent_states, num_clusters=num_pairs)

        # Step 4: Sample pairs
        first_indices, second_indices = self._sample_random_pair_indices(clusters=clusters, num_pairs=num_pairs)
        assert len(first_indices) == len(second_indices)

        # Step 5: Rank by uncertainty estimate
        ranked_pair_indices = self._rank_pairs(first_indices, second_indices, batch).to(self.device)

        top_indices = ranked_pair_indices[:num_pairs].to(self.device)
        top_first_indices = first_indices[top_indices].to(self.device)
        top_second_indices = second_indices[top_indices].to(self.device)

        # Step 6: Visualize latent space and clusters
        self.visualizer.visualize(
            metrics=metrics,
            latents=latent_states,
            first_indices=top_first_indices,
            second_indices=top_second_indices,
            clusters=clusters,
            global_step=global_step,
        )

        return TrajectoryPairBatch(
            first_obs=batch.obs[top_first_indices],
            first_acts=batch.acts[top_first_indices],
            first_rews=batch.rews[top_first_indices],
            first_dones=batch.dones[top_first_indices],
            second_obs=batch.obs[top_second_indices],
            second_acts=batch.acts[top_second_indices],
            second_rews=batch.rews[top_second_indices],
            second_dones=batch.dones[top_second_indices]
        )
    
    @staticmethod
    def _cluster_latent_space(latent_states: th.Tensor, num_clusters: int) -> List[List[int]]:
        latent_states = latent_states.detach().cpu().numpy()

        # Standardize the latent states
        latent_states = (latent_states - latent_states.mean(axis=0)) / latent_states.std(axis=0)

        # Fit k-means with multiple initializations
        kmeans = KMeans(
            n_clusters=num_clusters,
            random_state=42,
            n_init=10,  # Try multiple initializations
            max_iter=300  # Increase max iterations
        )
        cluster_labels = kmeans.fit_predict(latent_states)
        
        # Calculate silhouette score to evaluate clustering quality
        # if len(latent_states) > 1:
        #     score = silhouette_score(latent_states, cluster_labels)
        
        # Group indices by cluster
        clusters = [[] for _ in range(num_clusters)]
        for idx, label in enumerate(cluster_labels):
            clusters[label].append(idx)
            
        return clusters
    
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
    
    def _rank_pairs(self, first_indices: th.Tensor, second_indices: th.Tensor, batch: ReplayBufferBatch) -> th.Tensor:
        with th.no_grad():
            rewards = self.reward_ensemble(batch.obs, batch.acts)

        uncertainties = estimate_uncertainties(
            rewards=rewards,
            first_indices=first_indices,
            second_indices=second_indices,
            method="return_diff"
        )
        
        # Sort indices based on uncertainty values in descending order
        sorted_indices = th.argsort(uncertainties, descending=True)
        
        return sorted_indices
        
        
        

        