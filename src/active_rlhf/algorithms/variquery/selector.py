from typing import List, Tuple, Optional
from active_rlhf.algorithms.variquery.vae import MLPStateVAE, VAETrainer, GRUStateVAE, AttnStateVAE, EnhancedGRUStateVAE
from active_rlhf.algorithms.variquery.visualizer import VAEVisualizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from active_rlhf.data.running_stats import RunningStat
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
                 reward_norm: RunningStat,
                 vae_state_dim: int,
                 fragment_length: int = 50,
                 vae_latent_dim: int = 16,
                 vae_hidden_dims: List[int] = [128, 64, 32],
                 vae_lr: float = 1e-3,
                 vae_weight_decay: float = 1e-4,
                 vae_dropout: float = 0.1,
                 vae_batch_size: int = 32,
                 vae_num_epochs: int = 25,
                 vae_kl_weight: float = 1.0,
                 vae_kl_warmup_epochs: int = 40,
                 vae_kl_warmup_steps: int = 320_000,
                 vae_early_stopping_patience: Optional[int] = None,
                 vae_attn_dim: int = 128,
                 vae_attn_heads: int = 4,
                 vae_attn_blocks: int = 2,
                 vae_decoder_layers: int = 2,
                 total_steps: int = 1_000_000,
                 device: str = "cuda" if th.cuda.is_available() else "cpu"
                 ):
        self.reward_ensemble = reward_ensemble
        self.reward_norm = reward_norm
        self.fragment_length = fragment_length
        self.vae_state_dim = vae_state_dim
        self.vae_latent_dim = vae_latent_dim
        self.vae_hidden_dims = vae_hidden_dims
        self.vae_lr = vae_lr
        self.vae_weight_decay = vae_weight_decay
        self.vae_dropout = vae_dropout
        self.vae_batch_size = vae_batch_size
        self.vae_num_epochs = vae_num_epochs
        self.vae_kl_weight = vae_kl_weight
        self.vae_kl_warmup_epochs = vae_kl_warmup_epochs
        self.vae_kl_warmup_steps = vae_kl_warmup_steps
        self.vae_early_stopping_patience = vae_early_stopping_patience
        self.vae_attn_dim = vae_attn_dim
        self.vae_attn_heads = vae_attn_heads
        self.vae_attn_blocks = vae_attn_blocks
        self.vae_decoder_layers = vae_decoder_layers
        self.total_steps = total_steps

        self.device = device

        self.vae = MLPStateVAE(
            state_dim=vae_state_dim,
            latent_dim=self.vae_latent_dim,
            fragment_length=self.fragment_length,
            hidden_dims=self.vae_hidden_dims,
            dropout=self.vae_dropout,
        )

        self.vae = AttnStateVAE(
            state_dim=vae_state_dim,
            latent_dim=self.vae_latent_dim,
            fragment_length=self.fragment_length,
            # hidden_dims=vae_hidden_dims,
            # dropout=self.vae_dropout,
            attn_dim=vae_attn_dim,
            n_heads=vae_attn_heads,
            n_blocks=vae_attn_blocks,
            n_decoder_layers=vae_decoder_layers,
            attn_dropout=vae_dropout,
            device=device,
        )

        self.vae_trainer = VAETrainer(
            vae=self.vae,
            lr=self.vae_lr,
            weight_decay=self.vae_weight_decay,
            kl_warmup_epochs=self.vae_kl_warmup_epochs,
            kl_warmup_steps=self.vae_kl_warmup_steps,
            batch_size=self.vae_batch_size,
            num_epochs=self.vae_num_epochs,
            early_stopping_patience=self.vae_early_stopping_patience,
            total_steps=self.total_steps,
            device=self.device
        )

        self.visualizer = VAEVisualizer(writer=writer)
        

    def select_pairs(self, train_batch: ReplayBufferBatch, val_batch: ReplayBufferBatch, num_pairs: int, global_step: int) -> TrajectoryPairBatch:
        batch_size = train_batch.obs.shape[0]
        # vae = StateVAE(
        #     state_dim=batch.obs.shape[2],
        #     latent_dim=self.vae_latent_dim,
        #     fragment_length=self.fragment_length,
        #     hidden_dims=self.vae_hidden_dims,
        #     dropout=self.vae_dropout,
        # )
        #
        # vae_trainer = VAETrainer(
        #     vae=vae,
        #     lr=self.vae_lr,
        #     weight_decay=self.vae_weight_decay,
        #     batch_size=self.vae_batch_size,
        #     num_epochs=self.vae_num_epochs,
        #     kl_weight_beta=self.vae_kl_weight,
        #     kl_warmup_epochs=self.vae_kl_warmup_epochs,
        #     device=self.device
        # )

        # Step 2: Train VAE and encode states
        metrics = self.vae_trainer.train(train_batch, val_batch, global_step)

        with th.no_grad():
            latent_states, _, _ = self.vae.encode(train_batch.obs)

        # Step 3: Cluster and sample pairs
        clusters = self._cluster_latent_space(latent_states=latent_states, num_clusters=num_pairs)

        # Step 4: Sample pairs
        first_indices, second_indices = self._sample_random_pair_indices(clusters=clusters, num_pairs=batch_size//2)
        assert len(first_indices) == len(second_indices)

        # Step 5: Rank by uncertainty estimate
        with th.no_grad():
            rewards = self.reward_ensemble(train_batch.obs, train_batch.acts)
            # rewards_norm = self.reward_norm(rewards)
        ranked_pair_indices = self._rank_pairs(rewards, first_indices, second_indices).to(self.device)

        top_indices = ranked_pair_indices[:num_pairs].to(self.device)
        top_first_indices = first_indices[top_indices].to(self.device)
        top_second_indices = second_indices[top_indices].to(self.device)

        # Step 6: Visualize latent space and clusters
        returns = rewards.mean(dim=-1).sum(dim=-1)
        self.visualizer.visualize(
            metrics=metrics,
            latents=latent_states,
            rewards=returns,
            first_indices=top_first_indices,
            second_indices=top_second_indices,
            clusters=clusters,
            global_step=global_step,
        )

        return TrajectoryPairBatch(
            first_obs=train_batch.obs[top_first_indices],
            first_acts=train_batch.acts[top_first_indices],
            first_rews=train_batch.rews[top_first_indices],
            first_dones=train_batch.dones[top_first_indices],
            second_obs=train_batch.obs[top_second_indices],
            second_acts=train_batch.acts[top_second_indices],
            second_rews=train_batch.rews[top_second_indices],
            second_dones=train_batch.dones[top_second_indices]
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
    
    def _rank_pairs(self, rewards: th.Tensor, first_indices: th.Tensor, second_indices: th.Tensor) -> th.Tensor:
        uncertainties = estimate_uncertainties(
            rewards=rewards,
            first_indices=first_indices,
            second_indices=second_indices,
            method="return_diff"
        )
        
        # Sort indices based on uncertainty values in descending order
        sorted_indices = th.argsort(uncertainties, descending=True)
        
        return sorted_indices
        
        
        

        