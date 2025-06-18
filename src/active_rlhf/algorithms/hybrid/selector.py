from typing import List
from active_rlhf.algorithms.variquery.vae import StateVAE, VAETrainer
from active_rlhf.data.buffers import ReplayBufferBatch, TrajectoryPairBatch
import torch as th
from torch.utils.tensorboard import SummaryWriter

from active_rlhf.queries.selector import Selector, RandomSelector
from active_rlhf.rewards.reward_nets import RewardEnsemble


class HybridSelector(Selector):
    def __init__(self,
                 writer: SummaryWriter,
                 reward_ensemble: RewardEnsemble,
                 fragment_length: int = 50,
                 oversampling_factor: float = 10.0,
                 vae_latent_dim: int = 16,
                 vae_hidden_dims: List[int] = [128, 64, 32],
                 vae_lr: float = 1e-3,
                 vae_weight_decay: float = 1e-4,
                 vae_dropout: float = 0.1,
                 vae_batch_size: int = 32,
                 vae_num_epochs: int = 25,
                 gamma_z: float = 0.1,
                 gamma_r: float = 0.1,
                 device: str = "cuda" if th.cuda.is_available() else "cpu"
                 ):
        self.reward_ensemble = reward_ensemble
        self.fragment_length = fragment_length
        self.oversampling_factor = oversampling_factor
        self.vae_latent_dim = vae_latent_dim
        self.vae_hidden_dims = vae_hidden_dims
        self.vae_lr = vae_lr
        self.vae_weight_decay = vae_weight_decay
        self.vae_dropout = vae_dropout
        self.vae_batch_size = vae_batch_size
        self.vae_num_epochs = vae_num_epochs
        self.gamma_z = gamma_z
        self.gamma_r = gamma_r
        self.device = device

        self.random_selector = RandomSelector()

    def select_pairs(self, batch: ReplayBufferBatch, num_pairs: int, global_step: int) -> TrajectoryPairBatch:
        """
        Selects a batch of trajectory pairs from the replay buffer using a greedy DPP.
        First selects a larger random subset, then applies DPP selection on that subset.

        L_{ij} = exp(-gamma_z * ||z_i - z_j||^2 - gamma_r * ||Δr_i - Δr_j||^2)
        """
        # First, use random selector to get a larger subset
        batch_size = int(num_pairs * self.oversampling_factor)
        candidate_pairs = self.random_selector.select_pairs(batch, num_pairs=batch_size, global_step=global_step)

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

        metrics = vae_trainer.train(batch, global_step)

        # Encode trajectories with VAE and predict rewards
        with th.no_grad():
            # z_i, z_j: (batch_size, latent_dim)
            z_i, _, _ = vae.encode(candidate_pairs.first_obs)
            z_j, _, _ = vae.encode(candidate_pairs.second_obs)
            # r_i, r_j: (batch_size, fragment_length, ensemble_size)
            r_i = self.reward_ensemble(candidate_pairs.first_obs, candidate_pairs.first_acts)
            r_j = self.reward_ensemble(candidate_pairs.second_obs, candidate_pairs.second_acts)

        # Mean rewards over ensemble dimension: r_i, r_j: (batch_size, fragment_length)
        r_i = r_i.mean(dim=-1)
        r_j = r_j.mean(dim=-1)

        # Standardize rewards and latents
        # z_i, z_j: (batch_size, latent_dim)
        # r_i, r_j: (batch_size, fragment_length)
        z_i = (z_i - z_i.mean(dim=0)) / (z_i.std(dim=0) + 1e-8)
        z_j = (z_j - z_j.mean(dim=0)) / (z_j.std(dim=0) + 1e-8)
        r_i = (r_i - r_i.mean(dim=0)) / (r_i.std(dim=0) + 1e-8)
        r_j = (r_j - r_j.mean(dim=0)) / (r_j.std(dim=0) + 1e-8)

        # Pairwise squared distances
        Z2 = th.cdist(z_i, z_j, p=2) ** 2 # (batch_size, batch_size)
        R2 = th.cdist(r_i, r_j, p=2) ** 2 # (batch_size, batch_size)

        # L-ensemble kernel
        L = th.exp(-self.gamma_z * Z2 - self.gamma_r * R2)

        # Greedy DPP selection
        selected = []
        remaining = set(range(batch_size))
        while len(selected) < num_pairs and remaining:
            best_idx, best_ld = None, -float('inf')
            for i in list(remaining):
                S = selected + [i]
                L_sub = L[S][:, S]
                sign, logdet = th.slogdet(L_sub)
                if sign > 0 and logdet > best_ld:
                    best_ld, best_idx = logdet, i
            if best_idx is None:
                break
            selected.append(best_idx)
            remaining.remove(best_idx)

        return candidate_pairs[selected]

        

        