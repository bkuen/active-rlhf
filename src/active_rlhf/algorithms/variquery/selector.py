import os
from typing import List, Tuple, Optional

import torch
import umap
from matplotlib import pyplot as plt
from matplotlib.patches import ConnectionPatch

from active_rlhf.algorithms.variquery.vae import MLPStateVAE, VAETrainer, GRUStateVAE, AttnStateVAE, \
    EnhancedGRUStateVAE, MLPStateSkipVAE, ConvStateVAE
from active_rlhf.algorithms.variquery.visualizer import VAEVisualizer
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import silhouette_score

from active_rlhf.data.running_stats import RunningStat
from active_rlhf.queries.uncertainty import estimate_uncertainties, estimate_epistemic_uncertainties
from active_rlhf.rewards.reward_nets import RewardEnsemble, PreferenceModel
import numpy as np
import torch as th

from active_rlhf.data.buffers import TrajectoryPairBatch, ReplayBufferBatch
from active_rlhf.queries.selector import Selector
from torch.utils.tensorboard import SummaryWriter

from active_rlhf.video import video


class VARIQuerySelector(Selector):
    """Selector for the VariQuery algorithm, which selects pairs of trajectories based on a specific strategy."""

    def __init__(self,
                 writer: SummaryWriter,
                 reward_ensemble: RewardEnsemble,
                 preference_model: PreferenceModel,
                 reward_norm: RunningStat,
                 vae: ConvStateVAE,
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
                 vae_conv_kernel_size: int = 5,
                 vae_attn_dim: int = 128,
                 vae_attn_heads: int = 4,
                 vae_attn_blocks: int = 2,
                 vae_decoder_layers: int = 2,
                 vae_noise_sigma: float = 0.0,
                 total_steps: int = 1_000_000,
                 cluster_size: int = 10,
                 env_id: str = "HalfCheetah-v4",
                 device: str = "cuda" if th.cuda.is_available() else "cpu"
                 ):
        self.writer = writer
        self.reward_ensemble = reward_ensemble
        self.preference_model = preference_model
        self.reward_norm = reward_norm
        self.fragment_length = fragment_length
        self.vae = vae
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
        self.vae_conv_kernel_size = vae_conv_kernel_size
        self.vae_attn_dim = vae_attn_dim
        self.vae_attn_heads = vae_attn_heads
        self.vae_attn_blocks = vae_attn_blocks
        self.vae_decoder_layers = vae_decoder_layers
        self.vae_noise_sigma = vae_noise_sigma
        self.cluster_size = cluster_size
        self.total_steps = total_steps
        self.env_id = env_id
        self.device = device

        # self.vae = (MLPStateSkipVAE(
        #     state_dim=vae_state_dim,
        #     latent_dim=self.vae_latent_dim,
        #     fragment_length=self.fragment_length,
        #     hidden_dims=self.vae_hidden_dims,
        #     dropout=self.vae_dropout,
        # ))

        # self.vae = ConvStateVAE(
        #     state_dim=vae_state_dim,
        #     latent_dim=vae_latent_dim,
        #     hidden_dims=vae_hidden_dims,
        #     dropout=vae_dropout,
        #     device=device,
        #     kernel_size=vae_conv_kernel_size,
        #     # padding=args.vae_conv_padding,
        #     fragment_length=fragment_length,
        # )

        # self.vae = AttnStateVAE(
        #     state_dim=vae_state_dim,
        #     latent_dim=self.vae_latent_dim,
        #     fragment_length=self.fragment_length,
        #     # hidden_dims=vae_hidden_dims,
        #     # dropout=self.vae_dropout,
        #     attn_dim=vae_attn_dim,
        #     n_heads=vae_attn_heads,
        #     n_blocks=vae_attn_blocks,
        #     n_decoder_layers=vae_decoder_layers,
        #     attn_dropout=vae_dropout,
        #     device=device,
        # )

        # self.vae_trainer = VAETrainer(
        #     vae=self.vae,
        #     lr=self.vae_lr,
        #     weight_decay=self.vae_weight_decay,
        #     kl_warmup_epochs=self.vae_kl_warmup_epochs,
        #     kl_warmup_steps=self.vae_kl_warmup_steps,
        #     batch_size=self.vae_batch_size,
        #     num_epochs=self.vae_num_epochs,
        #     early_stopping_patience=self.vae_early_stopping_patience,
        #     noise_sigma=self.vae_noise_sigma,
        #     total_steps=self.total_steps,
        #     device=self.device
        # )

        self.visualizer = VAEVisualizer(writer=writer)
        

    def select_pairs(self, train_batch: ReplayBufferBatch, val_batch: ReplayBufferBatch, num_pairs: int, global_step: int) -> TrajectoryPairBatch:
        batch_size = train_batch.obs.shape[0]
        # self.vae = (MLPStateSkipVAE(
        #     state_dim=self.vae_state_dim,
        #     latent_dim=self.vae_latent_dim,
        #     fragment_length=self.fragment_length,
        #     hidden_dims=self.vae_hidden_dims,
        #     dropout=self.vae_dropout,
        # ))
        #
        # self.vae_trainer = VAETrainer(
        #     vae=self.vae,
        #     lr=self.vae_lr,
        #     weight_decay=self.vae_weight_decay,
        #     kl_warmup_epochs=self.vae_kl_warmup_epochs,
        #     kl_warmup_steps=self.vae_kl_warmup_steps,
        #     batch_size=self.vae_batch_size,
        #     num_epochs=self.vae_num_epochs,
        #     early_stopping_patience=self.vae_early_stopping_patience,
        #     noise_sigma=self.vae_noise_sigma,
        #     total_steps=self.total_steps,
        #     device=self.device
        # )

        # Step 2: Train VAE and encode states
        # metrics = self.vae_trainer.train(train_batch, val_batch, global_step)

        with th.no_grad():
            latent_states, mu, logvar = self.vae.encode(train_batch.obs)
            recon_states = self.vae.decode(latent_states)

        # Step 3: Cluster and sample pairs
        clusters = self._cluster_latent_space_knn(latent_states=latent_states, num_clusters=self.cluster_size, global_step=global_step)

        # Step 4: Sample pairs
        first_indices, second_indices = self._sample_random_pair_indices(clusters=clusters, num_pairs=batch_size//2)
        assert len(first_indices) == len(second_indices)

        # Step 5: Rank by uncertainty estimate
        with th.no_grad():
            rewards = self.reward_ensemble(train_batch.obs, train_batch.acts)
            # rewards = rewards.mean(dim=-1).sum(dim=-1)  # Aggregate rewards over time steps
        #     # rewards_norm = self.reward_norm(rewards)
        # ranked_pair_indices = self._rank_pairs(rewards, first_indices, second_indices).to(self.device)

        mean_rewards = rewards.mean(dim=-1)  # Mean rewards over ensemble dimension
        returns = mean_rewards.sum(dim=-1) # Sum over time steps to get total return
        avg_return = returns.mean(dim=0)
        max_return = returns.max(dim=0)
        min_return = returns.min(dim=0)

        self.writer.add_scalar("variquery2/avg_return", avg_return.item(), global_step)
        self.writer.add_scalar("variquery2/max_return", max_return.values.item(), global_step)
        self.writer.add_scalar("variquery2/min_return", min_return.values.item(), global_step)

        # Alternative: Use DUO uncertainty estimation
        with th.no_grad():
            first_rews, second_rews, probs = self.preference_model(
                train_batch.obs[first_indices],
                train_batch.acts[first_indices],
                train_batch.obs[second_indices],
                train_batch.acts[second_indices]
            )

        epistemic_uncertainties = estimate_epistemic_uncertainties(probs)
        ranked_pair_indices = th.argsort(epistemic_uncertainties, descending=True)

        top_indices = ranked_pair_indices[:num_pairs].to(self.device)
        top_first_indices = first_indices[top_indices].to(self.device)
        top_second_indices = second_indices[top_indices].to(self.device)

        # Visualizations

        num_show = 5

        # 0. Log latent states

        # Visualize latent space
        umap_mapper = umap.UMAP(
            n_neighbors=15,
            min_dist=0.1,
            metric='cosine',
            random_state=42
        )

        try:
            embedded = umap_mapper.fit_transform(latent_states)

            # Normalize the embedding to improve visualization
            embedded = (embedded - embedded.min(axis=0)) / (embedded.max(axis=0) - embedded.min(axis=0))

            # Create plot with improved styling
            fig0 = plt.figure(figsize=(12, 8))
            plt.style.use('default')  # Reset to default style
            # Set background color and grid
            plt.gca().set_facecolor('#f0f0f0')
            plt.grid(True, linestyle='--', alpha=0.7)

            # Plot each cluster with different colors and improved visibility
            colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728', '#9467bd',
                      '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']  # Better color palette

            # First plot all points with lower alpha for context
            plt.scatter(embedded[:, 0], embedded[:, 1], c='gray', alpha=0.1, s=50)
            self.writer.add_figure('variquery2/latent_space', fig0, global_step)
            plt.close(fig0)

        except Exception as e:
            print(f"UMAP visualization failed: {str(e)}")

        # 1. 2D Scatter: wrap your matplotlib figure
        latent_data = mu.cpu().numpy()
        latent_2d = latent_data[:, :2]
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(latent_2d[:, 0], latent_2d[:, 1], alpha=0.6, s=50)
        ax.set(xlabel='Latent 1', ylabel='Latent 2',
               title=f'VAE Latent Space (2D) — step {global_step}')
        self.writer.add_figure('variquery2/latent_space_scatter2d', fig, global_step)
        plt.close(fig)

        # 2. Heatmap: log the matplotlib figure

        fig2 = plt.figure(figsize=(12, 8))
        plt.imshow(latent_data.T, aspect='auto', cmap='viridis')
        plt.colorbar(label='Latent Value')
        plt.xlabel('Sample Index')
        plt.ylabel('Latent Dimension')
        self.writer.add_figure('variquery2/latent_space_heatmap', fig2, global_step)
        plt.close(fig2)
        # fig2, ax2 = plt.subplots(...)
        # ax2.imshow(latent_data.T, aspect='auto', cmap='viridis')
        # writer.add_figure('latent_space/heatmap_matplotlib', fig2, global_step)
        # plt.close(fig2)

        # 3. Reconstructions: stack originals & recons into a single figure
        fig3, axes = plt.subplots(num_show, 2, figsize=(8, 2 * num_show))
        for i in range(num_show):
            axes[i, 0].plot(train_batch.obs[i].cpu())
            axes[i, 0].set_title(f'Orig {i}')
            axes[i, 1].plot(recon_states[i].cpu())
            axes[i, 1].set_title(f'Recon {i}')
        plt.tight_layout()
        self.writer.add_figure('variquery2/reconstructions_comparison', fig3, global_step)
        plt.close(fig3)

        # 4. Reconstruction Error Distribution
        # you can log the raw array as a histogram:
        recon_error = (train_batch.obs - recon_states).abs().cpu().numpy()
        fig4 = plt.figure(figsize=(10, 6))
        plt.hist(recon_error.flatten(), bins=50, alpha=0.7, edgecolor='black')
        plt.xlabel('Absolute Reconstruction Error')
        plt.ylabel('Frequency')
        plt.title('Distribution of Reconstruction Errors')
        plt.grid(True, alpha=0.3)
        self.writer.add_figure('variquery2/reconstruction_error_distribution', fig4, global_step)
        plt.close(fig4)

        # 5. KL per dimension
        # Approach A: histogram of per-dim KLs
        kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        kl_per_dim_mean = kl_per_dim.mean(dim=0).cpu().numpy()
        fig5 = plt.figure(figsize=(10, 6))
        plt.bar(range(len(kl_per_dim_mean)), kl_per_dim_mean)
        plt.xlabel('Latent Dimension')
        plt.ylabel('Average KL Divergence')
        plt.title('KL Divergence per Latent Dimension')
        plt.grid(True, alpha=0.3)
        self.writer.add_figure('variquery2/kl_div_per_latent_dim', fig5, global_step)
        plt.close(fig5)

        # Visualize latent space and clusters
        self._plot_latent_clusters(latent_states, clusters, top_first_indices, top_second_indices, global_step=global_step)

        # Step 6: Visualize latent space and clusters
        # returns = rewards.mean(dim=-1).sum(dim=-1)
        # self.visualizer.visualize(
        #     metrics=metrics,
        #     latents=latent_states,
        #     rewards=returns,
        #     first_indices=top_first_indices,
        #     second_indices=top_second_indices,
        #     clusters=clusters,
        #     global_step=global_step,
        # )

        # vis_obs = torch.stack([
        #     train_batch.obs[top_first_indices[0]],
        #     train_batch.obs[top_second_indices[0]],
        #     train_batch.obs[top_first_indices[1]],
        #     train_batch.obs[top_second_indices[1]]
        # ], dim=0)
        #
        # with th.no_grad():
        #     vis_obs_latent, _, _ = self.vae.encode(vis_obs)
        #     vis_obs_recon = self.vae.decode(vis_obs_latent)
        #
        # # Step 7: Visualize VAE reconstructions
        # vis_save_path = os.path.join(self.writer.log_dir, f"visualizations/original_vs_reconstructed_{global_step}.mp4")
        # os.makedirs(os.path.dirname(vis_save_path), exist_ok=True)
        # video.render_observation_comparison(vis_obs, vis_obs_recon, save_path=vis_save_path, env_id=self.env_id, num_samples=len(vis_obs))

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

    def _cluster_latent_space_knn(
            self,
            latent_states: th.Tensor,
            num_clusters: int,
            global_step: int,
            neighbor_options: List[int] = (5, 10, 20, 30)
    ) -> List[List[int]]:
        """
        Cluster the latent representations by testing multiple k-NN graph densities.
        For each n_neighbors in neighbor_options, build the k-NN affinity, run
        spectral clustering, compute silhouette score, and pick the best setting.
        """

        # 1. Prepare data
        X = latent_states.detach().cpu().numpy()
        X = (X - X.mean(axis=0)) / X.std(axis=0)

        best_score = -1.0
        best_n = neighbor_options[0]
        best_labels = None

        # 2. Loop over different neighbor counts
        for n in neighbor_options:
            # Ensure n is valid
            n_eff = min(n, X.shape[0] - 1)

            spectral = SpectralClustering(
                n_clusters=num_clusters,
                affinity='nearest_neighbors',
                n_neighbors=n_eff,
                assign_labels='kmeans',
                random_state=42
            )
            labels = spectral.fit_predict(X)

            # Evaluate
            score = silhouette_score(X, labels)
            # Log silhouette for this neighbor setting
            self.writer.add_scalar(f"variquery2/silhouette_score_knn_n{n_eff}", score, global_step)

            # Track best
            if score > best_score:
                best_score = score
                best_n = n_eff
                best_labels = labels

        # 3. Log the chosen n_neighbors
        self.writer.add_scalar("variquery2/best_n_neighbors", best_n, global_step)
        self.writer.add_scalar("variquery2/best_silhouette_score", best_score, global_step)

        # 4. Group indices by the best clustering
        clusters: List[List[int]] = [[] for _ in range(num_clusters)]
        for idx, lbl in enumerate(best_labels):
            clusters[lbl].append(idx)

        return clusters

    def _cluster_latent_space(self, latent_states: th.Tensor, num_clusters: int, global_step: int) -> List[List[int]]:
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
        score = silhouette_score(latent_states, cluster_labels)
        self.writer.add_scalar("variquery/silhouette_score", score, global_step=global_step)
        
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

    def _plot_latent_clusters(self,
                             latents: th.Tensor,
                             clusters: List[List[int]],
                             first_indices: th.Tensor,
                             second_indices: th.Tensor,
                             global_step: int):
        """Plot and save UMAP visualization of latent clusters.

        Args:
            latents: Tensor of latents
            clusters: List of latent cluster indices
            first_indices: Indices of first trajectories in pairs
            second_indices: Indices of second trajectories in pairs
            global_step: Current training step for file naming
        """
        # Convert to numpy and normalize
        latent_vectors = latents.cpu().numpy()
        n_samples = latent_vectors.shape[0]

        # Normalize vectors
        norms = np.linalg.norm(latent_vectors, axis=1, keepdims=True)
        normalized_vectors = latent_vectors / (norms + 1e-8)

        # Skip visualization if too few samples
        if n_samples < 4:
            print(f"Warning: Too few samples ({n_samples}) for meaningful UMAP visualization")
            return

        # Create UMAP embedding
        umap_mapper = umap.UMAP(
            n_neighbors=15,
            min_dist=0.1,
            metric='cosine',
            random_state=42
        )

        try:
            embedded = umap_mapper.fit_transform(normalized_vectors)

            # Normalize the embedding to improve visualization
            embedded = (embedded - embedded.min(axis=0)) / (embedded.max(axis=0) - embedded.min(axis=0))

        except Exception as e:
            print(f"UMAP visualization failed: {str(e)}")
            return

        # Create plot with improved styling
        plt.style.use('default')  # Reset to default style
        plt.figure(figsize=(12, 8))

        # Set background color and grid
        plt.gca().set_facecolor('#f0f0f0')
        plt.grid(True, linestyle='--', alpha=0.7)

        # Plot each cluster with different colors and improved visibility
        colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728', '#9467bd',
                  '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']  # Better color palette

        # First plot all points with lower alpha for context
        plt.scatter(embedded[:, 0], embedded[:, 1], c='gray', alpha=0.1, s=50)

        # Then plot clusters
        for cluster_idx, cluster in enumerate(clusters):
            if not cluster:  # Skip empty clusters
                continue
            cluster_points = embedded[cluster]
            plt.scatter(
                cluster_points[:, 0],
                cluster_points[:, 1],
                alpha=0.6,
                c=[colors[cluster_idx % len(colors)]],
                label=f'Cluster {cluster_idx} (n={len(cluster)})',
                s=100,  # Larger point size
                edgecolors='white',  # White edges for better visibility
                linewidth=0.5
            )

        # Draw connection patches between paired indices
        # Convert indices to numpy arrays if they're tensors
        first_indices = first_indices.cpu().numpy() if isinstance(first_indices, th.Tensor) else first_indices
        second_indices = second_indices.cpu().numpy() if isinstance(second_indices, th.Tensor) else second_indices

        # Draw curved arrows between pairs
        for idx1, idx2 in zip(first_indices, second_indices):
            try:
                # Create curved arrow between pairs
                con = ConnectionPatch(
                    xyA=(embedded[idx1, 0], embedded[idx1, 1]),
                    xyB=(embedded[idx2, 0], embedded[idx2, 1]),
                    coordsA="data", coordsB="data",
                    axesA=plt.gca(), axesB=plt.gca(),
                    arrowstyle="-",
                    connectionstyle="arc3,rad=0.2",
                    edgecolor='red',
                    alpha=0.5,
                    linewidth=1.5
                )
                plt.gca().add_patch(con)

                # Highlight selected points
                plt.scatter(
                    [embedded[idx1, 0], embedded[idx2, 0]],
                    [embedded[idx1, 1], embedded[idx2, 1]],
                    c='red',
                    s=150,
                    alpha=0.8,
                    zorder=5,
                    edgecolors='white',
                    linewidth=0.5
                )
            except (IndexError, ValueError) as e:
                print(f"Warning: Could not plot pair due to invalid index: {str(e)}")
                continue

        # Improve title and labels
        plt.title(f'Latent Space Clusters with Paired Connections (n_samples={n_samples})',
                  pad=20, fontsize=12, fontweight='bold')
        plt.xlabel('UMAP Component 1', fontsize=10)
        plt.ylabel('UMAP Component 2', fontsize=10)

        # Improve legend
        legend = plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left',
                            borderaxespad=0., frameon=True, fancybox=True, shadow=True)
        legend.get_frame().set_facecolor('white')
        legend.get_frame().set_alpha(0.8)

        # Set figure background color
        plt.gcf().patch.set_facecolor('white')

        # Add a border around the plot
        plt.gca().spines['top'].set_visible(True)
        plt.gca().spines['right'].set_visible(True)
        plt.gca().spines['bottom'].set_visible(True)
        plt.gca().spines['left'].set_visible(True)

        plt.tight_layout()  # Adjust layout to prevent label clipping

        # Log to tensorboard
        self.writer.add_figure('variquery2/vae_clusters', plt.gcf(), global_step)

        plt.close()
        

        