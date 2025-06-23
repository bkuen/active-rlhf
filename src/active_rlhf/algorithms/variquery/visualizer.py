import os
import torch as th
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import umap
from typing import List
from torch.utils.tensorboard import SummaryWriter
from matplotlib.patches import ConnectionPatch

from active_rlhf.algorithms.variquery.vae import VAEMetrics

class VAEVisualizer:
    def __init__(self, writer: SummaryWriter):
        """Initialize the VAE visualizer.
        
        Args:
            writer: TensorBoard SummaryWriter instance
        """
        self.writer = writer

    def plot_metrics(self, metrics: List[VAEMetrics], global_step: int):
        """Plot and save training metrics.
        
        Args:
            metrics: List of VAEMetrics for each epoch
            global_step: Current training step for file naming
        """
        # Extract metrics
        epochs = range(len(metrics))
        recon_losses = [m.recon_loss for m in metrics]
        kl_losses = [m.kl_loss for m in metrics]
        total_losses = [m.total_loss for m in metrics]

        # Create figure
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, recon_losses, label='Reconstruction Loss')
        plt.plot(epochs, kl_losses, label='KL Loss')
        plt.plot(epochs, total_losses, label='Total Loss')
        
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('VAE Training Losses')
        plt.legend()
        plt.grid(True)
        
        # Log to tensorboard
        self.writer.add_figure('variquery/vae_losses', plt.gcf(), global_step)
        
        # Log individual metrics
        # for epoch, (recon, kl, total) in enumerate(zip(recon_losses, kl_losses, total_losses)):
        #     self.writer.add_scalars(f"variquery/losses/epoch/{epoch}", {
        #         'reconstruction': recon,
        #         'kl': kl,
        #         'total': total
        #     }, global_step)
        
        plt.close()

    def plot_latent_heatmap(self, metrics: List[VAEMetrics], global_step: int):
        """Plot and save heatmap of kl per dimension over epochs.
        
        Args:
            metrics: List of VAEMetrics for each epoch
            global_step: Current training step for file naming
        """
        # Stack kl per dimension across epochs
        kl_per_dim = th.stack([m.latent_stats.kl_per_dim for m in metrics], dim=0)
        
        # Convert to numpy for plotting
        kl_per_dim_np = kl_per_dim.numpy()
        
        # Create heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(kl_per_dim_np.T,
                   cmap='viridis',
                   xticklabels=range(len(metrics)),
                   yticklabels=range(kl_per_dim_np.shape[1]))
        
        plt.xlabel('Epoch')
        plt.ylabel('KL Per Dimension')
        plt.title('KL Divergence Per Dimension Over Epochs')
        
        # Log to tensorboard
        self.writer.add_figure('variquery/vae_latent', plt.gcf(), global_step)

        # # Log raw latent means as a tensor
        # self.writer.add_tensor('variquery/vae_latent_means', latent_means, global_step)
        
        plt.close()

    def plot_latent_clusters(self,
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
        self.writer.add_figure('variquery/vae_clusters', plt.gcf(), global_step)
        
        plt.close()

    def plot_reward_distribution_per_cluster(self,
                                             rewards: th.Tensor,
                                             clusters: List[List[int]],
                                             global_step: int):
        """
        Plots the distribution of rewards for each cluster as a boxplot.

        Args:
            rewards (th.Tensor): 1D tensor of reward values.
            clusters (List[List[int]]): List of clusters, each is a list of indices into rewards.
            global_step (int): Current training step (for title/labeling).
        """
        # Move to CPU numpy array
        rewards_np = rewards.detach().cpu().numpy()

        # Gather reward values per cluster
        cluster_values = [rewards_np[indices] for indices in clusters]

        # Prepare labels with cluster ID and size
        labels = [f"C{i}\n(n={len(indices)})" for i, indices in enumerate(clusters)]

        # Create the plot
        fig, ax = plt.subplots(figsize=(max(6, len(clusters) * 1.5), 6))
        ax.boxplot(cluster_values, labels=labels, showfliers=False)
        ax.set_title(f"Reward Distribution per Cluster (step {global_step})")
        ax.set_xlabel("Cluster")
        ax.set_ylabel("Reward")
        ax.grid(axis="y", linestyle="--", alpha=0.7)

        plt.tight_layout()

        self.writer.add_figure('variquery/vae_cluster_rewards', plt.gcf(), global_step)

        plt.close()

    def visualize(self,
                  metrics: List[VAEMetrics],
                  latents: th.Tensor,
                  rewards: th.Tensor,
                  clusters: List[List[int]],
                  first_indices: th.Tensor,
                  second_indices: th.Tensor,
                  global_step: int):
        """Generate all visualizations for the current training step.
        
        Args:
            metrics: List of VAEMetrics for each epoch
            latents: Tensor of latent states
            rewards: Tensor of rewards corresponding to latent states
            clusters: List of clusters derived from latent states
            first_indices: Indices of first trajectories in pairs
            second_indices: Indices of second trajectories in pairs
            global_step: Current training step for file naming
        """
        self.plot_metrics(metrics, global_step)
        self.plot_latent_heatmap(metrics, global_step)
        self.plot_latent_clusters(
            latents=latents,
            clusters=clusters,
            first_indices=first_indices,
            second_indices=second_indices,
            global_step=global_step,
        )
        self.plot_reward_distribution_per_cluster(
            rewards=rewards,
            clusters=clusters,
            global_step=global_step,
        )
