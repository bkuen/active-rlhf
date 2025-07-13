#!/usr/bin/env python3
"""
Script to train a VAE on saved replay buffer data.

This script demonstrates how to:
1. Load a saved replay buffer
2. Train a VAE on the collected trajectory data
3. Visualize the VAE's latent space

Usage:
    python train_vae_on_replay_buffer.py --replay_buffer_path runs/your_run/replay_buffer.pkl
"""

import os
import argparse

import imageio
import numpy as np
import torch as th
import gymnasium as gym
from gymnasium import Env
from gymnasium.vector import SyncVectorEnv
import moviepy
from torch.utils.tensorboard import SummaryWriter
import tyro
from dataclasses import dataclass, field
from typing import List, Optional
import matplotlib.pyplot as plt

from active_rlhf.data.buffers import ReplayBuffer
from active_rlhf.algorithms.variquery.vae import MLPStateVAE, MLPStateSkipVAE, AttnStateVAE, GRUStateVAE, \
    EnhancedGRUStateVAE, ConvStateVAE


@dataclass
class Args:
    replay_buffer_path: str
    """path to the saved replay buffer file"""
    
    # VAE training parameters
    vae_latent_dim: int = 32
    """dimension of the VAE latent space"""
    vae_hidden_dims: List[int] = field(default_factory=lambda: [128, 64, 32])
    """hidden dimensions of the VAE encoder/decoder"""
    vae_lr: float = 1e-3
    """learning rate for the VAE"""
    vae_weight_decay: float = 1e-4
    """weight decay for the VAE"""
    vae_batch_size: int = 32
    """batch size for VAE training"""
    vae_num_epochs: int = 50
    """number of epochs to train the VAE"""
    vae_dropout: float = 0.1
    """dropout rate for the VAE"""
    vae_kl_weight: float = 1.0
    """weight of the KL loss term in VAE training"""
    vae_kl_warmup_epochs: int = 10
    """number of epochs to warm up the KL loss term in VAE training"""
    vae_early_stopping_patience: Optional[int] = 10
    """number of epochs to wait for early stopping in VAE training"""
    vae_attention_dim: int = 128
    """dimension of the attention layer in the VAE"""
    vae_attention_heads: int = 4
    """number of attention heads in the VAE"""
    vae_attention_blocks: int = 2
    """number of attention blocks in the VAE"""
    vae_decoder_layers: int = 2
    """number of layers in the VAE decoder"""
    vae_conv_kernel_size: int = 3
    """kernel size for the convolutional layers in the VAE"""
    vae_conv_padding: int = 1
    """padding for the convolutional layers in the VAE"""
    
    # Training settings
    device: str = "cuda" if th.cuda.is_available() else "cpu"
    """device to use for training"""
    seed: int = 42
    """random seed"""
    fragment_length: int = 50
    """length of each trajectory fragment"""
    save_vae: bool = True
    """whether to save the trained VAE"""
    vae_save_path: Optional[str] = None
    """path to save the trained VAE (if None, will use default path)"""
    
    # Visualization
    create_visualizations: bool = True
    """whether to create VAE visualizations"""
    num_visualization_samples: int = 1000
    """number of samples to use for visualization"""


def make_env(env_id: str):
    """Create a simple environment for loading the replay buffer."""
    env = gym.make(env_id)
    env = gym.wrappers.FlattenObservation(env)
    return env


def main():
    args = tyro.cli(Args, config=(tyro.conf.FlagConversionOff,))
    
    # Set random seed
    th.manual_seed(args.seed)
    if th.cuda.is_available():
        th.cuda.manual_seed_all(args.seed)
    
    print(f"Loading replay buffer from {args.replay_buffer_path}")
    
    # Create a simple environment (we only need it for the observation/action spaces)
    # We'll try to infer the environment ID from the replay buffer path
    run_name = os.path.basename(os.path.dirname(args.replay_buffer_path))
    env_id = run_name.split('__')[0]  # Extract env_id from run_name
    
    print(f"Inferred environment ID: {env_id}")
    
    try:
        env = make_env(env_id)
        envs = gym.vector.SyncVectorEnv([lambda: make_env(env_id) for _ in range(1)])
    except Exception as e:
        print(f"Warning: Could not create environment {env_id}: {e}")
        print("You may need to specify the correct environment ID manually.")
        return
    
    # Load the replay buffer
    try:
        replay_buffer = ReplayBuffer.load(args.replay_buffer_path, envs, fragment_length=args.fragment_length, device=args.device)
    except Exception as e:
        print(f"Error loading replay buffer: {e}")
        return
    
    # Get buffer statistics
    stats = replay_buffer.get_trajectory_statistics()
    print(f"Buffer statistics: {stats}")
    
    # Check if we have enough data
    if stats['num_train_trajectories'] == 0:
        print("No training trajectories found. Trying to use all trajectories...")
        if stats['num_trajectories'] == 0:
            print("No trajectories found in the buffer!")
            return
        # Use all trajectories for training
        train_split = None
    else:
        train_split = "train"
    
    # Create output directory
    output_dir = os.path.dirname(args.replay_buffer_path)
    if args.vae_save_path is None:
        vae_save_path = os.path.join(output_dir, "trained_vae.pth")
    else:
        vae_save_path = args.vae_save_path
    
    # Create tensorboard writer
    writer = SummaryWriter(os.path.join(output_dir, "vae_training"))
    
    # Initialize VAE
    print("Initializing VAE...")
    state_dim = envs.single_observation_space.shape[0]
    # vae = MLPStateVAE(
    #     state_dim=state_dim,
    #     latent_dim=args.vae_latent_dim,
    #     hidden_dims=args.vae_hidden_dims,
    #     dropout=args.vae_dropout,
    #     device=args.device,
    #     fragment_length=args.fragment_length,
    # )

    vae = ConvStateVAE(
        state_dim=state_dim,
        latent_dim=args.vae_latent_dim,
        hidden_dims=args.vae_hidden_dims,
        dropout=args.vae_dropout,
        device=args.device,
        kernel_size=args.vae_conv_kernel_size,
        # padding=args.vae_conv_padding,
        fragment_length=args.fragment_length,
    )

    # vae = AttnStateVAE(
    #     state_dim=state_dim,
    #     latent_dim=args.vae_latent_dim,
    #     fragment_length=args.fragment_length,
    #     attn_dim=args.vae_attention_dim,
    #     n_heads=args.vae_attention_heads,
    #     n_blocks=args.vae_attention_blocks,
    #     n_decoder_layers=args.vae_decoder_layers,
    #     attn_dropout=args.vae_dropout,
    #     device=args.device,
    # )
    
    # Train VAE
    print("Training VAE...")
    
    # Debug: Check data shapes
    print(f"State dimension: {state_dim}")
    print(f"Expected VAE input shape: (batch_size, {args.fragment_length}, {state_dim})")
    
    # Create a simple training loop since the VAE trainer expects specific data format
    optimizer = th.optim.Adam(vae.parameters(), lr=args.vae_lr, weight_decay=args.vae_weight_decay)
    
    for epoch in range(args.vae_num_epochs):
        print(f"Epoch {epoch + 1}/{args.vae_num_epochs}")
        
        # Sample training and validation batches
        train_batch = replay_buffer.sample2(args.vae_batch_size, split=train_split)
        val_batch = replay_buffer.sample2(args.vae_batch_size, split="val")
        
        # Debug: Check batch shapes
        if epoch == 0:
            print(f"Train batch shape: {train_batch.obs.shape}")
            print(f"Val batch shape: {val_batch.obs.shape}")
        
        # Training
        vae.train()
        optimizer.zero_grad()
        
        # Forward pass
        x_hat, mu, log_var = vae(train_batch.obs)
        
        # Compute losses
        recon_loss = th.nn.functional.mse_loss(x_hat, train_batch.obs)
        kl_loss = -0.5 * th.sum(1 + log_var - mu.pow(2) - log_var.exp()) / train_batch.obs.shape[0]
        total_loss = recon_loss + args.vae_kl_weight * kl_loss
        
        # Backward pass
        total_loss.backward()
        optimizer.step()
        
        # Validation
        vae.eval()
        with th.no_grad():
            val_x_hat, val_mu, val_log_var = vae(val_batch.obs)
            val_recon_loss = th.nn.functional.mse_loss(val_x_hat, val_batch.obs)
            val_kl_loss = -0.5 * th.sum(1 + val_log_var - val_mu.pow(2) - val_log_var.exp()) / val_batch.obs.shape[0]
            val_total_loss = val_recon_loss + args.vae_kl_weight * val_kl_loss
        
        # Log metrics
        writer.add_scalar("vae/train_recon_loss", recon_loss.item(), epoch)
        writer.add_scalar("vae/train_kl_loss", kl_loss.item(), epoch)
        writer.add_scalar("vae/train_total_loss", total_loss.item(), epoch)
        writer.add_scalar("vae/val_recon_loss", val_recon_loss.item(), epoch)
        writer.add_scalar("vae/val_kl_loss", val_kl_loss.item(), epoch)
        writer.add_scalar("vae/val_total_loss", val_total_loss.item(), epoch)
        
        print(f"  Train - Recon: {recon_loss.item():.4f}, KL: {kl_loss.item():.4f}, Total: {total_loss.item():.4f}")
        print(f"  Val   - Recon: {val_recon_loss.item():.4f}, KL: {val_kl_loss.item():.4f}, Total: {val_total_loss.item():.4f}")
    
    # Save VAE if requested
    if args.save_vae:
        os.makedirs(os.path.dirname(vae_save_path), exist_ok=True)
        th.save({
            'vae_state_dict': vae.state_dict(),
            'vae_config': {
                'state_dim': state_dim,
                'latent_dim': args.vae_latent_dim,
                'hidden_dims': args.vae_hidden_dims,
                'dropout': args.vae_dropout,
            },
            'training_args': vars(args),
        }, vae_save_path)
        print(f"VAE saved to {vae_save_path}")
    
    # Create visualizations if requested
    if args.create_visualizations:
        print("Creating VAE visualizations...")
        
        # Create visualizations directory
        viz_dir = os.path.join(output_dir, "vae_visualizations")
        os.makedirs(viz_dir, exist_ok=True)

        num_samples = min(args.num_visualization_samples, args.vae_batch_size)
        # Sample some data for visualization
        sample_batch = replay_buffer.sample2(num_samples, split=train_split)
        sample_obs = sample_batch.obs
        
        # Create simple visualizations
        with th.no_grad():
            # Encode and decode samples
            z, mu, logvar = vae.encode(sample_obs)
            recon = vae.decode(z)

            # Print video of first observation
            print("Rendering first observation as video...")
            render_observation(sample_obs[0], os.path.join(viz_dir, "first_observation.mp4"), env_id)
            
            # Render comparison video showing original vs reconstructed observations
            print("Rendering comparison video (original vs reconstructed)...")
            render_observation_comparison(sample_obs, recon, os.path.join(viz_dir, "original_vs_reconstructed.mp4"), env_id, num_samples=5)
            
            # 1. Latent space visualization (2D scatter plot of first 2 dimensions)
            plt.figure(figsize=(10, 8))
            latent_2d = mu.cpu().numpy()[:, :2]  # Take first 2 dimensions
            plt.scatter(latent_2d[:, 0], latent_2d[:, 1], alpha=0.6, s=50)
            plt.xlabel('Latent Dimension 1')
            plt.ylabel('Latent Dimension 2')
            plt.title(f'VAE Latent Space (First 2 Dimensions)\n{num_samples} samples')
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(viz_dir, "latent_space_2d.png"), dpi=300, bbox_inches='tight')
            plt.close()
            
            # 2. Latent space heatmap (all dimensions)
            plt.figure(figsize=(12, 8))
            latent_data = mu.cpu().numpy()
            plt.imshow(latent_data.T, aspect='auto', cmap='viridis')
            plt.colorbar(label='Latent Value')
            plt.xlabel('Sample Index')
            plt.ylabel('Latent Dimension')
            plt.title(f'VAE Latent Space Heatmap\n{num_samples} samples, {args.vae_latent_dim} dimensions')
            plt.savefig(os.path.join(viz_dir, "latent_space_heatmap.png"), dpi=300, bbox_inches='tight')
            plt.close()
            
            # 3. Reconstruction comparison (first few samples)
            num_show = min(5, num_samples)
            fig, axes = plt.subplots(num_show, 2, figsize=(12, 3*num_show))
            if num_show == 1:
                axes = axes.reshape(1, -1)
            
            for i in range(num_show):
                # Original
                axes[i, 0].plot(sample_obs[i].cpu().numpy())
                axes[i, 0].set_title(f'Sample {i+1}: Original')
                axes[i, 0].set_ylabel('State Value')
                axes[i, 0].grid(True, alpha=0.3)
                
                # Reconstruction
                axes[i, 1].plot(recon[i].cpu().numpy())
                axes[i, 1].set_title(f'Sample {i+1}: Reconstruction')
                axes[i, 1].set_ylabel('State Value')
                axes[i, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, "reconstructions.png"), dpi=300, bbox_inches='tight')
            plt.close()
            
            # 4. Reconstruction error distribution
            recon_error = (sample_obs - recon).abs().cpu().numpy()
            plt.figure(figsize=(10, 6))
            plt.hist(recon_error.flatten(), bins=50, alpha=0.7, edgecolor='black')
            plt.xlabel('Absolute Reconstruction Error')
            plt.ylabel('Frequency')
            plt.title('Distribution of Reconstruction Errors')
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(viz_dir, "reconstruction_errors.png"), dpi=300, bbox_inches='tight')
            plt.close()
            
            # 5. KL divergence per dimension
            kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
            kl_per_dim_mean = kl_per_dim.mean(dim=0).cpu().numpy()
            
            plt.figure(figsize=(10, 6))
            plt.bar(range(len(kl_per_dim_mean)), kl_per_dim_mean)
            plt.xlabel('Latent Dimension')
            plt.ylabel('Average KL Divergence')
            plt.title('KL Divergence per Latent Dimension')
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(viz_dir, "kl_per_dimension.png"), dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Visualizations saved to {viz_dir}")
            
            # Log some metrics
            # Reconstruction loss
            recon_loss = th.nn.functional.mse_loss(recon, sample_obs)
            writer.add_scalar("vae/final_reconstruction_loss", recon_loss.item(), 0)
            
            # KL divergence
            kl_loss = kl_per_dim.sum(dim=1).mean()
            writer.add_scalar("vae/final_kl_loss", kl_loss.item(), 0)
            
            print(f"Final reconstruction loss: {recon_loss.item():.4f}")
            print(f"Final KL loss: {kl_loss.item():.4f}")
    
    writer.close()
    print("VAE training completed!")

def render_observation_frame(env: Env, obs: th.Tensor):
    mujoco_env = env.unwrapped
    is_mujoco = hasattr(mujoco_env, 'set_state') and hasattr(mujoco_env, 'data')

    frames = []
    if is_mujoco:
        # Dynamically determine qpos/qvel split
        model = getattr(mujoco_env, 'model', None)
        if model is not None and hasattr(model, 'nq') and hasattr(model, 'nv'):
            nq = model.nq
            nv = model.nv
        else:
            # Fallback: try to infer from data shapes
            nq = mujoco_env.data.qpos.shape[0]
            nv = mujoco_env.data.qvel.shape[0]
        for i in range(len(obs)):
            obs_i = obs[i].cpu().numpy() if hasattr(obs[i], 'cpu') else np.array(obs[i])
            qpos = mujoco_env.data.qpos.copy()
            qvel = mujoco_env.data.qvel.copy()
            # Try to fill qpos and qvel from obs
            if obs_i.shape[0] == nq + nv - 1:
                # Some envs (like HalfCheetah) hide root x (qpos[0])
                qpos[1:] = obs_i[:nq-1]
                qvel[:] = obs_i[nq-1:]
            elif obs_i.shape[0] == nq + nv:
                # Full state in obs
                qpos[:] = obs_i[:nq]
                qvel[:] = obs_i[nq:]
            else:
                # Fallback: try to fill as much as possible
                qpos[:min(nq, obs_i.shape[0])] = obs_i[:min(nq, obs_i.shape[0])]
            mujoco_env.set_state(qpos, qvel)
            frame = env.render()
            frames.append(frame)
    else:
        # Non-MuJoCo: just reset and render each obs (best effort)
        print("Warning: Environment does not support MuJoCo-style state setting. Rendering resets only.")
        for i in range(len(obs)):
            env.reset()
            frame = env.render()
            frames.append(frame)

    return frames

def render_observation(obs: th.Tensor, save_path: str, env_id: str):
    """
    Render a sequence of observations as a video, supporting MuJoCo and non-MuJoCo environments.
    For MuJoCo envs, sets the state using qpos/qvel. For others, just resets and renders.
    """
    import numpy as np
    env = gym.make(env_id, render_mode="rgb_array")
    env.reset()

    frames = render_observation_frame(env, obs)

    # Save frames as a video
    imageio.mimsave(save_path, frames, fps=30, codec="libx264")

def render_observation_comparison(original_obs: th.Tensor, recon_obs: th.Tensor, save_path: str, env_id: str, num_samples: int = 5):
    """
    Render a comparison video showing original vs reconstructed observations for multiple samples.
    Creates a video with num_samples rows, each showing original (left) vs reconstructed (right).
    """
    # Limit number of samples to available data
    num_samples = min(num_samples, original_obs.shape[0], recon_obs.shape[0])
    
    env = gym.make(env_id, render_mode="rgb_array")
    env.reset()
    
    # Get trajectory length
    traj_length = original_obs.shape[1]
    
    # For each timestep, create a frame showing all samples side by side
    frames = []
    for t in range(traj_length):
        # Create a list to store frames for each sample
        sample_frames = []
        
        for sample_idx in range(num_samples):
            # Get original and reconstructed observations for this timestep
            orig_obs_t = original_obs[sample_idx, t].unsqueeze(0)  # Add batch dimension
            recon_obs_t = recon_obs[sample_idx, t].unsqueeze(0)   # Add batch dimension
            
            # Render original observation
            orig_frames = render_observation_frame(env, orig_obs_t)
            orig_frame = orig_frames[0] if orig_frames else env.render()
            
            # Render reconstructed observation
            recon_frames = render_observation_frame(env, recon_obs_t)
            recon_frame = recon_frames[0] if recon_frames else env.render()
            
            # Concatenate original and reconstructed frames horizontally
            combined_frame = np.concatenate([orig_frame, recon_frame], axis=1)
            sample_frames.append(combined_frame)
        
        # Concatenate all samples vertically
        if sample_frames:
            combined_frame = np.concatenate(sample_frames, axis=0)
            frames.append(combined_frame)
    
    # Save frames as a video
    if frames:
        imageio.mimsave(save_path, frames, fps=30, codec="libx264")
        print(f"Comparison video saved to {save_path}")
    else:
        print("No frames generated for comparison video")
    
    env.close()

if __name__ == "__main__":
    main() 