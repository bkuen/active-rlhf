# Replay Buffer Saving and Analysis

This document explains how to save replay buffers during training and analyze them afterwards for VAE training and data exploration.

## Overview

The replay buffer now supports saving and loading functionality, allowing you to:
1. Save the replay buffer at the end of training
2. Load the saved buffer for analysis
3. Train VAEs on the collected data
4. Analyze trajectory statistics and data distributions

## Saving the Replay Buffer

### During Training

The replay buffer is automatically saved at the end of training if `save_replay_buffer=True` (default). You can control this behavior with the following arguments:

```bash
python src/active_rlhf/scripts/pref_ppo.py \
    --env_id HalfCheetah-v5 \
    --total_timesteps 1000000 \
    --save_replay_buffer True \
    --replay_buffer_save_path /path/to/custom/save/location.pkl
```

**Arguments:**
- `save_replay_buffer`: Whether to save the replay buffer (default: True)
- `replay_buffer_save_path`: Custom path to save the buffer (optional, defaults to `runs/{run_name}/replay_buffer.pkl`)

### What Gets Saved

The saved replay buffer contains:
- All observations, actions, rewards, and done flags
- Trajectory metadata (start positions, lengths, splits, on-policiness scores)
- Buffer configuration (capacity, fragment length, etc.)
- Environment information (observation and action space shapes)

## Loading and Analyzing the Replay Buffer

### Basic Analysis

Use the analysis script to explore the saved data:

```bash
python src/active_rlhf/scripts/analyze_replay_buffer.py \
    --replay_buffer_path runs/HalfCheetah-v5__pref_ppo__1__1234567890/replay_buffer.pkl
```

This will create visualizations including:
- Trajectory length distributions
- On-policiness score distributions
- Train/validation split analysis
- Reward distributions
- Observation dimension analysis

### Training a VAE on the Data

Train a VAE on the collected trajectory data:

```bash
python src/active_rlhf/scripts/train_vae_on_replay_buffer.py \
    --replay_buffer_path runs/HalfCheetah-v5__pref_ppo__1__1234567890/replay_buffer.pkl \
    --vae_latent_dim 32 \
    --vae_num_epochs 50 \
    --create_visualizations True
```

**Key VAE Parameters:**
- `vae_latent_dim`: Dimension of the latent space
- `vae_hidden_dims`: Hidden layer dimensions for encoder/decoder
- `vae_num_epochs`: Number of training epochs
- `vae_kl_weight`: Weight of KL divergence loss
- `create_visualizations`: Whether to create latent space and reconstruction visualizations

### VAE Visualizations

When `create_visualizations=True`, the script creates several types of visualizations:

1. **Latent Space 2D Scatter Plot** (`latent_space_2d.png`): Shows the first two dimensions of the latent space
2. **Latent Space Heatmap** (`latent_space_heatmap.png`): Visualizes all latent dimensions as a heatmap
3. **Reconstruction Comparison** (`reconstructions.png`): Shows original vs reconstructed trajectories
4. **Reconstruction Error Distribution** (`reconstruction_errors.png`): Histogram of reconstruction errors
5. **KL Divergence per Dimension** (`kl_per_dimension.png`): Shows how much each latent dimension is used

## Programmatic Usage

### Loading a Replay Buffer

```python
from active_rlhf.data.buffers import ReplayBuffer
import gymnasium as gym

# Create environment (needed for loading)
envs = gym.vector.SyncVectorEnv([lambda: gym.make("HalfCheetah-v5") for _ in range(1)])

# Load the replay buffer
replay_buffer = ReplayBuffer.load(
    "runs/HalfCheetah-v5__pref_ppo__1__1234567890/replay_buffer.pkl",
    envs=envs,
    device="cuda"
)

# Get statistics
stats = replay_buffer.get_trajectory_statistics()
print(f"Number of trajectories: {stats['num_trajectories']}")
print(f"Mean trajectory length: {stats['mean_length']:.2f}")
```

### Extracting Data for VAE Training

```python
# Get all training trajectories
train_data = replay_buffer.get_all_trajectories(split="train")
print(f"Training data shape: {train_data.obs.shape}")

# Get all validation trajectories
val_data = replay_buffer.get_all_trajectories(split="val")
print(f"Validation data shape: {val_data.obs.shape}")

# Get all trajectories (both train and val)
all_data = replay_buffer.get_all_trajectories(split=None)
print(f"All data shape: {all_data.obs.shape}")
```

### Sampling from the Buffer

```python
# Sample random fragments
batch = replay_buffer.sample2(batch_size=32, split="train")

# Sample using on-policiness priority (requires an agent)
batch = replay_buffer.sample_by_on_policiness(
    batch_size=32, 
    agent=your_agent, 
    split="train"
)
```

## File Structure

After running the analysis and VAE training scripts, you'll have:

```
runs/HalfCheetah-v5__pref_ppo__1__1234567890/
├── replay_buffer.pkl                    # Saved replay buffer
├── analysis/                            # Analysis plots
│   ├── trajectory_lengths.png
│   ├── on_policiness_scores.png
│   ├── split_distribution.png
│   ├── reward_distribution.png
│   └── observation_dimensions.png
├── vae_training/                        # VAE training logs
│   └── events.out.tfevents...
├── vae_visualizations/                  # VAE visualizations
│   ├── latent_space_2d.png
│   ├── latent_space_heatmap.png
│   ├── reconstructions.png
│   ├── reconstruction_errors.png
│   └── kl_per_dimension.png
└── trained_vae.pth                      # Trained VAE model
```

## Tips for VAE Training

1. **Data Quality**: The quality of your VAE depends on the quality of the collected data. Make sure your agent has explored the environment well.

2. **Latent Dimension**: Start with a small latent dimension (16-32) and increase if needed. Too large latent spaces can lead to poor reconstructions.

3. **Training Epochs**: Monitor the reconstruction and KL losses. Stop training when they stabilize.

4. **Batch Size**: Use larger batch sizes if memory allows for more stable training.

5. **KL Weight**: The KL weight controls the trade-off between reconstruction quality and latent space structure. Start with 1.0 and adjust as needed.

## Troubleshooting

### Common Issues

1. **Environment Mismatch**: If you get environment-related errors when loading, make sure you're using the same environment version that was used during training.

2. **Memory Issues**: For large replay buffers, consider using CPU for analysis or sampling smaller batches.

3. **Missing Dependencies**: Make sure you have matplotlib installed for visualizations:
   ```bash
   pip install matplotlib
   ```

4. **Index Out of Bounds Errors**: If you encounter index errors when sampling from a loaded replay buffer, this was likely due to circular buffer wrapping issues that have been fixed in recent versions.

### Testing Replay Buffer Loading

Before training a VAE, you can test that your replay buffer loads and samples correctly:

```bash
python src/active_rlhf/scripts/test_replay_buffer_loading.py \
    --replay_buffer_path runs/HalfCheetah-v5__pref_ppo__1__1234567890/replay_buffer.pkl
```

This script will:
- Load the replay buffer
- Test all sampling methods
- Verify that no index errors occur
- Display buffer statistics

### Getting Help

If you encounter issues:
1. Check that the replay buffer file exists and is not corrupted
2. Verify that you're using compatible versions of the environment
3. Check the console output for specific error messages
4. Ensure you have sufficient disk space for saving the buffer and visualizations 