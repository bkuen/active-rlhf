import os
import pickle
from random import random
from typing import TypedDict, Union, Sequence, List, Optional, Literal
import numpy as np
import torch as th
from gymnasium.vector import SyncVectorEnv
from dataclasses import dataclass

@dataclass
class ReplayBufferBatch:
    obs: th.Tensor
    acts: th.Tensor
    rews: th.Tensor
    dones: th.Tensor

@dataclass
class RolloutBufferBatch:
    obs: th.Tensor
    actions: th.Tensor
    logprobs: th.Tensor
    ground_truth_rewards: th.Tensor
    dones: th.Tensor
    values: th.Tensor

@dataclass
class TrajectoryPairBatch:
    first_obs: th.Tensor
    first_acts: th.Tensor
    first_rews: th.Tensor
    first_dones: th.Tensor
    second_obs: th.Tensor
    second_acts: th.Tensor
    second_rews: th.Tensor
    second_dones: th.Tensor

    def __len__(self) -> int:
        """Return the number of pairs in the batch."""
        return self.first_obs.shape[0]

    Index = Union[
        int,  # e.g. 3
        slice,  # e.g. 2:10
        Sequence[int],  # e.g. [0, 4, 7]
        th.Tensor  # 1-D bool or long tensor
    ]

    def __getitem__(self, idx: Index) -> 'TrajectoryPairBatch':
        return TrajectoryPairBatch(
            first_obs=self.first_obs[idx],
            first_acts=self.first_acts[idx],
            first_rews=self.first_rews[idx],
            first_dones=self.first_dones[idx],
            second_obs=self.second_obs[idx],
            second_acts=self.second_acts[idx],
            second_rews=self.second_rews[idx],
            second_dones=self.second_dones[idx],
        )


@dataclass
class PreferenceBufferBatch(TrajectoryPairBatch):
    prefs: th.Tensor

@dataclass
class TrajectoryInfo:
    """Information about a trajectory stored in the replay buffer."""
    start_pos: int
    length: int
    split: Literal["train", "val"]
    on_policiness_score: float
    last_updated_step: int

class ReplayBuffer:
    obs: th.Tensor
    acts: th.Tensor
    rews: th.Tensor
    dones: th.Tensor

    def __init__(self, envs: SyncVectorEnv, capacity: int = 500000, fragment_length: int = 50, device: str = "cuda" if th.cuda.is_available() else "cpu"):
        self.envs = envs
        self.capacity = capacity
        self.fragment_length = fragment_length
        self.size = 0
        self.pos = 0
        self.obs = th.zeros((capacity, envs.single_observation_space.shape[0]), device=device)
        self.acts = th.zeros((capacity, envs.single_action_space.shape[0]), device=device)
        self.rews = th.zeros((capacity, 1), device=device)
        self.dones = th.zeros((capacity, 1), dtype=th.bool, device=device)
        
        # Trajectory tracking
        self.trajectories: List[TrajectoryInfo] = []
        self.device = device
        self.global_step = 0

    def save(self, filepath: str) -> None:
        """Save the replay buffer to a file.
        
        Args:
            filepath: Path where to save the replay buffer.
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Prepare data for saving (move to CPU to avoid device issues)
        save_data = {
            'capacity': self.capacity,
            'fragment_length': self.fragment_length,
            'size': self.size,
            'pos': self.pos,
            'global_step': self.global_step,
            'obs': self.obs.cpu(),
            'acts': self.acts.cpu(),
            'rews': self.rews.cpu(),
            'dones': self.dones.cpu(),
            'trajectories': self.trajectories,
            'env_obs_shape': self.envs.single_observation_space.shape,
            'env_act_shape': self.envs.single_action_space.shape,
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)
        
        print(f"Replay buffer saved to {filepath}")
        print(f"Buffer statistics: {self.get_trajectory_statistics()}")

    @classmethod
    def load(cls, filepath: str, envs: SyncVectorEnv, fragment_length: int, device: str = "cuda" if th.cuda.is_available() else "cpu") -> 'ReplayBuffer':
        """Load a replay buffer from a file.
        
        Args:
            filepath: Path to the saved replay buffer file.
            envs: The environment vector (needed for initialization).
            device: Device to load the tensors on.
            
        Returns:
            Loaded ReplayBuffer instance.
        """
        with open(filepath, 'rb') as f:
            save_data = pickle.load(f)
        
        # Create new buffer instance
        buffer = cls(envs=envs, capacity=save_data['capacity'], 
                    fragment_length=fragment_length, device=device)
        
        # Restore data
        buffer.size = save_data['size']
        buffer.pos = save_data['pos']
        buffer.global_step = save_data['global_step']
        buffer.obs = save_data['obs'].to(device)
        buffer.acts = save_data['acts'].to(device)
        buffer.rews = save_data['rews'].to(device)
        buffer.dones = save_data['dones'].to(device)
        buffer.trajectories = save_data['trajectories']
        
        print(f"Replay buffer loaded from {filepath}")
        print(f"Buffer statistics: {buffer.get_trajectory_statistics()}")
        
        return buffer

    def add(self, obs: th.Tensor, act: th.Tensor, rew: th.Tensor, done: th.Tensor, split: Literal["train", "val"] = "train"):
        """Add a new transition to the buffer.
        Args:
            obs: The observation tensor of shape (num_steps, obs_dim).
            act: The action tensor of shape (num_steps, act_dim).
            rew: The reward tensor of shape (num_steps, 1).
            done: The done tensor of shape (num_steps, 1).
            split: The split to which this trajectory belongs, either "train" or "val".
        """
        fragment_length = obs.shape[0]
        start_idx = self.pos
        end_idx = start_idx + fragment_length

        rew = rew.unsqueeze(-1)  # Ensure rew is of shape (fragment_length,)
        done = done.unsqueeze(-1)  # Ensure done is of shape (fragment_length,)

        # Clean up trajectories that will be overwritten
        self._cleanup_overwritten_trajectories(start_idx, end_idx)

        if end_idx < self.capacity:
            self.obs[start_idx:end_idx] = obs[:]
            self.acts[start_idx:end_idx] = act[:]
            self.rews[start_idx:end_idx] = rew[:]
            self.dones[start_idx:end_idx] = done[:]

        else:
            overwrite_length = end_idx - self.capacity
            self.obs[start_idx:self.capacity] = obs[:(fragment_length - overwrite_length)]
            self.acts[start_idx:self.capacity] = act[:(fragment_length - overwrite_length)]
            self.rews[start_idx:self.capacity] = rew[:(fragment_length - overwrite_length)]
            self.dones[start_idx:self.capacity] = done[:(fragment_length - overwrite_length)]

            self.obs[:overwrite_length] = obs[(fragment_length - overwrite_length):]
            self.acts[:overwrite_length] = act[(fragment_length - overwrite_length):]
            self.rews[:overwrite_length] = rew[(fragment_length - overwrite_length):]
            self.dones[:overwrite_length] = done[(fragment_length - overwrite_length):]

        # Add trajectory info
        trajectory_info = TrajectoryInfo(
            start_pos=start_idx,
            length=fragment_length,
            on_policiness_score=0.0,  # Will be updated later
            last_updated_step=self.global_step,
            split=split
        )
        self.trajectories.append(trajectory_info)

        self.pos = (self.pos + fragment_length) % self.capacity
        self.size = min(self.size + fragment_length, self.capacity)
        self.global_step += fragment_length

    def _cleanup_overwritten_trajectories(self, start_idx: int, end_idx: int) -> None:
        """Remove trajectories that will be overwritten by the new data."""
        if end_idx <= self.capacity:
            # Simple case: no wrapping
            trajectories_to_remove = []
            for trajectory in self.trajectories:
                traj_start = trajectory.start_pos
                traj_end = traj_start + trajectory.length
                
                # Check if trajectory overlaps with the new data
                if (traj_start < end_idx and traj_end > start_idx):
                    trajectories_to_remove.append(trajectory)
        else:
            # Wrapping case: new data spans from start_idx to end_idx (wrapped)
            trajectories_to_remove = []
            for trajectory in self.trajectories:
                traj_start = trajectory.start_pos
                traj_end = traj_start + trajectory.length
                
                # Check if trajectory overlaps with either part of the wrapped data
                # Part 1: start_idx to capacity
                if (traj_start < self.capacity and traj_end > start_idx):
                    trajectories_to_remove.append(trajectory)
                # Part 2: 0 to (end_idx - capacity)
                elif (traj_start < (end_idx - self.capacity) and traj_end > 0):
                    trajectories_to_remove.append(trajectory)
        
        # Remove the overlapping trajectories
        for trajectory in trajectories_to_remove:
            self.trajectories.remove(trajectory)

    def add_rollout(self, rollout: RolloutBufferBatch, split: Literal["train", "val"] = "train"):
        obs = rollout.obs.reshape((-1,) + self.envs.single_observation_space.shape)
        acts = rollout.actions.reshape((-1,) + self.envs.single_action_space.shape)
        rews = rollout.ground_truth_rewards.reshape(-1)
        dones = rollout.dones.reshape(-1)

        """Add a rollout of transitions to the buffer."""
        self.add(
            obs=obs,
            act=acts,
            rew=rews,  # Ensure rew is of shape (fragment_length, 1)
            done=dones,  # Ensure done is of shape (fragment_length, 1)
            split=split
        )

    def update_on_policiness_scores(self, agent) -> None:
        """Update on-policiness scores for all trajectories using the current agent.
        
        Args:
            agent: The current agent to compute log probabilities with.
        """
        for trajectory in self.trajectories:
            # Get trajectory data
            start_pos = trajectory.start_pos
            length = trajectory.length
            
            # Handle circular buffer wrapping
            if start_pos + length <= self.capacity:
                traj_obs = self.obs[start_pos:start_pos + length]
                traj_acts = self.acts[start_pos:start_pos + length]
            else:
                # Handle wrapping case
                first_part_length = self.capacity - start_pos
                traj_obs = th.cat([
                    self.obs[start_pos:],
                    self.obs[:length - first_part_length]
                ], dim=0)
                traj_acts = th.cat([
                    self.acts[start_pos:],
                    self.acts[:length - first_part_length]
                ], dim=0)
            
            # Compute current log probabilities for the trajectory actions
            with th.no_grad():
                _, logprobs, _, _ = agent.get_action_and_value(traj_obs, traj_acts)
            
            # Compute on-policiness score: sum of log probabilities
            on_policiness_score = logprobs.sum().item()
            trajectory.on_policiness_score = on_policiness_score
            trajectory.last_updated_step = self.global_step

    def sample_by_on_policiness(self, batch_size: int, agent, split: Literal["train", "val"] = "train") -> ReplayBufferBatch:
        """Sample fragments using on-policiness priority sampling.
        
        Args:
            batch_size: Number of fragments to sample.
            agent: Current agent to compute on-policiness scores.
            split: The split to sample from, either "train" or "val".
            
        Returns:
            ReplayBufferBatch with sampled fragments.
        """
        # Update on-policiness scores first
        self.update_on_policiness_scores(agent)
        
        # Filter out trajectories that are too short for fragment_length
        valid_trajectories = [t for t in self.trajectories if t.length >= self.fragment_length and t.split == split]
        
        if len(valid_trajectories) == 0:
            # Fallback to random sampling if no valid trajectories
            return self.sample(batch_size, split=split)
        
        # Compute on-policiness scores and rectified Z-scores
        scores = np.array([t.on_policiness_score for t in valid_trajectories])
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        
        if std_score == 0:
            # If all scores are the same, use uniform sampling
            rectified_scores = np.ones(len(valid_trajectories))
        else:
            # Compute rectified Z-scores: max(0, (O(τ) - μ_O(B)) / σ_O(B))
            z_scores = (scores - mean_score) / std_score
            rectified_scores = np.maximum(0, z_scores)
        
        # Normalize to probabilities
        if np.sum(rectified_scores) == 0:
            # If all rectified scores are 0, use uniform sampling
            probs = np.ones(len(valid_trajectories)) / len(valid_trajectories)
        else:
            probs = rectified_scores / np.sum(rectified_scores)
        
        # Sample trajectories based on on-policiness probabilities
        sampled_trajectories = np.random.choice(
            valid_trajectories, 
            size=batch_size, 
            p=probs, 
            replace=True
        )
        
        # Sample fragments from the selected trajectories
        sampled_fragments = []
        for trajectory in sampled_trajectories:
            # Sample a random starting position within the trajectory
            max_start = trajectory.length - self.fragment_length
            if max_start <= 0:
                start_offset = 0
            else:
                start_offset = np.random.randint(0, max_start + 1)
            
            # Calculate actual buffer indices
            fragment_start = (trajectory.start_pos + start_offset) % self.capacity
            fragment_end = (fragment_start + self.fragment_length) % self.capacity
            
            # Extract fragment
            if fragment_end > fragment_start:
                fragment_obs = self.obs[fragment_start:fragment_end]
                fragment_acts = self.acts[fragment_start:fragment_end]
                fragment_rews = self.rews[fragment_start:fragment_end]
                fragment_dones = self.dones[fragment_start:fragment_end]
            else:
                # Handle wrapping case
                first_part_length = self.capacity - fragment_start
                fragment_obs = th.cat([
                    self.obs[fragment_start:],
                    self.obs[:self.fragment_length - first_part_length]
                ], dim=0)
                fragment_acts = th.cat([
                    self.acts[fragment_start:],
                    self.acts[:self.fragment_length - first_part_length]
                ], dim=0)
                fragment_rews = th.cat([
                    self.rews[fragment_start:],
                    self.rews[:self.fragment_length - first_part_length]
                ], dim=0)
                fragment_dones = th.cat([
                    self.dones[fragment_start:],
                    self.dones[:self.fragment_length - first_part_length]
                ], dim=0)
            
            sampled_fragments.append({
                'obs': fragment_obs,
                'acts': fragment_acts,
                'rews': fragment_rews,
                'dones': fragment_dones
            })
        
        # Stack fragments into batch
        batch_obs = th.stack([f['obs'] for f in sampled_fragments])
        batch_acts = th.stack([f['acts'] for f in sampled_fragments])
        batch_rews = th.stack([f['rews'] for f in sampled_fragments])
        batch_dones = th.stack([f['dones'] for f in sampled_fragments])
        
        return ReplayBufferBatch(
            obs=batch_obs,
            acts=batch_acts,
            rews=batch_rews,
            dones=batch_dones
        )

    def sample(self, batch_size: int) -> ReplayBufferBatch:
        """Sample a batch of transitions from the buffer."""
        max_size = min(self.size, self.capacity)
        start_indices = th.randint(0, max_size, (batch_size,))
        indices = (start_indices[:, None] + th.arange(self.fragment_length)[None, :]) % max_size
        return ReplayBufferBatch(
            obs=self.obs[indices].view(batch_size, self.fragment_length, -1),
            acts=self.acts[indices].view(batch_size, self.fragment_length, -1),
            rews=self.rews[indices].view(batch_size, self.fragment_length, -1),
            dones=self.dones[indices].view(batch_size, self.fragment_length, -1)
        )

    def sample2(self, batch_size: int, split: Literal["train", "val"] = "train") -> ReplayBufferBatch:
        """
        Sample `batch_size` length-`fragment_length` fragments from either
        the training or validation partition.  Every fragment is fully
        contained inside ONE episode; multiple fragments can come from the
        same episode.
        """
        eps = [t for t in self.trajectories if t.length >= self.fragment_length and t.split == split]
        chosen_idx = np.random.choice(len(eps), size=batch_size, replace=True)

        # 2. within each episode choose a start offset
        start_indices = []
        for ep_idx in chosen_idx:
            ep = eps[ep_idx]
            ep_len = ep.length
            assert ep_len >= self.fragment_length, "episode length must be at least fragment_length"

            start_indices.append(ep.start_pos + np.random.randint(0, ep_len - self.fragment_length + 1))

        start_indices = th.tensor(start_indices, device=self.device)
        indices = (start_indices[:, None] + th.arange(self.fragment_length, device=self.device)[None, :])
        
        # Apply modulo operation to handle circular buffer wrapping
        indices = indices % self.capacity
        
        return ReplayBufferBatch(
            obs=self.obs[indices].view(batch_size, self.fragment_length, -1),
            acts=self.acts[indices].view(batch_size, self.fragment_length, -1),
            rews=self.rews[indices].view(batch_size, self.fragment_length, -1),
            dones=self.dones[indices].view(batch_size, self.fragment_length, -1)
        )

    def get_trajectory_statistics(self) -> dict:
        """Get statistics about stored trajectories."""
        if not self.trajectories:
            return {
                'num_trajectories': 0,
                'mean_length': 0.0,
                'mean_on_policiness_score': 0.0,
                'std_on_policiness_score': 0.0,
                'min_on_policiness_score': 0.0,
                'max_on_policiness_score': 0.0
            }
        
        lengths = [t.length for t in self.trajectories]
        scores = [t.on_policiness_score for t in self.trajectories]
        
        return {
            'num_trajectories': len(self.trajectories),
            'num_train_trajectories': len([t for t in self.trajectories if t.split == 'train']),
            'num_val_trajectories': len([t for t in self.trajectories if t.split == 'val']),
            'mean_length': np.mean(lengths),
            'mean_on_policiness_score': np.mean(scores),
            'std_on_policiness_score': np.std(scores),
            'min_on_policiness_score': np.min(scores),
            'max_on_policiness_score': np.max(scores)
        }

    def log_trajectory_statistics(self, writer, global_step: int) -> None:
        """Log trajectory statistics to tensorboard."""
        stats = self.get_trajectory_statistics()
        
        writer.add_scalar("replay_buffer/num_trajectories", stats['num_trajectories'], global_step)
        writer.add_scalar("replay_buffer/num_train_trajectories", stats['num_train_trajectories'], global_step)
        writer.add_scalar("replay_buffer/num_val_trajectories", stats['num_val_trajectories'], global_step)
        writer.add_scalar("replay_buffer/mean_trajectory_length", stats['mean_length'], global_step)
        writer.add_scalar("replay_buffer/mean_on_policiness_score", stats['mean_on_policiness_score'], global_step)
        writer.add_scalar("replay_buffer/std_on_policiness_score", stats['std_on_policiness_score'], global_step)
        writer.add_scalar("replay_buffer/min_on_policiness_score", stats['min_on_policiness_score'], global_step)
        writer.add_scalar("replay_buffer/max_on_policiness_score", stats['max_on_policiness_score'], global_step)

    def get_all_trajectories(self, split: Optional[Literal["train", "val"]] = None) -> ReplayBufferBatch:
        """Get all trajectories from the buffer, optionally filtered by split.
        
        Args:
            split: If provided, only return trajectories from this split ("train" or "val").
                  If None, return all trajectories.
                  
        Returns:
            ReplayBufferBatch containing all matching trajectories.
        """
        if split is not None:
            valid_trajectories = [t for t in self.trajectories if t.split == split]
        else:
            valid_trajectories = self.trajectories
        
        if not valid_trajectories:
            raise ValueError(f"No trajectories found for split: {split}")
        
        # Collect all trajectory fragments
        all_obs = []
        all_acts = []
        all_rews = []
        all_dones = []
        
        for trajectory in valid_trajectories:
            start_pos = trajectory.start_pos
            length = trajectory.length
            
            # Handle circular buffer wrapping
            if start_pos + length <= self.capacity:
                traj_obs = self.obs[start_pos:start_pos + length]
                traj_acts = self.acts[start_pos:start_pos + length]
                traj_rews = self.rews[start_pos:start_pos + length]
                traj_dones = self.dones[start_pos:start_pos + length]
            else:
                # Handle wrapping case
                first_part_length = self.capacity - start_pos
                traj_obs = th.cat([
                    self.obs[start_pos:],
                    self.obs[:length - first_part_length]
                ], dim=0)
                traj_acts = th.cat([
                    self.acts[start_pos:],
                    self.acts[:length - first_part_length]
                ], dim=0)
                traj_rews = th.cat([
                    self.rews[start_pos:],
                    self.rews[:length - first_part_length]
                ], dim=0)
                traj_dones = th.cat([
                    self.dones[start_pos:],
                    self.dones[:length - first_part_length]
                ], dim=0)
            
            all_obs.append(traj_obs)
            all_acts.append(traj_acts)
            all_rews.append(traj_rews)
            all_dones.append(traj_dones)
        
        # Stack all trajectories
        return ReplayBufferBatch(
            obs=th.cat(all_obs, dim=0),
            acts=th.cat(all_acts, dim=0),
            rews=th.cat(all_rews, dim=0),
            dones=th.cat(all_dones, dim=0)
        )

class RolloutBuffer:
    def __init__(self,
                 num_steps: int,
                 num_envs: int,
                 envs: SyncVectorEnv,
                 device: str = 'cpu'):
        self.envs = envs
        self.obs = th.zeros((num_steps, num_envs) + envs.single_observation_space.shape).to(device)
        self.actions = th.zeros((num_steps, num_envs) + envs.single_action_space.shape).to(device)
        self.logprobs = th.zeros((num_steps, num_envs)).to(device)
        self.ground_truth_rewards = th.zeros((num_steps, num_envs)).to(device)
        self.dones = th.zeros((num_steps, num_envs)).to(device)
        self.values = th.zeros((num_steps, num_envs)).to(device)

    def store(self, step: int, obs: th.Tensor, action: th.Tensor, logprob: th.Tensor, ground_truth_reward: th.Tensor, done: th.Tensor, value: th.Tensor):
        self.obs[step] = obs
        self.actions[step] = action
        self.logprobs[step] = logprob
        self.ground_truth_rewards[step] = ground_truth_reward
        self.dones[step] = done
        self.values[step] = value

    def get_batch(self) -> RolloutBufferBatch:
        return RolloutBufferBatch(
            obs=self.obs,
            actions=self.actions,
            logprobs=self.logprobs,
            ground_truth_rewards=self.ground_truth_rewards,
            dones=self.dones,
            values=self.values
        )

    def get_flattened_batch(self) -> RolloutBufferBatch:
        """Returns a flattened version of the buffer."""
        return RolloutBufferBatch(
            obs=self.obs.reshape((-1,) + self.envs.single_observation_space.shape),
            actions=self.actions.reshape((-1,) + self.envs.single_action_space.shape),
            logprobs=self.logprobs.reshape(-1),
            ground_truth_rewards=self.ground_truth_rewards.reshape(-1),
            dones=self.dones.reshape(-1),
            values=self.values.reshape(-1),
        )

class PreferenceBuffer(th.utils.data.Dataset):
    def __init__(self, capacity: int = 1000):
        """Initialize the preference buffer.
        
        Args:
            capacity: Maximum number of preference pairs to store.
        """
        self.capacity = capacity
        self.size = 0
        self.pos = 0
        
        # Initialize tensors to store preference pairs
        self.first_obs = []
        self.first_acts = []
        self.first_rews = []
        self.first_dones = []
        self.second_obs = []
        self.second_acts = []
        self.second_rews = []
        self.second_dones = []
        self.prefs = []

    def add(self, batch: PreferenceBufferBatch):
        """Add preference pairs to the buffer.
        
        Args:
            batch: PreferenceBufferBatch containing the preference pairs to add
        """
        batch_size = batch.first_obs.shape[0]
        
        for i in range(batch_size):
            if self.size < self.capacity:
                self.first_obs.append(batch.first_obs[i])
                self.first_acts.append(batch.first_acts[i])
                self.first_rews.append(batch.first_rews[i])
                self.first_dones.append(batch.first_dones[i])
                self.second_obs.append(batch.second_obs[i])
                self.second_acts.append(batch.second_acts[i])
                self.second_rews.append(batch.second_rews[i])
                self.second_dones.append(batch.second_dones[i])
                self.prefs.append(batch.prefs[i])
                self.size += 1
            else:
                # Replace oldest entry
                self.first_obs[self.pos] = batch.first_obs[i]
                self.first_acts[self.pos] = batch.first_acts[i]
                self.first_rews[self.pos] = batch.first_rews[i]
                self.first_dones[self.pos] = batch.first_dones[i]
                self.second_obs[self.pos] = batch.second_obs[i]
                self.second_acts[self.pos] = batch.second_acts[i]
                self.second_rews[self.pos] = batch.second_rews[i]
                self.second_dones[self.pos] = batch.second_dones[i]
                self.prefs[self.pos] = batch.prefs[i]
                self.pos = (self.pos + 1) % self.capacity

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> PreferenceBufferBatch:
        """Get a single preference pair from the buffer.
        
        Args:
            idx: Index of the preference pair to get.
            
        Returns:
            PreferenceBufferBatch containing the preference pair.
        """
        return PreferenceBufferBatch(
            first_obs=self.first_obs[idx],
            first_acts=self.first_acts[idx],
            first_rews=self.first_rews[idx],
            first_dones=self.first_dones[idx],
            second_obs=self.second_obs[idx],
            second_acts=self.second_acts[idx],
            second_rews=self.second_rews[idx],
            second_dones=self.second_dones[idx],
            prefs=self.prefs[idx]
        )
