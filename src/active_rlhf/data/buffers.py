from typing import TypedDict, Union, Sequence

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

    def add(self, obs: th.Tensor, act: th.Tensor, rew: th.Tensor, done: th.Tensor):
        """Add a new transition to the buffer.
        Args:
            obs: The observation tensor of shape (num_steps, obs_dim).
            act: The action tensor of shape (num_steps, act_dim).
            rew: The reward tensor of shape (num_steps, 1).
            done: The done tensor of shape (num_steps, 1).
        """
        fragment_length = obs.shape[0]
        start_idx = self.pos
        end_idx = start_idx + fragment_length

        rew = rew.unsqueeze(-1)  # Ensure rew is of shape (fragment_length,)
        done = done.unsqueeze(-1)  # Ensure done is of shape (fragment_length,)

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

        self.pos = (self.pos + fragment_length) % self.capacity
        self.size = min(self.size + fragment_length, self.capacity)

    def add_rollout(self, rollout: RolloutBufferBatch):
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
