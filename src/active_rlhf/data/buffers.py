from typing import TypedDict

import torch as th
from gymnasium.vector import SyncVectorEnv
from dataclasses import dataclass

@dataclass
class ReplayBufferSample:
    obs: th.Tensor
    acts: th.Tensor
    rews: th.Tensor
    dones: th.Tensor

class ReplayBuffer:
    obs: th.Tensor
    acts: th.Tensor
    rews: th.Tensor
    dones: th.Tensor

    def __init__(self, capacity: int = 500000):
        self.capacity = capacity
        self.size = 0
        self.obs = th.zeros((capacity, 1))
        self.acts = th.zeros((capacity, 1))
        self.rews = th.zeros((capacity, 1))
        self.dones = th.zeros((capacity, 1), dtype=th.bool)

    def add(self, obs: th.Tensor, act: th.Tensor, rew: th.Tensor, done: th.Tensor):
        """Add a new transition to the buffer."""
        idx = self.size % self.capacity
        self.obs[idx] = obs
        self.acts[idx] = act
        self.rews[idx] = rew
        self.dones[idx] = done
        self.size += 1

    def sample(self, batch_size: int) -> ReplayBufferSample:
        """Sample a batch of transitions from the buffer."""
        max_size = min(self.size, self.capacity)
        indices = th.randint(0, max_size, (batch_size,))
        return ReplayBufferSample(
            obs=self.obs[indices],
            acts=self.acts[indices],
            rews=self.rews[indices],
            dones=self.dones[indices]
        )
    
@dataclass
class RolloutBufferSample:
    obs: th.Tensor
    actions: th.Tensor
    logprobs: th.Tensor
    ground_truth_rewards: th.Tensor
    dones: th.Tensor
    values: th.Tensor

class RolloutBuffer:
    def __init__(self,
                 num_steps: int,
                 num_envs: int,
                 envs: SyncVectorEnv,
                 device: str = 'cpu'):
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

    def get_batch(self) -> RolloutBufferSample:
        return RolloutBufferSample(
            obs=self.obs,
            actions=self.actions,
            logprobs=self.logprobs,
            ground_truth_rewards=self.ground_truth_rewards,
            dones=self.dones,
            values=self.values
        )
