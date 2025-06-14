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

@dataclass
class RolloutBufferSample:
    obs: th.Tensor
    actions: th.Tensor
    logprobs: th.Tensor
    ground_truth_rewards: th.Tensor
    dones: th.Tensor
    values: th.Tensor

class ReplayBuffer:
    obs: th.Tensor
    acts: th.Tensor
    rews: th.Tensor
    dones: th.Tensor

    def __init__(self, envs: SyncVectorEnv, capacity: int = 500000, fragment_length: int = 50):
        self.envs = envs
        self.capacity = capacity
        self.fragment_length = fragment_length
        self.size = 0
        self.pos = 0
        self.obs = th.zeros((capacity, envs.single_observation_space.shape[0]))
        self.acts = th.zeros((capacity, envs.single_action_space.shape[0]))
        self.rews = th.zeros((capacity, 1))
        self.dones = th.zeros((capacity, 1), dtype=th.bool)

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

            self.size = min(self.size + fragment_length, self.capacity)
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

    def add_rollout(self, rollout: RolloutBufferSample):
        obs = rollout.obs.reshape((-1,) + self.envs.single_observation_space.shape)
        acts = rollout.actions.reshape((-1,) + self.envs.single_action_space.shape)
        rews = rollout.ground_truth_rewards.reshape(-1)
        dones = rollout.dones.reshape(-1)

        print("Rollout observations shape:", obs.shape)
        print("Rollout actions shape:", acts.shape)
        print("Rollout rewards shape:", rews.shape)
        print("Rollout dones shape:", dones.shape)

        """Add a rollout of transitions to the buffer."""
        self.add(
            obs=obs,
            act=acts,
            rew=rews,  # Ensure rew is of shape (fragment_length, 1)
            done=dones,  # Ensure done is of shape (fragment_length, 1)
        )


    def sample(self, batch_size: int) -> ReplayBufferSample:
        """Sample a batch of transitions from the buffer."""
        max_size = min(self.size, self.capacity)
        start_indices = th.randint(0, max_size, (batch_size,))
        indices = (start_indices[:, None] + th.arange(self.fragment_length)[None, :]) % max_size
        return ReplayBufferSample(
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

    def get_batch(self) -> RolloutBufferSample:
        return RolloutBufferSample(
            obs=self.obs,
            actions=self.actions,
            logprobs=self.logprobs,
            ground_truth_rewards=self.ground_truth_rewards,
            dones=self.dones,
            values=self.values
        )

    def get_flattened_batch(self) -> RolloutBufferSample:
        """Returns a flattened version of the buffer."""
        return RolloutBufferSample(
            obs=self.obs.reshape((-1,) + self.envs.single_observation_space.shape),
            actions=self.actions.reshape((-1,) + self.envs.single_action_space.shape),
            logprobs=self.logprobs.reshape(-1),
            ground_truth_rewards=self.ground_truth_rewards.reshape(-1),
            dones=self.dones.reshape(-1),
            values=self.values.reshape(-1),
        )
