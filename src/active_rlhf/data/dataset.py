from dataclasses import dataclass
from typing import List, TypedDict

import torch as th
import torch.utils.data as th_data

@dataclass
class PreferenceBatch:
    first_obs: th.Tensor
    first_acts: th.Tensor
    first_rews: th.Tensor
    second_obs: th.Tensor
    second_acts: th.Tensor
    second_rews: th.Tensor
    prefs: th.Tensor  # shape: (batch_size, 2) representing distribution over {1, 2}

class PreferenceDataset(th.utils.data.Dataset):
    def __init__(self, first_obs: th.Tensor, first_acts: th.Tensor, first_rews: th.Tensor, first_dones: th.Tensor, 
                 second_obs: th.Tensor, second_acts: th.Tensor, second_rews: th.Tensor, second_dones: th.Tensor, 
                 prefs: th.Tensor):
        """Initialize the PreferenceDataset.
        Args:
            first_obs: Tensor of shape (N, obs_dim) for the first observations.
            first_acts: Tensor of shape (N, act_dim) for the first actions.
            first_rews: Tensor of shape (N, 1) for the first rewards.
            first_dones: Tensor of shape (N, 1) for the first done flags.
            second_obs: Tensor of shape (N, obs_dim) for the second observations.
            second_acts: Tensor of shape (N, act_dim) for the second actions.
            second_rews: Tensor of shape (N, 1) for the second rewards.
            second_dones: Tensor of shape (N, 1) for the second done flags.
            prefs: Tensor of shape (N, 2) representing distribution over {1, 2} where:
                  - [1, 0] means first segment is preferred
                  - [0, 1] means second segment is preferred
                  - [0.5, 0.5] means segments are equally preferable
        """
        self.first_obs = first_obs
        self.first_acts = first_acts
        self.first_rews = first_rews
        self.first_dones = first_dones
        self.second_obs = second_obs
        self.second_acts = second_acts
        self.second_rews = second_rews
        self.second_dones = second_dones
        self.prefs = prefs

    def __len__(self):
        return len(self.prefs)

    def __getitem__(self, idx: int) -> PreferenceBatch:
        return PreferenceBatch(
            first_obs=self.first_obs[idx],
            first_acts=self.first_acts[idx],
            first_rews=self.first_rews[idx],
            second_obs=self.second_obs[idx],
            second_acts=self.second_acts[idx],
            second_rews=self.second_rews[idx],
            prefs=self.prefs[idx],
        )

def make_dataloader(dataset: PreferenceDataset, batch_size: int) -> th_data.DataLoader:
    def collate_fn(batch):
        # batch is a list of PreferenceBatch objects
        return PreferenceBatch(
            first_obs=th.stack([b.first_obs for b in batch]),
            first_acts=th.stack([b.first_acts for b in batch]),
            first_rews=th.stack([b.first_rews for b in batch]),
            second_obs=th.stack([b.second_obs for b in batch]),
            second_acts=th.stack([b.second_acts for b in batch]),
            second_rews=th.stack([b.second_rews for b in batch]),
            prefs=th.stack([b.prefs for b in batch])
        )
    
    return th_data.DataLoader(
        dataset, 
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
