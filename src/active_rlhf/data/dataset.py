from dataclasses import dataclass
from typing import List, TypedDict

import torch as th
import torch.utils.data as th_data

from active_rlhf.data.buffers import PreferenceBuffer


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
    def __init__(self, buffer: PreferenceBuffer):
        """Initialize the PreferenceDataset.
        Args:
            buffer (PreferenceBuffer): The preference buffer containing the data.
        """
        self.first_obs = th.stack(buffer.first_obs, dim=0)
        self.first_acts = th.stack(buffer.first_acts, dim=0)
        self.first_rews = th.stack(buffer.first_rews, dim=0)
        self.second_obs = th.stack(buffer.second_obs, dim=0)
        self.second_acts = th.stack(buffer.second_acts, dim=0)
        self.second_rews = th.stack(buffer.second_rews, dim=0)
        self.prefs = th.stack(buffer.prefs, dim=0)

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

def make_dataloader(dataset: PreferenceDataset, batch_size: int, device: str = "cuda" if th.cuda.is_available() else "cpu") -> th_data.DataLoader:
    def collate_fn(batch):
        # batch is a list of PreferenceBatch objects
        return PreferenceBatch(
            first_obs=th.stack([b.first_obs for b in batch]).to(device),
            first_acts=th.stack([b.first_acts for b in batch]).to(device),
            first_rews=th.stack([b.first_rews for b in batch]).to(device),
            second_obs=th.stack([b.second_obs for b in batch]).to(device),
            second_acts=th.stack([b.second_acts for b in batch]).to(device),
            second_rews=th.stack([b.second_rews for b in batch]).to(device),
            prefs=th.stack([b.prefs for b in batch])
        )
    
    return th_data.DataLoader(
        dataset, 
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
