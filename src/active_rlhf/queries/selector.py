from abc import ABC, abstractmethod
import torch as th
from active_rlhf.data.buffers import ReplayBufferBatch, TrajectoryPairBatch

class Selector(ABC):
    """Abstract base class for trajectory pair selection strategies."""
    
    @abstractmethod
    def select_pairs(self, train_batch: ReplayBufferBatch, val_batch: ReplayBufferBatch, num_pairs: int, global_step: int) -> TrajectoryPairBatch:
        """Select pairs of trajectories from a batch.
        
        Args:
            train_batch: Batch of trajectories to select pairs from
            val_batch: Validation batch of trajectories (if needed for selection strategy)
            num_pairs: Number of pairs to select
            global_step: Current training step for logging or other purposes
            
        Returns:
            TrajectoryPairBatch containing the selected pairs
        """
        pass

class RandomSelector(Selector):
    """Randomly selects pairs of trajectories from a batch."""
    
    def select_pairs(self, train_batch: ReplayBufferBatch, val_batch: ReplayBufferBatch, num_pairs: int, global_step: int) -> TrajectoryPairBatch:
        """Randomly select pairs of trajectories from a batch.
        
        Args:
            train_batch: Batch of trajectories to select pairs from
            val_batch: Validation batch of trajectories (not used in this selector)
            num_pairs: Number of pairs to select
            global_step: Current training step for logging or other purposes
            
        Returns:
            TrajectoryPairBatch containing randomly selected pairs
        """
        # Randomly select indices for first and second trajectories
        batch_size = train_batch.obs.shape[0]
        perm = th.randperm(batch_size)
        first_indices = perm[:num_pairs]
        # Create second indices by shifting the first indices by a random offset
        # This ensures no self-pairing while maintaining randomness
        shift = th.randint(1, batch_size, (1,)).item()
        second_indices = perm[(shift + th.arange(num_pairs)) % batch_size]
        
        return TrajectoryPairBatch(
            first_obs=train_batch.obs[first_indices],
            first_acts=train_batch.acts[first_indices],
            first_rews=train_batch.rews[first_indices],
            first_dones=train_batch.dones[first_indices],
            second_obs=train_batch.obs[second_indices],
            second_acts=train_batch.acts[second_indices],
            second_rews=train_batch.rews[second_indices],
            second_dones=train_batch.dones[second_indices]
        )
