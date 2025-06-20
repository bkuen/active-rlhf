from abc import ABC, abstractmethod
import torch as th
from active_rlhf.data.buffers import ReplayBufferBatch, TrajectoryPairBatch

class Selector(ABC):
    """Abstract base class for trajectory pair selection strategies."""
    
    @abstractmethod
    def select_pairs(self, batch: ReplayBufferBatch, num_pairs: int, global_step: int) -> TrajectoryPairBatch:
        """Select pairs of trajectories from a batch.
        
        Args:
            batch: Batch of trajectories to select pairs from
            num_pairs: Number of pairs to select
            global_step: Current training step for logging or other purposes
            
        Returns:
            TrajectoryPairBatch containing the selected pairs
        """
        pass

class RandomSelector(Selector):
    """Randomly selects pairs of trajectories from a batch."""
    
    def select_pairs(self, batch: ReplayBufferBatch, num_pairs: int, global_step: int) -> TrajectoryPairBatch:
        """Randomly select pairs of trajectories from a batch.
        
        Args:
            batch: Batch of trajectories to select pairs from
            num_pairs: Number of pairs to select
            global_step: Current training step for logging or other purposes
            
        Returns:
            TrajectoryPairBatch containing randomly selected pairs
        """
        # Randomly select indices for first and second trajectories
        first_indices = th.randperm(num_pairs)
        
        # Create second indices by shifting the first indices by a random offset
        # This ensures no self-pairing while maintaining randomness
        shift = th.randint(1, num_pairs, (1,)).item()
        second_indices = (first_indices + shift) % num_pairs
        
        return TrajectoryPairBatch(
            first_obs=batch.obs[first_indices],
            first_acts=batch.acts[first_indices],
            first_rews=batch.rews[first_indices],
            first_dones=batch.dones[first_indices],
            second_obs=batch.obs[second_indices],
            second_acts=batch.acts[second_indices],
            second_rews=batch.rews[second_indices],
            second_dones=batch.dones[second_indices]
        )


class RandomSelectorSimple(Selector):
    """Randomly selects pairs of trajectories from a batch."""

    def select_pairs(self, batch: ReplayBufferBatch, num_pairs: int, global_step: int) -> TrajectoryPairBatch:
        """Randomly select pairs of trajectories from a batch.

        Args:
            batch: Batch of trajectories to select pairs from
            num_pairs: Number of pairs to select
            global_step: Current training step for logging or other purposes

        Returns:
            TrajectoryPairBatch containing randomly selected pairs
        """
        batch_size = batch.obs.shape[0]
        half_batch_size = batch_size // 2
        assert half_batch_size == num_pairs, "Batch size must be even and equal to 2 * num_pairs"

        return TrajectoryPairBatch(
            first_obs=batch.obs[:half_batch_size],
            first_acts=batch.acts[:half_batch_size],
            first_rews=batch.rews[:half_batch_size],
            first_dones=batch.dones[:half_batch_size],
            second_obs=batch.obs[half_batch_size:],
            second_acts=batch.acts[half_batch_size:],
            second_rews=batch.rews[half_batch_size:],
            second_dones=batch.dones[half_batch_size:]
        )
