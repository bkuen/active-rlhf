import os
from typing import List, Optional

from tqdm import tqdm
import torch as th
import torch.utils.data as th_data
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from active_rlhf.data import dataset
from active_rlhf.data.buffers import PreferenceBuffer
from active_rlhf.data.dataset import make_dataloader


class RewardNet(nn.Module):
    def __init__(self,
                 obs_dim: int,
                 act_dim: int,
                 hidden_dims: List[int] = [256, 256, 256],
                 dropout: float = 0.1,
                 ):
        super(RewardNet, self).__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_dims = hidden_dims

        layers = []
        input_dim = obs_dim + act_dim
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(nn.LeakyReLU())
        layers.append(nn.Dropout(dropout))

        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            layers.append(nn.LeakyReLU())
            layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(hidden_dims[-1], 1))

        self.net = nn.Sequential(*layers)

    def forward(self, obs: th.Tensor, act: th.Tensor) -> th.Tensor:
        """Compute the reward for a given observation and action.

        Args:
            obs: The observations of shape (batch_size, fragment_size, obs_dim).
            act: The actions of shape (batch_size, fragment_size, act_dim).

        Returns:
            The rewards of shape (batch_size, fragment_size, 1).
        """
        print("RewardNet forward pass:")
        x = self.net(th.cat([obs, act], dim=-1))
        print("x shape:", x.shape)
        x = x.squeeze(-1)
        print("x shape after squeeze:", x.shape)
        return x

class RewardEnsemble(nn.Module):
    def __init__(self,
                 obs_dim: int,
                 act_dim: int,
                 ensemble_size: int = 3,
                 hidden_dims: List[int] = [256, 256, 256],
                 dropout: float = 0.1,
                 ):
        super(RewardEnsemble, self).__init__()

        self.nets = nn.ModuleList([RewardNet(
            obs_dim=obs_dim,
            act_dim=act_dim,
            hidden_dims=hidden_dims,
            dropout=dropout
        ) for _ in range(ensemble_size)])

    def forward(self, obs: th.Tensor, act: th.Tensor) -> th.Tensor:
        """Compute the rewards for a given observation and action.

        Args:
            obs: The observations of shape (batch_size, fragment_size, obs_dim).
            act: The actions of shape (batch_size, fragment_size, act_dim).

        Returns:
            The rewards of shape (batch_size, fragment_size, ensemble_size).
        """
        return th.stack([net(obs, act) for net in self.nets], dim=-1)

    def mean_reward(self, obs: th.Tensor, act: th.Tensor) -> th.Tensor:
        """Compute the mean reward across the ensemble.

        Args:
            obs: The observations of shape (batch_size, fragment_size, obs_dim).
            act: The actions of shape (batch_size, fragment_size, act_dim).

        Returns:
            The mean rewards of shape (batch_size, fragment_size).
        """
        print("Computing mean reward across ensemble members.")
        print(f"obs shape: {obs.shape}, act shape: {act.shape}")

        rewards = self.forward(obs, act)
        mean = rewards.mean(dim=-1).squeeze(-1)

        return mean
    
class PreferenceModel(nn.Module):
    def __init__(self, ensemble: RewardEnsemble):
        super(PreferenceModel, self).__init__()
        self.ensemble = ensemble
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, first_obs: th.Tensor, first_acts: th.Tensor, second_obs: th.Tensor, second_acts: th.Tensor) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        """Computes the probability distribution over preferences.

        Args:
            first_obs: The observations of the first action of shape (batch_size, fragment_size, obs_dim).
            first_acts: The actions of the first action of shape (batch_size, fragment_size, act_dim).
            second_obs: The observations of the second action of shape (batch_size, fragment_size, obs_dim).
            second_acts: The actions of the second action of shape (batch_size, fragment_size, act_dim).

        Returns:
            first_rews: The rewards of the first action of shape (batch_size, fragment_size, ensemble_size).
            second_rews: The rewards of the second action of shape (batch_size, fragment_size, ensemble_size).
            probs: The probability distribution over preferences of shape (batch_size, ensemble_size, 2).
        """
        first_rews = self.ensemble(first_obs, first_acts) # shape (batch_size, fragment_size, ensemble_size)
        second_rews = self.ensemble(second_obs, second_acts) # shape (batch_size, fragment_size, ensemble_size)
        probs = self.preference_probs(first_rews, second_rews) # shape (batch_size, ensemble_size, 2)
        return first_rews, second_rews, probs

    def preference_probs(self, first_rews: th.Tensor, second_rews: th.Tensor) -> th.Tensor:
        """Compute the probability distribution based on the Bradley-Terry model.

        Args:
            first_rews: The rewards of the first action of shape (batch_size, fragment_size, ensemble_size).
            second_rews: The rewards of the second action of shape (batch_size, fragment_size, ensemble_size).

        Returns:
            The probability distribution over preferences of shape (batch_size, ensemble_size, 2).
        """
        # sum over fragment_size
        print("Computing preference probabilities:")
        print("First rews shape:", first_rews.shape)
        print("Second rews shape:", second_rews.shape)

        first_exp = first_rews.sum(dim=1)
        second_exp = second_rews.sum(dim=1)

        print("First exp shape:", first_exp.shape)
        print("Second exp shape:", second_exp.shape)

        probs = self.softmax(th.stack([first_exp, second_exp], dim=-1)) # shape (batch_size, ensemble_size, 2)

        print("Probs shape:", probs.shape)
        return probs
        
    def loss(self, probs: th.Tensor, prefs: th.Tensor, eps: float = 1e-8) -> th.Tensor:
        """
        Cross-entropy loss between predicted preference distribution
        and ground-truth preferences.

        Args
        ----
        probs : Tensor, shape (B, E, 2)
            Output of `preference_probs` – already passed through softmax.
        prefs : Tensor, shape (B, 2)
            Target distribution over the two trajectories.  Each row is
            either one-hot ([1,0] or [0,1]) or a soft target such as [0.5,0.5].

        Returns
        -------
        Tensor (scalar): Mean loss across batch and ensemble members.
        """
        print("Computing loss for preference model:")
        print("Prefs shape:", prefs.shape)
        print("Probs shape:", probs.shape)
        # Expand targets across the ensemble dimension → (B, E, 2)
        prefs_expanded = prefs.unsqueeze(1).expand_as(probs)

        # Safe log and element-wise cross entropy
        log_probs = th.log(probs.clamp_min(eps))  # (B, E, 2)
        per_member_ce = -(prefs_expanded * log_probs).sum(dim=-1)  # (B, E)

        # First average over ensemble, then over batch
        return per_member_ce.mean()

class RewardTrainer:
    def __init__(self, 
                 preference_model: PreferenceModel,
                 writer: SummaryWriter,
                 epochs: int = 1,
                 lr: float = 1e-3,
                 weight_decay: float = 0.0,
                 batch_size: int = 32,
                 minibatch_size: Optional[int] = None,
                 val_split: float = 0.2,  # Add validation split parameter
                 ):
        self.preference_model = preference_model
        self.writer = writer
        self.optimizer = th.optim.Adam(self.preference_model.ensemble.parameters(), lr=lr, weight_decay=weight_decay)
        self.epochs = epochs
        self.batch_size = batch_size
        self.minibatch_size = minibatch_size or batch_size
        self.val_split = val_split
        if self.batch_size % self.minibatch_size != 0:
            raise ValueError("Batch size must be a multiple of minibatch size.")

    def _calculate_reward_accuracy(self, first_pred_rews: th.Tensor, second_pred_rews: th.Tensor, batch: dataset.PreferenceBatch) -> float:
        """Calculate accuracy of reward predictions compared to ground truth rewards.
        
        Args:
            first_pred_rews: Predicted rewards for first trajectory of shape (batch_size, fragment_size, 1, ensemble_size)
            second_pred_rews: Predicted rewards for second trajectory of shape (batch_size, fragment_size, 1, ensemble_size)
            batch: A PreferenceBatch containing first and second trajectories with their ground truth rewards with shapes:
                - first_rews: (batch_size, fragment_size, 1)
                - second_rews: (batch_size, fragment_size, 1)
            
        Returns:
            float: Accuracy of reward predictions (0-1)
        """
        # Calculate MSE between predicted and ground truth rewards
        print("First predicted rewards shape:", first_pred_rews.shape)
        print("Second predicted rewards shape:", second_pred_rews.shape)
        print("First ground truth rewards shape:", batch.first_rews.shape)
        print("Second ground truth rewards shape:", batch.second_rews.shape)

        with th.no_grad():
            predictions = th.stack([first_pred_rews, second_pred_rews], dim=0)
            predictions = predictions.mean(dim=-1)  # Average over ensemble members
            predictions = predictions.reshape(-1) # Flatten to 1D for comparison

            print("Predictions shape after stacking and averaging:", predictions.shape)

            ground_truth = th.stack([batch.first_rews, batch.second_rews], dim=0)
            ground_truth = ground_truth.reshape(-1)  # Flatten to 1D for comparison

            print("Ground truth shape after stacking:", ground_truth.shape)

            mse = nn.MSELoss(reduction='mean')
            accuracy = 1 / (1 + mse(predictions, ground_truth))
            accuracy = accuracy.item()
            assert 0 <= accuracy <= 1, f"Accuracy out of bounds: {accuracy}"

        return accuracy

    def train(self, train_buffer: PreferenceBuffer, val_buffer: PreferenceBuffer, global_step: int):
        """Train the reward network using batches sampled from the training buffer and validated on the validation buffer.
        
        Args:
            train_buffer: The buffer containing training preference pairs
            val_buffer: The buffer containing validation preference pairs
            global_step: Current global step for logging
        """
        total_train_loss = 0.0
        total_val_loss = 0.0
        total_reward_accuracy = 0.0

        # Calculate number of batches per epoch
        num_train_batches = max(1, len(train_buffer) // self.batch_size)
        num_val_batches = max(1, len(val_buffer) // self.batch_size)
        
        for epoch in tqdm(range(1, self.epochs+1), desc="Training Reward Model"):
            # Training phase
            self.preference_model.train()
            epoch_train_loss = 0.0
            
            for _ in range(num_train_batches):
                try:
                    # Sample a new batch for each training step
                    batch = train_buffer.sample(self.batch_size)
                    
                    # Forward pass
                    first_rews, second_rews, probs = self.preference_model(
                        first_obs=batch.first_obs,
                        first_acts=batch.first_acts,
                        second_obs=batch.second_obs,
                        second_acts=batch.second_acts,
                    )

                    # Compute loss
                    loss = self.preference_model.loss(probs, batch.prefs)
                    epoch_train_loss += loss.item()

                    # Backward pass
                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.preference_model.ensemble.parameters(), max_norm=1.0)
                    self.optimizer.step()
                except ValueError as e:
                    print(f"Warning: {e}. Skipping this training batch.")
                    continue

            # Validation phase
            self.preference_model.eval()
            epoch_val_loss = 0.0
            epoch_reward_accuracy = 0.0
            num_val_batches_processed = 0
            
            with th.no_grad():
                for _ in range(num_val_batches):
                    try:
                        # Sample a new batch for validation
                        batch = val_buffer.sample(self.batch_size)
                        
                        # Forward pass
                        first_rews, second_rews, probs = self.preference_model(
                            first_obs=batch.first_obs,
                            first_acts=batch.first_acts,
                            second_obs=batch.second_obs,
                            second_acts=batch.second_acts,
                        )

                        # Compute validation loss
                        val_loss = self.preference_model.loss(probs, batch.prefs)
                        epoch_val_loss += val_loss.item()

                        # Calculate reward accuracy
                        reward_accuracy = self._calculate_reward_accuracy(first_rews, second_rews, batch)
                        epoch_reward_accuracy += reward_accuracy
                        num_val_batches_processed += 1
                    except ValueError as e:
                        print(f"Warning: {e}. Skipping this validation batch.")
                        continue

            # Calculate epoch averages
            avg_train_loss = epoch_train_loss / num_train_batches
            avg_val_loss = epoch_val_loss / num_val_batches_processed if num_val_batches_processed > 0 else 0
            avg_reward_accuracy = epoch_reward_accuracy / num_val_batches_processed if num_val_batches_processed > 0 else 0

            # Update running totals
            total_train_loss += avg_train_loss
            total_val_loss += avg_val_loss
            total_reward_accuracy += avg_reward_accuracy

            # Log metrics to TensorBoard
            self.writer.add_scalar("reward/train_loss", avg_train_loss, global_step)
            self.writer.add_scalar("reward/val_loss", avg_val_loss, global_step)
            self.writer.add_scalar("reward/accuracy", avg_reward_accuracy, global_step)

            # Print metrics
            tqdm.write(f"Epoch {epoch}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Reward Accuracy: {avg_reward_accuracy:.4f}")

        # Calculate final averages
        final_train_loss = total_train_loss / self.epochs
        final_val_loss = total_val_loss / self.epochs
        final_reward_accuracy = total_reward_accuracy / self.epochs

        return final_train_loss, final_val_loss, final_reward_accuracy

    def _make_dataloader(self, dataset: dataset.PreferenceDataset) -> th_data.DataLoader:
        return th_data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)