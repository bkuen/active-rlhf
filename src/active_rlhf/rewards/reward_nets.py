import os
from typing import List, Optional

from tqdm import tqdm
import torch as th
import torch.utils.data as th_data
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from active_rlhf.data import dataset
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
        
    def loss(self, probs: th.Tensor, prefs: th.Tensor) -> th.Tensor:
        """Compute the cross entropy loss for the preference model.
        
        Args:
            probs: The probability distribution over preferences of shape (batch_size, ensemble_size, 2).
            prefs: The ground truth preference distribution of shape (batch_size, 2).

        Returns:
            The average loss across all ensemble members.
        """
        # # Expand preferences to match ensemble size
        # prefs = prefs.unsqueeze(1).expand(-1, probs.shape[1], -1)  # shape (batch_size, ensemble_size, 2)
        # # Compute cross entropy loss for each ensemble member
        # probs = probs.mean(dim=2)
        # losses = -th.sum(prefs * th.log(probs + 1e-8), dim=-1)  # shape (batch_size, ensemble_size)
        # print("Prefs shape:", prefs.shape)
        # print("Probs shape:", probs.shape)
        #
        # # Average across ensemble members
        # return losses.mean()
        prefs = prefs.unsqueeze(1).expand(-1, probs.shape[1], -1)  # (B,E,2)

        log_probs = th.log(probs + 1e-8)  # (B,E,2)
        losses = -(prefs * log_probs).sum(-1)  # (B,E)
        return losses.mean()

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

    def train(self, dataset: dataset.PreferenceDataset, global_step: int):
        # Split dataset into train and validation using torch's random_split
        dataset_size = len(dataset)
        val_size = int(dataset_size * self.val_split)
        train_size = dataset_size - val_size
        
        train_dataset, val_dataset = th.utils.data.random_split(
            dataset, 
            [train_size, val_size],
            generator=th.Generator().manual_seed(42)  # For reproducibility
        )
        
        train_dataloader = make_dataloader(train_dataset, self.batch_size)
        val_dataloader = make_dataloader(val_dataset, self.batch_size)

        total_train_loss = 0.0
        total_val_loss = 0.0
        total_reward_accuracy = 0.0

        for epoch in tqdm(range(1, self.epochs+1), desc="Training Reward Model"):
            # Training phase
            self.preference_model.train()
            for batch in train_dataloader:
                # Forward pass
                first_rews, second_rews, probs = self.preference_model(
                    first_obs=batch.first_obs,
                    first_acts=batch.first_acts,
                    second_obs=batch.second_obs,
                    second_acts=batch.second_acts,
                )

                # Compute loss
                loss = self.preference_model.loss(probs, batch.prefs)
                total_train_loss += loss.item()

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.preference_model.ensemble.parameters(), max_norm=1.0)
                self.optimizer.step()

            # Validation phase
            self.preference_model.eval()
            with th.no_grad():
                for batch in val_dataloader:
                    # Forward pass
                    first_rews, second_rews, probs = self.preference_model(
                        first_obs=batch.first_obs,
                        first_acts=batch.first_acts,
                        second_obs=batch.second_obs,
                        second_acts=batch.second_acts,
                    )

                    # Compute validation loss
                    val_loss = self.preference_model.loss(probs, batch.prefs)
                    total_val_loss += val_loss.item()

                    # Calculate reward accuracy using already computed rewards
                    reward_accuracy = self._calculate_reward_accuracy(first_rews, second_rews, batch)
                    total_reward_accuracy += reward_accuracy

        # Calculate averages
        avg_train_loss = total_train_loss / len(train_dataloader)
        avg_val_loss = total_val_loss / len(val_dataloader)
        avg_reward_accuracy = total_reward_accuracy / (self.epochs * len(val_dataloader))

        # Log metrics to TensorBoard
        self.writer.add_scalar("reward/train_loss", avg_train_loss, global_step)
        self.writer.add_scalar("reward/val_loss", avg_val_loss, global_step)
        self.writer.add_scalar("reward/accuracy", avg_reward_accuracy, global_step)

        # Print metrics
        tqdm.write(f"Epoch {epoch}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Reward Accuracy: {avg_reward_accuracy:.4f}")

    def _make_dataloader(self, dataset: dataset.PreferenceDataset) -> th_data.DataLoader:
        return th_data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)