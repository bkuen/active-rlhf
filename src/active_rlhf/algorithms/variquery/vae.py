from dataclasses import dataclass
from typing import List, Tuple
import torch as th
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from tqdm import tqdm

from active_rlhf.data.buffers import ReplayBufferBatch

@dataclass
class VAEMetrics:
    recon_loss: float
    kl_loss: float
    total_loss: float
    latent_means: th.Tensor  # Store latent means for visualization


class ReplayBufferDataset(Dataset):
    def __init__(self, buffer_batch: ReplayBufferBatch):
        self.obs = buffer_batch.obs
        self.acts = buffer_batch.acts
        self.rews = buffer_batch.rews
        self.dones = buffer_batch.dones

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        return {
            'obs': self.obs[idx],
            'acts': self.acts[idx],
            'rews': self.rews[idx],
            'dones': self.dones[idx]
        }


class StateVAE(nn.Module):
    def __init__(self, 
                 state_dim: int,
                 latent_dim: int, 
                 fragment_length: int,
                 hidden_dims: List[int] = [128, 64, 32],
                 dropout: float = 0.1):
        super().__init__()
        self.state_dim = state_dim
        self.flat_dim = fragment_length * state_dim
        self.latent_dim = latent_dim
        self.fragment_length = fragment_length
        self.hidden_dims = hidden_dims
        self.dropout = dropout

        self.encoder = self._create_encoder()
        self.decoder = self._create_decoder()

        # Latent space projections
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)

    def _create_encoder(self):
        encoder_layers = []
        in_dim = self.flat_dim
        for hidden_dim in self.hidden_dims:
            encoder_layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
            ])
            in_dim = hidden_dim
        return nn.Sequential(*encoder_layers)
    
    def _create_decoder(self):
        decoder_layers = []
        in_dim = self.latent_dim
        for hidden_dim in reversed(self.hidden_dims):
            decoder_layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout),
            ])
            in_dim = hidden_dim
        final_layer = nn.Linear(in_dim, self.flat_dim)
        decoder_layers.extend([
            final_layer,
        ])

        return nn.Sequential(*decoder_layers)
    
    def encode(self, x: th.Tensor) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        """Encode state fragments into latent space.
        
        Args:
            x: (batch_size, fragment_length, state_dim)

        Returns:
            mu: (batch_size, latent_dim)
            log_var: (batch_size, latent_dim)
            z: (batch_size, latent_dim)
        """
        # Flatten input: (batch_size, fragment_length * state_dim)
        B = x.shape[0]
        x = x.view(B, -1)  # Flatten to (batch_size, fragment_length * state_dim)

        # Pass through encoder
        hidden = self.encoder(x)

        mu = self.fc_mu(hidden)
        log_var = self.fc_logvar(hidden)

        z = self.reparameterize(mu, log_var)

        return z, mu, log_var
    
    def decode(self, z: th.Tensor) -> th.Tensor:
        """Decode latent space samples into state fragments.
        
        Args:
            z: (batch_size, latent_dim)

        Returns:
            x_hat: (batch_size, fragment_length, state_dim)
        """
        x_hat = self.decoder(z)

        # Reshape to (batch_size, fragment_length, state_dim)
        x_hat = x_hat.view(-1, self.fragment_length, self.state_dim)

        return x_hat

    def reparameterize(self, mu: th.Tensor, log_var: th.Tensor) -> th.Tensor:
        """
        Reparameterization trick to sample from the latent space.
        """
        std = th.exp(0.5 * log_var)
        eps = th.randn_like(std)
        return mu + eps * std
    
    def forward(self, x: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """Forward pass through the VAE.

        Args:
            x: (batch_size, fragment_length, state_dim)

        Returns:
            x_hat: (batch_size, fragment_length, state_dim)
            mu: (batch_size, latent_dim)
            log_var: (batch_size, latent_dim)
        """
        z, mu, log_var = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, mu, log_var

class VAETrainer:
    def __init__(self,
                 vae: StateVAE,
                 lr: float = 1e-3,
                 weight_decay: float = 1e-4,
                 batch_size: int = 32,
                 num_epochs: int = 25,
                 device: str = "cuda" if th.cuda.is_available() else "cpu"):
        self.vae = vae
        self.device = device
        self.optimizer = th.optim.Adam(self.vae.parameters(), lr=lr, weight_decay=weight_decay)
        self.batch_size = batch_size
        self.num_epochs = num_epochs

    def train(self, 
              buffer_batch: ReplayBufferBatch, 
              global_step: int) -> List[VAEMetrics]:
        """Train the VAE on a batch of trajectories.
        
        Args:
            buffer_batch: ReplayBufferBatch
            global_step: int

        Returns:
            List of VAEMetrics for each epoch
        """
        dataset = ReplayBufferDataset(buffer_batch)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        epoch_metrics = []

        for epoch in tqdm(range(self.num_epochs), desc="Training VAE"):
            epoch_total_loss = 0.0
            epoch_recon_loss = 0.0
            epoch_kl_loss = 0.0
            all_latent_means = []

            for batch in dataloader:
                x = batch['obs'].to(self.device)

                # Forward pass
                x_hat, mu, log_var = self.vae(x)

                # Compute losses
                total_loss, recon_loss, kl_loss = self._loss(x, x_hat, mu, log_var)

                # Backward pass
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

                # Accumulate losses
                epoch_total_loss += total_loss.item()
                epoch_recon_loss += recon_loss.item()
                epoch_kl_loss += kl_loss.item()
                all_latent_means.append(mu.detach().cpu())

            # Average losses over batches
            num_batches = len(dataloader)
            epoch_metrics.append(VAEMetrics(
                recon_loss=epoch_recon_loss / num_batches,
                kl_loss=epoch_kl_loss / num_batches,
                total_loss=epoch_total_loss / num_batches,
                latent_means=th.cat(all_latent_means, dim=0)
            ))

        return epoch_metrics

    def _loss(self, 
              x: th.Tensor, 
              x_hat: th.Tensor,
              mu: th.Tensor,
              logvar: th.Tensor,
              kl_weight_beta: float = 1.0
              ) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """Compute the VAE loss.
        
        Args:
            x: (batch_size, fragment_length, state_dim)
            x_hat: (batch_size, fragment_length, state_dim)
            mu: (batch_size, latent_dim)
            logvar: (batch_size, latent_dim)
            kl_weight_beta: float

        Returns:
            total_loss: (batch_size)
            recon_loss: (batch_size)
            kl_loss: (batch_size)
        """
        B, _, _ = x.shape
        recon_loss = F.mse_loss(x_hat, x, reduction="sum") / B

        kl_loss = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(-1).mean()

        total_loss = recon_loss + kl_weight_beta * kl_loss

        return total_loss, recon_loss, kl_loss
        