from dataclasses import dataclass
from typing import List, Tuple, Optional
import torch as th
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from tqdm import tqdm

from active_rlhf.data.buffers import ReplayBufferBatch

@dataclass
class LatentStats:
    mu: th.Tensor
    sigma: th.Tensor
    kl_per_dim: th.Tensor  # KL divergence per latent dimension

@dataclass
class VAEMetrics:
    recon_loss: float
    kl_loss: float
    total_loss: float
    latent_stats: LatentStats


class ReplayBufferDataset(Dataset):
    def __init__(self, buffer_batch: ReplayBufferBatch):
        self.obs = buffer_batch.obs
        # self.acts = buffer_batch.acts
        # self.rews = buffer_batch.rews
        # self.dones = buffer_batch.dones

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        return {
            'obs': self.obs[idx],
            # 'acts': self.acts[idx],
            # 'rews': self.rews[idx],
            # 'dones': self.dones[idx]
        }

class GRUStateVAE(nn.Module):
    """
    Sequence-aware VAE for state fragments.
    Encodes a sequence with a stacked GRU and decodes the latent code with
    another stacked GRU that rolls the latent vector out over time.
    """

    def __init__(
        self,
        state_dim: int,
        latent_dim: int,
        fragment_length: int,
        hidden_dims: List[int] = [128, 64, 32],
        dropout: float = 0.1,
        device: str = "cuda" if th.cuda.is_available() else "cpu",
    ):
        super().__init__()
        self.state_dim = state_dim
        self.latent_dim = latent_dim
        self.fragment_length = fragment_length
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.device = device

        # ── Encoder ────────────────────────────────────────────────────────────
        # Same hidden size for every GRU layer (PyTorch limitation); we use
        # the *last* entry in hidden_dims so fc_mu/fc_logvar stay unchanged.
        enc_hidden_size = hidden_dims[-1]
        self.encoder_rnn = nn.GRU(
            input_size=state_dim,
            hidden_size=enc_hidden_size,
            num_layers=len(hidden_dims),
            batch_first=True,
            dropout=dropout if len(hidden_dims) > 1 else 0.0,
        ).to(device)

        # Latent projections
        self.fc_mu = nn.Linear(enc_hidden_size, latent_dim).to(device)
        self.fc_logvar = nn.Linear(enc_hidden_size, latent_dim).to(device)

        # ── Decoder ────────────────────────────────────────────────────────────
        # Takes the latent vector *at every timestep* as input.
        dec_hidden_size = hidden_dims[-1]
        self.decoder_rnn = nn.GRU(
            input_size=latent_dim,
            hidden_size=dec_hidden_size,
            num_layers=len(hidden_dims),
            batch_first=True,
            dropout=dropout if len(hidden_dims) > 1 else 0.0,
        ).to(device)

        # Map decoder hidden states back to state space
        self.output_layer = nn.Sequential(
            nn.Linear(dec_hidden_size, state_dim),
            nn.Tanh(),  # keep the output range in (-1, 1) like the original
        ).to(device)

    # ────────────────────────────────────────────────────────────────────────
    #                           VAE utilities
    # ────────────────────────────────────────────────────────────────────────
    @staticmethod
    def reparameterize(mu: th.Tensor, log_var: th.Tensor, device: str) -> th.Tensor:
        std = th.exp(0.5 * log_var)
        eps = th.randn_like(std, device=device)
        return mu + eps * std

    # ────────────────────────────────────────────────────────────────────────
    #                           Public API
    # ────────────────────────────────────────────────────────────────────────
    def encode(self, x: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Args
        ----
        x : Tensor, shape (B, L, state_dim)

        Returns
        -------
        z       : (B, latent_dim) — sampled latent
        mu      : (B, latent_dim)
        log_var : (B, latent_dim)
        """
        x = x.to(self.device)                               # (B, L, D)
        _, h_n = self.encoder_rnn(x)                        # h_n: (num_layers, B, H)
        h_last = h_n[-1]                                    # (B, H)

        mu = self.fc_mu(h_last)                             # (B, latent_dim)
        log_var = self.fc_logvar(h_last)                    # (B, latent_dim)
        z = self.reparameterize(mu, log_var, self.device)  # (B, latent_dim)

        return z, mu, log_var

    def decode(self, z: th.Tensor) -> th.Tensor:
        """
        Args
        ----
        z : Tensor, shape (B, latent_dim)

        Returns
        -------
        x_hat : (B, L, state_dim)
        """
        z = z.to(self.device)
        B = z.size(0)

        # Use the latent vector as the *input token* at every timestep
        z_seq = z.unsqueeze(1).repeat(1, self.fragment_length, 1)  # (B, L, latent_dim)
        noise = th.randn_like(z_seq) * 0.05
        z_seq = z_seq + noise # Add some noise to the latent vector to encourage exploration
        dec_out, _ = self.decoder_rnn(z_seq)                      # (B, L, H)

        x_hat = self.output_layer(dec_out)                        # (B, L, state_dim)
        return x_hat

    def forward(self, x: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        The familiar VAE forward pass.

        Args
        ----
        x : (B, L, state_dim)

        Returns
        -------
        x_hat : (B, L, state_dim)
        mu    : (B, latent_dim)
        log_var : (B, latent_dim)
        """
        z, mu, log_var = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, mu, log_var

class EnhancedGRUStateVAE(nn.Module):
    """
    Sequence‐aware VAE with input / output MLPs around a stacked GRU.
    - Projects each raw state up to `embed_dim` before the GRU.
    - Projects GRU outputs through an MLP before the final readout.
    """

    def __init__(
        self,
        state_dim: int,
        latent_dim: int,
        fragment_length: int,
        hidden_dims: List[int] = [64],
        embed_dim: Optional[int] = None,
        dropout: float = 0.1,
        device: str = "cuda" if th.cuda.is_available() else "cpu",
    ):
        super().__init__()
        self.state_dim = state_dim
        self.latent_dim = latent_dim
        self.fragment_length = fragment_length
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.device = device

        # If embed_dim not specified, default to last GRU hidden size
        self.embed_dim = embed_dim or hidden_dims[-1]

        # ── Input MLP ─────────────────────────────────────────────────────────
        # Projects raw state_dim → embed_dim per timestep
        self.input_proj = nn.Sequential(
            nn.Linear(state_dim, self.embed_dim),
            nn.ReLU(),
            nn.LayerNorm(self.embed_dim),
        )

        # ── Encoder GRU ───────────────────────────────────────────────────────
        enc_hidden = hidden_dims[-1]
        self.encoder_rnn = nn.GRU(
            input_size=self.embed_dim,
            hidden_size=enc_hidden,
            num_layers=len(hidden_dims),
            batch_first=True,
            dropout=dropout if len(hidden_dims) > 1 else 0.0,
        )

        # Latent projection heads
        self.fc_mu     = nn.Linear(enc_hidden, latent_dim)
        self.fc_logvar = nn.Linear(enc_hidden, latent_dim)

        # ── Decoder GRU ───────────────────────────────────────────────────────
        dec_hidden = hidden_dims[-1]
        self.decoder_rnn = nn.GRU(
            input_size=latent_dim,
            hidden_size=dec_hidden,
            num_layers=len(hidden_dims),
            batch_first=True,
            dropout=dropout if len(hidden_dims) > 1 else 0.0,
        )

        # ── Post‐RNN MLP ───────────────────────────────────────────────────────
        # Projects GRU hidden → same size before final readout
        self.post_rnn_proj = nn.Sequential(
            nn.Linear(dec_hidden, dec_hidden),
            nn.ReLU(),
            nn.LayerNorm(dec_hidden),
        )

        # ── Output readout ────────────────────────────────────────────────────
        # Projects to original state_dim, with tanh to match (-1,1) range
        self.output_layer = nn.Sequential(
            nn.Linear(dec_hidden, state_dim),
            nn.Tanh(),
        )

        # Move everything to device
        self.to(self.device)


    @staticmethod
    def reparameterize(mu: th.Tensor, log_var: th.Tensor) -> th.Tensor:
        std = th.exp(0.5 * log_var)
        eps = th.randn_like(std)
        return mu + eps * std


    def encode(self, x: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Encode a batch of fragments.
        Args:
            x: (B, L, state_dim)
        Returns:
            z      : (B, latent_dim)
            mu     : (B, latent_dim)
            log_var: (B, latent_dim)
        """
        x = x.to(self.device)                         # (B, L, D)
        h_in = self.input_proj(x)                     # (B, L, embed_dim)
        _, h_n = self.encoder_rnn(h_in)               # h_n: (num_layers, B, H)
        h_last = h_n[-1]                              # (B, H)

        mu      = self.fc_mu(h_last)                  # (B, latent_dim)
        log_var = self.fc_logvar(h_last)              # (B, latent_dim)
        z       = self.reparameterize(mu, log_var)    # (B, latent_dim)

        return z, mu, log_var


    def decode(self, z: th.Tensor) -> th.Tensor:
        """
        Decode latents back to state fragments.
        Args:
            z: (B, latent_dim)
        Returns:
            x_hat: (B, L, state_dim)
        """
        z = z.to(self.device)
        B = z.size(0)

        # Repeat latent at every timestep
        z_seq = z.unsqueeze(1).repeat(1, self.fragment_length, 1)  # (B, L, latent_dim)

        # Optionally add small noise to latent inputs (can help exploration)
        z_seq = z_seq + th.randn_like(z_seq) * 0.05

        dec_out, _ = self.decoder_rnn(z_seq)                       # (B, L, H)
        h_dec = self.post_rnn_proj(dec_out)                        # (B, L, H)
        x_hat = self.output_layer(h_dec)                           # (B, L, state_dim)

        return x_hat


    def forward(self, x: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Full VAE pass.
        Args:
            x: (B, L, state_dim)
        Returns:
            x_hat : (B, L, state_dim)
            mu    : (B, latent_dim)
            log_var: (B, latent_dim)
        """
        z, mu, log_var = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, mu, log_var

class StateVAE(nn.Module):
    def __init__(self, 
                 state_dim: int,
                 latent_dim: int, 
                 fragment_length: int,
                 hidden_dims: List[int] = [128, 64, 32],
                 dropout: float = 0.1,
                 device: str = "cuda" if th.cuda.is_available() else "cpu"):
        super().__init__()
        self.state_dim = state_dim
        self.flat_dim = fragment_length * state_dim
        self.latent_dim = latent_dim
        self.fragment_length = fragment_length
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.device = device

        self.encoder = self._create_encoder().to(self.device)
        self.decoder = self._create_decoder().to(self.device)

        # Latent space projections
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim).to(self.device)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim).to(self.device)

    def _create_encoder(self):
        encoder_layers = []
        in_dim = self.flat_dim
        for hidden_dim in self.hidden_dims:
            encoder_layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU()
            ])
            in_dim = hidden_dim
        return nn.Sequential(*encoder_layers)
    
    def _create_decoder(self):
        decoder_layers = []
        in_dim = self.latent_dim
        for hidden_dim in reversed(self.hidden_dims):
            decoder_layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU()
                # nn.Dropout(self.dropout),
            ])
            in_dim = hidden_dim
        final_layer = nn.Linear(in_dim, self.flat_dim)
        decoder_layers.extend([
            final_layer,
            nn.LayerNorm(self.flat_dim),
            nn.Tanh(),
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
        x = x.to(self.device)
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
        z = z.to(self.device)
        x_hat = self.decoder(z)

        # Reshape to (batch_size, fragment_length, state_dim)
        x_hat = x_hat.view(-1, self.fragment_length, self.state_dim)

        return x_hat

    def reparameterize(self, mu: th.Tensor, log_var: th.Tensor) -> th.Tensor:
        """
        Reparameterization trick to sample from the latent space.
        """
        std = th.exp(0.5 * log_var)
        eps = th.randn_like(std, device=self.device)
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

class TransBlock(nn.Module):
    """
    Transformer encoder block: self‐attention + feed‐forward with residuals & layernorm.
    """
    def __init__(self, attn_dim: int, n_heads: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=attn_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(attn_dim)
        self.ff = nn.Sequential(
            nn.Linear(attn_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, attn_dim),
        )
        self.norm2 = nn.LayerNorm(attn_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: th.Tensor) -> th.Tensor:
        # x: (B, L, attn_dim)
        attn_out, _ = self.attn(x, x, x)        # (B, L, attn_dim)
        x = self.norm1(x + self.dropout(attn_out))
        ff_out = self.ff(x)                     # (B, L, attn_dim)
        x = self.norm2(x + self.dropout(ff_out))
        return x


class AttnStateVAE(nn.Module):
    def __init__(
        self,
        state_dim: int,
        latent_dim: int,
        fragment_length: int,
        n_heads: int = 4,
        n_blocks: int = 2,
        attn_dim: int = 128,
        attn_dropout: float = 0.1,
        device: str = "cuda"
    ):
        super().__init__()
        self.state_dim = state_dim
        self.latent_dim = latent_dim
        self.fragment_length = fragment_length
        self.device = device

        # 1) Input projection: state_dim → attn_dim
        self.input_proj = nn.Linear(state_dim, attn_dim)

        # 2) Transformer blocks
        self.blocks = nn.Sequential(
            *[TransBlock(
                attn_dim=attn_dim,
                n_heads=n_heads,
                ff_dim=attn_dim * 4,
                dropout=attn_dropout,
            ) for _ in range(n_blocks)]
        )

        # 3) Latent projection heads
        self.fc_mu     = nn.Linear(attn_dim, latent_dim)
        self.fc_logvar = nn.Linear(attn_dim, latent_dim)

        # 4) Simple decoder MLP
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, attn_dim),
            nn.ReLU(),
            nn.Linear(attn_dim, state_dim * fragment_length)
        )

        # Move everything to the target device
        self.to(self.device)

    def encode(self, x: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Encode a batch of state‐sequence fragments.

        Args:
            x: (B, L, state_dim)
        Returns:
            z      : (B, latent_dim)
            mu     : (B, latent_dim)
            logvar : (B, latent_dim)
        """
        x = x.to(self.device)                         # (B, L, state_dim)

        h = self.input_proj(x)                        # (B, L, attn_dim)
        h = self.blocks(h)                            # (B, L, attn_dim)

        # Pool over the time axis
        pooled = h.mean(dim=1)                        # (B, attn_dim)

        # Project to Gaussian parameters
        mu     = self.fc_mu(pooled)                   # (B, latent_dim)
        logvar = self.fc_logvar(pooled)               # (B, latent_dim)

        # Reparameterization trick
        std = th.exp(0.5 * logvar)                    # (B, latent_dim)
        eps = th.randn_like(std)                      # (B, latent_dim)
        z = mu + eps * std                            # (B, latent_dim)

        return z, mu, logvar

    def decode(self, z: th.Tensor) -> th.Tensor:
        """
        Decode latents into reconstructed state‐sequence fragments.

        Args:
            z: (B, latent_dim)
        Returns:
            x_hat: (B, L, state_dim)
        """
        z = z.to(self.device)
        B = z.size(0)

        out = self.decoder(z)                         # (B, state_dim * L)
        x_hat = out.view(B, self.fragment_length, self.state_dim)
        return x_hat

    def forward(self, x: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Full VAE forward pass.

        Args:
            x: (B, L, state_dim)
        Returns:
            x_hat : (B, L, state_dim)
            mu    : (B, latent_dim)
            logvar: (B, latent_dim)
        """
        z, mu, logvar = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, mu, logvar

class VAETrainer:
    def __init__(self,
                 vae: AttnStateVAE,
                 lr: float = 1e-3,
                 weight_decay: float = 1e-4,
                 batch_size: int = 32,
                 num_epochs: int = 25,
                 kl_weight_beta: float = 1.0,
                 kl_warmup_epochs: int = 40,
                 device: str = "cuda" if th.cuda.is_available() else "cpu"):
        self.vae = vae
        self.device = device
        self.optimizer = th.optim.Adam(self.vae.parameters(), lr=lr, weight_decay=weight_decay)
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.kl_warmup_epochs = kl_warmup_epochs
        self.kl_weight_beta = kl_weight_beta

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

        self.vae.train()
        for epoch in tqdm(range(self.num_epochs), desc="Training VAE"):
            epoch_total_loss = 0.0
            epoch_recon_loss = 0.0
            epoch_kl_loss = 0.0
            all_mu = []
            all_logvar = []

            for batch in dataloader:
                x = batch['obs'].to(self.device)

                # Forward pass
                x_hat, mu, log_var = self.vae(x)

                # Compute losses
                kl_weight_beta = self._get_kl_weight(epoch)
                total_loss, recon_loss, kl_loss = self._loss(x, x_hat, mu, log_var, kl_weight_beta=kl_weight_beta)

                # Backward pass
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

                # Accumulate losses
                epoch_total_loss += total_loss.item()
                epoch_recon_loss += recon_loss.item()
                epoch_kl_loss += kl_loss.item()
                all_mu.append(mu.detach().cpu())
                all_logvar.append(log_var.detach().cpu())

            mu = th.cat(all_mu, dim=0)
            logvar = th.cat(all_logvar, dim=0)

            sigma = th.exp(0.5 * logvar)
            kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())

            # Average losses over batches
            num_batches = len(dataloader)
            epoch_metrics.append(VAEMetrics(
                recon_loss=epoch_recon_loss / num_batches,
                kl_loss=epoch_kl_loss / num_batches,
                total_loss=epoch_total_loss / num_batches,
                latent_stats=LatentStats(
                    mu=mu.mean(dim=0),
                    sigma=sigma.mean(dim=0),
                    kl_per_dim= kl_per_dim.mean(dim=0),
                )
            ))

        return epoch_metrics

    def _get_kl_weight(self, epoch: int) -> float:
        """Compute the current KL weight based on linear warmup.

        Args:
            epoch: Current epoch number

        Returns:
            Current KL weight to use in the loss function
        """
        if self.kl_warmup_epochs is None or epoch >= self.kl_warmup_epochs:
            return self.kl_weight_beta

        # Linear warmup: β = min(1.0, epoch / N_warmup) * β_target
        progress = epoch / self.kl_warmup_epochs
        return min(1.0, progress) * self.kl_weight_beta

    def _get_kl_weight_cyclic(self, epoch: int) -> float:
        """Compute the current KL weight using cyclic annealing.

        Args:
            epoch: Current epoch number

        Returns:
            Current KL weight to use in the loss function
        """
        cycle_length = self.kl_warmup_epochs  # One full cycle = warmup_epochs
        cycle_progress = (epoch % cycle_length) / cycle_length  # 0 to 1 within each cycle

        # Optional: change shape of the cycle — linear ramp up, cosine, etc.
        beta = self.kl_weight_beta * cycle_progress  # Linear ramp
        # beta = self.kl_weight_beta * (1 - math.cos(math.pi * cycle_progress)) / 2  # Cosine ramp

        return beta

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

        Returns:
            total_loss: (batch_size)
            recon_loss: (batch_size)
            kl_loss: (batch_size)
        """
        B, _, _ = x.shape
        recon_loss = F.mse_loss(x_hat, x, reduction="sum") / B

        kl_loss = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(-1).mean()

        # z = self.vae.reparameterize(mu, logvar, self.device)  # (B, latent_dim)
        # z_prior = th.randn_like(z)  # sample p(z)
        # mmd_loss = self.compute_mmd(z, z_prior)

        total_loss = recon_loss + 1.0 * kl_loss # + 0.5 * mmd_loss

        return total_loss, recon_loss, kl_loss

    def compute_mmd(self, z: th.Tensor, z_prior: th.Tensor, kernel_mul=2.0, kernel_num=5):
        """
        MMD with RBF-kernel mixture:
          k(x,y) = sum_r exp(-||x-y||^2 / (2 σ_r^2))
        where σ_r = σ0 * (kernel_mul**r), and σ0 is the median distance.
        """
        # 1) pairwise squared distances
        xx = z @ z.t()
        x2 = (z * z).sum(dim=1, keepdim=True)
        pdist = x2 + x2.t() - 2 * xx

        # 2) choose a bandwidth base (median heuristic)
        with th.no_grad():
            median = pdist.flatten()[pdist.numel() // 2]
            base = median.clamp(min=1e-3)

        # 3) build RBF kernels
        kernels = 0
        for i in range(kernel_num):
            sigma = base * (kernel_mul ** i)
            kernels += th.exp(-pdist / (2 * sigma))

        # 4) MMD = E[k(z,z)] + E[k(z',z')] - 2E[k(z,z')]
        m = z.size(0)
        k_xx = kernels.mean()
        yy = z_prior @ z_prior.t()
        y2 = (z_prior * z_prior).sum(dim=1, keepdim=True)
        pdist_y = y2 + y2.t() - 2 * yy
        kernels_y = sum(th.exp(-pdist_y / (2 * base * (kernel_mul ** i)))
                        for i in range(kernel_num))
        k_yy = kernels_y.mean()
        k_xy = (th.exp(-pdist / (2 * base))  # you could reuse `kernels` & pick one bandwidth
                .mean())
        return k_xx + k_yy - 2 * k_xy
        