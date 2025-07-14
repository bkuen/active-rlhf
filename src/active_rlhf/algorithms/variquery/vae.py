import copy
import math
from dataclasses import dataclass
from typing import List, Tuple, Optional, Sequence
import torch as th
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from active_rlhf.data.buffers import ReplayBufferBatch, ReplayBuffer


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

@dataclass
class VAETrainerMetrics:
    train_metrics: List[VAEMetrics]
    val_metrics: List[VAEMetrics]

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
        # noise = th.randn_like(z_seq) * 0.05
        # z_seq = z_seq + noise # Add some noise to the latent vector to encourage exploration
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

class MLPStateVAE(nn.Module):
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

        # self.input_norm = nn.LayerNorm(self.flat_dim)
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
                nn.ReLU(),
                nn.Dropout(self.dropout),
            ])
            in_dim = hidden_dim
        final_layer = nn.Linear(in_dim, self.flat_dim)
        decoder_layers.extend([
            final_layer,
            # nn.LayerNorm(self.flat_dim),
            # nn.Tanh(),
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
        # x = self.input_norm(x)

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

class MLPStateSkipVAE(nn.Module):
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

        # self.input_norm = nn.LayerNorm(self.flat_dim)
        self.encoder = self._create_encoder().to(self.device)
        self.decoder_layers = self._create_decoder_layers().to(self.device)

        # Latent space projections
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim).to(self.device)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim).to(self.device)

        # Final output layer
        self.final_layer = nn.Sequential(
            nn.Linear(hidden_dims[0], self.flat_dim),
            # nn.LayerNorm(self.flat_dim),
            # nn.Tanh(),
        ).to(self.device)

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
    
    def _create_decoder_layers(self):
        # Each decoder layer takes [h, z] as input, so input dim increases by latent_dim
        layers = nn.ModuleList()
        hidden_dims = list(reversed(self.hidden_dims))
        in_dim = self.latent_dim
        for i, hidden_dim in enumerate(hidden_dims):
            if i > 0:
                in_dim = hidden_dims[i-1] + self.latent_dim
            layers.append(nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout),
            ))
        return layers
    
    def encode(self, x: th.Tensor) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        B = x.shape[0]
        x = x.to(self.device)
        x = x.view(B, -1)
        # x = self.input_norm(x)

        hidden = self.encoder(x)
        mu = self.fc_mu(hidden)
        log_var = self.fc_logvar(hidden)
        z = self.reparameterize(mu, log_var)
        return z, mu, log_var
    
    def decode(self, z: th.Tensor) -> th.Tensor:
        z = z.to(self.device)
        h = z
        for i, layer in enumerate(self.decoder_layers):
            if i == 0:
                h = layer(h)
            else:
                h = layer(th.cat([h, z], dim=-1))
        x_hat = self.final_layer(h)
        x_hat = x_hat.view(-1, self.fragment_length, self.state_dim)
        return x_hat

    def reparameterize(self, mu: th.Tensor, log_var: th.Tensor) -> th.Tensor:
        std = th.exp(0.5 * log_var)
        eps = th.randn_like(std, device=self.device)
        return mu + eps * std
    
    def forward(self, x: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        z, mu, log_var = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, mu, log_var


class ConvStateVAE(nn.Module):
    """
    Convolution-MLP VAE for fixed-length state fragments.
    • LayerNorm on the flattened input (per sample) normalises scale.
    • Encoder funnel is strictly decreasing (128 → 64 → 32 → μ, log σ²).
    • Decoder mirrors the funnel and learns a per-state output scale.
    """

    def __init__(
        self,
        state_dim: int,
        latent_dim: int,
        fragment_length: int,
        hidden_dims: Sequence[int] = (128, 64, 32),
        kernel_size: int = 5,
        dropout: float = 0.05,
        device: str | th.device = "cuda" if th.cuda.is_available() else "cpu",
    ):
        super().__init__()
        self.state_dim = state_dim
        self.fragment_length = fragment_length
        self.flat_dim = state_dim * fragment_length
        self.latent_dim = latent_dim
        self.hidden_dims = tuple(hidden_dims)
        self.device = th.device(device)

        # --------------------------------------------------------------------- #
        # Normalisation / denormalisation
        # --------------------------------------------------------------------- #
        self.input_norm = nn.LayerNorm(self.flat_dim, elementwise_affine=False)

        # --------------------------------------------------------------------- #
        # Convolutional front-end
        # --------------------------------------------------------------------- #
        self.conv = nn.Sequential(
            nn.Conv1d(state_dim, 128, kernel_size, padding=(kernel_size - 1) // 2),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size, padding=(kernel_size - 1) // 2),
            nn.ReLU(),
        )

        # Project flattened conv features to the first hidden dim
        self.conv_proj = nn.Linear(256 * fragment_length, self.hidden_dims[0])

        # --------------------------------------------------------------------- #
        # Encoder MLP  (monotonic funnel)
        # --------------------------------------------------------------------- #
        enc_layers: list[nn.Module] = []
        in_dim = self.hidden_dims[0]
        for h in self.hidden_dims[1:]:
            enc_layers += [
                nn.Linear(in_dim, h),
                nn.LayerNorm(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            in_dim = h
        self.encoder = nn.Sequential(*enc_layers)

        self.fc_mu = nn.Linear(self.hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(self.hidden_dims[-1], latent_dim)

        # --------------------------------------------------------------------- #
        # Decoder MLP (mirror)
        # --------------------------------------------------------------------- #
        dec_layers: list[nn.Module] = [nn.Linear(latent_dim, self.hidden_dims[-1]), nn.ReLU()]
        in_dim = self.hidden_dims[-1]
        for h in reversed(self.hidden_dims[:-1]):
            dec_layers += [
                nn.Linear(in_dim, h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            in_dim = h
        dec_layers += [nn.Linear(in_dim, self.flat_dim)]
        self.decoder = nn.Sequential(*dec_layers)

        # Learnable per-state output scale / bias to undo input normalisation
        self.out_scale = nn.Parameter(th.ones(state_dim))
        self.out_bias = nn.Parameter(th.zeros(state_dim))

        # --------------------------------------------------------------------- #
        self.to(self.device)

    # ------------------------------------------------------------------------- #
    # Forward components
    # ------------------------------------------------------------------------- #
    def encode(self, x: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Args:
            x: (B, L, S)
        Returns:
            z, mu, logvar – (B, latent_dim)
        """
        B = x.size(0)
        # Apply sample-wise LayerNorm
        x = self.input_norm(x.view(B, -1)).view(B, self.fragment_length, self.state_dim)

        # → (B, S, L) for Conv1d
        h_conv = self.conv(x.transpose(1, 2))
        h_flat = h_conv.flatten(1)                    # (B, 256 × L)
        h_proj = self.conv_proj(h_flat)               # (B, hidden_dims[0])
        h = self.encoder(h_proj)                      # (B, hidden_dims[-1])

        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h).clamp(-6.0, 3.0)   # numerical safety
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def decode(self, z: th.Tensor) -> th.Tensor:
        """
        Args:
            z: (B, latent_dim)
        Returns:
            x_hat: (B, L, S)
        """
        B = z.size(0)
        flat = self.decoder(z)                        # (B, L × S)
        x_hat = flat.view(B, self.fragment_length, self.state_dim)
        # Undo normalisation with learned scale / bias
        x_hat = x_hat * self.out_scale + self.out_bias
        return x_hat

    # ------------------------------------------------------------------------- #
    # VAE utilities
    # ------------------------------------------------------------------------- #
    def reparameterize(self, mu: th.Tensor, logvar: th.Tensor) -> th.Tensor:
        std = (0.5 * logvar).exp()
        eps = th.randn_like(std)
        return mu + eps * std

    def forward(self, x: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        z, mu, logvar = self.encode(x.to(self.device))
        x_hat = self.decode(z)
        return x_hat, mu, logvar

# ──────────────────────────────────────────────────────────────
# Helper: a single Transformer encoder/decoder block
# (identical layout for both encoder & decoder)
# ──────────────────────────────────────────────────────────────
class TransBlock(nn.Module):
    def __init__(self, attn_dim: int, n_heads: int, ff_dim: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(attn_dim, n_heads, dropout=dropout, batch_first=True)
        self.ff   = nn.Sequential(
            nn.Linear(attn_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, attn_dim),
            nn.Dropout(dropout),
        )
        self.ln1 = nn.LayerNorm(attn_dim)
        self.ln2 = nn.LayerNorm(attn_dim)

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        # Multi-head self-attention
        _y, _ = self.attn(
            query=x, key=x, value=x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
        )
        x = x + _y
        x = self.ln1(x)

        # Feed-forward
        _y = self.ff(x)
        x = x + _y
        x = self.ln2(x)
        return x


# ──────────────────────────────────────────────────────────────
# Main VAE
# ──────────────────────────────────────────────────────────────
class AttnStateVAE(nn.Module):
    """
    Bidirectional Transformer encoder + causal Transformer decoder VAE
    for fixed-length trajectory fragments.
    Interface identical to the stub you supplied.
    """
    def __init__(
        self,
        state_dim: int,
        latent_dim: int,
        fragment_length: int,
        n_heads: int = 4,
        n_blocks: int = 4,
        n_decoder_layers: int = 2,
        attn_dim: int = 128,
        attn_dropout: float = 0.1,
        device: str = "cuda",
    ):
        super().__init__()
        self.state_dim       = state_dim
        self.latent_dim      = latent_dim
        self.fragment_length = fragment_length
        self.device          = device
        self.seq_len_plus_cls = fragment_length + 1      # +1 for CLS pooling token

        # 1) Input projection and learned positional embedding
        self.input_proj = nn.Linear(state_dim, attn_dim)
        self.pos_emb    = nn.Parameter(th.zeros(1, self.seq_len_plus_cls, attn_dim))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

        # Dedicated CLS (attention-pooling) token
        self.cls_token  = nn.Parameter(th.zeros(1, 1, attn_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # 2) Bidirectional Transformer encoder
        self.enc_blocks = nn.Sequential(
            *[TransBlock(attn_dim, n_heads, ff_dim=attn_dim * 4, dropout=attn_dropout)
              for _ in range(n_blocks)]
        )

        # 3) Latent projection heads
        self.fc_mu     = nn.Linear(attn_dim, latent_dim)
        self.fc_logvar = nn.Linear(attn_dim, latent_dim)

        # 4) Tiny **causal** Transformer decoder
        self.dec_input = nn.Linear(latent_dim, attn_dim)
        self.dec_blocks = nn.ModuleList(
            [TransBlock(attn_dim, n_heads, ff_dim=attn_dim * 4, dropout=attn_dropout)
             for _ in range(n_decoder_layers)]
        )
        self.dec_ln  = nn.LayerNorm(attn_dim)
        self.dec_out = nn.Linear(attn_dim, state_dim)

        # Prepare causal mask once (L × L, bool)
        causal_mask = th.triu(th.ones(fragment_length, fragment_length, dtype=th.bool), diagonal=1)
        self.register_buffer("causal_mask", causal_mask, persistent=False)

        self.to(device)


    # ─────────── Encoder ───────────
    def encode(self, x: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Args
        ----
        x : (B, L, state_dim)
        Returns
        -------
        z      : (B, latent_dim)
        mu     : (B, latent_dim)
        logvar : (B, latent_dim)
        """
        x = x.to(self.device)                                  # (B, L, D)
        B = x.size(0)

        # (1) project
        h = self.input_proj(x)                                 # (B, L, attn_dim)

        # (2) prepend CLS token and add positional enc.
        cls = self.cls_token.expand(B, -1, -1)                 # (B, 1, attn_dim)
        h = th.cat([cls, h], dim=1)                            # (B, L+1, attn_dim)
        h = h + self.pos_emb[:, : self.seq_len_plus_cls, :]    # broadcast

        # (3) bidirectional attention
        h = self.enc_blocks(h)                                 # (B, L+1, attn_dim)

        # (4) take CLS vector
        pooled = h[:, 0]                                       # (B, attn_dim)

        mu     = self.fc_mu(pooled)                            # (B, z)
        logvar = self.fc_logvar(pooled)                        # (B, z)

        std = th.exp(0.5 * logvar)
        eps = th.randn_like(std)
        z = mu + eps * std                                     # reparameterise

        return z, mu, logvar


    # ─────────── Decoder ───────────
    def decode(self, z: th.Tensor) -> th.Tensor:
        """
        Args
        ----
        z : (B, latent_dim)
        Returns
        -------
        x_hat : (B, L, state_dim)
        """
        z = z.to(self.device)
        B = z.size(0)

        # Repeat latent for each time-step and project
        h = self.dec_input(z).unsqueeze(1)                     # (B, 1, attn_dim)
        h = h.repeat(1, self.fragment_length, 1)               # (B, L, attn_dim)

        # Positional encoding reuse (skip CLS slot)
        h = h + self.pos_emb[:, 1:self.fragment_length + 1, :]

        # Causal self-attention
        for blk in self.dec_blocks:
            h = blk(h, attn_mask=self.causal_mask)     # (B, L, attn_dim)

        h = self.dec_ln(h)
        x_hat = self.dec_out(h)                                # (B, L, state_dim)
        return x_hat


    # ─────────── VAE forward ───────────
    def forward(self, x: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Args
        ----
        x : (B, L, state_dim)
        Returns
        -------
        x_hat : (B, L, state_dim)
        mu    : (B, latent_dim)
        logvar: (B, latent_dim)
        """
        z, mu, logvar = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, mu, logvar

class BetterVAETrainer:
    def __init__(self,
                 writer: SummaryWriter,
                 replay_buffer: ReplayBuffer,
                 vae: ConvStateVAE,
                 lr: float = 1e-3,
                 weight_decay: float = 1e-4,
                 batch_size: int = 1024,
                 minibatch_size: int = 64,
                 val_batch_size: int = 128,
                 num_epochs: int = 25,
                 kl_weight_beta: float = 1.0,
                 kl_warmup_steps: Optional[int] = None,
                 total_steps: int = 1_000_000,
                 noise_sigma: float = 0.00):
        self.writer = writer
        self.replay_buffer = replay_buffer
        self.vae = vae
        self.batch_size = batch_size
        self.minibatch_size = minibatch_size
        self.val_batch_size = val_batch_size
        self.num_epochs = num_epochs
        self.kl_weight_beta = kl_weight_beta
        self.kl_warmup_steps = kl_warmup_steps
        self.total_steps = total_steps
        self.noise_sigma = noise_sigma

        self.optimizer = th.optim.AdamW(self.vae.parameters(), lr=lr, weight_decay=weight_decay)

        self.vae_step = 0

    def get_kl_weight(self) -> float:
        if self.kl_warmup_steps is None or self.kl_warmup_steps <= 0:
            # no warm-up requested
            return self.kl_weight_beta

            # progress in [0, 1]
        progress = min(1.0, self.vae_step / float(self.kl_warmup_steps))
        return self.kl_weight_beta * progress

    def train(self, global_step: int):
        """
        Train the VAE on a replay buffer.

        Args:
            global_step: int, the global step for logging.
        """

        train_recon_losses = []
        train_kl_losses = []
        train_total_losses = []
        val_recon_losses = []
        val_kl_losses = []
        val_total_losses = []

        for epoch in tqdm(range(self.num_epochs), desc="Training VAE"):
            train_dataset = self.replay_buffer.sample2(self.batch_size, split="train")
            val_batch = self.replay_buffer.sample2(self.val_batch_size, split="val")

            train_loader = DataLoader(
                ReplayBufferDataset(train_dataset),
                batch_size=self.minibatch_size,
                shuffle=True,
                pin_memory=False,
            )

            for train_batch in train_loader:
                x_clean = train_batch["obs"].to(self.vae.device)
                if self.noise_sigma != 0.0:
                    x_corrupted = x_clean + th.randn_like(x_clean) * self.noise_sigma  # Add noise to the input
                else:
                    x_corrupted = x_clean

                # Training
                self.vae.train()
                self.optimizer.zero_grad()

                # Forward pass
                x_hat, mu, log_var = self.vae(x_corrupted)

                # Compute losses
                recon_loss = th.nn.functional.mse_loss(x_hat, x_clean)
                kl_loss = -0.5 * th.sum(1 + log_var - mu.pow(2) - log_var.exp()) / train_batch["obs"].shape[0]
                total_loss = recon_loss + self.get_kl_weight() * kl_loss

                train_recon_losses.append(recon_loss.item())
                train_kl_losses.append(kl_loss.item())
                train_total_losses.append(total_loss.item())

                # Backward pass
                total_loss.backward()
                self.optimizer.step()

            # Validation
            self.vae.eval()
            with th.no_grad():
                val_x_clean = val_batch.obs.to(self.vae.device)
                if self.noise_sigma != 0.0:
                    val_x_corrupted = val_x_clean + th.randn_like(val_x_clean) * self.noise_sigma  # Add noise to the input
                else:
                    val_x_corrupted = val_x_clean

                val_x_hat, val_mu, val_log_var = self.vae(val_x_corrupted)
                val_recon_loss = th.nn.functional.mse_loss(val_x_hat, val_x_clean)
                val_kl_loss = -0.5 * th.sum(1 + val_log_var - val_mu.pow(2) - val_log_var.exp()) / val_batch.obs.shape[0]
                val_total_loss = val_recon_loss + self.get_kl_weight() * val_kl_loss

                val_recon_losses.append(val_recon_loss.item())
                val_kl_losses.append(val_kl_loss.item())
                val_total_losses.append(val_total_loss.item())

            # Log per-epoch losses
            # self.writer.add_scalar("vae/epoch_train_recon_loss", recon_loss.item(), self.vae_step)
            # self.writer.add_scalar("vae/epoch_train_kl_loss", kl_loss.item(), self.vae_step)
            # self.writer.add_scalar("vae/epoch_train_total_loss", total_loss.item(), self.vae_step)
            # self.writer.add_scalar("vae/epoch_val_recon_loss", val_recon_loss.item(), self.vae_step)
            # self.writer.add_scalar("vae/epoch_val_kl_loss", val_kl_loss.item(), self.vae_step)
            # self.writer.add_scalar("vae/epoch_val_total_loss", val_total_loss.item(), self.vae_step)

            self.vae_step += 1

        # Log average losses
        avg_train_recon_loss = sum(train_recon_losses) / len(train_recon_losses)
        avg_train_kl_loss = sum(train_kl_losses) / len(train_kl_losses)
        avg_train_total_loss = sum(train_total_losses) / len(train_total_losses)

        avg_val_recon_loss = sum(val_recon_losses) / len(val_recon_losses)
        avg_val_kl_loss = sum(val_kl_losses) / len(val_kl_losses)
        avg_val_total_loss = sum(val_total_losses) / len(val_total_losses)

        self.writer.add_scalar("vae/train_recon_loss", avg_train_recon_loss, global_step)
        self.writer.add_scalar("vae/train_kl_loss", avg_train_kl_loss, global_step)
        self.writer.add_scalar("vae/train_total_loss", avg_train_total_loss, global_step)
        self.writer.add_scalar("vae/val_recon_loss", avg_val_recon_loss, global_step)
        self.writer.add_scalar("vae/val_kl_loss", avg_val_kl_loss, global_step)
        self.writer.add_scalar("vae/val_total_loss", avg_val_total_loss, global_step)


class VAETrainer:
    def __init__(self,
                 vae: MLPStateVAE,
                 lr: float = 1e-3,
                 weight_decay: float = 1e-4,
                 batch_size: int = 32,
                 num_epochs: int = 25,
                 kl_weight_beta: float = 1.0,
                 kl_warmup_epochs: int = 40,
                 kl_warmup_steps: int = 320_000,
                 total_steps: int = 1_000_000,
                 noise_sigma: float = 0.00,
                 early_stopping_patience: Optional[int] = None,
                 device: str = "cuda" if th.cuda.is_available() else "cpu"):
        self.vae = vae
        self.device = device
        self.optimizer = th.optim.AdamW(self.vae.parameters(), lr=lr, weight_decay=weight_decay)
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.kl_warmup_epochs = kl_warmup_epochs
        self.kl_warmup_steps = kl_warmup_steps
        self.kl_weight_beta = kl_weight_beta
        self.total_steps = total_steps
        self.noise_sigma = noise_sigma
        self.early_stopping_patience = early_stopping_patience,
        self.vae_step = 0

        self.free_bits = 0.1

    def train(self,
              train_batch: ReplayBufferBatch,
              val_batch: ReplayBufferBatch,
              global_step: int) -> VAETrainerMetrics:
        """Train the VAE on a batch of trajectories.
        
        Args:
            train_batch: ReplayBufferBatch
            global_step: int

        Returns:
            List of VAEMetrics for each epoch
        """
        train_dataset = ReplayBufferDataset(train_batch)
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        val_dataset = ReplayBufferDataset(val_batch)
        val_dataloader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        train_metrics = []
        val_metrics = []

        best_val_loss = float('inf')
        epochs_no_improve = 0
        best_state = copy.deepcopy(self.vae.state_dict())

        for epoch in tqdm(range(self.num_epochs), desc="Training VAE"):
            # kl_weight_beta = self._get_kl_weight(epoch)

            # Training
            self.vae.train()

            train_total_loss, train_recon_loss, train_kl_loss = 0.0, 0.0, 0.0
            train_all_mu, train_all_logvar = [], []

            for batch in train_dataloader:
                x_clean = batch['obs'].to(self.device)

                self.optimizer.zero_grad()

                if self.noise_sigma != 0.0:
                    x_corrupted = x_clean + th.randn_like(x_clean) * self.noise_sigma  # Add noise to the input
                else:
                    x_corrupted = x_clean

                # Forward pass
                x_hat, mu, log_var = self.vae(x_corrupted)

                # Compute losses
                kl_weight_beta = self._get_current_kl_weight()
                total_loss, recon_loss, kl_loss = self._loss(x_clean, x_hat, mu, log_var, kl_weight_beta=kl_weight_beta)

                # Backward pass
                total_loss.backward()
                # nn.utils.clip_grad_norm_(self.vae.parameters(), max_norm=1.0)
                self.optimizer.step()

                self.vae_step += 1

                # Accumulate losses
                train_total_loss += total_loss.item()
                train_recon_loss += recon_loss.item()
                train_kl_loss += kl_loss.item()
                train_all_mu.append(mu.detach().cpu())
                train_all_logvar.append(log_var.detach().cpu())

            mu = th.cat(train_all_mu, dim=0)
            logvar = th.cat(train_all_logvar, dim=0)

            sigma = th.exp(0.5 * logvar)
            kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())

            # Average losses over batches
            num_batches = len(train_dataloader)
            train_metrics.append(VAEMetrics(
                recon_loss=train_recon_loss / num_batches,
                kl_loss=train_kl_loss / num_batches,
                total_loss=train_total_loss / num_batches,
                latent_stats=LatentStats(
                    mu=mu.mean(dim=0),
                    sigma=sigma.mean(dim=0),
                    kl_per_dim= kl_per_dim.mean(dim=0),
                )
            ))

            # Validation
            self.vae.eval()

            val_total_loss, val_recon_loss, val_kl_loss = 0.0, 0.0, 0.0
            val_all_mu, val_all_logvar = [], []

            for batch in val_dataloader:
                x_clean = batch['obs'].to(self.device)
                if self.noise_sigma != 0.0:
                    x_corrupted = x_clean + th.randn_like(x_clean) * self.noise_sigma  # Add noise to the input
                else:
                    x_corrupted = x_clean

                # Forward pass
                with th.no_grad():
                    x_hat, mu, log_var = self.vae(x_corrupted)

                # Compute losses
                kl_weight_beta = self._get_current_kl_weight()
                total_loss, recon_loss, kl_loss = self._loss(x_clean, x_hat, mu, log_var, kl_weight_beta=kl_weight_beta)

                # Accumulate losses
                val_total_loss += total_loss.item()
                val_recon_loss += recon_loss.item()
                val_kl_loss += kl_loss.item()
                val_all_mu.append(mu.detach().cpu())
                val_all_logvar.append(log_var.detach().cpu())

            mu = th.cat(val_all_mu, dim=0)
            logvar = th.cat(val_all_logvar, dim=0)

            sigma = th.exp(0.5 * logvar)
            kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())

            # Average losses over batches
            num_batches = len(val_dataloader)
            val_metrics.append(VAEMetrics(
                recon_loss=val_recon_loss / num_batches,
                kl_loss=val_kl_loss / num_batches,
                total_loss=val_total_loss / num_batches,
                latent_stats=LatentStats(mu=mu.mean(dim=0), sigma=sigma.mean(dim=0), kl_per_dim=kl_per_dim.mean(dim=0))
            ))

            # Early stopping check
            if self.early_stopping_patience is not None:
                if isinstance(self.early_stopping_patience, tuple):
                    early_stopping_patience = self.early_stopping_patience[0]
                else:
                    early_stopping_patience = self.early_stopping_patience

                current_val = val_metrics[-1].total_loss
                if current_val < best_val_loss:
                    best_val_loss = current_val
                    epochs_no_improve = 0
                    best_state = copy.deepcopy(self.vae.state_dict())
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= early_stopping_patience:
                        print(f"[EarlyStopping] stopping at epoch {epoch}")
                        break

        if self.early_stopping_patience is not None:
            # Restore best model state
            self.vae.load_state_dict(best_state)

        return VAETrainerMetrics(train_metrics=train_metrics, val_metrics=val_metrics)

    def _get_current_kl_weight(self) -> float:
        # Alternative 1: cosine triangle warm-up from 0→β_target over kl_warmup_epochs:
        # pos = (self.vae_step % self.kl_warmup_epochs) / self.kl_warmup_epochs  # [0,1)
        # t = 0.5 * (1 - math.cos(2 * math.pi * pos))
        # return self.kl_weight_beta * t

        # Alternative 2: monotonic cosine warm-up from 0→β_target over kl_warmup_steps:
        # progress = min(self.vae_step / self.kl_warmup_steps, 1.0)
        # return self.kl_weight_beta * 0.5 * (1 - math.cos(math.pi * progress))

        # Alternative 3: static β_target:
        return self.kl_weight_beta

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
        recon_loss = F.mse_loss(x_hat, x, reduction="mean") # / B

        kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        # kl_per_dim = th.clamp(kl_per_dim, min=self.free_bits)  # free bits trick
        kl_loss = kl_per_dim.sum(-1).mean()

        # z = self.vae.reparameterize(mu, logvar, self.device)  # (B, latent_dim)
        # z_prior = th.randn_like(z)  # sample p(z)
        # mmd_loss = self.compute_mmd(z, z_prior)

        total_loss = recon_loss + kl_weight_beta * kl_loss # + 0.5 * mmd_loss

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
        