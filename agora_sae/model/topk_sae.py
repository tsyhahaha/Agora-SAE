"""
Module C: SAE Modeling

Top-K Sparse Autoencoder with Auxiliary Loss for dead latent recovery.
"""

import math
from typing import Tuple, Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class TopKSAE(nn.Module):
    """
    Top-K Sparse Autoencoder with Auxiliary Loss.
    
    Architecture:
    - Encoder: W_enc [d_model, d_sae], b_enc [d_sae]
    - Decoder: W_dec [d_sae, d_model], b_dec [d_model]
    - Activation: Top-K selection
    
    Loss:
    - L_recon: MSE reconstruction loss
    - L_aux: Dead latent reconstruction loss
    """
    
    def __init__(
        self,
        d_model: int,
        d_sae: int,
        k: int,
        aux_loss_weight: float = 1/32,
        dead_threshold: int = 10000  # Steps without activation to consider dead
    ):
        """
        Initialize the Top-K SAE.
        
        Args:
            d_model: Input/output dimension (model hidden size)
            d_sae: SAE hidden dimension (typically 32x d_model)
            k: Number of top activations to keep
            aux_loss_weight: Weight for auxiliary loss
            dead_threshold: Number of steps without activation to consider a latent dead
        """
        super().__init__()
        
        self.d_model = d_model
        self.d_sae = d_sae
        self.k = k
        self.aux_loss_weight = aux_loss_weight
        self.dead_threshold = dead_threshold
        
        # Encoder
        self.W_enc = nn.Parameter(torch.empty(d_model, d_sae))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        
        # Decoder
        self.W_dec = nn.Parameter(torch.empty(d_sae, d_model))
        self.b_dec = nn.Parameter(torch.zeros(d_model))
        
        # Track latent activation for dead feature detection
        self.register_buffer("latent_activation_count", torch.zeros(d_sae, dtype=torch.long))
        self.register_buffer("steps_since_activation", torch.zeros(d_sae, dtype=torch.long))
        
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights using Kaiming uniform."""
        # Kaiming uniform initialization
        nn.init.kaiming_uniform_(self.W_enc, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.W_dec, a=math.sqrt(5))
        
        # b_enc initialized to 0 (already done in __init__)
        # b_dec can be initialized to data mean later
        
        # Normalize decoder columns
        self.set_decoder_norm()
        
    @torch.no_grad()
    def set_decoder_norm(self):
        """Force unit norm on decoder columns (feature directions)."""
        # W_dec is [d_sae, d_model], normalize along dim=1 (each row is a feature direction)
        self.W_dec.data = F.normalize(self.W_dec.data, dim=1)
        
    @torch.no_grad()
    def init_b_dec_from_data(self, data_mean: torch.Tensor):
        """Initialize decoder bias from data geometric mean."""
        self.b_dec.data = data_mean.to(self.b_dec.device)
        
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input to pre-activation SAE hidden state.
        
        Args:
            x: Input tensor [batch_size, d_model]
            
        Returns:
            Pre-activation hidden state [batch_size, d_sae]
        """
        # Center by subtracting decoder bias
        x_centered = x - self.b_dec
        
        # Encode
        z = x_centered @ self.W_enc + self.b_enc
        
        return z
        
    def activate(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply ReLU and Top-K activation.
        
        Args:
            z: Pre-activation hidden state [batch_size, d_sae]
            
        Returns:
            Tuple of (sparse activations [batch_size, d_sae], top-k indices [batch_size, k])
        """
        # ReLU
        z_relu = F.relu(z)
        
        # Top-K selection
        topk_values, topk_indices = torch.topk(z_relu, self.k, dim=-1)
        
        # Create sparse activation tensor
        f = torch.zeros_like(z_relu)
        f.scatter_(-1, topk_indices, topk_values)
        
        return f, topk_indices
        
    def decode(self, f: torch.Tensor) -> torch.Tensor:
        """
        Decode sparse activations to reconstructed input.
        
        Args:
            f: Sparse activations [batch_size, d_sae]
            
        Returns:
            Reconstructed input [batch_size, d_model]
        """
        return f @ self.W_dec + self.b_dec
        
    def forward(
        self, 
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the SAE.
        
        Args:
            x: Input tensor [batch_size, d_model]
            
        Returns:
            Tuple of (x_hat, f, topk_indices, z)
            - x_hat: Reconstructed input [batch_size, d_model]
            - f: Sparse activations [batch_size, d_sae]
            - topk_indices: Indices of top-k activations [batch_size, k]
            - z: Pre-activation hidden state [batch_size, d_sae]
        """
        z = self.encode(x)
        f, topk_indices = self.activate(z)
        x_hat = self.decode(f)
        
        return x_hat, f, topk_indices, z
        
    def compute_loss(
        self,
        x: torch.Tensor,
        x_hat: torch.Tensor,
        f: torch.Tensor,
        topk_indices: torch.Tensor,
        z: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute training loss.
        
        Args:
            x: Original input [batch_size, d_model]
            x_hat: Reconstructed input [batch_size, d_model]
            f: Sparse activations [batch_size, d_sae]
            topk_indices: Top-k indices [batch_size, k]
            z: Pre-activation [batch_size, d_sae]
            
        Returns:
            Dictionary with loss components
        """
        batch_size = x.shape[0]
        
        # MSE Reconstruction Loss
        l_recon = F.mse_loss(x_hat, x)
        
        # Auxiliary Loss for dead latents
        # Compute residual
        e = x - x_hat  # [batch_size, d_model]
        
        # Find dead latents (not in current batch's top-k)
        active_mask = torch.zeros(self.d_sae, device=x.device, dtype=torch.bool)
        active_mask.scatter_(0, topk_indices.flatten(), True)
        dead_mask = ~active_mask
        
        # Auxiliary loss: use dead latents to predict residual
        if dead_mask.any():
            # Get pre-activations for dead latents only
            z_dead = z[:, dead_mask]  # [batch_size, n_dead]
            z_dead_relu = F.relu(z_dead)
            
            # Dead latent decoder weights
            W_dec_dead = self.W_dec[dead_mask, :]  # [n_dead, d_model]
            
            # Predict residual using dead latents
            e_pred = z_dead_relu @ W_dec_dead  # [batch_size, d_model]
            
            # Auxiliary loss
            l_aux = F.mse_loss(e_pred, e)
        else:
            l_aux = torch.tensor(0.0, device=x.device)
            
        # Total loss
        total_loss = l_recon + self.aux_loss_weight * l_aux
        
        # Compute metrics
        with torch.no_grad():
            l2_ratio = (torch.norm(x - x_hat, dim=-1).pow(2) / torch.norm(x, dim=-1).pow(2)).mean()
            l0 = (f > 0).float().sum(dim=-1).mean()  # Effective sparsity
            
        return {
            "loss": total_loss,
            "l_recon": l_recon,
            "l_aux": l_aux,
            "l2_ratio": l2_ratio,
            "l0": l0
        }
        
    @torch.no_grad()
    def update_activation_stats(self, topk_indices: torch.Tensor):
        """Update latent activation statistics."""
        # Increment steps since activation for all latents
        self.steps_since_activation += 1
        
        # Reset counter for activated latents
        unique_indices = topk_indices.unique()
        self.latent_activation_count[unique_indices] += 1
        self.steps_since_activation[unique_indices] = 0
        
    @torch.no_grad()
    def get_dead_latent_ratio(self) -> float:
        """Get the ratio of dead latents."""
        dead_mask = self.steps_since_activation > self.dead_threshold
        return dead_mask.float().mean().item()
        
    @torch.no_grad()
    def get_dead_latent_indices(self) -> torch.Tensor:
        """Get indices of dead latents."""
        return torch.where(self.steps_since_activation > self.dead_threshold)[0]
        
    def explained_variance(self, x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
        """Calculate explained variance."""
        total_var = torch.var(x)
        residual_var = torch.var(x - x_hat)
        return 1 - (residual_var / total_var)


class TopKSAEWithResampling(TopKSAE):
    """
    Top-K SAE with Anthropic-style dead latent resampling.
    
    When a latent is dead for too long, reinitialize it using high-loss examples.
    """
    
    @torch.no_grad()
    def resample_dead_latents(
        self,
        x: torch.Tensor,
        x_hat: torch.Tensor,
        n_resample: Optional[int] = None
    ):
        """
        Resample dead latents using high-loss examples.
        
        Args:
            x: Input batch [batch_size, d_model]
            x_hat: Reconstructed batch [batch_size, d_model]
            n_resample: Number of latents to resample (None = all dead)
        """
        dead_indices = self.get_dead_latent_indices()
        
        if len(dead_indices) == 0:
            return 0
            
        if n_resample is not None:
            dead_indices = dead_indices[:n_resample]
            
        # Compute per-example reconstruction error
        errors = (x - x_hat).pow(2).sum(dim=-1)  # [batch_size]
        
        # Sample examples with probability proportional to error
        probs = errors / errors.sum()
        sample_indices = torch.multinomial(probs, len(dead_indices), replacement=True)
        
        # Reinitialize dead latents
        for i, dead_idx in enumerate(dead_indices):
            # New encoder direction: normalized residual
            residual = x[sample_indices[i]] - x_hat[sample_indices[i]]
            new_direction = F.normalize(residual, dim=0)
            
            # Set encoder column
            self.W_enc[:, dead_idx] = new_direction
            
            # Set decoder row (transpose of encoder direction for tied weights intuition)
            self.W_dec[dead_idx, :] = new_direction
            
            # Reset bias
            self.b_enc[dead_idx] = 0
            
            # Reset activation counter
            self.steps_since_activation[dead_idx] = 0
            self.latent_activation_count[dead_idx] = 0
            
        return len(dead_indices)
