"""
Module D: SAE Trainer (Stage 2)

High-throughput offline trainer for SAE using pre-computed activations.
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from ..model.topk_sae import TopKSAE, TopKSAEWithResampling
from .shard_loader import ShardLoader


class SAETrainer:
    """
    High-throughput SAE trainer using pre-computed activations.
    
    Features:
    - AdamW optimizer with warmup and cosine decay
    - Decoder unit norm constraint
    - Dead latent monitoring and optional resampling
    - Wandb logging
    """
    
    def __init__(
        self,
        sae: TopKSAE,
        config,
        device: str = "cuda",
        use_wandb: bool = True
    ):
        """
        Initialize the trainer.
        
        Args:
            sae: SAE model to train
            config: Training configuration
            device: Device to train on
            use_wandb: Whether to log to wandb
        """
        self.sae = sae.to(device)
        self.config = config
        self.device = device
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        
        # Optimizer
        self.optimizer = AdamW(
            sae.parameters(),
            lr=config.training.lr,
            betas=(0.9, 0.999),
            weight_decay=config.training.weight_decay
        )
        
        # Learning rate scheduler: warmup + cosine decay
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=config.training.warmup_steps
        )
        cosine_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config.training.total_steps - config.training.warmup_steps,
            eta_min=config.training.lr * 0.01
        )
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[config.training.warmup_steps]
        )
        
        # Training state
        self.step = 0
        self.best_l2_ratio = float('inf')
        
        # Initialize wandb
        if self.use_wandb:
            run_name = config.wandb_run_name or f"sae_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            wandb.init(
                project=config.wandb_project,
                name=run_name,
                config={
                    "d_model": config.model.d_model,
                    "d_sae": config.d_sae,
                    "k": config.sae.k,
                    "lr": config.training.lr,
                    "batch_size": config.training.batch_size,
                    "expansion_factor": config.sae.expansion_factor,
                    "aux_loss_weight": config.sae.aux_loss_weight
                }
            )

    def _prepare_batch(self, batch: torch.Tensor) -> torch.Tensor:
        """Move a batch to the training device and cast it to the SAE dtype."""
        return batch.to(self.device, dtype=self.sae.W_enc.dtype)

    def train_step(self, batch: torch.Tensor) -> Dict[str, float]:
        """
        Execute a single training step.
        
        Args:
            batch: Activation batch [batch_size, d_model]
            
        Returns:
            Dictionary of metrics
        """
        batch = self._prepare_batch(batch)
        
        # Forward pass
        x_hat, f, topk_indices, z = self.sae(batch)
        
        # Compute loss
        loss_dict = self.sae.compute_loss(batch, x_hat, f, topk_indices, z)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss_dict["loss"].backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.sae.parameters(), max_norm=1.0)
        
        # Optimizer step
        self.optimizer.step()
        self.scheduler.step()
        
        # Apply decoder unit norm constraint
        self.sae.set_decoder_norm()
        
        # Update activation statistics
        self.sae.update_activation_stats(topk_indices)
        
        self.step += 1
        
        # Convert to Python floats
        metrics = {k: v.item() if torch.is_tensor(v) else v for k, v in loss_dict.items()}
        metrics["lr"] = self.scheduler.get_last_lr()[0]
        
        return metrics
        
    def train(
        self,
        shard_loader: ShardLoader,
        checkpoint_dir: Optional[Path] = None,
        checkpoint_interval: int = 5000,
        log_interval: int = 100
    ):
        """
        Main training loop.
        
        Args:
            shard_loader: DataLoader for activation shards
            checkpoint_dir: Directory to save checkpoints
            checkpoint_interval: Steps between checkpoints
            log_interval: Steps between logging
        """
        if checkpoint_dir:
            checkpoint_dir = Path(checkpoint_dir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
        pbar = tqdm(
            shard_loader,
            total=self.config.training.total_steps,
            desc="Training SAE"
        )
        
        running_metrics = {}
        
        for batch in pbar:
            if self.step >= self.config.training.total_steps:
                break
                
            # Training step
            metrics = self.train_step(batch)
            
            # Update running metrics
            for k, v in metrics.items():
                if k not in running_metrics:
                    running_metrics[k] = []
                running_metrics[k].append(v)
                
            # Check dead latents
            if self.step % self.config.training.dead_latent_check_interval == 0:
                dead_ratio = self.sae.get_dead_latent_ratio()
                metrics["dead_ratio"] = dead_ratio
                
                # Optional: trigger resampling if dead ratio is too high
                if (dead_ratio > self.config.training.dead_latent_threshold and 
                    isinstance(self.sae, TopKSAEWithResampling)):
                    with torch.no_grad():
                        batch_on_device = self._prepare_batch(batch)
                        x_hat, _, _, _ = self.sae(batch_on_device)
                        n_resampled = self.sae.resample_dead_latents(batch_on_device, x_hat)
                        print(f"Step {self.step}: Resampled {n_resampled} dead latents")
                        
            # Logging
            if self.step % log_interval == 0:
                avg_metrics = {
                    k: sum(v) / len(v) 
                    for k, v in running_metrics.items()
                }
                running_metrics = {}
                
                # Update progress bar
                pbar.set_postfix({
                    "loss": f"{avg_metrics.get('loss', 0):.4f}",
                    "l2": f"{avg_metrics.get('l2_ratio', 0):.4f}",
                    "l0": f"{avg_metrics.get('l0', 0):.1f}"
                })
                
                # Wandb logging
                if self.use_wandb:
                    wandb.log(avg_metrics, step=self.step)
                    
                # Track best L2 ratio
                if avg_metrics.get("l2_ratio", float('inf')) < self.best_l2_ratio:
                    self.best_l2_ratio = avg_metrics["l2_ratio"]
                    
            # Checkpointing
            if checkpoint_dir and self.step % checkpoint_interval == 0:
                self.save_checkpoint(checkpoint_dir / f"checkpoint_{self.step}.pt")
                
        # Final checkpoint
        if checkpoint_dir:
            self.save_checkpoint(checkpoint_dir / "checkpoint_final.pt")
            
        print(f"Training complete. Best L2 ratio: {self.best_l2_ratio:.4f}")
        
        if self.use_wandb:
            wandb.finish()
            
    def save_checkpoint(self, path: Path):
        """Save training checkpoint."""
        checkpoint = {
            "step": self.step,
            "model_state_dict": self.sae.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_l2_ratio": self.best_l2_ratio,
            "config": {
                "d_model": self.sae.d_model,
                "d_sae": self.sae.d_sae,
                "k": self.sae.k
            }
        }
        torch.save(checkpoint, path)
        print(f"Saved checkpoint to {path}")
        
    def load_checkpoint(self, path: Path):
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.sae.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.step = checkpoint["step"]
        self.best_l2_ratio = checkpoint.get("best_l2_ratio", float('inf'))
        print(f"Loaded checkpoint from {path} (step {self.step})")


def load_sae_from_checkpoint(
    checkpoint_path: Path,
    device: str = "cuda"
) -> TopKSAE:
    """
    Load a trained SAE from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load to
        
    Returns:
        Loaded SAE model
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["config"]
    
    sae = TopKSAE(
        d_model=config["d_model"],
        d_sae=config["d_sae"],
        k=config["k"]
    )
    sae.load_state_dict(checkpoint["model_state_dict"])
    sae.to(device)
    sae.eval()
    
    return sae
