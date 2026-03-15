#!/usr/bin/env python3
"""
Stage 2: SAE Training

Train SAE on pre-generated activation shards.

Usage:
    agora-train --preset deepseek-1.5b --shards ./buffer_shards/
"""

import argparse
from pathlib import Path

from agora_sae.config import Config, ModelConfig, SAEConfig, TrainingConfig, StorageConfig, get_config
from agora_sae.model.topk_sae import TopKSAE, TopKSAEWithResampling
from agora_sae.trainer.shard_loader import ShardLoader, InfiniteShardLoader
from agora_sae.trainer.sae_trainer import SAETrainer


def main():
    parser = argparse.ArgumentParser(description="Train SAE on activation shards")
    
    # Data arguments
    parser.add_argument(
        "--shards",
        type=str,
        default="./buffer_shards",
        help="Directory containing activation shards"
    )
    parser.add_argument(
        "--delete-after-read",
        action="store_true",
        help="Delete shards after reading (for rolling buffer mode)"
    )
    
    # Model arguments
    parser.add_argument(
        "--d-model",
        type=int,
        default=1536,
        help="Model hidden dimension"
    )
    parser.add_argument(
        "--expansion",
        type=int,
        default=32,
        help="SAE expansion factor"
    )
    parser.add_argument(
        "--k",
        type=int,
        default=32,
        help="Top-K sparsity"
    )
    parser.add_argument(
        "--aux-weight",
        type=float,
        default=1/32,
        help="Auxiliary loss weight"
    )
    parser.add_argument(
        "--use-resampling",
        action="store_true",
        help="Use dead latent resampling instead of just aux loss"
    )
    
    # Training arguments
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4096,
        help="Training batch size"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-5,
        help="Learning rate"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=100000,
        help="Total training steps"
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=5000,
        help="Warmup steps"
    )
    
    # Logging arguments
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="agora-sae",
        help="Wandb project name"
    )
    parser.add_argument(
        "--wandb-run",
        type=str,
        default=None,
        help="Wandb run name"
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable wandb logging"
    )
    
    # Checkpoint arguments
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="./checkpoints",
        help="Directory for checkpoints"
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=5000,
        help="Steps between checkpoints"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from checkpoint"
    )
    
    # Preset argument
    parser.add_argument(
        "--preset",
        type=str,
        choices=["deepseek-1.5b", "qwen3-8b", "qwq-32b", "math500-1.5b"],
        default=None,
        help="Use a preset configuration"
    )
    
    args = parser.parse_args()
    
    # Build configuration
    if args.preset:
        config = get_config(args.preset)

        config.storage.storage_path = Path(args.shards)
        config.storage.delete_after_read = args.delete_after_read
        config.training.batch_size = args.batch_size
        config.training.lr = args.lr
        config.training.total_steps = args.steps
        config.training.warmup_steps = args.warmup_steps
        config.wandb_project = args.wandb_project
        config.wandb_run_name = args.wandb_run
    else:
        config = Config(
            model=ModelConfig(d_model=args.d_model),
            sae=SAEConfig(
                expansion_factor=args.expansion,
                k=args.k,
                aux_loss_weight=args.aux_weight
            ),
            training=TrainingConfig(
                lr=args.lr,
                batch_size=args.batch_size,
                total_steps=args.steps,
                warmup_steps=args.warmup_steps
            ),
            storage=StorageConfig(
                storage_path=Path(args.shards),
                delete_after_read=args.delete_after_read
            ),
            wandb_project=args.wandb_project,
            wandb_run_name=args.wandb_run
        )

    config.storage.storage_path = Path(config.storage.storage_path)
    config.storage.storage_path.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("SAE TRAINING CONFIGURATION")
    print("="*60)
    print(f"d_model: {config.model.d_model}")
    print(f"d_sae: {config.d_sae}")
    print(f"k: {config.sae.k}")
    print(f"Expansion: {config.sae.expansion_factor}x")
    print(f"Batch Size: {config.training.batch_size}")
    print(f"Learning Rate: {config.training.lr}")
    print(f"Total Steps: {config.training.total_steps}")
    print(f"Warmup Steps: {config.training.warmup_steps}")
    print(f"Aux Loss Weight: {config.sae.aux_loss_weight}")
    print(f"Shards: {config.storage.storage_path}")
    print("="*60)
    
    # Create SAE model
    print("\nInitializing SAE model...")
    if args.use_resampling:
        sae = TopKSAEWithResampling(
            d_model=config.model.d_model,
            d_sae=config.d_sae,
            k=config.sae.k,
            aux_loss_weight=config.sae.aux_loss_weight
        )
        print("Using TopKSAE with resampling")
    else:
        sae = TopKSAE(
            d_model=config.model.d_model,
            d_sae=config.d_sae,
            k=config.sae.k,
            aux_loss_weight=config.sae.aux_loss_weight
        )
        print("Using TopKSAE with auxiliary loss only")
    
    # Create shard loader
    print("Creating shard loader...")
    if args.delete_after_read:
        shard_loader = ShardLoader(
            shard_dir=config.storage.storage_path,
            batch_size=config.training.batch_size,
            delete_after_read=True
        )
    else:
        shard_loader = InfiniteShardLoader(
            shard_dir=config.storage.storage_path,
            batch_size=config.training.batch_size,
            max_epochs=10
        )
    
    # Create trainer
    print("Initializing trainer...")
    trainer = SAETrainer(
        sae=sae,
        config=config,
        device="cuda",
        use_wandb=not args.no_wandb
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(Path(args.resume))
    
    # Train
    print("\nStarting training...")
    try:
        trainer.train(
            shard_loader=shard_loader,
            checkpoint_dir=Path(args.checkpoint_dir),
            checkpoint_interval=args.checkpoint_interval
        )
    except KeyboardInterrupt:
        print("\nInterrupted. Saving checkpoint...")
        trainer.save_checkpoint(Path(args.checkpoint_dir) / "checkpoint_interrupted.pt")
        
    print("\nTraining complete!")
    print(f"Best L2 ratio: {trainer.best_l2_ratio:.4f}")


if __name__ == "__main__":
    main()
