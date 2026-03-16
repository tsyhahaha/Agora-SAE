#!/usr/bin/env python3
"""
Stage 2: SAE Training

Train SAE on pre-generated activation shards.

Usage:
    agora-train --preset deepseek-1.5b --shards ./buffer_shards/
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import List

from agora_sae.config import Config, ModelConfig, SAEConfig, StorageConfig, TrainingConfig, get_config
from agora_sae.layer_scan import (
    LayerTrainingPlan,
    build_layer_training_plans,
    infer_final_layer,
    resolve_scan_layers,
    write_scan_manifest,
)
from agora_sae.model.topk_sae import TopKSAE, TopKSAEWithResampling
from agora_sae.trainer.sae_trainer import SAETrainer
from agora_sae.trainer.shard_loader import InfiniteShardLoader, ShardLoader


PRESET_CHOICES = ["deepseek-1.5b", "qwen3-8b", "qwq-32b", "math500-1.5b"]


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(description="Train SAE on activation shards")

    # Data arguments
    parser.add_argument(
        "--shards",
        type=str,
        default="./buffer_shards",
        help="Directory containing activation shards",
    )
    parser.add_argument(
        "--shards-template",
        type=str,
        default=None,
        help="Layer-specific shard template, e.g. ./data/acts/layer_{layer}",
    )
    parser.add_argument(
        "--delete-after-read",
        action="store_true",
        help="Delete shards after reading (for rolling buffer mode)",
    )

    # Layer arguments
    parser.add_argument(
        "--layer",
        type=int,
        default=None,
        help="Single hook layer override for this training run",
    )
    parser.add_argument(
        "--layers",
        type=str,
        default=None,
        help="Comma-separated layers to train, e.g. 12,16,20,24",
    )
    parser.add_argument(
        "--layer-range",
        type=str,
        default=None,
        help="Layer range to scan, e.g. 8:24 or 8:24:4",
    )
    parser.add_argument(
        "--layer-step",
        type=int,
        default=None,
        help="Step size for layer scans; if used alone, scans 0..final_layer",
    )
    parser.add_argument(
        "--final-layer",
        type=int,
        default=None,
        help="Final layer index override. Needed when scanning without a preset layer count",
    )

    # Model arguments
    parser.add_argument(
        "--d-model",
        type=int,
        default=1536,
        help="Model hidden dimension",
    )
    parser.add_argument(
        "--expansion",
        type=int,
        default=32,
        help="SAE expansion factor",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=32,
        help="Top-K sparsity",
    )
    parser.add_argument(
        "--aux-weight",
        type=float,
        default=1 / 32,
        help="Auxiliary loss weight",
    )
    parser.add_argument(
        "--use-resampling",
        action="store_true",
        help="Use dead latent resampling instead of just aux loss",
    )

    # Training arguments
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4096,
        help="Training batch size",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-5,
        help="Learning rate",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=100000,
        help="Total training steps",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=5000,
        help="Warmup steps",
    )

    # Logging arguments
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="agora-sae",
        help="Wandb project name",
    )
    parser.add_argument(
        "--wandb-run",
        type=str,
        default=None,
        help="Wandb run name",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable wandb logging",
    )

    # Checkpoint arguments
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="./checkpoints",
        help="Directory for checkpoints",
    )
    parser.add_argument(
        "--checkpoint-template",
        type=str,
        default=None,
        help="Layer-specific checkpoint template, e.g. ./checkpoints/layer_{layer}",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=5000,
        help="Steps between checkpoints",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from checkpoint or pattern containing {layer}",
    )
    parser.add_argument(
        "--resume-template",
        type=str,
        default=None,
        help="Layer-specific resume template, e.g. ./checkpoints/layer_{layer}/checkpoint_final.pt",
    )
    parser.add_argument(
        "--manifest-out",
        type=str,
        default=None,
        help="Where to write the multi-layer scan manifest",
    )

    # Preset argument
    parser.add_argument(
        "--preset",
        type=str,
        choices=PRESET_CHOICES,
        default=None,
        help="Use a preset configuration",
    )

    return parser


def build_base_config(args: argparse.Namespace) -> Config:
    """Construct the base training config from CLI arguments."""
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
                aux_loss_weight=args.aux_weight,
            ),
            training=TrainingConfig(
                lr=args.lr,
                batch_size=args.batch_size,
                total_steps=args.steps,
                warmup_steps=args.warmup_steps,
            ),
            storage=StorageConfig(
                storage_path=Path(args.shards),
                delete_after_read=args.delete_after_read,
            ),
            wandb_project=args.wandb_project,
            wandb_run_name=args.wandb_run,
        )

    if args.layer is not None:
        config.model.hook_layer = args.layer

    return config


def resolve_training_plans(args: argparse.Namespace, config: Config) -> List[LayerTrainingPlan]:
    """Resolve the layer-specific training plans for this invocation."""
    final_layer = infer_final_layer(
        model_num_layers=config.model.num_layers,
        final_layer_override=args.final_layer,
    )
    layers = resolve_scan_layers(
        default_layer=config.model.hook_layer,
        layers_spec=args.layers,
        layer_range_spec=args.layer_range,
        layer_step=args.layer_step,
        final_layer=final_layer,
    )

    is_multi_layer = len(layers) > 1
    if is_multi_layer and args.resume and not args.resume_template and "{layer}" not in args.resume:
        raise ValueError(
            "In multi-layer mode, --resume must contain '{layer}' or you must use --resume-template."
        )

    plans = build_layer_training_plans(
        layers=layers,
        shards_path=args.shards,
        checkpoint_dir=args.checkpoint_dir,
        shards_template=args.shards_template,
        checkpoint_template=args.checkpoint_template,
        resume=args.resume,
        resume_template=args.resume_template,
        append_layer_subdirs=is_multi_layer,
    )

    if is_multi_layer:
        manifest_base = Path(args.checkpoint_dir)
        if "{layer}" in args.checkpoint_dir:
            manifest_base = manifest_base.parent
        manifest_path = (
            Path(args.manifest_out)
            if args.manifest_out
            else manifest_base / "scan_manifest.json"
        )
        write_scan_manifest(
            path=manifest_path,
            preset=args.preset,
            model_name=config.model.model_name,
            layers=layers,
            plans=plans,
            metadata={
                "batch_size": config.training.batch_size,
                "lr": config.training.lr,
                "total_steps": config.training.total_steps,
                "warmup_steps": config.training.warmup_steps,
                "default_layer": config.model.hook_layer,
                "final_layer": final_layer,
            },
        )
        print(f"Wrote layer-scan manifest to {manifest_path}")

    return plans


def print_training_configuration(config: Config, plan: LayerTrainingPlan, run_index: int, num_runs: int) -> None:
    """Print the effective configuration for a layer training run."""
    print("=" * 60)
    print(f"SAE TRAINING CONFIGURATION ({run_index}/{num_runs})")
    print("=" * 60)
    print(f"Model: {config.model.model_name}")
    print(f"Layer: {config.model.hook_layer}")
    print(f"d_model: {config.model.d_model}")
    print(f"d_sae: {config.d_sae}")
    print(f"k: {config.sae.k}")
    print(f"Expansion: {config.sae.expansion_factor}x")
    print(f"Batch Size: {config.training.batch_size}")
    print(f"Learning Rate: {config.training.lr}")
    print(f"Total Steps: {config.training.total_steps}")
    print(f"Warmup Steps: {config.training.warmup_steps}")
    print(f"Aux Loss Weight: {config.sae.aux_loss_weight}")
    print(f"Shards: {plan.shards_dir}")
    print(f"Checkpoint Dir: {plan.checkpoint_dir}")
    if plan.resume_path:
        print(f"Resume: {plan.resume_path}")
    print("=" * 60)


def create_sae_model(config: Config, use_resampling: bool):
    """Instantiate the requested SAE model."""
    print("\nInitializing SAE model...")
    if use_resampling:
        sae = TopKSAEWithResampling(
            d_model=config.model.d_model,
            d_sae=config.d_sae,
            k=config.sae.k,
            aux_loss_weight=config.sae.aux_loss_weight,
        )
        print("Using TopKSAE with resampling")
        return sae

    sae = TopKSAE(
        d_model=config.model.d_model,
        d_sae=config.d_sae,
        k=config.sae.k,
        aux_loss_weight=config.sae.aux_loss_weight,
    )
    print("Using TopKSAE with auxiliary loss only")
    return sae


def create_shard_loader(config: Config):
    """Create the shard loader for a single training run."""
    shard_dir = Path(config.storage.storage_path)
    if not shard_dir.exists():
        raise FileNotFoundError(f"Activation shard directory does not exist: {shard_dir}")

    print("Creating shard loader...")
    if config.storage.delete_after_read:
        return ShardLoader(
            shard_dir=shard_dir,
            batch_size=config.training.batch_size,
            delete_after_read=True,
        )
    return InfiniteShardLoader(
        shard_dir=shard_dir,
        batch_size=config.training.batch_size,
        max_epochs=10,
    )


def train_one_layer(
    *,
    args: argparse.Namespace,
    base_config: Config,
    plan: LayerTrainingPlan,
    run_index: int,
    num_runs: int,
) -> None:
    """Train one SAE for a single layer plan."""
    config = copy.deepcopy(base_config)
    config.model.hook_layer = plan.layer
    config.storage.storage_path = Path(plan.shards_dir)
    config.storage.delete_after_read = args.delete_after_read
    if num_runs > 1 and config.wandb_run_name:
        config.wandb_run_name = f"{config.wandb_run_name}-layer{plan.layer}"

    print_training_configuration(config, plan, run_index=run_index, num_runs=num_runs)

    sae = create_sae_model(config, use_resampling=args.use_resampling)
    shard_loader = create_shard_loader(config)

    print("Initializing trainer...")
    trainer = SAETrainer(
        sae=sae,
        config=config,
        device="cuda",
        use_wandb=not args.no_wandb,
    )

    if plan.resume_path:
        trainer.load_checkpoint(Path(plan.resume_path))

    print("\nStarting training...")
    try:
        trainer.train(
            shard_loader=shard_loader,
            checkpoint_dir=plan.checkpoint_dir,
            checkpoint_interval=args.checkpoint_interval,
        )
    except KeyboardInterrupt:
        print("\nInterrupted. Saving checkpoint...")
        trainer.save_checkpoint(plan.checkpoint_dir / "checkpoint_interrupted.pt")
        raise

    print("\nTraining complete!")
    print(f"Best L2 ratio: {trainer.best_l2_ratio:.4f}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    base_config = build_base_config(args)
    plans = resolve_training_plans(args, base_config)

    if len(plans) > 1:
        print(f"Resolved multi-layer training plan for layers: {[plan.layer for plan in plans]}")

    for run_index, plan in enumerate(plans, start=1):
        train_one_layer(
            args=args,
            base_config=base_config,
            plan=plan,
            run_index=run_index,
            num_runs=len(plans),
        )


if __name__ == "__main__":
    main()
