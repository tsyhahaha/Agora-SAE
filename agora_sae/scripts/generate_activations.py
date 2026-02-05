#!/usr/bin/env python3
"""
Stage 1: Offline Activation Generation

Generate and save activations from a target layer of the LLM.

Usage:
    agora-generate --preset deepseek-1.5b --output ./buffer_shards/
"""

import argparse
from pathlib import Path

from transformers import AutoTokenizer

from agora_sae.config import Config, ModelConfig, DataConfig, StorageConfig, get_config
from agora_sae.data.mixed_source import create_dataloader
from agora_sae.activation.generator import OfflineActivationGenerator


def main():
    parser = argparse.ArgumentParser(description="Generate activations for SAE training")
    
    # Model arguments
    parser.add_argument(
        "--model", 
        type=str, 
        default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        help="HuggingFace model name"
    )
    parser.add_argument(
        "--layer", 
        type=int, 
        default=12,
        help="Layer to extract activations from"
    )
    parser.add_argument(
        "--d-model",
        type=int,
        default=1536,
        help="Model hidden dimension"
    )
    
    # Data arguments
    parser.add_argument(
        "--reasoning-datasets",
        type=str,
        nargs="+",
        default=["open-r1/OpenR1-Math-220k"],
        help="Reasoning dataset names"
    )
    parser.add_argument(
        "--general-datasets",
        type=str,
        nargs="+",
        default=["HuggingFaceFW/fineweb-edu"],
        help="General dataset names"
    )
    parser.add_argument(
        "--reasoning-ratio",
        type=float,
        default=0.8,
        help="Ratio of reasoning data (default: 0.8)"
    )
    
    # Storage arguments
    parser.add_argument(
        "--output",
        type=str,
        default="./buffer_shards",
        help="Output directory for shard files"
    )
    parser.add_argument(
        "--buffer-size-mb",
        type=int,
        default=500,
        help="Size of in-memory shuffle buffer in MB"
    )
    parser.add_argument(
        "--shard-size-mb",
        type=int,
        default=100,
        help="Size of each shard file in MB"
    )
    parser.add_argument(
        "--max-disk-gb",
        type=int,
        default=200,
        help="Maximum disk usage in GB before pausing"
    )
    
    # Generation arguments
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for inference"
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Maximum number of batches (None for unlimited)"
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=2048,
        help="Maximum sequence length"
    )
    
    # Preset argument
    parser.add_argument(
        "--preset",
        type=str,
        choices=["deepseek-1.5b", "qwen3-8b", "qwq-32b"],
        default=None,
        help="Use a preset configuration"
    )
    
    args = parser.parse_args()
    
    # Build configuration
    if args.preset:
        config = get_config(args.preset)
    else:
        config = Config(
            model=ModelConfig(
                model_name=args.model,
                hook_layer=args.layer,
                d_model=args.d_model
            ),
            data=DataConfig(
                reasoning_datasets=args.reasoning_datasets,
                general_datasets=args.general_datasets,
                reasoning_ratio=args.reasoning_ratio,
                max_seq_length=args.max_seq_length
            ),
            storage=StorageConfig(
                storage_path=Path(args.output),
                buffer_size_mb=args.buffer_size_mb,
                shard_size_mb=args.shard_size_mb,
                max_disk_usage_gb=args.max_disk_gb
            )
        )
    
    print("="*60)
    print("ACTIVATION GENERATION CONFIGURATION")
    print("="*60)
    print(f"Model: {config.model.model_name}")
    print(f"Hook Layer: {config.model.hook_layer}")
    print(f"d_model: {config.model.d_model}")
    print(f"Reasoning Ratio: {config.data.reasoning_ratio}")
    print(f"Output: {config.storage.storage_path}")
    print(f"Buffer Size: {config.storage.buffer_size_mb} MB")
    print(f"Shard Size: {config.storage.shard_size_mb} MB")
    print("="*60)
    
    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.model_name,
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create dataloader
    print("Creating data pipeline...")
    dataloader = create_dataloader(
        config,
        tokenizer,
        batch_size=args.batch_size
    )
    
    # Create generator
    print("Initializing activation generator...")
    generator = OfflineActivationGenerator(
        model_name=config.model.model_name,
        hook_layer=config.model.hook_layer,
        d_model=config.model.d_model,
        storage_path=config.storage.storage_path,
        buffer_size_mb=config.storage.buffer_size_mb,
        shard_size_mb=config.storage.shard_size_mb,
        max_disk_usage_gb=config.storage.max_disk_usage_gb
    )
    
    # Run generation
    print("\nStarting activation generation...")
    try:
        generator.run_generation_loop(
            dataloader,
            max_batches=args.max_batches
        )
    except KeyboardInterrupt:
        print("\nInterrupted by user. Finalizing...")
        generator.buffer.finalize()
    finally:
        generator.cleanup()
        
    print("\nGeneration complete!")
    print(f"Shards saved to: {config.storage.storage_path}")
    print(f"Total shards: {generator.buffer.shard_counter}")


if __name__ == "__main__":
    main()
