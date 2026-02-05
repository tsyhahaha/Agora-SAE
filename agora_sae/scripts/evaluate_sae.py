#!/usr/bin/env python3
"""
Evaluate trained SAE on reconstruction and interpretability.

Usage:
    agora-eval --checkpoint ./checkpoints/checkpoint_final.pt
"""

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from agora_sae.trainer.sae_trainer import load_sae_from_checkpoint
from agora_sae.eval.eval_sae import (
    evaluate_reconstruction, 
    browse_features, 
    print_feature_analysis,
    analyze_feature_utilization
)
from agora_sae.trainer.shard_loader import ShardLoader


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained SAE")
    
    parser.add_argument(
        "--checkpoint", 
        type=str, 
        required=True,
        help="Path to SAE checkpoint"
    )
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
        help="Layer to hook for evaluation"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=500,
        help="Maximum samples for perplexity evaluation"
    )
    parser.add_argument(
        "--shards",
        type=str,
        default=None,
        help="Shard directory for feature utilization analysis"
    )
    parser.add_argument(
        "--test-text",
        type=str,
        default=None,
        help="Test text for feature browsing"
    )
    parser.add_argument(
        "--skip-ppl",
        action="store_true",
        help="Skip perplexity evaluation"
    )
    
    args = parser.parse_args()
    
    # Load SAE
    print(f"Loading SAE from {args.checkpoint}...")
    sae = load_sae_from_checkpoint(args.checkpoint)
    print(f"SAE loaded: d_model={sae.d_model}, d_sae={sae.d_sae}, k={sae.k}")
    
    # 1. Perplexity evaluation
    if not args.skip_ppl:
        print("\n" + "="*60)
        print("PERPLEXITY EVALUATION")
        print("="*60)
        
        results = evaluate_reconstruction(
            model_name=args.model,
            sae=sae,
            hook_layer=args.layer,
            max_samples=args.max_samples
        )
        
        # Summary
        all_passed = all(r["passed"] for r in results.values())
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        for ds, metrics in results.items():
            status = "✓ PASS" if metrics["passed"] else "✗ FAIL"
            print(f"{ds}: {metrics['ppl_increase_pct']:.2f}% increase - {status}")
        print(f"\nOverall: {'✓ PASS' if all_passed else '✗ FAIL'}")
    
    # 2. Feature utilization analysis
    if args.shards:
        print("\n" + "="*60)
        print("FEATURE UTILIZATION ANALYSIS")
        print("="*60)
        
        shard_loader = ShardLoader(
            shard_dir=Path(args.shards),
            batch_size=4096,
            delete_after_read=False
        )
        
        stats = analyze_feature_utilization(sae, shard_loader)
        
        print(f"Always On (>90% frequency): {stats['always_on']} ({stats['always_on_pct']:.1f}%)")
        print(f"Always Off (<0.1% frequency): {stats['always_off']} ({stats['always_off_pct']:.1f}%)")
        print(f"Healthy: {stats['healthy']} ({stats['healthy_pct']:.1f}%)")
    
    # 3. Feature browsing
    if args.test_text:
        print("\n" + "="*60)
        print("FEATURE BROWSING")
        print("="*60)
        
        print("Loading model and tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        model.eval()
        
        print(f"\nAnalyzing text: {args.test_text[:100]}...")
        
        results = browse_features(
            sae=sae,
            text=args.test_text,
            tokenizer=tokenizer,
            model=model,
            hook_layer=args.layer
        )
        
        print_feature_analysis(results)
    
    # Interactive mode
    if args.test_text is None and not args.skip_ppl:
        print("\nEntering interactive feature browsing mode...")
        print("Enter text to analyze (or 'quit' to exit):")
        
        tokenizer = None
        model = None
        
        while True:
            try:
                text = input("\n> ")
                if text.lower() == 'quit':
                    break
                    
                if tokenizer is None:
                    print("Loading model and tokenizer...")
                    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
                    model = AutoModelForCausalLM.from_pretrained(
                        args.model,
                        torch_dtype=torch.bfloat16,
                        device_map="auto",
                        trust_remote_code=True
                    )
                    model.eval()
                    
                results = browse_features(
                    sae=sae,
                    text=text,
                    tokenizer=tokenizer,
                    model=model,
                    hook_layer=args.layer
                )
                
                print_feature_analysis(results)
                
            except KeyboardInterrupt:
                break
                
    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
