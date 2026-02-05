"""
Module E: Analysis & Evaluation

Evaluate SAE reconstruction quality and feature interpretability.
"""

from pathlib import Path
from typing import List, Dict, Tuple, Optional
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

from ..model.topk_sae import TopKSAE


class SAEWrapper(nn.Module):
    """
    Wrapper that replaces a layer in the LLM with SAE reconstruction.
    """
    
    def __init__(self, sae: TopKSAE):
        super().__init__()
        self.sae = sae
        
    def forward(self, x):
        # Apply SAE reconstruction
        x_hat, _, _, _ = self.sae(x.float())
        return x_hat.to(x.dtype)


def evaluate_reconstruction(
    model_name: str,
    sae: TopKSAE,
    hook_layer: int,
    datasets: List[str] = ["wikitext", "openai/gsm8k"],
    max_samples: int = 500,
    max_length: int = 512,
    device: str = "cuda"
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate SAE reconstruction quality by comparing perplexity.
    
    Args:
        model_name: HuggingFace model name
        sae: Trained SAE model
        hook_layer: Layer to insert SAE
        datasets: List of evaluation datasets
        max_samples: Maximum number of samples per dataset
        max_length: Maximum sequence length
        device: Device to run on
        
    Returns:
        Dictionary of results per dataset
    """
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    results = {}
    
    for ds_name in datasets:
        print(f"\nEvaluating on {ds_name}...")
        
        # Load dataset
        try:
            if ds_name == "wikitext":
                ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
                texts = [x["text"] for x in ds if len(x["text"]) > 50]
            else:
                ds = load_dataset(ds_name, split="test")
                # Try different text fields
                texts = []
                for x in ds:
                    for field in ["text", "question", "problem"]:
                        if field in x and x[field]:
                            texts.append(x[field])
                            break
        except Exception as e:
            print(f"Could not load {ds_name}: {e}")
            continue
            
        texts = texts[:max_samples]
        
        # Evaluate without SAE
        ppl_original = _compute_perplexity(
            model_name, tokenizer, texts, max_length, device,
            hook_layer=None, sae=None
        )
        
        # Evaluate with SAE
        ppl_with_sae = _compute_perplexity(
            model_name, tokenizer, texts, max_length, device,
            hook_layer=hook_layer, sae=sae
        )
        
        # Calculate increase
        ppl_increase = ((ppl_with_sae - ppl_original) / ppl_original) * 100
        
        results[ds_name] = {
            "ppl_original": ppl_original,
            "ppl_with_sae": ppl_with_sae,
            "ppl_increase_pct": ppl_increase,
            "passed": ppl_increase < 5.0  # Acceptance: < 5% increase
        }
        
        print(f"  Original PPL: {ppl_original:.2f}")
        print(f"  With SAE PPL: {ppl_with_sae:.2f}")
        print(f"  Increase: {ppl_increase:.2f}%")
        print(f"  Status: {'PASS' if results[ds_name]['passed'] else 'FAIL'}")
        
    return results


def _compute_perplexity(
    model_name: str,
    tokenizer,
    texts: List[str],
    max_length: int,
    device: str,
    hook_layer: Optional[int] = None,
    sae: Optional[TopKSAE] = None
) -> float:
    """Compute perplexity on a list of texts."""
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()
    
    hook_handle = None
    
    if hook_layer is not None and sae is not None:
        sae = sae.to(device)
        sae.eval()
        
        # Get target layer
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            target_layer = model.model.layers[hook_layer]
        elif hasattr(model, 'layers'):
            target_layer = model.layers[hook_layer]
        else:
            raise ValueError("Could not find layers in model")
            
        def hook_fn(module, input, output):
            # Modify the output with SAE reconstruction
            if isinstance(output, tuple):
                hidden_states = output[0]
                original_dtype = hidden_states.dtype
                x_hat, _, _, _ = sae(hidden_states.float())
                return (x_hat.to(original_dtype),) + output[1:]
            else:
                original_dtype = output.dtype
                x_hat, _, _, _ = sae(output.float())
                return x_hat.to(original_dtype)
                
        hook_handle = target_layer.register_forward_hook(hook_fn)
        
    total_loss = 0
    total_tokens = 0
    
    with torch.inference_mode():
        for text in tqdm(texts, desc="Computing PPL"):
            tokens = tokenizer(
                text,
                return_tensors="pt",
                max_length=max_length,
                truncation=True
            ).to(device)
            
            if tokens["input_ids"].shape[1] < 2:
                continue
                
            outputs = model(**tokens, labels=tokens["input_ids"])
            loss = outputs.loss
            
            n_tokens = tokens["input_ids"].shape[1] - 1
            total_loss += loss.item() * n_tokens
            total_tokens += n_tokens
            
    if hook_handle:
        hook_handle.remove()
        
    del model
    torch.cuda.empty_cache()
    
    return math.exp(total_loss / total_tokens) if total_tokens > 0 else float('inf')


def browse_features(
    sae: TopKSAE,
    text: str,
    tokenizer,
    model,
    hook_layer: int,
    top_k: int = 10,
    device: str = "cuda"
) -> List[Dict]:
    """
    Browse top activated features for a given text.
    
    Args:
        sae: Trained SAE model
        text: Input text (e.g., CoT reasoning)
        tokenizer: Tokenizer
        model: LLM model
        hook_layer: Layer to extract activations from
        top_k: Number of top features to return
        device: Device
        
    Returns:
        List of feature activations with metadata
    """
    sae = sae.to(device)
    sae.eval()
    
    # Tokenize
    tokens = tokenizer(text, return_tensors="pt").to(device)
    
    # Capture activations
    captured = {}
    
    def hook_fn(module, input, output):
        if isinstance(input, tuple):
            captured["activations"] = input[0].detach()
        else:
            captured["activations"] = input.detach()
            
    # Get target layer
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        target_layer = model.model.layers[hook_layer]
    elif hasattr(model, 'layers'):
        target_layer = model.layers[hook_layer]
    else:
        raise ValueError("Could not find layers in model")
        
    hook_handle = target_layer.register_forward_hook(hook_fn)
    
    with torch.inference_mode():
        model(**tokens)
        
    hook_handle.remove()
    
    # Get activations
    activations = captured["activations"]  # [1, seq_len, d_model]
    
    # Run through SAE
    x_hat, f, topk_indices, z = sae(activations.float().squeeze(0))
    
    # Aggregate feature activations across sequence
    feature_activations = f.sum(dim=0)  # [d_sae]
    
    # Get top features
    top_values, top_indices = torch.topk(feature_activations, top_k)
    
    results = []
    for i, (idx, val) in enumerate(zip(top_indices.tolist(), top_values.tolist())):
        # Find positions where this feature is most active
        per_position = f[:, idx]  # [seq_len]
        top_positions = torch.topk(per_position, min(5, len(per_position))).indices.tolist()
        
        # Get corresponding tokens
        token_ids = tokens["input_ids"].squeeze(0)
        active_tokens = [tokenizer.decode([token_ids[p]]) for p in top_positions]
        
        results.append({
            "rank": i + 1,
            "feature_index": idx,
            "total_activation": val,
            "active_positions": top_positions,
            "active_tokens": active_tokens
        })
        
    return results


def print_feature_analysis(results: List[Dict]):
    """Pretty print feature analysis results."""
    print("\n" + "="*60)
    print("TOP ACTIVATED FEATURES")
    print("="*60)
    
    for r in results:
        print(f"\n{r['rank']}. Feature #{r['feature_index']}")
        print(f"   Total activation: {r['total_activation']:.2f}")
        print(f"   Active at positions: {r['active_positions']}")
        print(f"   Tokens: {r['active_tokens']}")
        
    print("\n" + "="*60)


def analyze_feature_utilization(
    sae: TopKSAE,
    shard_loader,
    n_batches: int = 100,
    device: str = "cuda"
) -> Dict[str, any]:
    """
    Analyze feature utilization across the dataset.
    
    Returns statistics about feature activation frequency.
    """
    sae = sae.to(device)
    sae.eval()
    
    feature_counts = torch.zeros(sae.d_sae, device=device)
    total_samples = 0
    
    with torch.inference_mode():
        for i, batch in enumerate(shard_loader):
            if i >= n_batches:
                break
                
            batch = batch.to(device)
            x_hat, f, _, _ = sae(batch)
            
            # Count activations per feature
            active = (f > 0).float().sum(dim=0)
            feature_counts += active
            total_samples += batch.shape[0]
            
    # Normalize
    feature_freq = feature_counts / total_samples
    
    # Compute statistics
    always_on = (feature_freq > 0.9).sum().item()
    always_off = (feature_freq < 0.001).sum().item()
    healthy = sae.d_sae - always_on - always_off
    
    return {
        "always_on": always_on,
        "always_off": always_off,
        "healthy": healthy,
        "always_on_pct": always_on / sae.d_sae * 100,
        "always_off_pct": always_off / sae.d_sae * 100,
        "healthy_pct": healthy / sae.d_sae * 100,
        "feature_frequencies": feature_freq.cpu().numpy()
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate SAE")
    parser.add_argument("--checkpoint", type=str, required=True, help="SAE checkpoint path")
    parser.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    parser.add_argument("--layer", type=int, default=12)
    parser.add_argument("--max-samples", type=int, default=500)
    
    args = parser.parse_args()
    
    # Load SAE
    from ..trainer.sae_trainer import load_sae_from_checkpoint
    sae = load_sae_from_checkpoint(args.checkpoint)
    
    # Run evaluation
    results = evaluate_reconstruction(
        args.model,
        sae,
        args.layer,
        max_samples=args.max_samples
    )
    
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    for ds, metrics in results.items():
        status = "✓ PASS" if metrics["passed"] else "✗ FAIL"
        print(f"{ds}: {metrics['ppl_increase_pct']:.2f}% increase - {status}")
