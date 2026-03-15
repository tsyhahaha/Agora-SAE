"""
Module B: Activation Generator (Stage 1)

Offline activation generation with global shuffle buffer and safetensors storage.
"""

import time
from pathlib import Path
from typing import Callable, List, Optional
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from safetensors.torch import save_file
from tqdm import tqdm
from transformers import AutoModel


class ActivationBuffer:
    """
    Global shuffle buffer for activations.
    
    Maintains a fixed-size buffer, shuffles when full, and writes to disk.
    """
    
    def __init__(
        self,
        buffer_size_mb: int = 500,
        d_model: int = 1536,
        storage_path: Path = Path("./buffer_shards"),
        shard_size_mb: int = 100,
        storage_dtype: torch.dtype = torch.bfloat16,
    ):
        """
        Initialize the activation buffer.
        
        Args:
            buffer_size_mb: Size of the in-memory buffer in MB
            d_model: Model hidden dimension
            storage_path: Path to store shard files
            shard_size_mb: Size of each shard file in MB
        """
        self.d_model = d_model
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.storage_dtype = storage_dtype
        bytes_per_activation = d_model * torch.tensor([], dtype=storage_dtype).element_size()
        self.buffer_capacity = (buffer_size_mb * 1024 * 1024) // bytes_per_activation
        self.shard_capacity = (shard_size_mb * 1024 * 1024) // bytes_per_activation

        self.buffer: List[torch.Tensor] = []
        self.buffer_rows = 0
        self.shard_counter = 0

    def add(self, activations: torch.Tensor):
        """
        Add activations to buffer.
        
        Args:
            activations: Tensor of shape [batch_size, seq_len, d_model] or [N, d_model]
        """
        # Flatten to [N, d_model] if needed
        if activations.dim() == 3:
            activations = activations.reshape(-1, self.d_model)

        activations = activations.to(dtype=self.storage_dtype, device="cpu")
        self.buffer.append(activations)
        self.buffer_rows += activations.shape[0]

        if self.buffer_rows >= self.buffer_capacity:
            self._flush()

    def _flush(self):
        """Shuffle buffer and write to disk."""
        if not self.buffer:
            return

        all_activations = torch.cat(self.buffer, dim=0)
        perm = torch.randperm(all_activations.shape[0])
        all_activations = all_activations[perm]

        n_shards = (all_activations.shape[0] + self.shard_capacity - 1) // self.shard_capacity

        for i in range(n_shards):
            start_idx = i * self.shard_capacity
            end_idx = min((i + 1) * self.shard_capacity, all_activations.shape[0])
            shard_data = all_activations[start_idx:end_idx]

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            shard_path = self.storage_path / f"shard_{timestamp}_{self.shard_counter:06d}.safetensors"
            save_file({"activations": shard_data}, str(shard_path))
            self.shard_counter += 1

        self.buffer = []
        self.buffer_rows = 0

    def finalize(self):
        """Flush any remaining data in buffer."""
        if self.buffer:
            self._flush()

    def get_disk_usage_gb(self) -> float:
        """Get current disk usage in GB."""
        total_size = 0
        for f in self.storage_path.glob("*.safetensors"):
            total_size += f.stat().st_size
        return total_size / (1024 ** 3)


class OfflineActivationGenerator:
    """
    Offline activation generator for LLM hidden states.
    
    Runs inference on the model, extracts activations from a target layer,
    and saves them as shuffled safetensors shards.
    """
    
    def __init__(
        self,
        model_name: str,
        hook_layer: int,
        d_model: int,
        storage_path: Path,
        buffer_size_mb: int = 500,
        shard_size_mb: int = 100,
        max_disk_usage_gb: int = 200,
        device: str = "cuda"
    ):
        """
        Initialize the activation generator.
        
        Args:
            model_name: HuggingFace model name
            hook_layer: Layer index to extract activations from
            d_model: Model hidden dimension
            storage_path: Path to store shard files
            buffer_size_mb: Size of in-memory shuffle buffer
            shard_size_mb: Size of each shard file
            max_disk_usage_gb: Maximum disk usage before pausing
            device: Device to run model on
        """
        self.model_name = model_name
        self.hook_layer = hook_layer
        self.d_model = d_model
        self.max_disk_usage_gb = max_disk_usage_gb
        self.device = device

        # Initialize buffer
        self.buffer = ActivationBuffer(
            buffer_size_mb=buffer_size_mb,
            d_model=d_model,
            storage_path=storage_path,
            shard_size_mb=shard_size_mb,
        )

        # Will be set during model loading
        self.model = None
        self.input_device = None
        self._hook_handle = None
        self._captured_activations = None

    def load_model(self):
        """Load the model in BF16 with inference mode."""
        print(f"Loading model: {self.model_name}")

        self.model = AutoModel.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        self.model.eval()
        self.input_device = self._get_input_device()
        self._register_hook()

        print(f"Model loaded. Hook registered on layer {self.hook_layer}")

    def _get_input_device(self) -> torch.device:
        """Get the device that should receive model inputs."""
        if hasattr(self.model, "get_input_embeddings"):
            input_embeddings = self.model.get_input_embeddings()
            if input_embeddings is not None:
                return input_embeddings.weight.device
        return next(self.model.parameters()).device

    def _register_hook(self):
        """Register forward hook on target layer."""
        # Get the target layer
        # Most models use model.layers[i] or model.model.layers[i]
        try:
            if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
                target_layer = self.model.model.layers[self.hook_layer]
            elif hasattr(self.model, 'layers'):
                target_layer = self.model.layers[self.hook_layer]
            elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
                target_layer = self.model.transformer.h[self.hook_layer]
            else:
                raise ValueError(f"Could not find layers in model architecture")
        except IndexError:
            raise ValueError(f"Layer {self.hook_layer} does not exist in model")

        def hook_fn(module, inputs, output):
            # Capture the residual stream (input to the layer)
            if isinstance(inputs, tuple):
                self._captured_activations = inputs[0].detach()
            else:
                self._captured_activations = inputs.detach()

        self._hook_handle = target_layer.register_forward_hook(hook_fn)

    def _check_disk_usage(self) -> bool:
        """Check if disk usage is below threshold."""
        usage = self.buffer.get_disk_usage_gb()
        return usage < self.max_disk_usage_gb

    def run_generation_loop(
        self,
        dataloader: DataLoader,
        max_batches: Optional[int] = None,
        progress_callback: Optional[Callable] = None
    ):
        """
        Main generation loop.
        
        Args:
            dataloader: DataLoader providing input_ids batches
            max_batches: Maximum number of batches to process (None for unlimited)
            progress_callback: Optional callback for progress updates
        """
        if self.model is None:
            self.load_model()

        batch_count = 0
        total_tokens = 0

        pbar = tqdm(dataloader, desc="Generating activations")

        with torch.inference_mode():
            for batch in pbar:
                # Check disk usage
                if not self._check_disk_usage():
                    print(f"Disk usage exceeded {self.max_disk_usage_gb}GB. Pausing...")
                    # Wait for trainer to consume some shards
                    while not self._check_disk_usage():
                        time.sleep(60)

                # Move batch to device
                input_ids = batch["input_ids"].to(self.input_device)
                attention_mask = batch.get("attention_mask", None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.input_device)

                self._captured_activations = None
                self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                )

                if self._captured_activations is not None:
                    target_device = self._captured_activations.device
                    extraction_mask = batch.get("extraction_mask", None)
                    if extraction_mask is not None:
                        extraction_mask = extraction_mask.to(target_device).bool()
                        valid_activations = self._captured_activations[extraction_mask]
                    elif attention_mask is not None:
                        mask = attention_mask.to(target_device).bool()
                        valid_activations = self._captured_activations[mask]
                    else:
                        valid_activations = self._captured_activations.reshape(-1, self.d_model)

                    if valid_activations.numel() > 0:
                        self.buffer.add(valid_activations)
                        total_tokens += valid_activations.shape[0]

                batch_count += 1
                pbar.set_postfix({
                    "tokens": f"{total_tokens/1e6:.1f}M",
                    "shards": self.buffer.shard_counter
                })

                if progress_callback:
                    progress_callback(batch_count, total_tokens)

                if max_batches and batch_count >= max_batches:
                    break

        self.buffer.finalize()
        print(f"Generation complete. Total tokens: {total_tokens/1e6:.1f}M, Shards: {self.buffer.shard_counter}")

    def cleanup(self):
        """Clean up resources."""
        if self._hook_handle:
            self._hook_handle.remove()
        self.model = None
        self.input_device = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
