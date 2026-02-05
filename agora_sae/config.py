"""
Global configuration for SAE training system.
"""

from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path


@dataclass
class ModelConfig:
    """Configuration for the target LLM model."""
    model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    hook_layer: int = 12  # Layer to extract activations from
    d_model: int = 1536   # Model hidden dimension
    

@dataclass
class DataConfig:
    """Configuration for data ingestion."""
    reasoning_datasets: List[str] = field(default_factory=lambda: [
        "open-r1/OpenR1-Math-220k",
        "openai/gsm8k"
    ])
    general_datasets: List[str] = field(default_factory=lambda: [
        "HuggingFaceFW/fineweb-edu"
    ])
    reasoning_ratio: float = 0.8  # 80% reasoning, 20% general
    max_seq_length: int = 2048
    question_sample_prob: float = 0.1  # Probability to keep pure question parts


@dataclass
class SAEConfig:
    """Configuration for SAE architecture."""
    expansion_factor: int = 32  # d_sae = expansion_factor * d_model
    k: int = 32  # Top-K sparsity
    aux_loss_weight: float = 1/32  # Lambda for auxiliary loss
    
    @property
    def d_sae(self) -> int:
        """Calculate SAE dimension based on expansion factor."""
        # This will be set when combined with ModelConfig
        return None


@dataclass
class TrainingConfig:
    """Configuration for training."""
    lr: float = 5e-5
    batch_size: int = 4096
    total_steps: int = 100000
    warmup_steps: int = 5000  # ~5% of total steps
    weight_decay: float = 0.0
    dead_latent_check_interval: int = 2500
    dead_latent_threshold: float = 0.10  # 10%


@dataclass
class StorageConfig:
    """Configuration for activation storage."""
    storage_path: Path = field(default_factory=lambda: Path("./buffer_shards"))
    shard_size_mb: int = 100  # Size per shard file in MB
    buffer_size_mb: int = 500  # Global shuffle buffer size
    max_disk_usage_gb: int = 200  # Max disk usage before pausing generation
    delete_after_read: bool = True  # Delete shards after training consumes them


@dataclass
class Config:
    """Main configuration combining all sub-configs."""
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    sae: SAEConfig = field(default_factory=SAEConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    
    # Wandb settings
    wandb_project: str = "agora-sae"
    wandb_run_name: Optional[str] = None
    
    @property
    def d_sae(self) -> int:
        """Calculate SAE dimension."""
        return self.sae.expansion_factor * self.model.d_model
    
    def __post_init__(self):
        """Ensure storage path exists."""
        self.storage.storage_path = Path(self.storage.storage_path)
        self.storage.storage_path.mkdir(parents=True, exist_ok=True)


# Preset configurations for different models
PRESETS = {
    "deepseek-1.5b": Config(
        model=ModelConfig(
            model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
            hook_layer=12,
            d_model=1536
        ),
        sae=SAEConfig(expansion_factor=32, k=32)
    ),
    "qwen3-8b": Config(
        model=ModelConfig(
            model_name="Qwen/Qwen3-8B",
            hook_layer=16,
            d_model=4096
        ),
        sae=SAEConfig(expansion_factor=32, k=64)
    ),
    "qwq-32b": Config(
        model=ModelConfig(
            model_name="Qwen/QwQ-32B",
            hook_layer=24,
            d_model=5120
        ),
        sae=SAEConfig(expansion_factor=32, k=64)
    )
}


def get_config(preset: str = "deepseek-1.5b") -> Config:
    """Get a preset configuration."""
    if preset not in PRESETS:
        raise ValueError(f"Unknown preset: {preset}. Available: {list(PRESETS.keys())}")
    return PRESETS[preset]
