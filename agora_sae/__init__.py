# Agora-SAE: Staged Offline SAE Training System
# For reasoning model interpretability

__version__ = "0.1.0"

from .config import Config, get_config, PRESETS
from .model.topk_sae import TopKSAE, TopKSAEWithResampling
from .trainer.sae_trainer import SAETrainer, load_sae_from_checkpoint
from .trainer.shard_loader import ShardLoader, InfiniteShardLoader
from .data.mixed_source import MixedTokenSource
from .activation.generator import OfflineActivationGenerator

__all__ = [
    "Config",
    "get_config",
    "PRESETS",
    "TopKSAE",
    "TopKSAEWithResampling",
    "SAETrainer",
    "load_sae_from_checkpoint",
    "ShardLoader",
    "InfiniteShardLoader",
    "MixedTokenSource",
    "OfflineActivationGenerator",
]
