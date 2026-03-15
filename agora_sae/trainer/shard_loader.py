"""
Shard Loader for offline SAE training.

Multi-threaded loading of safetensors activation shards with optional delete-after-read.
"""

import os
import random
from pathlib import Path
from typing import Iterator, List
from concurrent.futures import ThreadPoolExecutor

import torch
from torch.utils.data import IterableDataset
from safetensors.torch import load_file


class ShardLoader(IterableDataset):
    """
    Multi-threaded shard loader for activation files.
    
    Features:
    - Loads .safetensors files from a directory
    - Optional delete-after-read mode
    - Multi-threaded prefetching
    - Random shard ordering
    """
    
    def __init__(
        self,
        shard_dir: Path,
        batch_size: int = 4096,
        delete_after_read: bool = False,
        prefetch_count: int = 2,
        shuffle_shards: bool = True,
        num_workers: int = 4
    ):
        """
        Initialize the shard loader.
        
        Args:
            shard_dir: Directory containing .safetensors shard files
            batch_size: Batch size for yielding data
            delete_after_read: Whether to delete shards after reading
            prefetch_count: Number of shards to prefetch
            shuffle_shards: Whether to shuffle shard order
            num_workers: Number of worker threads for prefetching
        """
        self.shard_dir = Path(shard_dir)
        self.batch_size = batch_size
        self.delete_after_read = delete_after_read
        self.prefetch_count = prefetch_count
        self.shuffle_shards = shuffle_shards
        self.num_workers = num_workers
        self._executor = None
        
    def get_shard_files(self) -> List[Path]:
        """Get list of shard files."""
        files = list(self.shard_dir.glob("*.safetensors"))
        if self.shuffle_shards:
            random.shuffle(files)
        return files
        
    def _load_shard(self, shard_path: Path) -> torch.Tensor:
        """Load a single shard file."""
        data = load_file(str(shard_path))
        return data["activations"]
        
    def __iter__(self) -> Iterator[torch.Tensor]:
        """
        Iterate over batches of activations.
        
        Yields:
            Batches of activations [batch_size, d_model]
        """
        shard_files = self.get_shard_files()
        
        if not shard_files:
            raise ValueError(f"No shard files found in {self.shard_dir}")
            
        # Start prefetch executor
        self._executor = ThreadPoolExecutor(max_workers=self.num_workers)
        prefetched_futures = {}
        
        # Submit initial prefetch jobs
        prefetch_idx = 0
        for i in range(min(self.prefetch_count, len(shard_files))):
            shard_path = shard_files[i]
            prefetched_futures[shard_path] = self._executor.submit(self._load_shard, shard_path)
            prefetch_idx = i + 1
            
        # Buffer for partial batches
        buffer = []
        
        for shard_idx, shard_path in enumerate(shard_files):
            # Submit next prefetch
            if prefetch_idx < len(shard_files):
                next_shard = shard_files[prefetch_idx]
                prefetched_futures[next_shard] = self._executor.submit(self._load_shard, next_shard)
                prefetch_idx += 1

            future = prefetched_futures.pop(shard_path, None)
            if future is not None:
                activations = future.result()
            else:
                activations = self._load_shard(shard_path)
                
            # Shuffle within shard
            perm = torch.randperm(activations.shape[0])
            activations = activations[perm]
            
            # Add to buffer
            buffer.append(activations)
            total_size = sum(a.shape[0] for a in buffer)
            
            # Yield batches
            while total_size >= self.batch_size:
                # Concatenate buffer
                all_data = torch.cat(buffer, dim=0)
                
                # Yield batch
                yield all_data[:self.batch_size]
                
                # Keep remainder in buffer
                if all_data.shape[0] > self.batch_size:
                    buffer = [all_data[self.batch_size:]]
                    total_size = buffer[0].shape[0]
                else:
                    buffer = []
                    total_size = 0
                    
            # Delete shard if requested
            if self.delete_after_read:
                try:
                    os.remove(shard_path)
                except Exception as e:
                    print(f"Error deleting {shard_path}: {e}")
                    
        # Yield final partial batch
        if buffer:
            yield torch.cat(buffer, dim=0)
            
        # Cleanup
        self._executor.shutdown(wait=True)
        
    def __len__(self) -> int:
        """Estimate number of batches (may not be exact)."""
        total_size = 0
        for shard_path in self.shard_dir.glob("*.safetensors"):
            # Estimate based on file size
            file_size = shard_path.stat().st_size
            # Rough estimate: assuming d_model=1536, float32
            n_activations = file_size // (1536 * 4)
            total_size += n_activations
        return (total_size + self.batch_size - 1) // self.batch_size


class InfiniteShardLoader(ShardLoader):
    """
    Infinite shard loader that cycles through shards.
    
    Useful for training when you want to iterate multiple epochs.
    """
    
    def __init__(self, *args, max_epochs: int = 10, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_epochs = max_epochs
        # Disable delete_after_read for infinite loader
        self.delete_after_read = False
        
    def __iter__(self) -> Iterator[torch.Tensor]:
        """Iterate infinitely over shards."""
        for epoch in range(self.max_epochs):
            for batch in super().__iter__():
                yield batch
