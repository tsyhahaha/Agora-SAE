"""
Module A: Data Ingestion & Formatting

Implements MixedTokenSource for constructing 80% Reasoning / 20% General data mix.
"""

import re
import random
from typing import List, Iterator, Optional
from torch.utils.data import IterableDataset
from datasets import load_dataset, interleave_datasets
from transformers import AutoTokenizer
import torch


class MixedTokenSource(IterableDataset):
    """
    Mixed token source combining reasoning and general datasets.
    
    Key features:
    - 80/20 ratio mixing between reasoning and general data
    - Extracts <think>...</think> content from reasoning data
    - Filters out pure question parts (or samples with low probability)
    """
    
    # Pattern to extract <think>...</think> content
    THINK_PATTERN = re.compile(r'<think>(.*?)</think>', re.DOTALL)
    # Pattern to find Solution/Answer sections
    SOLUTION_PATTERN = re.compile(
        r'(?:Solution|Answer|解答|答案)[:\s]*(.*?)(?:\n\n|$)', 
        re.DOTALL | re.IGNORECASE
    )
    
    def __init__(
        self,
        reasoning_datasets: List[str],
        general_datasets: List[str],
        tokenizer: AutoTokenizer,
        reasoning_ratio: float = 0.8,
        max_seq_length: int = 2048,
        question_sample_prob: float = 0.1,
        seed: int = 42
    ):
        """
        Initialize the mixed token source.
        
        Args:
            reasoning_datasets: List of HuggingFace dataset names for reasoning data
            general_datasets: List of HuggingFace dataset names for general data
            tokenizer: Tokenizer to use
            reasoning_ratio: Ratio of reasoning data (default 0.8)
            max_seq_length: Maximum sequence length
            question_sample_prob: Probability to keep pure question parts
            seed: Random seed for reproducibility
        """
        self.reasoning_datasets = reasoning_datasets
        self.general_datasets = general_datasets
        self.tokenizer = tokenizer
        self.reasoning_ratio = reasoning_ratio
        self.max_seq_length = max_seq_length
        self.question_sample_prob = question_sample_prob
        self.seed = seed
        
        random.seed(seed)
        
    def _load_reasoning_datasets(self) -> Iterator:
        """Load and combine reasoning datasets."""
        datasets_list = []
        for ds_name in self.reasoning_datasets:
            try:
                ds = load_dataset(ds_name, split="train", streaming=True)
                datasets_list.append(ds)
            except Exception as e:
                print(f"Warning: Could not load dataset {ds_name}: {e}")
                
        if not datasets_list:
            raise ValueError("No reasoning datasets could be loaded")
            
        # Interleave with equal probability
        if len(datasets_list) > 1:
            return interleave_datasets(datasets_list)
        return iter(datasets_list[0])
    
    def _load_general_datasets(self) -> Iterator:
        """Load and combine general datasets."""
        datasets_list = []
        for ds_name in self.general_datasets:
            try:
                ds = load_dataset(ds_name, split="train", streaming=True)
                datasets_list.append(ds)
            except Exception as e:
                print(f"Warning: Could not load dataset {ds_name}: {e}")
                
        if not datasets_list:
            raise ValueError("No general datasets could be loaded")
            
        if len(datasets_list) > 1:
            return interleave_datasets(datasets_list)
        return iter(datasets_list[0])
    
    def _parse_reasoning(self, example: dict) -> Optional[str]:
        """
        Parse reasoning data to extract <think> content and solution.
        
        Returns:
            Extracted text or None if should be skipped
        """
        # Try different common field names
        text = None
        for field in ['text', 'content', 'response', 'output', 'completion']:
            if field in example:
                text = example[field]
                break
                
        if text is None:
            return None
            
        extracted_parts = []
        
        # Extract <think>...</think> content
        think_matches = self.THINK_PATTERN.findall(text)
        if think_matches:
            extracted_parts.extend(think_matches)
            
        # Extract Solution/Answer sections
        solution_matches = self.SOLUTION_PATTERN.findall(text)
        if solution_matches:
            extracted_parts.extend(solution_matches)
            
        # If we found thinking or solution content, return it
        if extracted_parts:
            return "\n".join(extracted_parts)
            
        # Check if this is just a question (no reasoning)
        # Sample with low probability
        if random.random() < self.question_sample_prob:
            return text
            
        return None
    
    def _parse_general(self, example: dict) -> Optional[str]:
        """Parse general data to extract text content."""
        for field in ['text', 'content', 'document']:
            if field in example:
                return example[field]
        return None
    
    def _tokenize(self, text: str) -> torch.Tensor:
        """Tokenize text and return input_ids."""
        tokens = self.tokenizer(
            text,
            max_length=self.max_seq_length,
            truncation=True,
            return_tensors="pt"
        )
        return tokens["input_ids"].squeeze(0)
    
    def __iter__(self) -> Iterator[torch.Tensor]:
        """
        Iterate over mixed token data.
        
        Yields:
            input_ids tensor for each sample
        """
        reasoning_iter = self._load_reasoning_datasets()
        general_iter = self._load_general_datasets()
        
        while True:
            try:
                # Decide which source to sample from based on ratio
                if random.random() < self.reasoning_ratio:
                    # Sample from reasoning data
                    example = next(reasoning_iter)
                    text = self._parse_reasoning(example)
                else:
                    # Sample from general data
                    example = next(general_iter)
                    text = self._parse_general(example)
                    
                if text is None or len(text.strip()) == 0:
                    continue
                    
                input_ids = self._tokenize(text)
                
                # Skip very short sequences
                if len(input_ids) < 10:
                    continue
                    
                yield input_ids
                
            except StopIteration:
                # Reload iterators when exhausted
                reasoning_iter = self._load_reasoning_datasets()
                general_iter = self._load_general_datasets()


def create_dataloader(
    config,
    tokenizer: AutoTokenizer,
    batch_size: int = 64
) -> torch.utils.data.DataLoader:
    """
    Create a DataLoader for mixed token source.
    
    Args:
        config: Configuration object
        tokenizer: Tokenizer to use
        batch_size: Batch size for dataloader
        
    Returns:
        DataLoader instance
    """
    dataset = MixedTokenSource(
        reasoning_datasets=config.data.reasoning_datasets,
        general_datasets=config.data.general_datasets,
        tokenizer=tokenizer,
        reasoning_ratio=config.data.reasoning_ratio,
        max_seq_length=config.data.max_seq_length,
        question_sample_prob=config.data.question_sample_prob
    )
    
    def collate_fn(batch):
        """Pad sequences to same length."""
        max_len = max(len(seq) for seq in batch)
        padded = torch.zeros(len(batch), max_len, dtype=torch.long)
        attention_mask = torch.zeros(len(batch), max_len, dtype=torch.long)
        
        for i, seq in enumerate(batch):
            padded[i, :len(seq)] = seq
            attention_mask[i, :len(seq)] = 1
            
        return {"input_ids": padded, "attention_mask": attention_mask}
    
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn
    )
