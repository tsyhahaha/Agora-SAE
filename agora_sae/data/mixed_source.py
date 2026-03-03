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
        extraction_split_token: Optional[str] = None,
        retain_query: bool = False,
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
            extraction_split_token: Token to split on for fine-grained extraction (e.g. "\n\n")
            retain_query: Whether to keep the full query context
            seed: Random seed for reproducibility
        """
        self.reasoning_datasets = reasoning_datasets
        self.general_datasets = general_datasets
        self.tokenizer = tokenizer
        self.reasoning_ratio = reasoning_ratio
        self.max_seq_length = max_seq_length
        self.question_sample_prob = question_sample_prob
        self.extraction_split_token = extraction_split_token
        self.retain_query = retain_query
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
    
    def _parse_reasoning(self, example: dict) -> Optional[dict]:
        """
        Parse reasoning data to extract <think> content and solution.
        
        Returns:
            Dictionary with 'text' and 'extraction_targets' (list of text bounds) or None
        """
        text = None
        for field in ['text', 'content', 'response', 'output', 'completion']:
            if field in example:
                text = example[field]
                break
                
        if text is None:
            return None
            
        extracted_parts = []
        
        # Extract <think>...</think> content and Solution sections for targets
        think_matches = self.THINK_PATTERN.findall(text)
        if think_matches:
            extracted_parts.extend(think_matches)
            
        solution_matches = self.SOLUTION_PATTERN.findall(text)
        if solution_matches:
            extracted_parts.extend(solution_matches)
            
        # If we found reasoning content, return it
        if extracted_parts:
            targets = []
            if self.extraction_split_token:
                # Split parts into smaller targets based on split token
                for part in extracted_parts:
                    segments = part.split(self.extraction_split_token)
                    targets.extend([s.strip() for s in segments if len(s.strip()) > 0])
            else:
                targets = extracted_parts

            return {
                "text": text if self.retain_query else "\n".join(extracted_parts),
                "targets": targets
            }
            
        # Check if this is just a question (no reasoning)
        if random.random() < self.question_sample_prob:
            return {
                "text": text,
                "targets": [] if self.extraction_split_token else [text]
            }
            
        return None
    
    def _parse_general(self, example: dict) -> Optional[dict]:
        """Parse general data to extract text content."""
        for field in ['text', 'content', 'document']:
            if field in example:
                return {
                    "text": example[field],
                    "targets": [example[field]]
                }
        return None
    
    def _tokenize(self, text_data: dict) -> dict:
        """Tokenize text and calculate the extraction_mask based on targets."""
        text = text_data["text"]
        targets = text_data["targets"]
        
        encoded = self.tokenizer(
            text,
            max_length=self.max_seq_length,
            truncation=True,
            return_offsets_mapping=True,
            return_tensors="pt"
        )
        input_ids = encoded["input_ids"].squeeze(0)
        offsets = encoded["offset_mapping"].squeeze(0)
        
        extraction_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        
        # If no targets are specified (or it's general data without split), extract everywhere
        if not self.extraction_split_token or not targets:
            extraction_mask[:] = True
        else:
            # Find the token indices that correspond to the target substrings
            for target in targets:
                # Find occurrences of target in the text
                start_char = 0
                while True:
                    start_char = text.find(target, start_char)
                    if start_char == -1:
                        break
                    end_char = start_char + len(target)
                    
                    # Map char bounds to token indices
                    # A token is included if its character span overlaps with the target
                    in_target = (offsets[:, 0] < end_char) & (offsets[:, 1] > start_char)
                    extraction_mask |= in_target
                    
                    start_char = end_char
                    
        return {
            "input_ids": input_ids,
            "extraction_mask": extraction_mask
        }
    
    def __iter__(self) -> Iterator[dict]:
        """
        Iterate over mixed token data.
        
        Yields:
            Dictionary with input_ids and extraction_mask tensors
        """
        reasoning_iter = self._load_reasoning_datasets()
        general_iter = self._load_general_datasets()
        
        while True:
            try:
                # Decide which source to sample from based on ratio
                if random.random() < self.reasoning_ratio:
                    # Sample from reasoning data
                    example = next(reasoning_iter)
                    text_data = self._parse_reasoning(example)
                else:
                    # Sample from general data
                    example = next(general_iter)
                    text_data = self._parse_general(example)
                    
                if text_data is None or len(text_data["text"].strip()) == 0:
                    continue
                    
                token_data = self._tokenize(text_data)
                
                # Skip very short sequences
                if len(token_data["input_ids"]) < 10:
                    continue
                    
                yield token_data
                
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
        question_sample_prob=config.data.question_sample_prob,
        extraction_split_token=config.data.extraction_split_token,
        retain_query=config.data.retain_query
    )
    
    def collate_fn(batch):
        """Pad sequences to same length."""
        # Find max length of input_ids in this batch
        max_len = max(len(seq["input_ids"]) for seq in batch)
        padded = torch.zeros(len(batch), max_len, dtype=torch.long)
        attention_mask = torch.zeros(len(batch), max_len, dtype=torch.long)
        extraction_mask = torch.zeros(len(batch), max_len, dtype=torch.bool)
        
        for i, seq in enumerate(batch):
            ids = seq["input_ids"]
            padded[i, :len(ids)] = ids
            attention_mask[i, :len(ids)] = 1
            extraction_mask[i, :len(ids)] = seq["extraction_mask"]
            
        return {
            "input_ids": padded, 
            "attention_mask": attention_mask,
            "extraction_mask": extraction_mask
        }
    
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn
    )
