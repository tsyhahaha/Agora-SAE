"""
Module A: Data ingestion and step-level activation-point selection.
"""

import random
import re
from pathlib import Path
from typing import Iterator, List, Optional, Sequence

import torch
from datasets import Dataset, DatasetDict, interleave_datasets, load_dataset, load_from_disk
from torch.utils.data import IterableDataset
from transformers import AutoTokenizer

from agora_sae.data.reasoning_steps import (
    ActivationPointSelector,
    DelimiterStepSegmenter,
)


class MixedTokenSource(IterableDataset):
    """
    Mixed token source combining reasoning and general datasets.

    Reasoning examples keep the full prompt context but only expose
    sentence/step-level activation points to the generator.
    """

    THINK_PATTERN = re.compile(r"<think>(.*?)</think>", re.DOTALL)
    SOLUTION_HEADER_PATTERN = re.compile(
        r"(?:Solution|Answer|解答|答案)\s*[:：]\s*",
        re.IGNORECASE,
    )
    COMBINED_TEXT_FIELDS = ("text", "content", "response", "output", "completion")
    QUERY_FIELDS = ("problem", "question", "query", "prompt", "instruction", "input")
    SOLUTION_FIELDS = ("solution", "answer", "final_answer", "target")
    SUPPORTED_SUFFIXES = {
        ".parquet": "parquet",
        ".json": "json",
        ".jsonl": "json",
        ".csv": "csv",
    }
    IGNORED_JSON_FILES = {"dataset_infos.json", "dataset_info.json", "state.json"}

    def __init__(
        self,
        reasoning_datasets: List[str],
        general_datasets: Optional[List[str]],
        tokenizer: AutoTokenizer,
        reasoning_ratio: float = 0.8,
        max_seq_length: int = 2048,
        question_sample_prob: float = 0.1,
        step_delimiter: Optional[str] = "\n\n",
        activation_point_strategy: str = "step_delimiter",
        retain_query: bool = True,
        repeat_dataset: bool = False,
        seed: int = 42,
    ):
        self.reasoning_datasets = list(reasoning_datasets or [])
        self.general_datasets = list(general_datasets or [])
        self.tokenizer = tokenizer
        self.reasoning_ratio = reasoning_ratio
        self.max_seq_length = max_seq_length
        self.question_sample_prob = question_sample_prob
        self.step_delimiter = step_delimiter or "\n\n"
        self.retain_query = retain_query
        self.repeat_dataset = repeat_dataset
        self.seed = seed
        self._rng = random.Random(seed)
        self._step_segmenter = DelimiterStepSegmenter(self.step_delimiter)
        self._activation_selector = ActivationPointSelector(activation_point_strategy)

    @staticmethod
    def _normalize_text_value(value) -> Optional[str]:
        """Convert a dataset field into text when possible."""
        if value is None:
            return None
        if isinstance(value, str):
            return value
        if isinstance(value, (list, tuple)):
            parts = [
                str(part).strip()
                for part in value
                if part is not None and str(part).strip()
            ]
            return "\n".join(parts) if parts else None
        if isinstance(value, dict):
            return None
        return str(value)

    def _get_first_text_field(self, example: dict, field_names: Sequence[str]) -> Optional[str]:
        """Return the first usable text field from an example."""
        for field_name in field_names:
            if field_name in example:
                normalized = self._normalize_text_value(example[field_name])
                if normalized:
                    return normalized
        return None

    @staticmethod
    def _select_split(dataset_obj, split_name: str):
        """Select a split from a loaded dataset object."""
        if isinstance(dataset_obj, DatasetDict):
            if split_name in dataset_obj:
                return dataset_obj[split_name]
            if len(dataset_obj) == 1:
                return next(iter(dataset_obj.values()))
            available = ", ".join(dataset_obj.keys())
            raise ValueError(f"Split '{split_name}' not found. Available splits: {available}")
        return dataset_obj

    @staticmethod
    def _to_iterable_dataset(dataset_obj):
        """Convert a map-style dataset into an iterable dataset when possible."""
        if isinstance(dataset_obj, Dataset):
            return dataset_obj.to_iterable_dataset()
        return dataset_obj

    def _load_dataset_source(self, dataset_name: str, split_name: str = "train"):
        """Load either a remote dataset name or a local dataset path."""
        dataset_path = Path(dataset_name).expanduser()
        load_errors = []

        if dataset_path.exists():
            try:
                dataset_obj = load_from_disk(str(dataset_path))
                return self._to_iterable_dataset(self._select_split(dataset_obj, split_name))
            except Exception as exc:
                load_errors.append(f"load_from_disk failed: {exc}")

            try:
                dataset_obj = load_dataset(str(dataset_path), split=split_name, streaming=True)
                return self._to_iterable_dataset(dataset_obj)
            except Exception as exc:
                load_errors.append(f"load_dataset(path, streaming=True) failed: {exc}")

            try:
                dataset_obj = load_dataset(str(dataset_path), split=split_name)
                return self._to_iterable_dataset(dataset_obj)
            except Exception as exc:
                load_errors.append(f"load_dataset(path) failed: {exc}")

            try:
                format_name, data_files = self._discover_local_data_files(dataset_path)
                dataset_obj = load_dataset(format_name, data_files=data_files, split=split_name)
                return self._to_iterable_dataset(dataset_obj)
            except Exception as exc:
                load_errors.append(f"data_files load failed: {exc}")

            errors = "\n".join(f"- {message}" for message in load_errors)
            raise ValueError(f"Could not load local dataset {dataset_name}:\n{errors}")

        try:
            dataset_obj = load_dataset(dataset_name, split=split_name, streaming=True)
            return self._to_iterable_dataset(dataset_obj)
        except Exception as exc:
            load_errors.append(f"streaming load failed: {exc}")

        try:
            dataset_obj = load_dataset(dataset_name, split=split_name)
            return self._to_iterable_dataset(dataset_obj)
        except Exception as exc:
            load_errors.append(f"standard load failed: {exc}")

        errors = "\n".join(f"- {message}" for message in load_errors)
        raise ValueError(f"Could not load dataset {dataset_name}:\n{errors}")

    def _infer_split_name(self, file_path: Path, root_path: Path) -> str:
        """Infer a split name from a local dataset file path."""
        relative_parts = [part.lower() for part in file_path.relative_to(root_path).parts]
        candidate_parts = []

        for part in relative_parts:
            candidate_parts.append(Path(part).stem)
            candidate_parts.append(part)

        for part in reversed(candidate_parts):
            for split_name in ("train", "test", "validation", "valid", "val", "dev"):
                if part == split_name:
                    if split_name in {"valid", "val"}:
                        return "validation"
                    return split_name
                if part.startswith(f"{split_name}-") or part.startswith(f"{split_name}_"):
                    if split_name in {"valid", "val"}:
                        return "validation"
                    return split_name

        return "train"

    def _discover_local_data_files(self, dataset_path: Path):
        """Discover local data files and group them by split."""
        matched_files = {}
        for suffix, format_name in self.SUPPORTED_SUFFIXES.items():
            files = sorted(
                path
                for path in dataset_path.rglob(f"*{suffix}")
                if path.is_file() and path.name not in self.IGNORED_JSON_FILES
            )
            if files:
                matched_files[format_name] = files

        if not matched_files:
            raise ValueError(f"No supported dataset files found under {dataset_path}")

        if len(matched_files) > 1:
            formats = ", ".join(sorted(matched_files))
            raise ValueError(f"Found multiple data formats in {dataset_path}: {formats}")

        format_name, files = next(iter(matched_files.items()))
        data_files = {}
        for file_path in files:
            split = self._infer_split_name(file_path, dataset_path)
            data_files.setdefault(split, []).append(str(file_path))

        return format_name, data_files

    def _build_dataset_iterator(self, dataset_names: Sequence[str]) -> Iterator:
        """Load and combine datasets, returning a plain Python iterator."""
        datasets_list = []
        for dataset_name in dataset_names:
            try:
                datasets_list.append(self._load_dataset_source(dataset_name, split_name="train"))
            except Exception as exc:
                print(f"Warning: Could not load dataset {dataset_name}: {exc}")

        if not datasets_list:
            raise ValueError("No datasets could be loaded")

        if len(datasets_list) > 1:
            return iter(interleave_datasets(datasets_list))
        return iter(datasets_list[0])

    def _load_reasoning_datasets(self) -> Iterator:
        return self._build_dataset_iterator(self.reasoning_datasets)

    def _load_general_datasets(self) -> Iterator:
        return self._build_dataset_iterator(self.general_datasets)

    def _build_reasoning_only_text(self, full_text: str, reasoning_spans: Sequence[tuple[int, int]]) -> str:
        """Collapse reasoning spans into a standalone text when query retention is disabled."""
        chunks = [full_text[start:end].strip() for start, end in reasoning_spans if full_text[start:end].strip()]
        return self.step_delimiter.join(chunks)

    def _make_reasoning_text_data(
        self,
        full_text: str,
        reasoning_spans: Sequence[tuple[int, int]],
    ) -> Optional[dict]:
        """Create the text payload plus step annotations for reasoning data."""
        if not reasoning_spans:
            return None

        if self.retain_query:
            text = full_text
            spans_for_segmentation = reasoning_spans
        else:
            text = self._build_reasoning_only_text(full_text, reasoning_spans)
            spans_for_segmentation = [(0, len(text))] if text else []

        if not text or not spans_for_segmentation:
            return None

        steps = []
        for span in spans_for_segmentation:
            steps.extend(self._step_segmenter.segment(text, span))

        if not steps:
            return None

        return {
            "text": text,
            "steps": steps,
        }

    def _parse_reasoning(self, example: dict) -> Optional[dict]:
        """
        Parse reasoning data and mark step-level activation points.

        Returns:
            Dictionary with "text" and optional step annotations.
        """
        query_text = self._get_first_text_field(example, self.QUERY_FIELDS)
        solution_text = self._get_first_text_field(example, self.SOLUTION_FIELDS)
        combined_text = self._get_first_text_field(example, self.COMBINED_TEXT_FIELDS)

        if query_text and solution_text:
            full_text = f"Problem:\n{query_text}\n\nSolution:\n{solution_text}"
            solution_start = len(full_text) - len(solution_text)
            return self._make_reasoning_text_data(full_text, [(solution_start, len(full_text))])

        if combined_text is not None:
            think_spans = [match.span(1) for match in self.THINK_PATTERN.finditer(combined_text)]
            if think_spans:
                return self._make_reasoning_text_data(combined_text, think_spans)

            solution_match = self.SOLUTION_HEADER_PATTERN.search(combined_text)
            if solution_match is not None:
                return self._make_reasoning_text_data(
                    combined_text,
                    [(solution_match.end(), len(combined_text))],
                )

        fallback_text = query_text or combined_text
        if fallback_text is not None and self._rng.random() < self.question_sample_prob:
            return {
                "text": fallback_text,
                "steps": None,
            }

        return None

    def _parse_general(self, example: dict) -> Optional[dict]:
        """Parse general data to extract text content."""
        for field in ("text", "content", "document"):
            if field in example:
                text = self._normalize_text_value(example[field])
                if text:
                    return {
                        "text": text,
                        "steps": None,
                    }
        return None

    def _tokenize(self, text_data: dict) -> dict:
        """Tokenize text and calculate activation points."""
        text = text_data["text"]
        steps = text_data.get("steps")

        encoded = self.tokenizer(
            text,
            max_length=self.max_seq_length,
            truncation=True,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        if "offset_mapping" not in encoded:
            raise ValueError(
                "Tokenizer must support offset mappings for step-delimiter activation extraction."
            )
        input_ids = encoded["input_ids"].squeeze(0)
        offsets = encoded["offset_mapping"].squeeze(0)

        if steps is None:
            activation_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            activation_mask = self._activation_selector.select_mask(offsets, steps)

        return {
            "input_ids": input_ids,
            "activation_mask": activation_mask,
        }

    def __iter__(self) -> Iterator[dict]:
        """
        Iterate over mixed token data.

        Yields:
            Dictionary with input_ids and activation_mask tensors.
        """
        if not self.reasoning_datasets and not self.general_datasets:
            raise ValueError("At least one reasoning or general dataset must be provided")

        use_reasoning = bool(self.reasoning_datasets) and (
            not self.general_datasets or self.reasoning_ratio > 0.0
        )
        use_general = bool(self.general_datasets) and (
            not self.reasoning_datasets or self.reasoning_ratio < 1.0
        )

        reasoning_iter = self._load_reasoning_datasets() if use_reasoning else None
        general_iter = self._load_general_datasets() if use_general else None

        while reasoning_iter is not None or general_iter is not None:
            if reasoning_iter is not None and general_iter is not None:
                source = "reasoning" if self._rng.random() < self.reasoning_ratio else "general"
            elif reasoning_iter is not None:
                source = "reasoning"
            else:
                source = "general"

            try:
                if source == "reasoning":
                    example = next(reasoning_iter)
                    text_data = self._parse_reasoning(example)
                else:
                    example = next(general_iter)
                    text_data = self._parse_general(example)
            except StopIteration:
                if source == "reasoning":
                    reasoning_iter = self._load_reasoning_datasets() if self.repeat_dataset else None
                else:
                    general_iter = self._load_general_datasets() if self.repeat_dataset else None
                continue

            if text_data is None or len(text_data["text"].strip()) == 0:
                continue

            token_data = self._tokenize(text_data)

            if len(token_data["input_ids"]) < 10:
                continue
            if not token_data["activation_mask"].any():
                continue

            yield token_data


def create_dataloader(
    config,
    tokenizer: AutoTokenizer,
    batch_size: int = 64,
) -> torch.utils.data.DataLoader:
    """
    Create a DataLoader for mixed token source.
    """
    dataset = MixedTokenSource(
        reasoning_datasets=config.data.reasoning_datasets,
        general_datasets=config.data.general_datasets,
        tokenizer=tokenizer,
        reasoning_ratio=config.data.reasoning_ratio,
        max_seq_length=config.data.max_seq_length,
        question_sample_prob=config.data.question_sample_prob,
        step_delimiter=config.data.step_delimiter,
        activation_point_strategy=config.data.activation_point_strategy,
        retain_query=config.data.retain_query,
        repeat_dataset=config.data.repeat_dataset,
    )

    def collate_fn(batch):
        """Pad sequences to the same length."""
        max_len = max(len(seq["input_ids"]) for seq in batch)
        pad_token_id = tokenizer.pad_token_id or 0
        padded = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros(len(batch), max_len, dtype=torch.long)
        activation_mask = torch.zeros(len(batch), max_len, dtype=torch.bool)

        for i, seq in enumerate(batch):
            ids = seq["input_ids"]
            padded[i, : len(ids)] = ids
            attention_mask[i, : len(ids)] = 1
            activation_mask[i, : len(ids)] = seq["activation_mask"]

        return {
            "input_ids": padded,
            "attention_mask": attention_mask,
            "activation_mask": activation_mask,
        }

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
    )
