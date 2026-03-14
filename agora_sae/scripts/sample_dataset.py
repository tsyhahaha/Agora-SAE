#!/usr/bin/env python3
"""
Sample a smaller local Hugging Face dataset from a cloned dataset directory.

Usage:
    python -m agora_sae.scripts.sample_dataset \
        --dataset-path /path/to/competition_math \
        --output-path /path/to/competition_math_sampled \
        --num-samples 500
"""

import argparse
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Union

from datasets import Dataset, DatasetDict, load_dataset, load_from_disk


DatasetLike = Union[Dataset, DatasetDict]
SUPPORTED_SUFFIXES = {
    ".parquet": "parquet",
    ".json": "json",
    ".jsonl": "json",
    ".csv": "csv",
}
IGNORED_JSON_FILES = {
    "dataset_infos.json",
    "dataset_info.json",
    "state.json",
}
SPLIT_ALIASES = {
    "val": "validation",
    "valid": "validation",
}
KNOWN_SPLITS = ("train", "test", "validation", "valid", "val", "dev")


def normalize_split_name(split_name: str) -> str:
    """Normalize common split aliases to canonical names."""
    return SPLIT_ALIASES.get(split_name, split_name)


def infer_split_name(file_path: Path, root_path: Path) -> str:
    """Infer the split name from a local data file path."""
    relative_parts = [part.lower() for part in file_path.relative_to(root_path).parts]
    candidate_parts = []

    for part in relative_parts:
        candidate_parts.append(Path(part).stem)
        candidate_parts.append(part)

    for part in reversed(candidate_parts):
        for split_name in KNOWN_SPLITS:
            if part == split_name:
                return normalize_split_name(split_name)
            if part.startswith(f"{split_name}-") or part.startswith(f"{split_name}_"):
                return normalize_split_name(split_name)

    return "train"


def discover_data_files(dataset_path: Path) -> Tuple[str, Dict[str, List[str]]]:
    """Discover local data files and group them by split."""
    if dataset_path.is_file():
        suffix = dataset_path.suffix.lower()
        if suffix not in SUPPORTED_SUFFIXES:
            raise ValueError(f"Unsupported dataset file format: {dataset_path.suffix}")
        return SUPPORTED_SUFFIXES[suffix], {"train": [str(dataset_path)]}

    matched_files = {}
    for suffix, format_name in SUPPORTED_SUFFIXES.items():
        files = sorted(
            path
            for path in dataset_path.rglob(f"*{suffix}")
            if path.is_file() and path.name not in IGNORED_JSON_FILES
        )
        if files:
            matched_files[format_name] = files

    if not matched_files:
        raise ValueError(f"No supported dataset files found under {dataset_path}")

    if len(matched_files) > 1:
        formats = ", ".join(sorted(matched_files))
        raise ValueError(
            "Found multiple data formats in the same directory. "
            f"Please keep only one format type. Found: {formats}"
        )

    format_name, files = next(iter(matched_files.items()))
    data_files: Dict[str, List[str]] = {}
    for file_path in files:
        split_name = infer_split_name(file_path, dataset_path)
        data_files.setdefault(split_name, []).append(str(file_path))

    return format_name, data_files


def load_local_dataset(dataset_path: Path) -> DatasetLike:
    """Load a local dataset from save_to_disk output, a local dataset repo, or raw files."""
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")

    load_errors = []

    try:
        return load_from_disk(str(dataset_path))
    except Exception as exc:
        load_errors.append(f"load_from_disk failed: {exc}")

    if dataset_path.is_dir():
        try:
            return load_dataset(str(dataset_path))
        except Exception as exc:
            load_errors.append(f"load_dataset(path) failed: {exc}")

    try:
        format_name, data_files = discover_data_files(dataset_path)
        return load_dataset(format_name, data_files=data_files)
    except Exception as exc:
        load_errors.append(f"file discovery load failed: {exc}")

    error_message = "\n".join(f"- {message}" for message in load_errors)
    raise ValueError(f"Could not load dataset from {dataset_path}:\n{error_message}")


def sample_dataset_split(dataset: Dataset, num_samples: int, seed: int) -> Dataset:
    """Sample rows from a single dataset split."""
    if num_samples <= 0:
        raise ValueError("--num-samples must be greater than 0")

    total_rows = len(dataset)
    if total_rows == 0:
        raise ValueError("Cannot sample from an empty dataset")

    if num_samples >= total_rows:
        return dataset

    import random

    rng = random.Random(seed)
    indices = rng.sample(range(total_rows), num_samples)
    return dataset.select(indices)


def sample_dataset_object(dataset_obj: DatasetLike, num_samples: int, seed: int) -> DatasetLike:
    """Sample a Dataset or DatasetDict while preserving schema and split names."""
    if isinstance(dataset_obj, DatasetDict):
        sampled_splits = {}
        for offset, (split_name, split_dataset) in enumerate(dataset_obj.items()):
            sampled_splits[split_name] = sample_dataset_split(split_dataset, num_samples, seed + offset)
        return DatasetDict(sampled_splits)

    return sample_dataset_split(dataset_obj, num_samples, seed)


def prepare_output_directory(output_path: Path, overwrite: bool) -> None:
    """Create an output directory, optionally overwriting an existing one."""
    if output_path.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output path already exists: {output_path}. "
                "Use --overwrite if you want to replace it."
            )
        if output_path.is_dir():
            shutil.rmtree(output_path)
        else:
            output_path.unlink()

    output_path.mkdir(parents=True, exist_ok=True)


def save_sampled_dataset(
    dataset_obj: DatasetLike,
    output_path: Path,
    single_split_name: str = "train",
) -> None:
    """Save the sampled dataset as local parquet files under a Hugging Face-style data/ layout."""
    data_path = output_path / "data"
    data_path.mkdir(parents=True, exist_ok=True)

    if isinstance(dataset_obj, DatasetDict):
        for split_name, split_dataset in dataset_obj.items():
            split_dataset.to_parquet(str(data_path / f"{split_name}-00000-of-00001.parquet"))
        return

    dataset_obj.to_parquet(str(data_path / f"{single_split_name}-00000-of-00001.parquet"))


def print_dataset_summary(dataset_obj: DatasetLike, title: str) -> None:
    """Print a concise summary of dataset splits and sizes."""
    print(f"\n{title}")
    print("-" * len(title))

    if isinstance(dataset_obj, DatasetDict):
        for split_name, split_dataset in dataset_obj.items():
            print(f"{split_name}: {len(split_dataset)} rows, columns={split_dataset.column_names}")
        return

    print(f"rows={len(dataset_obj)}, columns={dataset_obj.column_names}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample a smaller local dataset")
    parser.add_argument(
        "--dataset-path",
        type=str,
        required=True,
        help="Path to a local Hugging Face dataset directory or data file",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Directory where the sampled dataset will be saved",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=500,
        help="Number of samples to keep from each split (default: 500)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling",
    )
    parser.add_argument(
        "--single-split-name",
        type=str,
        default="train",
        help="Split name to use when the source is a single Dataset (default: train)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output directory if it already exists",
    )

    args = parser.parse_args()

    dataset_path = Path(args.dataset_path).expanduser().resolve()
    output_path = Path(args.output_path).expanduser().resolve()

    print("=" * 60)
    print("DATASET SAMPLING CONFIGURATION")
    print("=" * 60)
    print(f"Input: {dataset_path}")
    print(f"Output: {output_path}")
    print(f"Samples per split: {args.num_samples}")
    print(f"Seed: {args.seed}")
    print("=" * 60)

    print("\nLoading dataset...")
    dataset_obj = load_local_dataset(dataset_path)
    print_dataset_summary(dataset_obj, "Source dataset")

    print("\nSampling dataset...")
    sampled_dataset = sample_dataset_object(dataset_obj, args.num_samples, args.seed)
    print_dataset_summary(sampled_dataset, "Sampled dataset")

    print("\nSaving sampled dataset...")
    prepare_output_directory(output_path, args.overwrite)
    save_sampled_dataset(
        sampled_dataset,
        output_path,
        single_split_name=args.single_split_name,
    )

    print("\nSampling complete!")
    print(f"Sampled dataset saved to: {output_path}")
    print(f"You can load it with: load_dataset('{output_path}', split='train')")


if __name__ == "__main__":
    main()
