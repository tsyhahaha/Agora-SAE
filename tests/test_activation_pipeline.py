import time
from argparse import Namespace
from pathlib import Path

import torch

from agora_sae.config import get_config
from agora_sae.data.mixed_source import MixedTokenSource
from agora_sae.scripts.generate_activations import build_config_from_args
from agora_sae.trainer.shard_loader import ShardLoader


class DummyTokenizer:
    pad_token_id = 0

    def __call__(
        self,
        text,
        max_length,
        truncation,
        return_offsets_mapping,
        return_tensors,
    ):
        text = text[:max_length]
        input_ids = torch.arange(1, len(text) + 1, dtype=torch.long).unsqueeze(0)
        offset_mapping = torch.tensor(
            [[(idx, idx + 1) for idx in range(len(text))]],
            dtype=torch.long,
        )
        return {
            "input_ids": input_ids,
            "offset_mapping": offset_mapping,
        }


def make_args(tmp_path: Path, **overrides) -> Namespace:
    args = Namespace(
        model=None,
        layer=None,
        d_model=None,
        reasoning_datasets=None,
        general_datasets=None,
        reasoning_ratio=None,
        output=None,
        buffer_size_mb=None,
        shard_size_mb=None,
        max_disk_gb=None,
        batch_size=64,
        max_batches=None,
        max_seq_length=None,
        preset=None,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def test_build_config_from_args_applies_cli_overrides_to_preset(tmp_path):
    output_path = tmp_path / "buffer_shards"
    args = make_args(
        tmp_path,
        preset="math500-1.5b",
        model="/models/local-deepseek",
        reasoning_datasets=["/datasets/competition_math_sampled"],
        output=str(output_path),
        buffer_size_mb=256,
        shard_size_mb=64,
        max_disk_gb=12,
        max_seq_length=1024,
    )

    config = build_config_from_args(args)

    assert config.model.model_name == "/models/local-deepseek"
    assert config.data.reasoning_datasets == ["/datasets/competition_math_sampled"]
    assert config.storage.storage_path == output_path
    assert config.storage.buffer_size_mb == 256
    assert config.storage.shard_size_mb == 64
    assert config.storage.max_disk_usage_gb == 12
    assert config.data.max_seq_length == 1024

    fresh_config = get_config("math500-1.5b")
    assert fresh_config.model.model_name == "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    assert fresh_config.data.reasoning_datasets == ["MATH500"]


def test_parse_reasoning_supports_problem_solution_schema():
    source = MixedTokenSource(
        reasoning_datasets=["dummy"],
        general_datasets=[],
        tokenizer=DummyTokenizer(),
        reasoning_ratio=1.0,
        extraction_split_token="\n\n",
        retain_query=True,
    )

    text_data = source._parse_reasoning(
        {
            "problem": "What is 1 + 1?",
            "solution": "First compute the sum.\n\nThen conclude the answer is 2.",
        }
    )

    assert text_data is not None
    assert text_data["text"].startswith("Problem:\nWhat is 1 + 1?")
    extracted_segments = [text_data["text"][start:end] for start, end in text_data["target_spans"]]
    assert extracted_segments == [
        "First compute the sum.",
        "Then conclude the answer is 2.",
    ]


def test_reasoning_only_iteration_does_not_touch_general_loader(monkeypatch):
    source = MixedTokenSource(
        reasoning_datasets=["dummy"],
        general_datasets=[],
        tokenizer=DummyTokenizer(),
        reasoning_ratio=1.0,
        retain_query=True,
        question_sample_prob=0.0,
    )

    monkeypatch.setattr(
        source,
        "_load_reasoning_datasets",
        lambda: iter(
            [
                {
                    "problem": "abcdefghijk",
                    "solution": "lmnopqrstuv",
                }
            ]
        ),
    )

    def fail_general_loader():
        raise AssertionError("general loader should not be called for reasoning-only runs")

    monkeypatch.setattr(source, "_load_general_datasets", fail_general_loader)

    token_data = next(iter(source))

    assert token_data["input_ids"].shape[0] >= 10
    assert token_data["extraction_mask"].any().item() is True


def test_shard_loader_reuses_prefetched_results_for_out_of_order_completion(tmp_path):
    class TrackingShardLoader(ShardLoader):
        def __init__(self):
            super().__init__(
                shard_dir=tmp_path,
                batch_size=2,
                delete_after_read=False,
                prefetch_count=2,
                shuffle_shards=False,
                num_workers=2,
            )
            self.calls = []

        def get_shard_files(self):
            return [
                tmp_path / "a.safetensors",
                tmp_path / "b.safetensors",
            ]

        def _load_shard(self, shard_path):
            self.calls.append(shard_path.name)
            if shard_path.name == "a.safetensors":
                time.sleep(0.05)
            return torch.ones(2, 4)

    loader = TrackingShardLoader()
    list(loader)

    assert loader.calls.count("a.safetensors") == 1
    assert loader.calls.count("b.safetensors") == 1
