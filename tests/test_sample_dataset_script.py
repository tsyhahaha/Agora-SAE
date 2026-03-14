from datasets import Dataset, DatasetDict, load_dataset

from agora_sae.scripts.sample_dataset import (
    infer_split_name,
    load_local_dataset,
    sample_dataset_object,
    sample_dataset_split,
    save_sampled_dataset,
)


def test_sample_dataset_split_is_deterministic():
    dataset = Dataset.from_dict(
        {
            "problem": [f"problem-{idx}" for idx in range(10)],
            "level": [idx % 5 for idx in range(10)],
            "type": ["algebra"] * 10,
            "solution": [f"solution-{idx}" for idx in range(10)],
        }
    )

    sampled_once = sample_dataset_split(dataset, num_samples=4, seed=7)
    sampled_twice = sample_dataset_split(dataset, num_samples=4, seed=7)

    assert len(sampled_once) == 4
    assert sampled_once.column_names == dataset.column_names
    assert sampled_once["problem"] == sampled_twice["problem"]


def test_load_and_save_local_parquet_dataset(tmp_path):
    source_path = tmp_path / "competition_math"
    source_path.mkdir()
    (source_path / "dataset_infos.json").write_text("{}", encoding="utf-8")

    dataset = DatasetDict(
        {
            "train": Dataset.from_dict(
                {
                    "problem": [f"train-problem-{idx}" for idx in range(6)],
                    "level": [idx for idx in range(6)],
                    "type": ["algebra"] * 6,
                    "solution": [f"train-solution-{idx}" for idx in range(6)],
                }
            ),
            "validation": Dataset.from_dict(
                {
                    "problem": [f"val-problem-{idx}" for idx in range(4)],
                    "level": [idx for idx in range(4)],
                    "type": ["geometry"] * 4,
                    "solution": [f"val-solution-{idx}" for idx in range(4)],
                }
            ),
        }
    )

    dataset["train"].to_parquet(str(source_path / "train.parquet"))
    dataset["validation"].to_parquet(str(source_path / "validation.parquet"))

    loaded_dataset = load_local_dataset(source_path)
    sampled_dataset = sample_dataset_object(loaded_dataset, num_samples=3, seed=11)

    output_path = tmp_path / "sampled_competition_math"
    output_path.mkdir()
    save_sampled_dataset(sampled_dataset, output_path)

    reloaded_dataset = load_dataset(str(output_path))

    assert set(reloaded_dataset.keys()) == {"train", "validation"}
    assert len(reloaded_dataset["train"]) == 3
    assert len(reloaded_dataset["validation"]) == 3
    assert reloaded_dataset["train"].column_names == dataset["train"].column_names


def test_infer_split_name_handles_huggingface_parquet_layout(tmp_path):
    dataset_root = tmp_path / "dataset"
    file_path = dataset_root / "data" / "train-00000-of-00001.parquet"
    file_path.parent.mkdir(parents=True)
    file_path.touch()

    assert infer_split_name(file_path, dataset_root) == "train"
