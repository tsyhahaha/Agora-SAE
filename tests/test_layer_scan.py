import importlib.util
import sys
from pathlib import Path

import pytest


def _load_layer_scan_module():
    module_name = "agora_sae_layer_scan_test"
    spec = importlib.util.spec_from_file_location(
        module_name,
        Path(__file__).resolve().parents[1] / "agora_sae" / "layer_scan.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


layer_scan = _load_layer_scan_module()


def test_parse_layers_spec_deduplicates_and_sorts():
    assert layer_scan.parse_layers_spec("12, 4,12, 20") == [4, 12, 20]


def test_parse_layer_range_spec_supports_embedded_step():
    assert layer_scan.parse_layer_range_spec("8:24:4") == (8, 24, 4)


def test_infer_final_layer_uses_model_depth():
    assert layer_scan.infer_final_layer(model_num_layers=28) == 27


def test_resolve_scan_layers_appends_final_layer_to_step_grid():
    layers = layer_scan.resolve_scan_layers(
        default_layer=12,
        layer_range_spec="0:24",
        layer_step=4,
        final_layer=27,
    )
    assert layers == [0, 4, 8, 12, 16, 20, 24, 27]


def test_resolve_scan_layers_supports_step_only_mode():
    layers = layer_scan.resolve_scan_layers(
        default_layer=12,
        layer_step=6,
        final_layer=27,
    )
    assert layers == [0, 6, 12, 18, 24, 27]


def test_resolve_scan_layers_rejects_conflicting_step_spec():
    with pytest.raises(ValueError):
        layer_scan.resolve_scan_layers(
            default_layer=12,
            layer_range_spec="8:24:4",
            layer_step=4,
            final_layer=27,
        )


def test_resolve_layer_path_formats_template():
    path = layer_scan.resolve_layer_path(
        layer=12,
        base_path="./data/acts",
        explicit_template="./data/acts_layer_{layer}",
    )
    assert path == Path("./data/acts_layer_12")


def test_build_layer_training_plans_uses_default_layer_subdirs():
    plans = layer_scan.build_layer_training_plans(
        layers=[12, 27],
        shards_path="./data/math500_activations",
        checkpoint_dir="./checkpoints/math500",
    )
    assert plans[0].shards_dir == Path("./data/math500_activations/layer_12")
    assert plans[1].checkpoint_dir == Path("./checkpoints/math500/layer_27")


def test_build_layer_training_plans_preserves_single_layer_paths():
    plans = layer_scan.build_layer_training_plans(
        layers=[12],
        shards_path="./data/math500_activations",
        checkpoint_dir="./checkpoints/math500",
        append_layer_subdirs=False,
    )
    assert plans[0].shards_dir == Path("./data/math500_activations")
    assert plans[0].checkpoint_dir == Path("./checkpoints/math500")
