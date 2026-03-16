"""Utilities for planning layer-wise activation and SAE scan runs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence


@dataclass(frozen=True)
class LayerTrainingPlan:
    """Resolved IO plan for training an SAE on a single model layer."""

    layer: int
    shards_dir: Path
    checkpoint_dir: Path
    resume_path: Optional[Path] = None


def parse_layers_spec(spec: str) -> List[int]:
    """Parse a comma-separated layer list."""
    layers: List[int] = []
    for raw_part in spec.split(","):
        part = raw_part.strip()
        if not part:
            continue
        layer = int(part)
        if layer < 0:
            raise ValueError(f"Layer indices must be >= 0, got {layer}")
        layers.append(layer)

    if not layers:
        raise ValueError("Expected at least one layer in --layers")

    return sorted(set(layers))


def parse_layer_range_spec(spec: str) -> tuple[int, int, Optional[int]]:
    """
    Parse a layer range specification.

    Supported formats:
    - start:end
    - start:end:step
    """
    parts = [part.strip() for part in spec.split(":")]
    if len(parts) not in (2, 3):
        raise ValueError(
            f"Invalid --layer-range '{spec}'. Expected start:end or start:end:step"
        )

    start = int(parts[0])
    end = int(parts[1])
    step = int(parts[2]) if len(parts) == 3 else None

    if start < 0 or end < 0:
        raise ValueError("Layer indices must be >= 0")
    if end < start:
        raise ValueError(
            f"Invalid --layer-range '{spec}': end layer must be >= start layer"
        )
    if step is not None and step <= 0:
        raise ValueError("Layer step must be > 0")

    return start, end, step


def infer_final_layer(
    *,
    model_num_layers: Optional[int] = None,
    final_layer_override: Optional[int] = None,
) -> Optional[int]:
    """Resolve the final layer index for scan planning."""
    if final_layer_override is not None:
        if final_layer_override < 0:
            raise ValueError("--final-layer must be >= 0")
        return final_layer_override

    if model_num_layers is None:
        return None

    if model_num_layers <= 0:
        raise ValueError("model_num_layers must be > 0 when provided")

    return model_num_layers - 1


def resolve_scan_layers(
    *,
    default_layer: int,
    layers_spec: Optional[str] = None,
    layer_range_spec: Optional[str] = None,
    layer_step: Optional[int] = None,
    final_layer: Optional[int] = None,
) -> List[int]:
    """
    Resolve the ordered set of layers to train/evaluate.

    Scan mode is triggered by any of --layers / --layer-range / --layer-step.
    When scan mode is active and final_layer is known, the final layer is always
    appended even if it does not lie on the requested step grid.
    """
    if default_layer < 0:
        raise ValueError("default_layer must be >= 0")
    if layer_step is not None and layer_step <= 0:
        raise ValueError("--layer-step must be > 0")
    if layers_spec and layer_range_spec:
        raise ValueError("Use either --layers or --layer-range, not both")

    scan_requested = (
        layers_spec is not None
        or layer_range_spec is not None
        or layer_step is not None
    )
    if not scan_requested:
        return [default_layer]

    if layers_spec is not None:
        layers = parse_layers_spec(layers_spec)
    elif layer_range_spec is not None:
        start, end, embedded_step = parse_layer_range_spec(layer_range_spec)
        if embedded_step is not None and layer_step is not None:
            raise ValueError(
                "Specify the step either in --layer-range start:end:step "
                "or with --layer-step, not both"
            )
        step = layer_step or embedded_step or 1
        layers = list(range(start, end + 1, step))
    else:
        if final_layer is None:
            raise ValueError(
                "--layer-step requires a known final layer. Pass --final-layer."
            )
        layers = list(range(0, final_layer + 1, layer_step))

    if final_layer is not None:
        layers.append(final_layer)

    return sorted(set(layers))


def resolve_layer_path(
    *,
    layer: int,
    base_path: str,
    explicit_template: Optional[str] = None,
    append_layer_subdir: bool = True,
) -> Path:
    """
    Resolve a layer-specific path.

    Rules:
    - explicit template wins
    - otherwise, if base_path already contains {layer}, format it
    - otherwise append /layer_{layer}
    """
    template = explicit_template or base_path
    if "{layer}" in template:
        return Path(template.format(layer=layer))
    if append_layer_subdir:
        return Path(base_path) / f"layer_{layer}"
    return Path(base_path)


def build_layer_training_plans(
    *,
    layers: Sequence[int],
    shards_path: str,
    checkpoint_dir: str,
    shards_template: Optional[str] = None,
    checkpoint_template: Optional[str] = None,
    resume: Optional[str] = None,
    resume_template: Optional[str] = None,
    append_layer_subdirs: bool = True,
) -> List[LayerTrainingPlan]:
    """Build one training plan per target layer."""
    plans: List[LayerTrainingPlan] = []
    for layer in layers:
        shards_dir = resolve_layer_path(
            layer=layer,
            base_path=shards_path,
            explicit_template=shards_template,
            append_layer_subdir=append_layer_subdirs,
        )
        checkpoint_path = resolve_layer_path(
            layer=layer,
            base_path=checkpoint_dir,
            explicit_template=checkpoint_template,
            append_layer_subdir=append_layer_subdirs,
        )

        resume_path: Optional[Path] = None
        if resume_template:
            resume_path = Path(resume_template.format(layer=layer))
        elif resume and "{layer}" in resume:
            resume_path = Path(resume.format(layer=layer))
        elif resume and len(layers) == 1:
            resume_path = Path(resume)

        plans.append(
            LayerTrainingPlan(
                layer=layer,
                shards_dir=shards_dir,
                checkpoint_dir=checkpoint_path,
                resume_path=resume_path,
            )
        )

    return plans


def write_scan_manifest(
    *,
    path: Path,
    preset: Optional[str],
    model_name: str,
    layers: Sequence[int],
    plans: Sequence[LayerTrainingPlan],
    metadata: Optional[dict] = None,
) -> None:
    """Persist a layer-scan manifest for downstream eval tooling."""
    payload = {
        "preset": preset,
        "model_name": model_name,
        "layers": list(layers),
        "runs": [
            {
                "layer": plan.layer,
                "shards_dir": str(plan.shards_dir),
                "checkpoint_dir": str(plan.checkpoint_dir),
                "resume_path": str(plan.resume_path) if plan.resume_path else None,
            }
            for plan in plans
        ],
    }
    if metadata:
        payload["metadata"] = metadata

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
