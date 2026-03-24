#!/usr/bin/env python3
"""Paper-aligned evaluation workflow for the MATH500 reproduction path."""

import argparse
import json
from pathlib import Path

from agora_sae.eval.paper_math500 import (
    build_behavior_vector,
    capture_step_activations,
    create_reasoning_samples,
    embed_decoder_features,
    get_step_judge,
    load_geometry_summary,
    load_step_records,
    parse_condition_spec,
    run_intervention_eval,
    save_geometry_outputs,
    score_behavior_features,
    write_step_labels,
    compute_silhouette,
)
from agora_sae.jsonl_resume import load_jsonl_records, prepare_jsonl_output
from agora_sae.trainer.sae_trainer import load_sae_from_checkpoint


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(
        description="Paper-style MATH500 evaluation: judge labeling, geometry, and intervention."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    label_parser = subparsers.add_parser(
        "label-steps",
        help="Label reasoning steps as reflection/backtracking/other.",
    )
    label_parser.add_argument("--dataset-path", required=True, help="Dataset name or local path")
    label_parser.add_argument("--output", required=True, help="Output JSONL for labeled steps")
    label_parser.add_argument(
        "--response-cache",
        default=None,
        help=(
            "Optional JSONL cache for model-generated responses. "
            "Defaults to <output>.responses.jsonl when --response-source=model."
        ),
    )
    label_parser.add_argument(
        "--response-source",
        choices=["dataset", "model"],
        default="dataset",
        help="Use dataset responses or generate fresh model responses",
    )
    label_parser.add_argument("--model", default=None, help="Model name or path for response generation")
    label_parser.add_argument(
        "--prompt-template",
        default="{question}",
        help="Prompt template used when --response-source=model",
    )
    label_parser.add_argument("--delimiter", default="\n\n", help="Step delimiter")
    label_parser.add_argument("--judge", choices=["openai", "minimax", "heuristic"], default="heuristic")
    label_parser.add_argument(
        "--judge-model",
        default=None,
        help="Judge model for external judges. Defaults: gpt-5 for OpenAI, MiniMax-M2.5 for MiniMax.",
    )
    label_parser.add_argument(
        "--judge-timeout",
        type=int,
        default=60,
        help="HTTP timeout in seconds for external judge requests.",
    )
    label_parser.add_argument(
        "--judge-max-retries",
        type=int,
        default=5,
        help="Max retry attempts for transient external judge failures.",
    )
    label_parser.add_argument(
        "--minimax-max-output-tokens",
        type=int,
        default=128,
        help="Cap MiniMax judge output tokens to keep label responses short and stable.",
    )
    label_parser.add_argument(
        "--disable-minimax-reasoning-split",
        action="store_true",
        help="Disable MiniMax reasoning_split if the compatible endpoint is unstable for long prompts.",
    )
    label_parser.add_argument("--max-samples", type=int, default=500)
    label_parser.add_argument("--max-new-tokens", type=int, default=512)
    label_parser.add_argument("--temperature", type=float, default=0.0)
    label_parser.add_argument("--top-p", type=float, default=1.0)
    label_parser.add_argument("--seed", type=int, default=42)
    label_parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from an existing output JSONL and skip already labeled steps",
    )
    label_parser.add_argument(
        "--overwrite-output",
        action="store_true",
        help="Overwrite an existing output JSONL instead of resuming",
    )

    geometry_parser = subparsers.add_parser(
        "analyze-geometry",
        help="Score behavior-specific features and build decoder geometry artifacts.",
    )
    geometry_parser.add_argument("--labels", required=True, help="Labeled step JSONL")
    geometry_parser.add_argument("--checkpoint", required=True, help="SAE checkpoint path")
    geometry_parser.add_argument("--model", required=True, help="Model name or path")
    geometry_parser.add_argument("--layer", type=int, required=True, help="Hook layer")
    geometry_parser.add_argument("--output-dir", required=True, help="Directory for geometry outputs")
    geometry_parser.add_argument("--max-seq-length", type=int, default=4096)
    geometry_parser.add_argument("--top-features-per-label", type=int, default=20)
    geometry_parser.add_argument("--embedding-method", choices=["umap", "pca"], default="umap")
    geometry_parser.add_argument("--plot-path", default=None, help="Optional scatter plot output path")

    intervention_parser = subparsers.add_parser(
        "run-intervention",
        help="Run causal interventions using behavior vectors derived from decoder columns.",
    )
    intervention_parser.add_argument("--dataset-path", required=True, help="Dataset name or local path")
    intervention_parser.add_argument("--geometry-summary", required=True, help="Path to geometry_summary.json")
    intervention_parser.add_argument("--checkpoint", required=True, help="SAE checkpoint path")
    intervention_parser.add_argument("--model", required=True, help="Model name or path")
    intervention_parser.add_argument("--layer", type=int, required=True, help="Hook layer")
    intervention_parser.add_argument(
        "--behavior",
        choices=["reflection", "backtracking", "other"],
        required=True,
        help="Behavior vector to apply",
    )
    intervention_parser.add_argument("--output", required=True, help="JSONL output for intervention runs")
    intervention_parser.add_argument("--delimiter", default="\n\n", help="Step delimiter")
    intervention_parser.add_argument(
        "--judge",
        choices=["openai", "minimax", "heuristic"],
        default="heuristic",
        help="Judge used to count generated behaviors",
    )
    intervention_parser.add_argument(
        "--judge-model",
        default=None,
        help="Judge model for external judges. Defaults: gpt-5 for OpenAI, MiniMax-M2.5 for MiniMax.",
    )
    intervention_parser.add_argument(
        "--judge-timeout",
        type=int,
        default=60,
        help="HTTP timeout in seconds for external judge requests.",
    )
    intervention_parser.add_argument(
        "--judge-max-retries",
        type=int,
        default=5,
        help="Max retry attempts for transient external judge failures.",
    )
    intervention_parser.add_argument(
        "--minimax-max-output-tokens",
        type=int,
        default=128,
        help="Cap MiniMax judge output tokens to keep label responses short and stable.",
    )
    intervention_parser.add_argument(
        "--disable-minimax-reasoning-split",
        action="store_true",
        help="Disable MiniMax reasoning_split if the compatible endpoint is unstable for long prompts.",
    )
    intervention_parser.add_argument("--max-samples", type=int, default=32)
    intervention_parser.add_argument("--top-features", type=int, default=8)
    intervention_parser.add_argument(
        "--conditions",
        default="negative:1.0,vanilla:0.0,positive:-1.0",
        help="Comma-separated name:alpha pairs",
    )
    intervention_parser.add_argument("--max-new-tokens", type=int, default=384)
    intervention_parser.add_argument("--temperature", type=float, default=0.0)
    intervention_parser.add_argument("--top-p", type=float, default=1.0)
    intervention_parser.add_argument("--seed", type=int, default=42)
    intervention_parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from an existing intervention JSONL and skip completed sample/condition pairs",
    )
    intervention_parser.add_argument(
        "--overwrite-output",
        action="store_true",
        help="Overwrite an existing intervention JSONL instead of resuming",
    )

    return parser


def _load_prefetched_responses_from_label_output(output_path: Path) -> dict[str, dict]:
    """Recover cached sample responses from an existing label JSONL."""
    records, skipped_invalid_lines = load_jsonl_records(output_path)
    prefetched = {}
    for record in records:
        sample_id = str(record["sample_id"])
        prefetched.setdefault(
            sample_id,
            {
                "sample_id": sample_id,
                "question": record.get("question", ""),
                "response": record.get("response", ""),
                "reference_answer": record.get("reference_answer"),
            },
        )
    if prefetched:
        print(
            f"Recovered model responses for {len(prefetched)} sample(s) "
            f"from existing label output {output_path}."
        )
    if skipped_invalid_lines:
        print(
            f"Ignored {skipped_invalid_lines} malformed trailing JSONL line(s) "
            f"while reading {output_path}."
        )
    return prefetched


def run_label_steps(args: argparse.Namespace):
    """Execute the label-steps subcommand."""
    _, existing_state = prepare_jsonl_output(
        Path(args.output),
        key_fields=("sample_id", "step_id"),
        resume=args.resume,
        overwrite=args.overwrite_output,
    )
    if existing_state.loaded_records:
        print(f"Found {existing_state.loaded_records} existing labeled steps in {args.output}.")
    response_cache_path = None
    prefetched_responses = None
    if args.response_source == "model":
        response_cache_path = (
            Path(args.response_cache)
            if args.response_cache
            else Path(args.output).with_suffix(".responses.jsonl")
        )
        if args.resume and Path(args.output).exists():
            prefetched_responses = _load_prefetched_responses_from_label_output(Path(args.output))
    judge = get_step_judge(
        args.judge,
        judge_model=args.judge_model,
        timeout=args.judge_timeout,
        max_retries=args.judge_max_retries,
        minimax_max_output_tokens=args.minimax_max_output_tokens,
        minimax_reasoning_split=not args.disable_minimax_reasoning_split,
    )
    samples = create_reasoning_samples(
        dataset_path=args.dataset_path,
        delimiter=args.delimiter,
        max_samples=args.max_samples,
        response_source=args.response_source,
        model_name=args.model,
        prompt_template=args.prompt_template,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
        response_cache_path=response_cache_path,
        resume_response_cache=args.resume,
        overwrite_response_cache=args.overwrite_output,
        prefetched_responses=prefetched_responses,
    )
    print(f"Prepared {len(samples)} reasoning samples.")
    if args.judge in {"openai", "minimax"}:
        print(
            f"Starting external step labeling via {args.judge} "
            f"(model={getattr(judge, 'model', args.judge_model)})..."
        )
    else:
        print("Starting heuristic step labeling...")
    label_summary = write_step_labels(
        samples=samples,
        judge=judge,
        output_path=Path(args.output),
        judge_name=args.judge,
        judge_model=getattr(judge, "model", None) if args.judge in {"openai", "minimax"} else None,
        resume=args.resume,
        overwrite_output=args.overwrite_output,
    )
    if label_summary["recovered_records"]:
        print(
            f"Recovered {label_summary['recovered_records']} existing labeled steps "
            f"from {args.output}."
        )
    if label_summary["skipped_invalid_lines"]:
        print(
            f"Ignored {label_summary['skipped_invalid_lines']} malformed trailing JSONL line(s) "
            f"while resuming {args.output}."
        )
    print(
        f"Wrote {label_summary['written_records']} new labeled steps "
        f"({label_summary['total_records']} total) across {len(samples)} samples."
    )
    print(f"Saved labels to: {args.output}")


def run_analyze_geometry(args: argparse.Namespace):
    """Execute the analyze-geometry subcommand."""
    records = load_step_records(Path(args.labels))
    sae = load_sae_from_checkpoint(Path(args.checkpoint))
    step_activations = capture_step_activations(
        records=records,
        model_name=args.model,
        hook_layer=args.layer,
        max_seq_length=args.max_seq_length,
    )
    summary = score_behavior_features(
        step_activations=step_activations,
        sae=sae,
        top_features_per_label=args.top_features_per_label,
    )
    points = embed_decoder_features(
        sae=sae,
        feature_assignments=summary["feature_assignments"],
        method=args.embedding_method,
    )
    summary["embedding_method"] = args.embedding_method
    summary["num_labeled_steps"] = len(records)
    summary["num_captured_steps"] = len(step_activations)
    summary["silhouette_score"] = compute_silhouette(points)
    save_geometry_outputs(
        output_dir=Path(args.output_dir),
        summary=summary,
        points=points,
        plot_path=Path(args.plot_path) if args.plot_path else None,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Saved geometry artifacts to: {args.output_dir}")


def run_intervention(args: argparse.Namespace):
    """Execute the run-intervention subcommand."""
    _, existing_state = prepare_jsonl_output(
        Path(args.output),
        key_fields=("sample_id", "condition"),
        resume=args.resume,
        overwrite=args.overwrite_output,
    )
    if existing_state.loaded_records:
        print(f"Found {existing_state.loaded_records} existing intervention records in {args.output}.")
    judge = get_step_judge(
        args.judge,
        judge_model=args.judge_model,
        timeout=args.judge_timeout,
        max_retries=args.judge_max_retries,
        minimax_max_output_tokens=args.minimax_max_output_tokens,
        minimax_reasoning_split=not args.disable_minimax_reasoning_split,
    )
    summary = load_geometry_summary(Path(args.geometry_summary))
    sae = load_sae_from_checkpoint(Path(args.checkpoint))
    behavior_vector = build_behavior_vector(
        sae=sae,
        summary=summary,
        behavior=args.behavior,
        top_n_features=args.top_features,
    )
    samples = create_reasoning_samples(
        dataset_path=args.dataset_path,
        delimiter=args.delimiter,
        max_samples=args.max_samples,
        response_source="dataset",
    )
    samples = [sample for sample in samples if sample.question.strip()]
    if not samples:
        raise ValueError("No question-bearing samples were available for intervention.")
    results = run_intervention_eval(
        samples=samples,
        model_name=args.model,
        hook_layer=args.layer,
        behavior_vector=behavior_vector,
        conditions=parse_condition_spec(args.conditions),
        judge=judge,
        delimiter=args.delimiter,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        output_path=Path(args.output),
        seed=args.seed,
        resume=args.resume,
        overwrite_output=args.overwrite_output,
    )
    print(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"Saved intervention runs to: {args.output}")


def main():
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "label-steps":
        run_label_steps(args)
    elif args.command == "analyze-geometry":
        run_analyze_geometry(args)
    elif args.command == "run-intervention":
        run_intervention(args)
    else:
        raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
