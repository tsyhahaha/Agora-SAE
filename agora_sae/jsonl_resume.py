"""Helpers for safe append/resume workflows over JSONL outputs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence, Tuple


@dataclass(frozen=True)
class JsonlResumeState:
    """Existing JSONL progress recovered from disk."""

    completed_keys: frozenset[Tuple[Any, ...]]
    loaded_records: int
    skipped_invalid_lines: int = 0


def load_jsonl_records(path: Path) -> tuple[list[dict], int]:
    """Load JSONL records while tolerating a malformed final line."""
    if not path.exists():
        return [], 0

    lines = path.read_text(encoding="utf-8").splitlines()
    records = []
    skipped_invalid_lines = 0

    for line_number, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped:
            continue

        try:
            records.append(json.loads(stripped))
        except json.JSONDecodeError as exc:
            if line_number == len(lines):
                skipped_invalid_lines += 1
                continue
            raise ValueError(
                f"Malformed JSONL in {path} at line {line_number}. "
                "Use --overwrite-output after inspecting the file."
            ) from exc

    return records, skipped_invalid_lines


def load_jsonl_resume_state(path: Path, key_fields: Sequence[str]) -> JsonlResumeState:
    """Load existing JSONL records and recover their logical keys.

    A malformed final line is treated as a partial write and ignored. Any earlier
    malformed line still raises because it likely indicates file corruption.
    """
    if not path.exists():
        return JsonlResumeState(completed_keys=frozenset(), loaded_records=0)

    records, skipped_invalid_lines = load_jsonl_records(path)
    completed_keys = set()
    loaded_records = 0

    for record in records:
        missing_fields = [field for field in key_fields if field not in record]
        if missing_fields:
            raise ValueError(
                f"Existing JSONL record in {path} is missing key fields {missing_fields}. "
                "Use --overwrite-output if this file came from a different command."
            )

        completed_keys.add(tuple(record[field] for field in key_fields))
        loaded_records += 1

    return JsonlResumeState(
        completed_keys=frozenset(completed_keys),
        loaded_records=loaded_records,
        skipped_invalid_lines=skipped_invalid_lines,
    )


def prepare_jsonl_output(
    path: Path,
    *,
    key_fields: Sequence[str],
    resume: bool,
    overwrite: bool,
) -> tuple[str, JsonlResumeState]:
    """Validate the requested output mode and return the file mode plus resume state."""
    if resume and overwrite:
        raise ValueError("Use either --resume or --overwrite-output, not both.")

    if path.exists():
        if overwrite:
            path.unlink()
            return "w", JsonlResumeState(completed_keys=frozenset(), loaded_records=0)
        if resume:
            return "a", load_jsonl_resume_state(path, key_fields)
        raise FileExistsError(
            f"Output file already exists: {path}. "
            "Use --resume to continue from it or --overwrite-output to restart."
        )

    return "w", JsonlResumeState(completed_keys=frozenset(), loaded_records=0)
