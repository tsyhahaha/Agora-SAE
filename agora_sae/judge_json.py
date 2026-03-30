"""Utilities for parsing and repairing judge JSON outputs."""

from __future__ import annotations

import json
import re
from typing import Dict, Optional


ALLOWED_LABELS = frozenset({"reflection", "backtracking", "other"})
JSON_BLOCK_PATTERN = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
JSON_LABEL_PATTERN = re.compile(
    r'"label"\s*:\s*"(?P<label>reflection|backtracking|other)"',
    re.IGNORECASE,
)
JSON_RATIONALE_PATTERN = re.compile(
    r'"rationale"\s*:\s*"(?P<rationale>.*)',
    re.IGNORECASE | re.DOTALL,
)
FREEFORM_LABEL_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r'\blabel\s+should\s+be\s+"?(reflection|backtracking|other)"?',
        r'\blabel\s+is\s+"?(reflection|backtracking|other)"?',
        r'\bthe\s+label\s+should\s+be\s+"?(reflection|backtracking|other)"?',
        r'\bthe\s+label\s+is\s+"?(reflection|backtracking|other)"?',
        r'\blabel\s*:\s*"?(reflection|backtracking|other)"?',
    )
)


def parse_and_repair_label_payload(raw_text: str) -> Dict[str, Optional[str]]:
    """Parse a judge response into a validated label payload.

    The parser first tries strict JSON candidates, then JSON-like repair, and
    finally a free-form label extraction fallback.
    """
    if not raw_text or not raw_text.strip():
        raise ValueError("Judge response was empty.")

    candidates = [raw_text.strip()]

    fenced_match = JSON_BLOCK_PATTERN.search(raw_text)
    if fenced_match:
        candidates.append(fenced_match.group(1).strip())

    start_index = raw_text.find("{")
    end_index = raw_text.rfind("}")
    if start_index != -1 and end_index != -1 and end_index > start_index:
        candidates.append(raw_text[start_index : end_index + 1].strip())

    for candidate in candidates:
        parsed = _try_parse_json(candidate)
        if parsed is not None:
            return parsed

    repaired = _extract_json_like_payload(raw_text)
    if repaired is not None:
        return repaired

    fallback = _extract_freeform_label_payload(raw_text)
    if fallback is not None:
        return fallback

    raise ValueError(f"Could not parse judge response as label JSON: {raw_text}")


def _try_parse_json(candidate: str) -> Optional[Dict[str, Optional[str]]]:
    for variant in _iter_repaired_json_candidates(candidate):
        try:
            parsed = json.loads(variant)
        except json.JSONDecodeError:
            continue
        validated = _validate_payload(parsed)
        if validated is not None:
            return validated
    return None


def _iter_repaired_json_candidates(candidate: str):
    normalized = candidate.strip()
    if not normalized:
        return

    yield normalized

    compact = normalized.replace("\r", "")
    if compact != normalized:
        yield compact

    if compact.startswith("{") and '"rationale"' in compact:
        repaired = compact
        if repaired.count('"') % 2 == 1:
            repaired = repaired + '"'
        if not repaired.rstrip().endswith("}"):
            repaired = repaired.rstrip() + "}"
        repaired = repaired.replace("\n", " ")
        yield repaired


def _extract_json_like_payload(raw_text: str) -> Optional[Dict[str, Optional[str]]]:
    label_match = JSON_LABEL_PATTERN.search(raw_text)
    if not label_match:
        return None

    label = label_match.group("label").lower()
    rationale_match = JSON_RATIONALE_PATTERN.search(raw_text)
    rationale = None
    if rationale_match:
        rationale = rationale_match.group("rationale")
        rationale = rationale.split('"}', 1)[0]
        rationale = rationale.split('", "', 1)[0]
        rationale = _normalize_text(rationale)
    return {"label": label, "rationale": rationale or None}


def _extract_freeform_label_payload(raw_text: str) -> Optional[Dict[str, Optional[str]]]:
    for pattern in FREEFORM_LABEL_PATTERNS:
        match = pattern.search(raw_text)
        if not match:
            continue
        label = match.group(1).lower()
        trailing = raw_text[match.end() :].strip()
        rationale_hint = re.search(
            r"rationale(?:\s+\w+){0,4}\s+(?:that\s+)?(?P<reason>.+)",
            trailing,
            re.IGNORECASE | re.DOTALL,
        )
        if rationale_hint:
            rationale = _normalize_text(rationale_hint.group("reason"))
            if rationale:
                return {"label": label, "rationale": rationale}

        rationale_source = raw_text[: match.start()].strip() or trailing or raw_text.strip()
        rationale = _summarize_freeform_rationale(rationale_source)
        return {"label": label, "rationale": rationale}
    return None


def _validate_payload(parsed: object) -> Optional[Dict[str, Optional[str]]]:
    if not isinstance(parsed, dict):
        return None
    label = parsed.get("label")
    if not isinstance(label, str):
        return None
    label = label.lower()
    if label not in ALLOWED_LABELS:
        return None
    rationale = parsed.get("rationale")
    if rationale is not None:
        rationale = _normalize_text(str(rationale))
    return {"label": label, "rationale": rationale}


def _summarize_freeform_rationale(text: str) -> str:
    normalized = _normalize_text(text)
    if not normalized:
        return "Recovered from non-JSON judge output."
    sentences = re.split(r"(?<=[.!?])\s+", normalized)
    summary = normalized
    for sentence in reversed(sentences):
        sentence = sentence.strip()
        if len(sentence) >= 20:
            summary = sentence
            break
    if len(summary) > 220:
        summary = summary[:217].rstrip() + "..."
    return summary


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().strip('"}')
