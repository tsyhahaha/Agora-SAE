import importlib.util
import sys
from pathlib import Path


def _load_judge_json_module():
    module_name = "agora_sae_judge_json_test"
    spec = importlib.util.spec_from_file_location(
        module_name,
        Path(__file__).resolve().parents[1] / "agora_sae" / "judge_json.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


judge_json = _load_judge_json_module()


def test_parse_and_repair_label_payload_accepts_valid_json():
    parsed = judge_json.parse_and_repair_label_payload(
        '{"label":"other","rationale":"routine step"}'
    )
    assert parsed == {"label": "other", "rationale": "routine step"}


def test_parse_and_repair_label_payload_repairs_truncated_json():
    parsed = judge_json.parse_and_repair_label_payload(
        '{"label": "other", "rationale": "This step introduces the setup'
    )
    assert parsed["label"] == "other"
    assert "introduces the setup" in parsed["rationale"]


def test_parse_and_repair_label_payload_extracts_label_from_freeform_text():
    raw_text = (
        "This is a routine computational continuation. "
        'So the label should be "other" with a short rationale.'
    )
    parsed = judge_json.parse_and_repair_label_payload(raw_text)
    assert parsed["label"] == "other"
    assert parsed["rationale"]


def test_parse_and_repair_label_payload_extracts_fenced_json():
    parsed = judge_json.parse_and_repair_label_payload(
        '```json\n{"label":"reflection","rationale":"re-checking arithmetic"}\n```'
    )
    assert parsed == {
        "label": "reflection",
        "rationale": "re-checking arithmetic",
    }
