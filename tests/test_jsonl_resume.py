import importlib.util
import sys
from pathlib import Path


def _load_jsonl_resume_module():
    module_name = "agora_sae_jsonl_resume_test"
    spec = importlib.util.spec_from_file_location(
        module_name,
        Path(__file__).resolve().parents[1] / "agora_sae" / "jsonl_resume.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


jsonl_resume = _load_jsonl_resume_module()


def test_prepare_jsonl_output_for_new_file(tmp_path):
    output_path = tmp_path / "labels.jsonl"
    mode, state = jsonl_resume.prepare_jsonl_output(
        output_path,
        key_fields=("sample_id", "step_id"),
        resume=False,
        overwrite=False,
    )
    assert mode == "w"
    assert state.loaded_records == 0
    assert len(state.completed_keys) == 0


def test_prepare_jsonl_output_requires_explicit_resume_or_overwrite(tmp_path):
    output_path = tmp_path / "labels.jsonl"
    output_path.write_text('{"sample_id":"0","step_id":0}\n', encoding="utf-8")

    try:
        jsonl_resume.prepare_jsonl_output(
            output_path,
            key_fields=("sample_id", "step_id"),
            resume=False,
            overwrite=False,
        )
    except FileExistsError as exc:
        assert "--resume" in str(exc)
        assert "--overwrite-output" in str(exc)
    else:
        raise AssertionError("Expected FileExistsError when output exists without resume/overwrite")


def test_load_jsonl_resume_state_recovers_existing_keys(tmp_path):
    output_path = tmp_path / "labels.jsonl"
    output_path.write_text(
        '{"sample_id":"0","step_id":0}\n{"sample_id":"0","step_id":1}\n',
        encoding="utf-8",
    )

    state = jsonl_resume.load_jsonl_resume_state(output_path, ("sample_id", "step_id"))

    assert state.loaded_records == 2
    assert state.skipped_invalid_lines == 0
    assert ("0", 0) in state.completed_keys
    assert ("0", 1) in state.completed_keys


def test_load_jsonl_resume_state_ignores_malformed_final_line(tmp_path):
    output_path = tmp_path / "labels.jsonl"
    output_path.write_text(
        '{"sample_id":"0","step_id":0}\n{"sample_id":"1"',
        encoding="utf-8",
    )

    state = jsonl_resume.load_jsonl_resume_state(output_path, ("sample_id", "step_id"))

    assert state.loaded_records == 1
    assert state.skipped_invalid_lines == 1
    assert ("0", 0) in state.completed_keys


def test_prepare_jsonl_output_resume_mode_returns_append_and_state(tmp_path):
    output_path = tmp_path / "intervention.jsonl"
    output_path.write_text(
        '{"sample_id":"3","condition":"vanilla"}\n',
        encoding="utf-8",
    )

    mode, state = jsonl_resume.prepare_jsonl_output(
        output_path,
        key_fields=("sample_id", "condition"),
        resume=True,
        overwrite=False,
    )

    assert mode == "a"
    assert state.loaded_records == 1
    assert ("3", "vanilla") in state.completed_keys
