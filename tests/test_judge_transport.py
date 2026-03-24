import importlib.util
import io
import sys
import urllib.error
from pathlib import Path

import pytest


def _load_judge_transport_module():
    module_name = "agora_sae_judge_transport_test"
    spec = importlib.util.spec_from_file_location(
        module_name,
        Path(__file__).resolve().parents[1] / "agora_sae" / "judge_transport.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


judge_transport = _load_judge_transport_module()


class _FakeResponse:
    def __init__(self, payload: str):
        self._payload = payload.encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self):
        return self._payload


def _http_error(url: str, code: int, body: str) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        url=url,
        code=code,
        msg=f"HTTP {code}",
        hdrs=None,
        fp=io.BytesIO(body.encode("utf-8")),
    )


def test_should_retry_http_status():
    assert judge_transport.should_retry_http_status(500) is True
    assert judge_transport.should_retry_http_status(429) is True
    assert judge_transport.should_retry_http_status(401) is False
    assert judge_transport.should_retry_http_status(None) is False


def test_compute_retry_delay_without_jitter_is_exact():
    assert judge_transport.compute_retry_delay_seconds(0, jitter_ratio=0.0) == 1.0
    assert judge_transport.compute_retry_delay_seconds(1, jitter_ratio=0.0) == 2.0
    assert judge_transport.compute_retry_delay_seconds(10, jitter_ratio=0.0, max_delay=20.0) == 20.0


def test_post_json_with_retry_retries_retryable_http_errors(monkeypatch):
    attempts = []
    sleeps = []
    events = [
        _http_error("https://example.test", 500, '{"error":"temporary"}'),
        _FakeResponse('{"ok": true}'),
    ]

    def fake_urlopen(request, timeout):
        attempts.append((request.full_url, timeout))
        event = events.pop(0)
        if isinstance(event, Exception):
            raise event
        return event

    monkeypatch.setattr(judge_transport.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(judge_transport.time, "sleep", lambda seconds: sleeps.append(seconds))
    monkeypatch.setattr(judge_transport, "compute_retry_delay_seconds", lambda attempt: 0.5)

    response = judge_transport.post_json_with_retry(
        url="https://example.test",
        payload={"x": 1},
        headers={"Authorization": "Bearer test"},
        timeout=30,
        max_retries=3,
        provider_name="MiniMax judge",
    )

    assert response == {"ok": True}
    assert len(attempts) == 2
    assert sleeps == [0.5]


def test_post_json_with_retry_fails_fast_on_non_retryable_http_error(monkeypatch):
    sleeps = []

    def fake_urlopen(request, timeout):
        raise _http_error("https://example.test", 401, '{"error":"unauthorized"}')

    monkeypatch.setattr(judge_transport.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(judge_transport.time, "sleep", lambda seconds: sleeps.append(seconds))

    with pytest.raises(RuntimeError, match="401"):
        judge_transport.post_json_with_retry(
            url="https://example.test",
            payload={"x": 1},
            headers={"Authorization": "Bearer test"},
            timeout=30,
            max_retries=3,
            provider_name="MiniMax judge",
        )

    assert sleeps == []
