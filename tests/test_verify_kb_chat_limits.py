"""scripts/verify_kb_chat_limits.py 的单元测试。"""

from __future__ import annotations

import json

import pytest
from verify_kb_chat_limits import (
    derive_health_url,
    main,
    run_burst_probe,
    run_security_probes,
    validate_endpoint_url,
)


def test_validate_endpoint_url_rejects_unsafe_schemes() -> None:
    assert validate_endpoint_url("https://demo.workers.dev/chat") == "https://demo.workers.dev/chat"
    assert validate_endpoint_url("http://127.0.0.1:8787/chat") == "http://127.0.0.1:8787/chat"
    for bad in (
        "http://demo.workers.dev/chat",
        "file:///etc/passwd",
        "https://user:pass@demo.workers.dev/chat",
        "https://demo.workers.dev/other",
    ):
        with pytest.raises(ValueError):
            validate_endpoint_url(bad)


def test_derive_health_url() -> None:
    assert derive_health_url("https://demo.workers.dev/chat") == "https://demo.workers.dev/health"
    assert derive_health_url("https://demo.workers.dev") == "https://demo.workers.dev/health"


def test_run_security_probes_and_burst_probe() -> None:
    seen: list[tuple[str, str]] = []

    def fake_requester(url: str, method: str, body: str | None, timeout: float) -> tuple[int, str]:
        seen.append((method, url))
        if url.endswith("/health"):
            return 200, json.dumps({"ok": True, "mode": "readonly-kb-chat"})
        assert body is not None
        parsed = json.loads(body)
        if len(parsed.get("question", "")) > 4000:
            return 413, json.dumps({"ok": False, "code": "PAYLOAD_TOO_LARGE"})
        if "tools" in parsed:
            return 400, json.dumps({"ok": False, "code": "INVALID_PAYLOAD"})
        if any(".." in h for h in parsed.get("contentHashes", [])):
            return 400, json.dumps({"ok": False, "code": "INVALID_PAYLOAD"})
        return 403, json.dumps({"ok": False, "code": "TURNSTILE_FAILED"})

    results = run_security_probes("https://demo.workers.dev/chat", requester=fake_requester)
    assert all(r.passed for r in results)
    assert ("GET", "https://demo.workers.dev/health") in seen

    calls = 0

    def burst_requester(
        _url: str, _method: str, _body: str | None, _timeout: float
    ) -> tuple[int, str]:
        nonlocal calls
        calls += 1
        return (429 if calls > 3 else 403), json.dumps({"ok": False})

    ok, statuses = run_burst_probe(
        "https://demo.workers.dev/chat", count=6, requester=burst_requester
    )
    assert ok is True
    assert statuses == [403, 403, 403, 429, 429, 429]


def test_run_security_probes_detects_secret_leak(monkeypatch: pytest.MonkeyPatch) -> None:
    def leaky_requester(
        _url: str, _method: str, _body: str | None, _timeout: float
    ) -> tuple[int, str]:
        return 200, json.dumps({"error": "TURNSTILE_SECRET_KEY leaked"})

    results = run_security_probes("https://demo.workers.dev/chat", requester=leaky_requester)
    assert results[0].passed is False
    assert "sensitive keyword" in results[0].detail

    monkeypatch.setattr(
        "verify_kb_chat_limits.run_security_probes",
        lambda *_args, **_kwargs: results,
    )
    assert main(["--endpoint", "https://demo.workers.dev/chat", "--mode", "security"]) == 1
