#!/usr/bin/env python3
"""站长上线后验证只读 AI 问答 Worker 网关的安全拦截、限流与配额降级契约。"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable

DUMMY_HASH = "a" * 64


@dataclass(frozen=True)
class ProbeCase:
    name: str
    method: str
    path: str
    body: str | None
    expected_statuses: tuple[int, ...]


@dataclass(frozen=True)
class ProbeResult:
    name: str
    status: int
    passed: bool
    detail: str


Requester = Callable[[str, str, str | None, float], tuple[int, str]]


def validate_endpoint_url(raw_url: str) -> str:
    """校验待测试的 Worker /chat 端点地址（仅允许 HTTPS 或本地回环调试地址）。"""
    parsed = urllib.parse.urlparse(raw_url.strip())
    is_https = parsed.scheme == "https" and bool(parsed.netloc)
    is_local_http = parsed.scheme == "http" and parsed.hostname in {"127.0.0.1", "localhost"}
    if not (is_https or is_local_http):
        raise ValueError("endpoint must use https:// (or http://127.0.0.1 / localhost)")
    if parsed.username or parsed.password:
        raise ValueError("endpoint must not embed credentials")
    if parsed.path not in {"", "/", "/chat"}:
        raise ValueError("endpoint path should point to /chat")
    return raw_url.strip().rstrip("/")


def derive_health_url(chat_endpoint: str) -> str:
    """从 /chat 端点推导出对应的 /health 端点 URL。"""
    parsed = urllib.parse.urlparse(chat_endpoint)
    base_path = parsed.path[:-5] if parsed.path.endswith("/chat") else parsed.path.rstrip("/")
    return urllib.parse.urlunparse(
        (parsed.scheme, parsed.netloc, f"{base_path}/health", "", "", "")
    )


def default_requester(url: str, method: str, body: str | None, timeout: float) -> tuple[int, str]:
    """使用标准库 urllib 向指定端点发送验证请求。"""
    headers = {"Accept": "application/json"}
    data: bytes | None = None
    if body is not None:
        headers["Content-Type"] = "application/json"
        data = body.encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return int(resp.status), resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        return int(exc.code), exc.read().decode("utf-8", errors="replace")


def build_security_probes() -> list[ProbeCase]:
    """构造负向安全与隔离回归测试用例（不消耗任何 AI Neurons）。"""
    return [
        ProbeCase(
            name="health-endpoint",
            method="GET",
            path="/health",
            body=None,
            expected_statuses=(200,),
        ),
        ProbeCase(
            name="reject-oversized-body",
            method="POST",
            path="/chat",
            body=json.dumps(
                {
                    "question": "x" * 5000,
                    "lang": "zh",
                    "turnstileToken": "probe",
                    "contentHashes": [DUMMY_HASH],
                }
            ),
            expected_statuses=(413,),
        ),
        ProbeCase(
            name="reject-tool-injection-field",
            method="POST",
            path="/chat",
            body=json.dumps(
                {
                    "question": "What is WBC?",
                    "lang": "en",
                    "turnstileToken": "probe",
                    "contentHashes": [DUMMY_HASH],
                    "tools": [{"name": "bash"}],
                }
            ),
            expected_statuses=(400,),
        ),
        ProbeCase(
            name="reject-path-traversal-hash",
            method="POST",
            path="/chat",
            body=json.dumps(
                {
                    "question": "What is WBC?",
                    "lang": "en",
                    "turnstileToken": "probe",
                    "contentHashes": ["../../etc/passwd"],
                }
            ),
            expected_statuses=(400,),
        ),
        ProbeCase(
            name="reject-unverified-turnstile",
            method="POST",
            path="/chat",
            body=json.dumps(
                {
                    "question": "什么是全身控制 WBC？",
                    "lang": "zh",
                    "turnstileToken": "invalid-probe-token",
                    "contentHashes": [DUMMY_HASH],
                }
            ),
            expected_statuses=(403, 429, 503),
        ),
    ]


def _check_response_safe(raw_body: str) -> str | None:
    """确认响应不泄露栈追踪或 Secret 关键字。"""
    lower = raw_body.lower()
    for banned in ("turnstile_secret_key", "github_token", "traceback", "stack"):
        if banned in lower:
            return f"response contains sensitive keyword: {banned}"
    try:
        parsed: Any = json.loads(raw_body)
    except json.JSONDecodeError:
        return "response is not valid JSON"
    if not isinstance(parsed, dict):
        return "response JSON is not an object"
    return None


def run_security_probes(
    endpoint: str,
    requester: Requester = default_requester,
    timeout: float = 8.0,
) -> list[ProbeResult]:
    """执行负向安全与输入约束测试。"""
    chat_url = validate_endpoint_url(endpoint)
    if not chat_url.endswith("/chat"):
        chat_url = f"{chat_url}/chat"
    health_url = derive_health_url(chat_url)

    results: list[ProbeResult] = []
    for case in build_security_probes():
        target = health_url if case.path == "/health" else chat_url
        status, raw = requester(target, case.method, case.body, timeout)
        safety_issue = _check_response_safe(raw)
        ok = status in case.expected_statuses and safety_issue is None
        detail = safety_issue or f"status={status} (expected {case.expected_statuses})"
        results.append(ProbeResult(name=case.name, status=status, passed=ok, detail=detail))
    return results


def run_burst_probe(
    endpoint: str,
    count: int = 12,
    requester: Requester = default_requester,
    timeout: float = 8.0,
) -> tuple[bool, list[int]]:
    """连发请求验证 429 边缘/Isolate 限速触发，且无 500 内部错误。"""
    chat_url = validate_endpoint_url(endpoint)
    if not chat_url.endswith("/chat"):
        chat_url = f"{chat_url}/chat"
    payload = json.dumps(
        {
            "question": "什么是全身控制 WBC？",
            "lang": "zh",
            "turnstileToken": "burst-probe-token",
            "contentHashes": [DUMMY_HASH],
        }
    )
    statuses: list[int] = []
    for _ in range(max(1, count)):
        status, _ = requester(chat_url, "POST", payload, timeout)
        statuses.append(status)
    has_rate_limit = 429 in statuses
    no_server_crash = all(code < 500 or code == 503 for code in statuses)
    return (has_rate_limit and no_server_crash), statuses


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True, help="Worker /chat URL")
    parser.add_argument(
        "--mode",
        choices=("security", "burst", "all"),
        default="all",
        help="Verification mode",
    )
    parser.add_argument("--count", type=int, default=12, help="Request count for burst probe")
    parser.add_argument("--timeout", type=float, default=8.0, help="Per-request timeout seconds")
    args = parser.parse_args(argv)

    failed = False
    if args.mode in ("security", "all"):
        for res in run_security_probes(args.endpoint, timeout=args.timeout):
            mark = "PASS" if res.passed else "FAIL"
            print(f"[{mark}] {res.name}: {res.detail}")
            if not res.passed:
                failed = True

    if args.mode in ("burst", "all"):
        ok, statuses = run_burst_probe(args.endpoint, count=args.count, timeout=args.timeout)
        mark = "PASS" if ok else "FAIL"
        print(f"[{mark}] burst-rate-limit: statuses={statuses}")
        if not ok:
            failed = True

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
