#!/usr/bin/env python3
"""Classify PR changes for fast, fail-closed ingest CI checks (issue #3507)."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONTENT_ROOTS = ("wiki/", "sources/", "log.d/")


def parse_name_status(data: bytes) -> list[tuple[str, tuple[str, ...]]]:
    """Parse git diff --name-status -z, including rename/copy old and new paths."""
    fields = data.decode("utf-8", "surrogateescape").split("\0")
    if fields[-1] != "":
        raise ValueError("git diff output is not NUL-terminated")
    fields.pop()
    entries = []
    index = 0
    while index < len(fields):
        status = fields[index]
        index += 1
        count = 2 if status.startswith(("R", "C")) else 1
        if not status or status[0] not in "ACDMRTUXB" or index + count > len(fields):
            raise ValueError(f"invalid git diff status: {status!r}")
        paths = tuple(fields[index : index + count])
        if not all(paths):
            raise ValueError("empty git diff path")
        entries.append((status, paths))
        index += count
    return entries


def classify(entries: list[tuple[str, tuple[str, ...]]]) -> dict:
    changed = sorted({path for _, paths in entries for path in paths})
    content_only = bool(changed) and all(
        path.endswith(".md") and path.startswith(CONTENT_ROOTS) for path in changed
    )
    # Deletions and renames need special handling by incremental lint; no silent omission.
    removed = sorted({paths[0] for status, paths in entries if status.startswith(("D", "R"))})
    mode = "ingest" if content_only else "full"
    return {
        "mode": mode,
        "reason": (
            "Only wiki/sources/log.d Markdown changed"
            if content_only
            else "Code, configuration, non-Markdown or empty changes: full CI"
        ),
        "paths": changed,
        "changed_wiki": [p for p in changed if p.startswith("wiki/")],
        "changed_sources": [p for p in changed if p.startswith("sources/")],
        "removed": removed,
        "base": "",
    }


def detect(base: str) -> dict:
    try:
        result = subprocess.run(
            ["git", "diff", "--name-status", "-z", "--find-renames", f"{base}...HEAD"],
            cwd=ROOT,
            capture_output=True,
            check=True,
        )
        scope = classify(parse_name_status(result.stdout))
        scope["base"] = base
        return scope
    except (subprocess.CalledProcessError, ValueError) as exc:
        print(f"Change detection failed; using full CI: {exc}", file=sys.stderr)
        return full_scope("Cannot safely classify git changes")


def full_scope(reason: str = "Push to main or manual dispatch") -> dict:
    return {
        "mode": "full",
        "reason": reason,
        "paths": [],
        "changed_wiki": [],
        "changed_sources": [],
        "removed": [],
        "base": "",
    }


def validate_mermaid(scope: dict) -> None:
    if scope["mode"] != "ingest":
        raise ValueError("incremental Mermaid validation requires ingest mode")
    files = [
        path for path in scope["changed_wiki"] if path.endswith(".md") and (ROOT / path).is_file()
    ]
    if not files:
        print("No changed published Markdown: Mermaid check skipped")
        return
    print(f"Checking Mermaid in {len(files)} changed wiki pages", flush=True)
    subprocess.run(["node", "scripts/check_mermaid.cjs", *files], cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--base", help="PR base ref, e.g. origin/main")
    mode.add_argument("--full", action="store_true", help="Require full CI")
    mode.add_argument("--validate-mermaid-from", type=Path, metavar="SCOPE_JSON")
    parser.add_argument("--output-file", type=Path)
    parser.add_argument("--github-output", type=Path)
    args = parser.parse_args()
    if args.validate_mermaid_from:
        validate_mermaid(json.loads(args.validate_mermaid_from.read_text(encoding="utf-8")))
        return
    scope = detect(args.base) if args.base else full_scope()
    if args.output_file:
        args.output_file.parent.mkdir(parents=True, exist_ok=True)
        args.output_file.write_text(json.dumps(scope, ensure_ascii=False), encoding="utf-8")
    if args.github_output:
        with args.github_output.open("a", encoding="utf-8") as output:
            output.write(f"mode={scope['mode']}\n")
            output.write(f"has_wiki={str(bool(scope['changed_wiki'])).lower()}\n")
    print(f"CI scope: {scope['mode']} — {scope['reason']}")
    print(f"Changed paths: {len(scope['paths'])}")
    if os.getenv("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as summary:
            summary.write(
                f"\n### CI change scope\nMode: **{scope['mode']}** — {scope['reason']}. "
                f"Changed paths: {len(scope['paths'])}.\n"
            )


if __name__ == "__main__":
    main()
