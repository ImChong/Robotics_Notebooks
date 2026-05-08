#!/usr/bin/env python3
"""Run the local preflight sequence used to avoid GitHub Actions drift.

The repository has several derived outputs: page catalog, JSON exports, search
index, graph exports, home stats, README badges, and docs hero stats. Running
only part of the chain is the common cause of Actions failures, so this script
keeps the order in one place.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

GENERATED_PATHS = [
    "index.md",
    "README.md",
    "docs/index.html",
    "docs/sitemap.xml",
    "docs/search-index.json",
    "docs/exports",
    "exports",
]


def run(cmd: list[str], description: str) -> None:
    print(f"\n==> {description}", flush=True)
    print("$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def check_generated_clean() -> None:
    result = subprocess.run(
        ["git", "diff", "--name-only", "HEAD", "--", *GENERATED_PATHS],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    changed = [line for line in result.stdout.splitlines() if line.strip()]
    if not changed:
        print("\n==> Generated outputs are already committed", flush=True)
        return

    print("\nGenerated outputs changed after preflight. Commit these files:", flush=True)
    for path in changed:
        print(f"- {path}", flush=True)
    sys.exit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regenerate exports/stats and run the same quality gates as CI."
    )
    parser.add_argument(
        "--check-generated-clean",
        action="store_true",
        help="fail if generated outputs differ from the working tree after regeneration",
    )
    parser.add_argument(
        "--skip-quality",
        action="store_true",
        help="only regenerate derived outputs; skip lint/search/export quality checks",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    run(["python3", "scripts/generate_page_catalog.py"], "Update page catalog")
    run(["python3", "scripts/export_minimal.py"], "Export wiki JSON, sitemap, and search index")
    run(
        ["python3", "scripts/sync_all_stats.py"], "Update graph, home stats, badges, and docs stats"
    )

    if not args.skip_quality:
        run(["python3", "scripts/eval_search_quality.py"], "Run search regression")
        run(["python3", "scripts/lint_wiki.py", "--report"], "Run wiki lint")
        run(["python3", "scripts/check_export_quality.py"], "Check export consistency")

    if args.check_generated_clean:
        check_generated_clean()

    print("\nPreflight complete.", flush=True)


if __name__ == "__main__":
    main()
