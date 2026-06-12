#!/usr/bin/env python3
"""Run the local preflight sequence used to avoid GitHub Actions drift.

The repository has several derived outputs: page catalog, JSON exports, search
index, graph exports, home stats, README badges, and docs hero stats. Running
only part of the chain is the common cause of Actions failures, so this script
keeps the order in one place.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# 仅含仍入库的派生文件；大体积站点 JSON 与 sitemap 已 gitignore，
# 由 pages.yml 部署时生成（docs/exports、exports 目录内被 ignore 的文件不会出现在 git diff）
GENERATED_PATHS = [
    "index.md",
    "README.md",
    "docs/index.html",
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
    if (REPO_ROOT / "package-lock.json").is_file():
        run(["npm", "ci"], "Install Node dependencies (ESLint)")

    run(["python3", "scripts/generate_link_graph.py"], "Update link graph and graph-stats")

    lint_results = None
    coverage_json: Path | None = None
    if not args.skip_quality:
        # 单次 lint：home-stats 与 lint-report 共用结果，避免 sync_all_stats 内二次全库扫描。
        import lint_wiki

        print("\n==> Run wiki lint (once, shared with home-stats)", flush=True)
        lint_results = lint_wiki.lint()
        coverage = lint_wiki.coverage_stats(lint_results)
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".json",
            delete=False,
            encoding="utf-8",
        ) as tmp:
            json.dump(coverage, tmp)
            coverage_json = Path(tmp.name)

    if coverage_json is not None:
        run(
            [
                "python3",
                "scripts/generate_home_stats.py",
                "--coverage-json",
                str(coverage_json),
            ],
            "Update home-stats from graph + lint coverage",
        )
        coverage_json.unlink(missing_ok=True)
    else:
        run(["python3", "scripts/generate_home_stats.py"], "Update home-stats")

    run(
        ["python3", "scripts/sync_all_stats.py", "--skip-graph", "--skip-home-stats"],
        "Sync graph exports, README badges, and docs hero stats",
    )

    if not args.skip_quality:
        import lint_wiki

        assert lint_results is not None
        run(["python3", "scripts/eval_search_quality.py"], "Run search regression")
        print(lint_wiki.format_report(lint_results))
        failing = lint_wiki._failing_total(lint_results)
        info = lint_wiki._info_total(lint_results)
        if failing == 0:
            if info:
                print(f"✅ 所有检查通过！（另含 {info} 条信息型预警，不阻塞 CI）")
            else:
                print("✅ 所有检查通过！")
        else:
            print(f"⚠️  共发现 {failing} 个问题，请参考上方报告处理。")
        report_path = lint_wiki.save_lint_report(lint_results)
        print(f"\n已将健康报告保存到 {report_path}")
        if failing > 0:
            sys.exit(1)
        run(["python3", "scripts/check_export_quality.py"], "Check export consistency")

    if args.check_generated_clean:
        check_generated_clean()

    print("\nPreflight complete.", flush=True)


if __name__ == "__main__":
    main()
