#!/usr/bin/env python3
"""将 sources/papers 中链接到的 wiki 页 frontmatter `updated:` 同步为今日。

ingest 在更新 source 摘录并交叉引用多个 wiki 后，先运行本脚本再 commit +
`make ci-preflight`，可避免 lint stale 多轮 retry。

用法：
  python3 scripts/bump_wiki_updated_for_sources.py
  python3 scripts/bump_wiki_updated_for_sources.py sources/papers/rma_arxiv_2107_04034.md
"""

from __future__ import annotations

import argparse
import re
import sys
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SOURCES_PAPERS = REPO_ROOT / "sources" / "papers"
WIKI_LINK_RE = re.compile(r"\]\(([^)]*wiki/[^)]+\.md)\)")


def _resolve_wiki_target(src_file: Path, href: str) -> Path | None:
    target = (src_file.parent / href.split("#")[0]).resolve()
    if target.is_file() and "wiki" in target.parts:
        return target
    return None


def collect_wiki_targets(source_files: list[Path]) -> list[Path]:
    seen: set[Path] = set()
    ordered: list[Path] = []
    for src_file in source_files:
        content = src_file.read_text(encoding="utf-8")
        for m in WIKI_LINK_RE.finditer(content):
            wiki = _resolve_wiki_target(src_file, m.group(1))
            if wiki and wiki not in seen:
                seen.add(wiki)
                ordered.append(wiki)
    return ordered


def bump_updated(path: Path, today: str) -> bool:
    content = path.read_text(encoding="utf-8")
    fm_match = re.match(r"^---\n(.*?)\n---", content, re.DOTALL)
    if not fm_match:
        return False

    fm_block = fm_match.group(1)
    if re.search(r"^updated:\s*", fm_block, re.MULTILINE):
        new_fm, n = re.subn(
            r"^updated:\s*\d{4}-\d{2}-\d{2}\s*$",
            f"updated: {today}",
            fm_block,
            count=1,
            flags=re.MULTILINE,
        )
        if n == 0:
            return False
        if new_fm == fm_block:
            return False
        fm_block = new_fm
    else:
        fm_block = fm_block.rstrip() + f"\nupdated: {today}\n"

    new_content = f"---\n{fm_block}\n---" + content[fm_match.end() :]
    if new_content == content:
        return False
    path.write_text(new_content, encoding="utf-8")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Bump wiki updated: from sources/papers links")
    parser.add_argument(
        "sources",
        nargs="*",
        help="指定 sources/papers 文件；默认扫描全部 *.md",
    )
    args = parser.parse_args()

    if args.sources:
        source_files = []
        for raw in args.sources:
            p = Path(raw)
            if not p.is_absolute():
                p = REPO_ROOT / p
            if not p.is_file():
                print(f"跳过不存在的源文件: {raw}", file=sys.stderr)
                continue
            source_files.append(p)
    else:
        source_files = sorted(SOURCES_PAPERS.glob("*.md"))

    if not source_files:
        print("没有可处理的 sources/papers 文件", file=sys.stderr)
        sys.exit(1)

    today = date.today().isoformat()
    targets = collect_wiki_targets(source_files)
    changed: list[str] = []
    for wiki in targets:
        if bump_updated(wiki, today):
            changed.append(str(wiki.relative_to(REPO_ROOT)))

    if changed:
        print(f"已 bump updated: {today} → {len(changed)} 个 wiki 页")
        for rel in changed:
            print(f"  - {rel}")
    else:
        print("无需更新（链接到的 wiki 页 updated 已为今日或无可写 frontmatter）")


if __name__ == "__main__":
    main()
