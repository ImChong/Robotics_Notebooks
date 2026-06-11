#!/usr/bin/env python3
"""scaffold_wiki_page.py — 按全库 frontmatter 规范生成 wiki 页面骨架。

把 query 答案 / 选题候选沉淀回 wiki 时，手工拼 frontmatter、速查区块、三段式
正文骨架成本不低。本脚本给定 type 与标题，输出符合 lint 结构要求的骨架：
含「英文缩写速查」锚点（落在规范位置）、`related`/`sources` 占位与三段式正文。

用法：
    python3 scripts/scaffold_wiki_page.py concept "视觉伺服" --slug visual-servoing-intro
    python3 scripts/scaffold_wiki_page.py query "机器人感知选型" --slug perception-pick --dry-run

自带 `--dry-run`（只打印不落盘）与生成后结构自检（复用 lint_wiki 判据）。
"""

from __future__ import annotations

import argparse
import re
import sys
from datetime import date
from pathlib import Path

from lint_wiki import has_section, word_count
from wiki_abbrev_section import is_abbrev_glossary_well_placed

REPO_ROOT = Path(__file__).resolve().parents[1]

# type → wiki 子目录
TYPE_DIRS: dict[str, str] = {
    "concept": "concepts",
    "method": "methods",
    "task": "tasks",
    "comparison": "comparisons",
    "query": "queries",
    "formalization": "formalizations",
    "entity": "entities",
    "overview": "overview",
}


def slugify(title: str, override: str | None) -> str:
    """从标题或 --slug 生成 kebab-case slug；纯中文标题需显式 --slug。"""
    if override:
        return override.strip().lower()
    ascii_only = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    return ascii_only


def _frontmatter(page_type: str, title: str) -> str:
    today = date.today().isoformat()
    return (
        "---\n"
        f"type: {page_type}\n"
        "tags:\n"
        "  - TODO\n"
        "status: draft\n"
        f"updated: {today}\n"
        f'summary: "TODO：一句话概括「{title}」是什么、为什么重要。"\n'
        "related:\n"
        "  - ../TODO/related-page.md\n"
        "sources:\n"
        "  - ../../sources/TODO/source.md\n"
        "---\n"
    )


_ABBREV_BLOCK = (
    "## 英文缩写速查\n\n"
    "| 缩写 | 英文全称 | 简要说明 |\n"
    "|------|----------|----------|\n"
    "| TODO | TODO Full Name | TODO 待补全 |\n"
)

_TAIL_BLOCK = (
    "## 关联页面\n\n"
    "- [TODO 相关页面标题](../TODO/related-page.md)\n\n"
    "## 参考来源\n\n"
    "- [TODO 来源标题](../../sources/TODO/source.md)\n\n"
    "## 推荐继续阅读\n\n"
    "- [TODO 外部资料标题](https://example.com)\n"
)


def _body_standard(title: str) -> str:
    return (
        f"# {title}\n\n"
        "## 一句话定义\n\n"
        f"**{title}**：TODO 一句话核心定义（在机器人语境下它解决什么问题）。\n\n"
        f"{_ABBREV_BLOCK}\n"
        "## 为什么重要\n\n"
        "- **TODO 要点一：** 待补全。\n"
        "- **TODO 要点二：** 待补全。\n\n"
        "## 核心内容\n\n"
        "TODO 三段式骨架：**是什么 → 怎么做 → 取舍与边界**，逐段补全。\n\n"
        "## 常见误区或局限\n\n"
        "- **误区：「TODO」** TODO 纠正。\n"
        "- **局限：** TODO 待补全。\n\n"
        f"{_TAIL_BLOCK}"
    )


def _body_query(title: str) -> str:
    return (
        f"# Query：{title}\n\n"
        "> **Query 产物**：本页由以下问题触发：「TODO 触发问题」\n"
        "> 综合来源：[TODO 页面](../TODO/related-page.md)\n\n"
        f"{_ABBREV_BLOCK}\n"
        "## TL;DR 决策结论\n\n"
        "- TODO 第一刀砍在哪。\n"
        "- TODO 第二刀砍在哪。\n\n"
        "## 详细分析\n\n"
        "TODO 分类对比 / 决策树 / 推荐组合，逐段补全。\n\n"
        "## 关联页面\n\n"
        "- [TODO 相关页面标题](../TODO/related-page.md)\n\n"
        "## 参考来源\n\n"
        "- [TODO 来源标题](../../sources/TODO/source.md)\n"
    )


def build_skeleton(page_type: str, title: str) -> str:
    body = _body_query(title) if page_type == "query" else _body_standard(title)
    return _frontmatter(page_type, title) + "\n" + body


def self_check(content: str, page_type: str) -> list[str]:
    """复用 lint_wiki 判据对骨架做结构自检；返回问题列表（空=通过）。"""
    problems: list[str] = []
    fm = re.match(r"^---\n(.*?)\n---", content, re.DOTALL)
    if not fm:
        problems.append("缺 frontmatter 区块")
    else:
        for key in ("type", "updated", "summary", "related", "sources"):
            if not re.search(rf"^{key}\s*:", fm.group(1), re.MULTILINE):
                problems.append(f"frontmatter 缺 {key}")

    if not has_section(content, ["英文缩写速查"]):
        problems.append("缺「英文缩写速查」区块")
    elif not is_abbrev_glossary_well_placed(content):
        problems.append("「英文缩写速查」位置不规范")
    if not has_section(content, ["关联", "related"]):
        problems.append("缺「关联页面」区块")
    if not has_section(content, ["参考来源", "sources"]):
        problems.append("缺「参考来源」区块")

    if page_type == "query":
        for needle, label in (
            ("**Query 产物**", "Query 产物说明"),
            ("## 参考来源", "## 参考来源"),
            ("## 关联页面", "## 关联页面"),
        ):
            if needle not in content:
                problems.append(f"query 页缺 {label}")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="生成符合 lint 规范的 wiki 页面骨架")
    parser.add_argument("type", choices=sorted(TYPE_DIRS), help="页面类型")
    parser.add_argument("title", help="页面标题（h1 与 summary 中使用）")
    parser.add_argument("--slug", help="文件名 slug（纯中文标题必填）")
    parser.add_argument("--dry-run", action="store_true", help="只打印骨架，不落盘")
    parser.add_argument("--force", action="store_true", help="允许覆盖已存在文件")
    args = parser.parse_args(argv)

    slug = slugify(args.title, args.slug)
    if not slug:
        print("✗ 无法从标题推断 slug（纯中文？请用 --slug 指定）", file=sys.stderr)
        return 2

    content = build_skeleton(args.type, args.title)

    problems = self_check(content, args.type)
    wc = word_count(content)

    target = REPO_ROOT / "wiki" / TYPE_DIRS[args.type] / f"{slug}.md"

    if args.dry_run:
        print(content)
        print(f"--- 自检：{'通过' if not problems else '发现问题'} ---", file=sys.stderr)
    if problems:
        for p in problems:
            print(f"✗ 结构自检：{p}", file=sys.stderr)
        return 1

    print(
        f"✓ 结构自检通过（骨架 {wc} 字，待补全正文以超过 200 字下限）",
        file=sys.stderr,
    )

    if args.dry_run:
        print(f"（--dry-run：未写入 {target.relative_to(REPO_ROOT)}）", file=sys.stderr)
        return 0

    if target.exists() and not args.force:
        print(f"✗ 已存在 {target.relative_to(REPO_ROOT)}（--force 可覆盖）", file=sys.stderr)
        return 1

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    print(f"✓ 已生成 {target.relative_to(REPO_ROOT)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
