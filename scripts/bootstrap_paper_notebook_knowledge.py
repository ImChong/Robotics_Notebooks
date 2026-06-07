#!/usr/bin/env python3
"""Bootstrap Paper Notebooks category tree + ingest unmapped papers into wiki."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import yaml

# Reuse matching helpers from link sync
from sync_paper_notebook_links import (  # noqa: E402
    SCHEMA_DIR,
    auto_match,
    build_paper_index,
    collect_wiki_index,
    load_manual_map,
    short_label,
)

ROOT = Path(__file__).resolve().parents[1]
WIKI = ROOT / "wiki"
SOURCES = ROOT / "sources" / "papers"
CATEGORIES_PATH = SCHEMA_DIR / "paper-notebook-categories.json"
FULL_MAP_PATH = SCHEMA_DIR / "paper-notebook-wiki-full-map.yml"
INDEX_OVERVIEW = WIKI / "overview" / "humanoid-paper-notebooks-index.md"
NOTEBOOK_SITE = "https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks"
PAPERS_JSON_URL = (
    "https://raw.githubusercontent.com/ImChong/"
    "Humanoid_Robot_Learning_Paper_Notebooks/main/_data/papers.json"
)

GENERIC_ABBREV = """| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略 |
| WBC | Whole-Body Control | 协调全身关节满足多任务/约束的控制基础设施 |
| Sim2Real | Simulation to Real | 把仿真中学到的策略迁移落地真机的工程主线 |"""


def fetch_papers_json() -> dict:
    import urllib.request

    with urllib.request.urlopen(PAPERS_JSON_URL, timeout=60) as resp:
        return json.load(resp)


def slugify(text: str, max_len: int = 56) -> str:
    text = text.split("__")[0]
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:max_len].rstrip("-")


def category_wiki_slug(cat_id: str) -> str:
    num = cat_id.split("_", 1)[0]
    tail = cat_id.split("_", 1)[1] if "_" in cat_id else cat_id
    return f"paper-notebook-category-{num}-{slugify(tail, 40)}"


def paper_entity_slug(dir_name: str) -> str:
    return f"paper-notebook-{slugify(dir_name, 48)}"


def source_filename(dir_name: str) -> str:
    return f"humanoid_pnb_{slugify(dir_name, 48)}.md"


def valid_papers(papers: list[dict]) -> list[dict]:
    return [
        p
        for p in papers
        if p.get("category") and p["dir"] != "papers" and "/" in p.get("folder", "")
    ]


def paper_meta_by_dir(papers_json: dict) -> dict[str, dict]:
    meta: dict[str, dict] = {}
    for cat_id, section in papers_json.items():
        for paper in section.get("papers", []):
            paper = dict(paper)
            paper["_category"] = cat_id
            meta[paper["dir"]] = paper
        for sub in section.get("subcategories") or []:
            sub_zh = sub.get("zhname", "")
            for paper in sub.get("papers", []):
                paper = dict(paper)
                paper["_category"] = cat_id
                paper["_subcategory_zh"] = sub_zh
                meta[paper["dir"]] = paper
    return meta


def resolve_primary_wiki(
    paper: dict,
    manual: dict[str, list[str]],
    wiki_index: dict,
    planned: dict[str, str],
) -> str:
    key = paper["dir"]
    if key in manual:
        return manual[key][0]
    if key in planned:
        return planned[key]
    auto = auto_match(paper, wiki_index)
    if auto:
        return auto[0]
    rel = f"wiki/entities/{paper_entity_slug(key)}.md"
    planned[key] = rel
    return rel


def render_source(paper: dict, meta: dict, wiki_rel: str) -> str:
    arxiv = paper.get("arxiv") or meta.get("arxiv")
    arxiv_line = f"- **arXiv：** <https://arxiv.org/abs/{arxiv}>\n" if arxiv else ""
    sub = meta.get("_subcategory_zh") or ""
    sub_line = f"- **子分类：** {sub}\n" if sub else ""
    return f"""# {paper["title"]}

> 来源归档（ingest · Humanoid Paper Notebooks 深读笔记）

- **标题：** {paper["title"]}
- **类型：** paper
- **笔记链接：** <{paper["url"]}>
- **分类：** {paper.get("category", meta.get("_category", ""))}
{sub_line}{arxiv_line}- **入库日期：** 2026-06-07
- **一句话说明：** 来自 [Humanoid Robot Learning Paper Notebooks]({NOTEBOOK_SITE}/index.html) 的深读笔记索引；正文以笔记站与 arXiv 为准。

## 核心摘录（策展，非全文）

- 本文件为 **Paper Notebooks → 本库 wiki** 的溯源锚点；方法细节请读笔记页与论文 PDF。
- 知识归纳见 wiki 实体页：[{Path(wiki_rel).stem}](../../{wiki_rel}).

## 对 wiki 的映射

- [{Path(wiki_rel).stem}](../../{wiki_rel})
- 分类父节点：[{category_wiki_slug(paper.get("category") or meta.get("_category", ""))}](../../wiki/overview/{category_wiki_slug(paper.get("category") or meta.get("_category", ""))}.md)

## 参考来源（原始）

- 深读笔记：<{paper["url"]}>
{f"- 论文：<https://arxiv.org/abs/{arxiv}>" if arxiv else ""}
"""


def render_entity_stub(paper: dict, meta: dict, wiki_rel: str, category_rel: str) -> str:
    src_rel = f"../../sources/papers/{source_filename(paper['dir'])}"
    arxiv = paper.get("arxiv") or meta.get("arxiv")
    fm_arxiv = f'arxiv: "{arxiv}"\n' if arxiv else ""
    title_short = short_label(paper["title"])
    return f"""---
type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-stub]
status: stub
updated: 2026-06-07
{fm_arxiv}related:
  - ../overview/{Path(category_rel).stem}.md
  - ../overview/humanoid-paper-notebooks-index.md
sources:
  - {src_rel}
summary: "{title_short}：Humanoid Paper Notebooks 深读笔记索引实体；待从笔记与论文 PDF 深化归纳。"
---

# {title_short}

**{paper["title"]}** 收录于 [Humanoid Robot Learning Paper Notebooks]({NOTEBOOK_SITE}/index.html)（分类：{paper.get("category", meta.get("_category", ""))}）。本页为 **索引级实体**，链向深读笔记与原始论文；详细机制待从笔记消化后补充。

## 一句话定义

{title_short} 的人形机器人学习论文条目，以 Paper Notebooks 深读笔记为首要编译来源。

## 英文缩写速查

{GENERIC_ABBREV}

## 为什么重要

- 列入 Paper Notebooks 策展清单，便于与全库 [人形论文笔记总索引](../overview/humanoid-paper-notebooks-index.md) 及分类父节点交叉检索。
- 深读笔记提供比摘要更贴近实现的阅读路径，适合作为后续 ingest 深化起点。

## 核心信息

| 字段 | 内容 |
|------|------|
| 分类 | {paper.get("category", meta.get("_category", ""))} |
| 深读笔记 | <{paper["url"]}> |
{f"| arXiv | <https://arxiv.org/abs/{arxiv}> |" if arxiv else ""}

## 实验与评测

- 本页为 **策展索引级** 摘要；量化 benchmark、消融与实机指标以 **深读笔记与论文 PDF** 为准（链接见 [参考来源](#参考来源)）。

## 与其他页面的关系

- 分类父节点：[{Path(category_rel).stem}](../overview/{Path(category_rel).stem}.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)

## 参考来源

- [{source_filename(paper["dir"])}]({src_rel})
- 深读笔记：<{paper["url"]}>
{f"- 论文：<https://arxiv.org/abs/{arxiv}>" if arxiv else ""}

## 推荐继续阅读

- [机器人论文阅读笔记：{title_short}]({paper["url"]})
"""


def render_category_page(
    cat_id: str,
    section: dict,
    papers_in_cat: list[tuple[dict, str, dict]],
    subcategories: list[dict] | None,
) -> str:
    display = section.get("display_name") or cat_id
    zh = section.get("zhname") or display
    subtitle = section.get("subtitle") or section.get("subtitle_zh") or ""
    num = cat_id.split("_", 1)[0]

    lines = [
        "---",
        "type: overview",
        "tags: [humanoid-paper-notebooks, paper-index, overview]",
        "status: complete",
        "updated: 2026-06-07",
        "related:",
        "  - ./humanoid-paper-notebooks-index.md",
        f'summary: "Paper Notebooks 分类 {num}：{zh}（{len(papers_in_cat)} 篇深读笔记索引）。"',
        "---",
        "",
        f"# Paper Notebooks · {display}",
        "",
        f"**{display}**（`{cat_id}`）是 [Humanoid Robot Learning Paper Notebooks]({NOTEBOOK_SITE}/index.html) 主页面的第 **{num}** 类。{subtitle}",
        "",
        "## 英文缩写速查",
        "",
        GENERIC_ABBREV,
        "",
        "## 本类论文索引",
        "",
    ]

    if subcategories:
        by_sub: dict[str, list[tuple[dict, str, dict]]] = {}
        no_sub: list[tuple[dict, str, dict]] = []
        for paper, wiki_rel, meta in papers_in_cat:
            sub_zh = meta.get("_subcategory_zh")
            if sub_zh:
                by_sub.setdefault(sub_zh, []).append((paper, wiki_rel, meta))
            else:
                no_sub.append((paper, wiki_rel, meta))
        for sub in subcategories:
            sub_zh = sub.get("zhname") or sub.get("display_name") or "其他"
            items = by_sub.get(sub_zh, [])
            if not items:
                continue
            lines.append(f"### {sub_zh}")
            lines.append("")
            for paper, wiki_rel, _ in sorted(items, key=lambda x: x[0]["title"]):
                label = short_label(paper["title"])
                wiki_path = wiki_rel.replace("wiki/", "../")
                lines.append(f"- [{label}]({wiki_path}) — [深读笔记]({paper['url']})")
            lines.append("")
        if no_sub:
            lines.append("### 其他")
            lines.append("")
            for paper, wiki_rel, _ in sorted(no_sub, key=lambda x: x[0]["title"]):
                label = short_label(paper["title"])
                wiki_path = wiki_rel.replace("wiki/", "../")
                lines.append(f"- [{label}]({wiki_path}) — [深读笔记]({paper['url']})")
            lines.append("")
    else:
        for paper, wiki_rel, _ in sorted(papers_in_cat, key=lambda x: x[0]["title"]):
            label = short_label(paper["title"])
            wiki_path = wiki_rel.replace("wiki/", "../")
            lines.append(f"- [{label}]({wiki_path}) — [深读笔记]({paper['url']})")
        lines.append("")

    lines.extend(
        [
            "## 与其他页面的关系",
            "",
            "- 总索引：[humanoid-paper-notebooks-index.md](./humanoid-paper-notebooks-index.md)",
            f"- 笔记主页：<{NOTEBOOK_SITE}/index.html>",
            "",
            "## 参考来源",
            "",
            f"- [Humanoid Robot Learning Paper Notebooks]({NOTEBOOK_SITE}/index.html)",
            "- [schema/paper-notebook-index.json](../../schema/paper-notebook-index.json)",
            "",
            "## 推荐继续阅读",
            "",
            f"- [机器人论文阅读笔记总站]({NOTEBOOK_SITE}/index.html)",
        ]
    )
    return "\n".join(lines) + "\n"


def render_root_index(categories: list[tuple[str, dict, int]]) -> str:
    lines = [
        "---",
        "type: overview",
        "tags: [humanoid-paper-notebooks, paper-index, overview]",
        "status: complete",
        "updated: 2026-06-07",
        "related:",
    ]
    for cat_id, _, _ in categories:
        lines.append(f"  - ./{category_wiki_slug(cat_id)}.md")
    total = sum(n for _, _, n in categories)
    lines.extend(
        [
            f'summary: "Humanoid Paper Notebooks 137+ 篇深读笔记在本库的分类父节点与 wiki 子节点总索引（共 {total} 篇）。"',
            "---",
            "",
            "# Humanoid Paper Notebooks 知识库索引",
            "",
            f"本页把 [Humanoid Robot Learning Paper Notebooks]({NOTEBOOK_SITE}/index.html) 的 **14 类主页分类** 映射为本仓库 `wiki/overview/paper-notebook-category-*` **父节点**；每篇论文对应 **子节点**（已有深度 wiki 或 `wiki/entities/paper-notebook-*` 索引实体）。",
            "",
            "## 英文缩写速查",
            "",
            GENERIC_ABBREV,
            "",
            "## 分类父节点（与笔记主页面一致）",
            "",
        ]
    )
    for cat_id, section, count in categories:
        slug = category_wiki_slug(cat_id)
        display = section.get("display_name") or cat_id
        zh = section.get("zhname") or display
        lines.append(f"- [{display}（{zh}）](./{slug}.md) — `{cat_id}`，{count} 篇")
    lines.extend(
        [
            "",
            "## 维护说明",
            "",
            "- 笔记 URL 与分类元数据：`schema/paper-notebook-index.json`、`schema/paper-notebook-categories.json`",
            "- 论文 → wiki 完整映射：`schema/paper-notebook-wiki-full-map.yml`",
            "- 向已有 wiki 页注入深读链接：`make paper-notebook-links`",
            "- 补齐未映射论文的 sources/实体与分类树：`make paper-notebook-bootstrap`",
            "",
            "## 参考来源",
            "",
            f"- [Humanoid Robot Learning Paper Notebooks]({NOTEBOOK_SITE}/index.html)",
            "- [sources/sites/rl-sim2sim-demo-website.md](../../sources/sites/rl-sim2sim-demo-website.md)（姊妹演示站，非本索引范围）",
            "",
            "## 推荐继续阅读",
            "",
            f"- [机器人论文阅读笔记总站]({NOTEBOOK_SITE}/index.html)",
            "- [人形 RL 身体系统栈](./humanoid-rl-motion-control-body-system-stack.md)",
            "- [BFM 41 篇技术地图](./bfm-41-papers-technology-map.md)",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    papers_json = fetch_papers_json()
    papers = valid_papers(build_paper_index())
    meta_by_dir = paper_meta_by_dir(papers_json)
    manual = load_manual_map()
    wiki_index = collect_wiki_index()
    planned: dict[str, str] = {}
    full_map: dict[str, list[str]] = {}

    created_src = created_wiki = updated_wiki = 0

    for paper in papers:
        key = paper["dir"]
        meta = meta_by_dir.get(key, {})
        primary = resolve_primary_wiki(paper, manual, wiki_index, planned)
        targets = manual.get(key) or [primary]
        full_map[key] = targets

        src_path = SOURCES / source_filename(key)
        wiki_path = ROOT / primary
        category_rel = f"wiki/overview/{category_wiki_slug(paper['category'])}.md"

        if primary.startswith("wiki/entities/paper-notebook-"):
            if not src_path.exists():
                content = render_source(paper, meta, primary)
                if not args.dry_run:
                    src_path.write_text(content, encoding="utf-8")
                created_src += 1
            if not wiki_path.exists():
                content = render_entity_stub(paper, meta, primary, category_rel)
                if not args.dry_run:
                    wiki_path.write_text(content, encoding="utf-8")
                created_wiki += 1
            elif wiki_path.exists() and not args.dry_run:
                text = wiki_path.read_text(encoding="utf-8")
                cat_link = f"../overview/{Path(category_rel).stem}.md"
                if cat_link not in text and "humanoid-paper-notebooks-index" not in text:
                    # minimal patch related block
                    if "related:" in text:
                        text = text.replace(
                            "related:\n",
                            f"related:\n  - {cat_link}\n  - ../overview/humanoid-paper-notebooks-index.md\n",
                            1,
                        )
                        wiki_path.write_text(text, encoding="utf-8")
                        updated_wiki += 1

    # Category pages
    cat_entries: list[tuple[str, dict, int]] = []
    for cat_id in sorted(papers_json.keys()):
        section = papers_json[cat_id]
        papers_in_cat: list[tuple[dict, str, dict]] = []
        for paper in papers:
            if paper.get("category") != cat_id:
                continue
            wiki_rel = full_map[paper["dir"]][0]
            papers_in_cat.append((paper, wiki_rel, meta_by_dir.get(paper["dir"], {})))
        if not papers_in_cat:
            continue
        subs = section.get("subcategories")
        cat_path = WIKI / "overview" / f"{category_wiki_slug(cat_id)}.md"
        content = render_category_page(cat_id, section, papers_in_cat, subs)
        if not args.dry_run:
            cat_path.write_text(content, encoding="utf-8")
        cat_entries.append((cat_id, section, len(papers_in_cat)))

    root_content = render_root_index(cat_entries)
    if not args.dry_run:
        INDEX_OVERVIEW.write_text(root_content, encoding="utf-8")
        CATEGORIES_PATH.write_text(
            json.dumps(
                {
                    cat_id: {
                        "display_name": sec.get("display_name"),
                        "zhname": sec.get("zhname"),
                        "wiki": f"wiki/overview/{category_wiki_slug(cat_id)}.md",
                        "count": count,
                    }
                    for cat_id, sec, count in cat_entries
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        FULL_MAP_PATH.write_text(
            yaml.safe_dump({"overrides": full_map}, allow_unicode=True, sort_keys=True),
            encoding="utf-8",
        )

    print(
        f"{'would create' if args.dry_run else 'created'} "
        f"{created_src} sources, {created_wiki} wiki entities; "
        f"updated {updated_wiki}; "
        f"{len(cat_entries)} category pages + root index; "
        f"full map {len(full_map)} / {len(papers)} papers"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
