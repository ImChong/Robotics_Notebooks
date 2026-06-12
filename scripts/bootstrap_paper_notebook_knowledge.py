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
    norm_title,
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
PROGRESS_JSON_URL = (
    "https://raw.githubusercontent.com/ImChong/"
    "Humanoid_Robot_Learning_Paper_Notebooks/main/progress.json"
)
PROGRESS_MD_URL = (
    "https://raw.githubusercontent.com/ImChong/"
    "Humanoid_Robot_Learning_Paper_Notebooks/main/papers/PROGRESS.md"
)
ARXIV_ID_RE = re.compile(r"^\d{4}\.\d{4,5}(v\d+)?$")
PROGRESS_SECTION_CATEGORY: list[tuple[str, str]] = [
    ("基础路线图", "01_Foundational_RL"),
    ("Whole-Body Control", "03_High_Impact_Selection"),
    ("遥操作与模仿学习", "03_High_Impact_Selection"),
    ("Locomotion 经典", "03_High_Impact_Selection"),
    ("Sim-to-Real & Foundation Model", "03_High_Impact_Selection"),
    ("仿真平台 & 工具", "03_High_Impact_Selection"),
    ("Loco-Manipulation and Whole-Body-Control", "04_Loco-Manipulation_and_WBC"),
    ("Locomotion（", "05_Locomotion"),
    ("Manipulation（", "06_Manipulation"),
    ("Teleoperation（", "07_Teleoperation"),
    ("Navigation（", "08_Navigation"),
    ("State Estimation（", "09_State_Estimation"),
    ("Sim-to-Real（", "10_Sim-to-Real"),
    ("Simulation Benchmark（", "11_Simulation_Benchmark"),
    ("Hardware Design（", "12_Hardware_Design"),
    ("Physics-Based Character Animation（", "13_Physics-Based_Animation"),
    ("Human Motion Analysis and Synthesis（", "14_Human_Motion"),
]

GENERIC_ABBREV = """| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略 |
| WBC | Whole-Body Control | 协调全身关节满足多任务/约束的控制基础设施 |
| Sim2Real | Simulation to Real | 把仿真中学到的策略迁移落地真机的工程主线 |"""


def fetch_papers_json() -> dict:
    import urllib.request

    with urllib.request.urlopen(PAPERS_JSON_URL, timeout=60) as resp:
        return json.load(resp)


def fetch_progress_json() -> dict:
    import urllib.request

    with urllib.request.urlopen(PROGRESS_JSON_URL, timeout=60) as resp:
        return json.load(resp)


def normalize_arxiv(value: str | None) -> str | None:
    if not value:
        return None
    cleaned = value.strip()
    return cleaned if ARXIV_ID_RE.match(cleaned) else None


def clean_progress_title(title: str) -> str:
    title = title.strip()
    link = re.match(r"^\[([^\]]+)\]\([^)]+\)$", title)
    if link:
        title = link.group(1).strip()
    if title.lower().startswith("[website],"):
        return title.split(",", 1)[1].strip()
    return title


def progress_entry_to_paper(entry: dict) -> dict:
    folder = entry["folder"]
    dir_name = Path(folder).name
    note_file = entry.get("note_file") or f"{dir_name}.md"
    html_name = note_file.rsplit(".", 1)[0] + ".html"
    parts = folder.split("/")
    category = parts[1] if len(parts) > 1 else ""
    return {
        "folder": folder,
        "dir": dir_name,
        "title": clean_progress_title(entry["title"]),
        "arxiv": normalize_arxiv(entry.get("arxiv")),
        "url": f"{NOTEBOOK_SITE}/{folder}/{html_name}",
        "category": category,
        "planned": True,
        "route": entry.get("route", ""),
    }


def fetch_progress_pending(existing_folders: set[str]) -> list[dict]:
    progress = fetch_progress_json()
    pending: list[dict] = []
    for entry in progress.get("papers", []):
        if entry.get("status") != "pending":
            continue
        folder = entry.get("folder", "")
        if not folder or folder in existing_folders:
            continue
        pending.append(progress_entry_to_paper(entry))
    pending.sort(key=lambda p: p["folder"])
    return pending


def fetch_progress_md() -> str:
    import urllib.request

    with urllib.request.urlopen(PROGRESS_MD_URL, timeout=60) as resp:
        return resp.read().decode("utf-8")


def category_for_progress_section(section: str) -> str:
    for needle, cat_id in PROGRESS_SECTION_CATEGORY:
        if needle in section:
            return cat_id
    return "03_High_Impact_Selection"


def parse_progress_md(text: str) -> list[dict]:
    entries: list[dict] = []
    current_section = ""
    for line in text.splitlines():
        if line.startswith("### "):
            current_section = line[4:].strip()
            continue
        if line.startswith("#### "):
            current_section = line[5:].strip()
            continue
        if not line.startswith("|") or "---" in line:
            continue
        cols = [c.strip() for c in line.strip("|").split("|")]
        if len(cols) < 2:
            continue
        num = cols[0]
        if not re.match(r"^(\d+|H\d+)$", num):
            continue
        paper_col = cols[1]
        link = re.search(r"\[([^\]]+)\]\((https://arxiv\.org/abs/[^)]+)\)", paper_col)
        if link:
            title = clean_progress_title(link.group(1))
            arxiv = normalize_arxiv(link.group(2).rsplit("/", 1)[-1])
        else:
            generic = re.search(r"\[([^\]]+)\]\(([^)]+)\)", paper_col)
            if generic:
                title = clean_progress_title(generic.group(1))
                arxiv = None
            else:
                title = clean_progress_title(re.sub(r"✅.*$", "", paper_col))
                title = title.replace("🌟", "").strip()
                arxiv = None
        note_match = re.search(r"\[笔记\]\(([^)]+)\)", paper_col)
        entries.append(
            {
                "num": num,
                "title": title,
                "arxiv": arxiv,
                "note_path": note_match.group(1) if note_match else None,
                "category": category_for_progress_section(current_section),
                "section": current_section,
            }
        )
    return entries


def progress_md_entry_to_paper(entry: dict) -> dict:
    note_path = entry.get("note_path")
    if note_path:
        parts = note_path.split("/")
        dir_name = parts[-2] if len(parts) >= 2 else slugify(entry["title"], 48)
        folder = f"papers/{'/'.join(parts[:-1])}"
        html_path = note_path.rsplit(".", 1)[0] + ".html"
        url = f"{NOTEBOOK_SITE}/papers/{html_path}"
    else:
        dir_name = slugify(entry["title"], 48)
        folder = f"papers/{entry['category']}/{dir_name}"
        url = f"{NOTEBOOK_SITE}/{folder}/{dir_name}.html"
    return {
        "folder": folder,
        "dir": dir_name,
        "title": entry["title"],
        "arxiv": entry.get("arxiv"),
        "url": url,
        "category": entry["category"],
        "planned": True,
        "from_progress_md": True,
    }


def fetch_progress_md_papers(existing_keys: set[str]) -> list[dict]:
    entries = parse_progress_md(fetch_progress_md())
    papers: list[dict] = []
    for entry in entries:
        paper = progress_md_entry_to_paper(entry)
        key = paper_dedup_key(paper)
        if key in existing_keys:
            continue
        papers.append(paper)
        existing_keys.add(key)
    papers.sort(key=lambda p: (p.get("category", ""), p["title"]))
    return papers


def paper_dedup_key(paper: dict) -> str:
    if paper.get("arxiv"):
        return f"arxiv:{paper['arxiv']}"
    return f"title:{norm_title(paper['title'])}"


def paper_catalog_score(paper: dict) -> int:
    score = 0
    if not paper.get("planned"):
        score += 100
    if paper.get("folder") and "/" in paper["folder"]:
        score += 20
    if not paper.get("from_progress_md"):
        score += 10
    return score


def merge_paper_catalog(*groups: list[dict]) -> list[dict]:
    """Merge papers.json, progress.json, and PROGRESS.md without duplicate index rows.

    Completed deep-read notes win over PROGRESS.md slug aliases in the same category
    (same folder slug or short title), which otherwise produced paired 深读笔记/待深读 lines.
    """
    merged: dict[str, dict] = {}
    for group in groups:
        for paper in group:
            key = paper_dedup_key(paper)
            if key not in merged or paper_catalog_score(paper) > paper_catalog_score(merged[key]):
                merged[key] = paper

    completed = [p for p in merged.values() if not p.get("planned")]
    planned_papers = [p for p in merged.values() if p.get("planned")]
    completed_dir_slugs: set[tuple[str, str]] = set()
    completed_label_slugs: set[tuple[str, str]] = set()
    for paper in completed:
        cat = paper.get("category", "")
        completed_dir_slugs.add((cat, slugify(paper["dir"], 48)))
        completed_label_slugs.add((cat, slugify(short_label(paper["title"]), 48)))

    catalog = list(completed)
    for paper in planned_papers:
        cat = paper.get("category", "")
        dir_slug = slugify(paper["dir"], 48)
        label_slug = slugify(short_label(paper["title"]), 48)
        if (cat, dir_slug) in completed_dir_slugs or (cat, label_slug) in completed_label_slugs:
            continue
        catalog.append(paper)
    return sorted(catalog, key=lambda p: (p.get("category", ""), p["title"]))


def dedupe_category_entries(
    papers_in_cat: list[tuple[dict, str, dict]],
) -> list[tuple[dict, str, dict]]:
    """One list row per wiki target; prefer completed notes over planned stubs."""
    by_wiki: dict[str, tuple[dict, str, dict]] = {}
    for paper, wiki_rel, meta in papers_in_cat:
        existing = by_wiki.get(wiki_rel)
        if existing is None or paper_catalog_score(paper) > paper_catalog_score(existing[0]):
            by_wiki[wiki_rel] = (paper, wiki_rel, meta)
    return list(by_wiki.values())


def category_entry_suffix(paper: dict) -> str:
    if paper.get("planned"):
        return "待深读"
    return f"[深读笔记]({paper['url']})"


def progress_source_label(paper: dict) -> tuple[str, str]:
    if paper.get("from_progress_md"):
        return (
            "PROGRESS.md",
            "https://github.com/ImChong/Humanoid_Robot_Learning_Paper_Notebooks/blob/main/papers/PROGRESS.md",
        )
    return (
        "progress.json",
        "https://github.com/ImChong/Humanoid_Robot_Learning_Paper_Notebooks/blob/main/progress.json",
    )


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
    if paper.get("arxiv"):
        from sync_paper_notebook_links import pick_entity_or_single

        picked = pick_entity_or_single(wiki_index["arxiv"].get(paper["arxiv"], set()))
        if picked and "paper-notebook-" not in picked[0]:
            return picked[0]
    auto = auto_match(paper, wiki_index)
    if auto and "paper-notebook-" not in auto[0]:
        return auto[0]
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
    if paper.get("planned"):
        route = paper.get("route") or ""
        route_line = f"- **路线：** {route}\n" if route else ""
        progress_src = (
            "[papers/PROGRESS.md](https://github.com/ImChong/"
            "Humanoid_Robot_Learning_Paper_Notebooks/blob/main/papers/PROGRESS.md)"
            if paper.get("from_progress_md")
            else "[progress.json](https://github.com/ImChong/"
            "Humanoid_Robot_Learning_Paper_Notebooks/blob/main/progress.json)"
        )
        return f"""# {paper["title"]}

> 来源归档（ingest · Humanoid Paper Notebooks progress 待深读）

- **标题：** {paper["title"]}
- **类型：** paper
- **深读状态：** 待撰写（见 {progress_src}）
- **计划笔记路径：** `{paper["folder"]}/{Path(paper["folder"]).name}.md`
- **分类：** {paper.get("category", meta.get("_category", ""))}
{sub_line}{route_line}{arxiv_line}- **入库日期：** 2026-06-11
- **一句话说明：** 列入 Paper Notebooks 阅读进度，深读笔记尚未完成；本文件为 **进度 → wiki** 溯源锚点。

## 核心摘录（策展，非全文）

- 本文件锚定 **待深读** 论文在姊妹仓库 `progress.json` 中的条目；笔记完成后应改用笔记页链接并深化 wiki 归纳。
- 知识归纳见 wiki 实体页：[{Path(wiki_rel).stem}](../../{wiki_rel}).

## 对 wiki 的映射

- [{Path(wiki_rel).stem}](../../{wiki_rel})
- 分类父节点：[{category_wiki_slug(paper.get("category") or meta.get("_category", ""))}](../../wiki/overview/{category_wiki_slug(paper.get("category") or meta.get("_category", ""))}.md)

## 参考来源（原始）

- [Humanoid Robot Learning Paper Notebooks · {"PROGRESS.md" if paper.get("from_progress_md") else "progress.json"}]({"https://github.com/ImChong/Humanoid_Robot_Learning_Paper_Notebooks/blob/main/papers/PROGRESS.md" if paper.get("from_progress_md") else "https://github.com/ImChong/Humanoid_Robot_Learning_Paper_Notebooks/blob/main/progress.json"})
{f"- 论文：<https://arxiv.org/abs/{arxiv}>" if arxiv else ""}
"""
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
    if paper.get("planned"):
        prog_label, prog_url = progress_source_label(paper)
        return f"""---
type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-planned]
status: planned
updated: 2026-06-11
{fm_arxiv}related:
  - ../overview/{Path(category_rel).stem}.md
  - ../overview/humanoid-paper-notebooks-index.md
sources:
  - {src_rel}
summary: "{title_short}：列入 Paper Notebooks {prog_label} 待深读清单；深读笔记完成后升格为完整索引实体。"
---

# {title_short}

**{paper["title"]}** 已列入 [Humanoid Robot Learning Paper Notebooks]({NOTEBOOK_SITE}/index.html) 的 **{prog_label} 待深读** 清单（分类：{paper.get("category", meta.get("_category", ""))}）。本页为 **计划索引实体**，深读笔记尚未撰写；笔记完成后应链向笔记站并深化归纳。

## 一句话定义

{title_short} 的人形机器人学习论文条目，当前处于 Paper Notebooks 阅读进度（待深读）阶段。

## 英文缩写速查

{GENERIC_ABBREV}

## 为什么重要

- 列入 Paper Notebooks **progress 待深读** 清单，便于与全库 [人形论文笔记总索引](../overview/humanoid-paper-notebooks-index.md) 及分类父节点交叉检索。
- 在深读笔记完成前，本页作为 **占位子节点**，避免知识图谱缺失该论文实体。

## 核心信息

| 字段 | 内容 |
|------|------|
| 分类 | {paper.get("category", meta.get("_category", ""))} |
| 深读状态 | 待撰写（[{prog_label}]({prog_url})） |
| 计划文件夹 | `{paper["folder"]}` |
{f"| arXiv | <https://arxiv.org/abs/{arxiv}> |" if arxiv else ""}

## 实验与评测

- 深读笔记尚未完成；量化 benchmark、消融与实机指标待笔记撰写后补充。

## 与其他页面的关系

- 分类父节点：[{Path(category_rel).stem}](../overview/{Path(category_rel).stem}.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)

## 参考来源

- [{source_filename(paper["dir"])}]({src_rel})
- [Humanoid Robot Learning Paper Notebooks · {prog_label}]({prog_url})
{f"- 论文：<https://arxiv.org/abs/{arxiv}>" if arxiv else ""}

## 推荐继续阅读

- [Paper Notebooks 阅读进度（PROGRESS.md）](https://github.com/ImChong/Humanoid_Robot_Learning_Paper_Notebooks/blob/main/papers/PROGRESS.md)
"""
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
                lines.append(f"- [{label}]({wiki_path}) — {category_entry_suffix(paper)}")
            lines.append("")
        if no_sub:
            lines.append("### 其他")
            lines.append("")
            for paper, wiki_rel, _ in sorted(no_sub, key=lambda x: x[0]["title"]):
                label = short_label(paper["title"])
                wiki_path = wiki_rel.replace("wiki/", "../")
                lines.append(f"- [{label}]({wiki_path}) — {category_entry_suffix(paper)}")
            lines.append("")
    else:
        for paper, wiki_rel, _ in sorted(papers_in_cat, key=lambda x: x[0]["title"]):
            label = short_label(paper["title"])
            wiki_path = wiki_rel.replace("wiki/", "../")
            lines.append(f"- [{label}]({wiki_path}) — {category_entry_suffix(paper)}")
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
            "- 补齐未映射论文的 sources/实体与分类树：`make paper-notebook-bootstrap`（含 progress.json 与 papers/PROGRESS.md）",
            "",
            "## 与其他页面的关系",
            "",
            "- [人形 RL 身体系统栈](./humanoid-rl-motion-control-body-system-stack.md)",
            "- [BFM 41 篇技术地图](./bfm-41-papers-technology-map.md)",
            "- [Ego 9 篇技术地图](./ego-9-papers-technology-map.md)",
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
    note_folders = {p["folder"] for p in papers}
    existing_keys = {paper_dedup_key(p) for p in papers}
    progress_pending = fetch_progress_pending(note_folders)
    for paper in progress_pending:
        existing_keys.add(paper_dedup_key(paper))
    progress_md_papers = fetch_progress_md_papers(existing_keys)
    all_papers = merge_paper_catalog(papers, progress_pending, progress_md_papers)
    meta_by_dir = paper_meta_by_dir(papers_json)
    manual = load_manual_map()
    wiki_index = collect_wiki_index()
    planned: dict[str, str] = {}
    full_map: dict[str, list[str]] = {}

    created_src = created_wiki = updated_wiki = 0

    for paper in all_papers:
        key = paper["dir"]
        meta = meta_by_dir.get(key, {})
        paper_for_resolve = dict(paper)
        if not paper_for_resolve.get("arxiv") and meta.get("arxiv"):
            paper_for_resolve["arxiv"] = normalize_arxiv(meta["arxiv"])
        if meta.get("title"):
            paper_for_resolve["title"] = meta["title"]
        primary = resolve_primary_wiki(paper_for_resolve, manual, wiki_index, planned)
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
    cat_ids = sorted(
        {cat_id for cat_id in papers_json}
        | {p["category"] for p in all_papers if p.get("category")}
    )
    cat_entries: list[tuple[str, dict, int]] = []
    for cat_id in cat_ids:
        section = papers_json.get(cat_id) or {
            "display_name": cat_id.split("_", 1)[-1].replace("_", " "),
            "zhname": cat_id.split("_", 1)[-1].replace("_", " "),
        }
        papers_in_cat: list[tuple[dict, str, dict]] = []
        for paper in all_papers:
            if paper.get("category") != cat_id:
                continue
            wiki_rel = full_map[paper["dir"]][0]
            papers_in_cat.append((paper, wiki_rel, meta_by_dir.get(paper["dir"], {})))
        papers_in_cat = dedupe_category_entries(papers_in_cat)
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
        f"full map {len(full_map)} / {len(all_papers)} papers "
        f"({len(progress_pending)} progress.json pending, "
        f"{len(progress_md_papers)} PROGRESS.md additions)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
