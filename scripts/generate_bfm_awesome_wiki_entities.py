#!/usr/bin/env python3
"""Generate wiki/entities pages for awesome-bfm-papers 41 papers + 10 datasets."""

from __future__ import annotations

import importlib.util
import re
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENTITIES_DIR = ROOT / "wiki" / "entities"
TECH_MAP = ROOT / "wiki" / "overview" / "bfm-41-papers-technology-map.md"
TODAY = date.today().isoformat()

_SPEC = importlib.util.spec_from_file_location(
    "bfm_sources",
    ROOT / "scripts" / "generate_bfm_awesome_sources.py",
)
_bfm = importlib.util.module_from_spec(_SPEC)
assert _SPEC and _SPEC.loader
_SPEC.loader.exec_module(_bfm)
PAPERS: list[dict] = _bfm.PAPERS
DATASETS: list[dict] = _bfm.DATASETS
GROUP_LABEL: dict[str, str] = _bfm.GROUP_LABEL

VENUE_SUFFIXES = (
    "_arxiv",
    "_icml",
    "_iclr",
    "_neurips",
    "_corl",
    "_siggraph",
    "_tog",
    "_cvpr",
    "_eccv",
)


def _strip_venue_suffix(slug: str) -> str:
    for suffix in VENUE_SUFFIXES:
        if suffix in slug:
            return slug.split(suffix)[0]
    return slug


def paper_entity_name(p: dict) -> str | None:
    if p.get("skip") and p["id"] == 13:
        return None
    short = _strip_venue_suffix(p["slug"]).replace("_", "-")
    return f"paper-bfm-{p['id']:02d}-{short}.md"


def paper_wiki_relpath(p: dict) -> str:
    if p.get("skip") and p["id"] == 13:
        return "wiki/entities/paper-behavior-foundation-model-humanoid.md"
    name = paper_entity_name(p)
    assert name
    return f"wiki/entities/{name}"


def dataset_entity_name(d: dict) -> str | None:
    if d.get("skip") and d.get("existing"):
        return None
    short = _strip_venue_suffix(d["slug"].replace("dataset_", "")).replace("_", "-")
    return f"dataset-bfm-{short}.md"


def dataset_wiki_relpath(d: dict) -> str:
    if d.get("skip") and d.get("existing"):
        return "wiki/entities/amass.md"
    name = dataset_entity_name(d)
    assert name
    return f"wiki/entities/{name}"


def _wiki_to_entity_rel(wiki_path: str) -> str:
    assert wiki_path.startswith("wiki/")
    return "../" + wiki_path[len("wiki/") :]


def _related_for_paper(p: dict) -> list[str]:
    out = [
        "../concepts/behavior-foundation-model.md",
        "../overview/bfm-41-papers-technology-map.md",
    ]
    for w in p.get("wiki", []):
        if not w.startswith("wiki/"):
            continue
        rel = _wiki_to_entity_rel(w)
        if rel not in out:
            out.append(rel)
    return out[:12]


def _related_for_dataset(d: dict) -> list[str]:
    out = [
        "../concepts/behavior-foundation-model.md",
        "../overview/bfm-41-papers-technology-map.md",
    ]
    for w in d.get("wiki", []):
        if not w.startswith("wiki/"):
            continue
        rel = _wiki_to_entity_rel(w)
        if rel not in out:
            out.append(rel)
    return out[:12]


def _short_title(title: str) -> str:
    if ":" in title:
        return title.split(":", 1)[0].strip()
    return title


def _yaml_list(items: list[str], indent: int = 0) -> str:
    pad = " " * indent
    return "\n".join(f"{pad}- {x}" for x in items)


def paper_entity_md(p: dict) -> str:
    rel_path = paper_wiki_relpath(p)
    source_rel = f"../../sources/papers/bfm_awesome_{p['slug']}.md"
    related = _related_for_paper(p)
    short = _short_title(p["title"])
    code_line = ""
    if p.get("code"):
        code_line = f"\n- **代码/项目：** <{p['code']}>"
    return f"""---
type: entity
tags: [paper, bfm, behavior-foundation-model, awesome-bfm-papers]
status: complete
updated: {TODAY}
summary: "{p["note"].replace('"', "'")}"
related:
{_yaml_list(related, 2)}
sources:
  - {source_rel}
  - ../../sources/papers/bfm_awesome_41_catalog.md
  - ../../sources/blogs/wechat_embodied_ai_lab_bfm_41_papers_survey.md
---

# {short}

**{short}** 收录于 [awesome-bfm-papers](https://github.com/friedrichyuan/awesome-bfm-papers) **第 {p["id"]:02d}/41** 篇，归类为 **{GROUP_LABEL[p["group"]]}**（{p["year"]} · {p["venue"]}）。本页为知识库 **策展摘要**；方法细节以论文 PDF 与项目页为准。

## 为什么重要

- {p["note"]}
- 在 [BFM 41 篇技术地图](../overview/bfm-41-papers-technology-map.md) 的五类问题坐标中，属于 **{GROUP_LABEL[p["group"]]}** 簇，可与 [Behavior Foundation Model](../concepts/behavior-foundation-model.md) taxonomy 对照阅读。

## 核心信息（索引级）

| 字段 | 内容 |
|------|------|
| 编号 | {p["id"]:02d}/41 |
| 分组 | {GROUP_LABEL[p["group"]]} |
| 出处 | {p["year"]} · {p["venue"]} |
| 论文 | <{p["paper"]}> |{code_line}

## 与其他页面的关系

- 技术地图：[bfm-41-papers-technology-map.md](../overview/bfm-41-papers-technology-map.md)
- BFM 概念：[behavior-foundation-model.md](../concepts/behavior-foundation-model.md)
- 原始 source：[bfm_awesome_{p["slug"]}.md](../../sources/papers/bfm_awesome_{p["slug"]}.md)

## 参考来源

- [bfm_awesome_{p["slug"]}.md](../../sources/papers/bfm_awesome_{p["slug"]}.md) — awesome-bfm 策展摘录
- [bfm_awesome_41_catalog.md](../../sources/papers/bfm_awesome_41_catalog.md) — 41+10 总表
- [wechat_embodied_ai_lab_bfm_41_papers_survey.md](../../sources/blogs/wechat_embodied_ai_lab_bfm_41_papers_survey.md) — 微信公众号编译导读
- 论文：<{p["paper"]}>

## 推荐继续阅读

- [awesome-bfm-papers](https://github.com/friedrichyuan/awesome-bfm-papers) — 完整列表与数据集表
- [A Survey of Behavior Foundation Model](https://arxiv.org/abs/2506.20487) — TPAMI 2025 综述
"""


def dataset_entity_md(d: dict) -> str:
    source_rel = f"../../sources/papers/bfm_awesome_{d['slug']}.md"
    related = _related_for_dataset(d)
    clips = d.get("clips", "-")
    hours = d.get("hours", "-")
    return f"""---
type: entity
tags: [dataset, bfm, behavior-foundation-model, human-motion, awesome-bfm-papers]
status: complete
updated: {TODAY}
summary: "{d["note"].replace('"', "'")}"
related:
{_yaml_list(related, 2)}
sources:
  - {source_rel}
  - ../../sources/papers/bfm_awesome_41_catalog.md
  - ../../sources/blogs/wechat_embodied_ai_lab_bfm_41_papers_survey.md
---

# {d["name"]}（BFM 行为数据）

**{d["name"]}** 列入 [awesome-bfm-papers](https://github.com/friedrichyuan/awesome-bfm-papers) 数据集表（{d["year"]} · {d["venue"]}）。本页为 **索引级** 说明；规模与许可以官方页面为准。

## 为什么重要

- {d["note"]}
- BFM 数据链路的瓶颈往往在 **能否变成机器人可信、可执行、可迁移** 的训练材料，而非单纯 clip 数量（见 [BFM 技术地图](../overview/bfm-41-papers-technology-map.md) § 数据集）。

## 核心信息（索引级）

| 字段 | 内容 |
|------|------|
| 规模（列表标注） | {clips} clips · {hours} h |
| 论文/说明 | <{d["paper"]}> |
| 入口 | <{d["code"]}> |

## 与其他页面的关系

- [behavior-foundation-model.md](../concepts/behavior-foundation-model.md)
- [bfm-41-papers-technology-map.md](../overview/bfm-41-papers-technology-map.md)

## 参考来源

- [bfm_awesome_{d["slug"]}.md](../../sources/papers/bfm_awesome_{d["slug"]}.md)
- [bfm_awesome_41_catalog.md](../../sources/papers/bfm_awesome_41_catalog.md#数据集10)
- 数据集页：<{d["paper"]}>

## 推荐继续阅读

- [awesome-bfm-papers § Datasets](https://github.com/friedrichyuan/awesome-bfm-papers#datasets)
"""


def entity_index_section() -> str:
    lines = [
        "",
        "## Wiki 实体索引（站内详情页）",
        "",
        "> 41 篇论文与 10 个数据集均已升格为 `wiki/entities/` 详情页（可搜索、进图谱）；#13 与 AMASS 复用既有深读页。",
        "",
        "### 论文（41）",
        "",
        "| # | 工作 | Wiki 实体 | Source |",
        "|---|------|-----------|--------|",
    ]
    for p in PAPERS:
        wiki = paper_wiki_relpath(p)
        ent = Path(wiki).name.replace(".md", "")
        short = _short_title(p["title"])
        lines.append(
            f"| {p['id']:02d} | {short} | "
            f"[{ent}](../entities/{Path(wiki).name}) | "
            f"[source](../../sources/papers/bfm_awesome_{p['slug']}.md) |"
        )
    lines.extend(
        [
            "",
            "### 数据集（10）",
            "",
            "| 数据集 | Wiki 实体 | Source |",
            "|--------|-----------|--------|",
        ]
    )
    for d in DATASETS:
        wiki = dataset_wiki_relpath(d)
        ent = Path(wiki).name.replace(".md", "")
        lines.append(
            f"| {d['name']} | [{ent}](../entities/{Path(wiki).name}) | "
            f"[source](../../sources/papers/bfm_awesome_{d['slug']}.md) |"
        )
    lines.append("")
    return "\n".join(lines)


def patch_tech_map() -> bool:
    if not TECH_MAP.is_file():
        return False
    text = TECH_MAP.read_text(encoding="utf-8")
    marker = "## Wiki 实体索引（站内详情页）"
    section = entity_index_section()
    if marker in text:
        text = re.sub(
            r"\n## Wiki 实体索引（站内详情页）[\s\S]*?(?=\n## |\Z)",
            "\n" + section.strip() + "\n",
            text,
            count=1,
        )
    else:
        anchor = "## 五组论文地图（41 篇）"
        if anchor not in text:
            return False
        text = text.replace(
            anchor,
            section.strip() + "\n\n" + anchor,
            1,
        )
    note = (
        "- **Wiki 实体：** 每篇/每项均有站内详情页，见下节 "
        "[Wiki 实体索引](#wiki-实体索引站内详情页)；图谱与搜索已收录。\n"
    )
    old = (
        "- **总表：** [bfm_awesome_41_catalog.md](../../sources/papers/bfm_awesome_41_catalog.md) "
        "— 每篇/每项对应独立 `sources/papers/bfm_awesome_<slug>.md`（策展摘录 + 公众号导读要点，非 PDF 全文）。\n"
    )
    if note.strip() not in text and old in text:
        text = text.replace(old, old + note)
    if f"updated: {TODAY}" not in text.split("---", 2)[0]:
        text = re.sub(r"^updated: \d{4}-\d{2}-\d{2}", f"updated: {TODAY}", text, count=1, flags=re.M)
    TECH_MAP.write_text(text, encoding="utf-8")
    return True


def main() -> int:
    ENTITIES_DIR.mkdir(parents=True, exist_ok=True)
    created = 0
    updated = 0

    for p in PAPERS:
        name = paper_entity_name(p)
        if not name:
            continue
        path = ENTITIES_DIR / name
        content = paper_entity_md(p)
        if path.exists() and path.read_text(encoding="utf-8") == content:
            continue
        if path.exists():
            updated += 1
        else:
            created += 1
        path.write_text(content, encoding="utf-8")

    for d in DATASETS:
        name = dataset_entity_name(d)
        if not name:
            continue
        path = ENTITIES_DIR / name
        content = dataset_entity_md(d)
        if path.exists() and path.read_text(encoding="utf-8") == content:
            continue
        if path.exists():
            updated += 1
        else:
            created += 1
        path.write_text(content, encoding="utf-8")

    patched = patch_tech_map()
    print(
        f"papers={sum(1 for p in PAPERS if paper_entity_name(p))} "
        f"datasets={sum(1 for d in DATASETS if dataset_entity_name(d))} "
        f"created={created} updated={updated} tech_map_patched={patched}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
