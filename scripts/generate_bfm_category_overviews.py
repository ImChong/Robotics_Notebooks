#!/usr/bin/env python3
"""Generate wiki/overview BFM category hub pages (5 groups) and cross-link paper entities."""

from __future__ import annotations

import importlib.util
import re
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OVERVIEW_DIR = ROOT / "wiki" / "overview"
ENTITIES_DIR = ROOT / "wiki" / "entities"
TECH_MAP = ROOT / "wiki" / "overview" / "bfm-41-papers-technology-map.md"
BFM_CONCEPT = ROOT / "wiki" / "concepts" / "behavior-foundation-model.md"
WECHAT_BLOG = ROOT / "sources" / "blogs" / "wechat_embodied_ai_lab_bfm_41_papers_survey.md"
TODAY = date.today().isoformat()
WECHAT_URL = "https://mp.weixin.qq.com/s/Ei32la_vo0UW9Y_QCAqB2g"


def _load_bfm_sources_module():
    spec = importlib.util.spec_from_file_location(
        "bfm_sources",
        ROOT / "scripts" / "generate_bfm_awesome_sources.py",
    )
    if spec is None or spec.loader is None:
        raise ImportError("cannot load generate_bfm_awesome_sources.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_bfm = _load_bfm_sources_module()
PAPERS: list[dict] = _bfm.PAPERS
GROUP_LABEL: dict[str, str] = _bfm.GROUP_LABEL

# Order matches公众号 / awesome-bfm-papers
CATEGORIES: list[dict] = [
    {
        "group": "forward-backward",
        "slug": "bfm-category-01-forward-backward-representation",
        "title": "BFM 分类 01：Forward-backward 表征",
        "question": "多任务能否压进**可调用的身体潜空间**（latent prompt / FB 嵌入），而非每换一个任务就重训全身策略？",
        "representatives": "BFM-Zero、MetaMotivo、FB-AW、Fast Imitation、Learning One Representation、Successor States",
    },
    {
        "group": "goal-conditioned",
        "slug": "bfm-category-02-goal-conditioned-learning",
        "title": "BFM 分类 02：Goal-conditioned 学习",
        "question": "运控基座的**动作覆盖面**——跟踪、全身技能、遥操作、人-物交互（HOI）能否在统一目标条件下扩展？",
        "representatives": "SONIC、OpenTrack、AMS、TWIST/TWIST2、BFM4Humanoid、HOVER、InterMimic、MaskedMimic、ASE/CALM/CASE…",
    },
    {
        "group": "intrinsic-reward",
        "slug": "bfm-category-03-intrinsic-reward-pretraining",
        "title": "BFM 分类 03：Intrinsic reward 预训练",
        "question": "在**尚无明确下游任务**时，身体应先通过内在奖励积累何种**可迁移探索经验**？",
        "representatives": "APS、Proto-RL、RE3、RND、DIAYN",
    },
    {
        "group": "adaptation",
        "slug": "bfm-category-04-adaptation",
        "title": "BFM 分类 04：Adaptation",
        "question": "预训练 BFM 如何以**低成本**适配新任务、新动力学或新机体（样本与工程摩擦）？",
        "representatives": "Task Tokens、Unseen Dynamics、Fast Adaptation",
    },
    {
        "group": "hierarchical",
        "slug": "bfm-category-05-hierarchical-control",
        "title": "BFM 分类 05：Hierarchical control",
        "question": "语言、VLA、扩散与规划器如何通过**层次接口**（技能 token、action chunk）调用已训练好的底层身体，并由 WBC / 执行器承担关节级闭环？",
        "representatives": "SENTINEL、BeyondMimic、LeVerb、LangWBC、TokenHSI、CLoSD、UniPhys、UniHSI",
    },
]


def category_overview_path(cat: dict) -> str:
    return f"wiki/overview/{cat['slug']}.md"


def category_rel_path(cat: dict) -> str:
    return f"../overview/{cat['slug']}.md"


def paper_entity_name(p: dict) -> str | None:
    if p.get("skip") and p["id"] == 13:
        return "paper-behavior-foundation-model-humanoid.md"
    short = p["slug"]
    for suffix in (
        "_arxiv",
        "_icml",
        "_iclr",
        "_neurips",
        "_corl",
        "_siggraph",
        "_tog",
        "_cvpr",
        "_eccv",
    ):
        if suffix in short:
            short = short.split(suffix)[0]
            break
    short = short.replace("_", "-")
    return f"paper-bfm-{p['id']:02d}-{short}.md"


def paper_entity_rel(p: dict) -> str:
    name = paper_entity_name(p)
    assert name
    return f"../entities/{name}"


def _yaml_list(items: list[str], indent: int = 0) -> str:
    pad = " " * indent
    return "\n".join(f"{pad}- {x}" for x in items)


def _related_for_category(cat: dict, papers_in_group: list[dict]) -> list[str]:
    out = [
        "./bfm-41-papers-technology-map.md",
        "../concepts/behavior-foundation-model.md",
        "../../sources/blogs/wechat_embodied_ai_lab_bfm_41_papers_survey.md",
        "../../sources/papers/bfm_awesome_41_catalog.md",
    ]
    for other in CATEGORIES:
        if other["group"] != cat["group"]:
            out.append(f"./{other['slug']}.md")
    for p in papers_in_group[:8]:
        rel = paper_entity_rel(p)
        if rel not in out:
            out.append(rel)
    return out[:14]


def category_overview_md(cat: dict, papers: list[dict]) -> str:
    label = GROUP_LABEL[cat["group"]]
    related = _related_for_category(cat, papers)
    paper_rows = []
    for p in sorted(papers, key=lambda x: x["id"]):
        ent = paper_entity_name(p)
        short = p["title"].split(":", 1)[0].strip() if ":" in p["title"] else p["title"]
        paper_rows.append(
            f"| {p['id']:02d} | {short} | "
            f"[{ent}](../entities/{ent}) | "
            f"[source](../../sources/papers/bfm_awesome_{p['slug']}.md) |"
        )
    paper_table = "\n".join(paper_rows)
    return f"""---
type: overview
tags: [bfm, behavior-foundation-model, category-hub, awesome-bfm-papers, {cat["group"]}]
status: complete
updated: {TODAY}
summary: "具身智能研究室 BFM 41 篇专题 · {label}（{len(papers)} 篇）— {cat["question"].replace("**", "")[:120]}"
related:
{_yaml_list(related, 2)}
sources:
  - ../../sources/blogs/wechat_embodied_ai_lab_bfm_41_papers_survey.md
  - ../../sources/papers/bfm_awesome_41_catalog.md
  - ../../sources/repos/awesome_bfm_papers.md
---

# {cat["title"]}

> **图谱分类节点**：对应 [具身智能研究室 · BFM 41 篇专题]({WECHAT_URL}) 与 [awesome-bfm-papers](https://github.com/friedrichyuan/awesome-bfm-papers) 的 **{label}** 分组；本页汇集该组 **{len(papers)} 篇** 论文的站内实体与 source 索引。总地图见 [BFM 技术地图](./bfm-41-papers-technology-map.md)。

## 核心问题（公众号分类）

{cat["question"]}

**代表工作（策展）：** {cat["representatives"]}

## 本组论文（{len(papers)} 篇）

| # | 工作 | Wiki 实体 | Source |
|---|------|-----------|--------|
{paper_table}

## 在 BFM taxonomy 中的位置

| 字段 | 内容 |
|------|------|
| 分组 | {label} |
| 篇数 | {len(papers)}/41 |
| 概念对照 | [Behavior Foundation Model](../concepts/behavior-foundation-model.md) |
| 姊妹分类 | 见 [BFM 技术地图 · 五类问题](./bfm-41-papers-technology-map.md#流程总览五类问题--身体-api) |

## 关联页面

- [BFM 41 篇技术地图](./bfm-41-papers-technology-map.md)
- [Behavior Foundation Model](../concepts/behavior-foundation-model.md)
- [人形 RL 身体系统栈](./humanoid-rl-motion-control-body-system-stack.md)
- [AMP 运动先验综述](./humanoid-amp-motion-prior-survey.md)

## 参考来源

- [wechat_embodied_ai_lab_bfm_41_papers_survey.md](../../sources/blogs/wechat_embodied_ai_lab_bfm_41_papers_survey.md) — 微信公众号编译（<{WECHAT_URL}>）
- [bfm_awesome_41_catalog.md](../../sources/papers/bfm_awesome_41_catalog.md)
- [awesome-bfm-papers](https://github.com/friedrichyuan/awesome-bfm-papers)
- [A Survey of Behavior Foundation Model](https://arxiv.org/abs/2506.20487)（TPAMI 2025）
"""


def patch_paper_entity_related(p: dict, cat: dict) -> bool:
    name = paper_entity_name(p)
    if not name:
        return False
    path = ENTITIES_DIR / name
    if not path.is_file():
        return False
    text = path.read_text(encoding="utf-8")
    cat_rel = category_rel_path(cat)
    if cat_rel in text:
        return False
    # Insert into frontmatter related (after tech map line)
    marker = "../overview/bfm-41-papers-technology-map.md"
    if marker in text:
        text = text.replace(
            f"  - {marker}\n",
            f"  - {marker}\n  - {cat_rel}\n",
            1,
        )
    else:
        text = re.sub(
            r"(related:\n(?:  - [^\n]+\n)+)",
            lambda m: m.group(1) + f"  - {cat_rel}\n",
            text,
            count=1,
        )
    # Update body cluster line to link category hub
    old_cluster = f"属于 **{GROUP_LABEL[p['group']]}** 簇"
    new_cluster = (
        f"属于 **[{cat['title'].split('：', 1)[-1]}]({cat_rel})**（{GROUP_LABEL[p['group']]}）"
    )
    if old_cluster in text:
        text = text.replace(old_cluster, new_cluster)
    path.write_text(text, encoding="utf-8")
    return True


def category_hub_section() -> str:
    lines = [
        "",
        "## 五类问题分类节点（图谱 hub）",
        "",
        "> 每组对应一个独立 `wiki/overview/bfm-category-*` 页面，作为图谱中的**分类枢纽**；组内论文实体经 `[text](../overview/...)` 与本页互链。",
        "",
        "| 组 | 分类节点 | 篇数 |",
        "|----|----------|------|",
    ]
    by_group: dict[str, list[dict]] = defaultdict(list)
    for p in PAPERS:
        by_group[p["group"]].append(p)
    for cat in CATEGORIES:
        n = len(by_group[cat["group"]])
        lines.append(
            f"| {GROUP_LABEL[cat['group']]} | "
            f"[{cat['title'].split('：', 1)[-1]}](./{cat['slug']}.md) | {n} |"
        )
    lines.append("")
    return "\n".join(lines)


def patch_tech_map() -> bool:
    if not TECH_MAP.is_file():
        return False
    text = TECH_MAP.read_text(encoding="utf-8")
    section = category_hub_section()
    if "## 五类问题分类节点（图谱 hub）" in text:
        text = re.sub(
            r"\n## 五类问题分类节点（图谱 hub）[\s\S]*?(?=\n## 原始资料索引|\n## Wiki 实体索引|\n## 五组论文地图|\Z)",
            "\n" + section.strip() + "\n",
            text,
            count=1,
        )
    else:
        anchor = "## 原始资料索引（41 论文 + 10 数据集）"
        if anchor not in text:
            anchor = "## Wiki 实体索引（站内详情页）"
        text = text.replace(anchor, section.strip() + "\n\n" + anchor, 1)
    # Add category links to frontmatter related
    for cat in CATEGORIES:
        rel = f"  - ./{cat['slug']}.md\n"
        if rel.strip() not in text.split("---", 2)[1]:
            text = text.replace(
                "  - ./humanoid-amp-motion-prior-survey.md\n",
                "  - ./humanoid-amp-motion-prior-survey.md\n" + rel,
                1,
            )
    text = re.sub(
        r"^updated: \d{4}-\d{2}-\d{2}",
        f"updated: {TODAY}",
        text,
        count=1,
        flags=re.M,
    )
    TECH_MAP.write_text(text, encoding="utf-8")
    return True


def patch_bfm_concept() -> bool:
    if not BFM_CONCEPT.is_file():
        return False
    text = BFM_CONCEPT.read_text(encoding="utf-8")
    if "bfm-category-01-forward-backward" in text:
        return False
    anchor = "- [BFM 41 篇技术地图](../overview/bfm-41-papers-technology-map.md)"
    if anchor in text:
        block = anchor + "\n"
        for cat in CATEGORIES:
            block += f"- [{cat['title']}](../overview/{cat['slug']}.md)\n"
        text = text.replace(anchor, block.strip(), 1)
        BFM_CONCEPT.write_text(text, encoding="utf-8")
        return True
    return False


def patch_wechat_blog() -> bool:
    if not WECHAT_BLOG.is_file():
        return False
    text = WECHAT_BLOG.read_text(encoding="utf-8")
    if "bfm-category-01-forward-backward" in text:
        return False
    lines = [
        "",
        "### 五类问题分类节点（wiki / 图谱）",
        "",
        "| 组 | 分类页 |",
        "|----|--------|",
    ]
    by_group: dict[str, list[dict]] = defaultdict(list)
    for p in PAPERS:
        by_group[p["group"]].append(p)
    for cat in CATEGORIES:
        n = len(by_group[cat["group"]])
        lines.append(
            f"| {GROUP_LABEL[cat['group']]}（{n} 篇） | "
            f"[{cat['slug']}](../../wiki/overview/{cat['slug']}.md) |"
        )
    block = "\n".join(lines)
    anchor = "## 对 wiki 的映射"
    if anchor in text:
        text = text.replace(anchor, block + "\n\n" + anchor, 1)
        WECHAT_BLOG.write_text(text, encoding="utf-8")
        return True
    return False


def main() -> int:
    by_group: dict[str, list[dict]] = defaultdict(list)
    for p in PAPERS:
        by_group[p["group"]].append(p)

    created = 0
    for cat in CATEGORIES:
        papers = sorted(by_group[cat["group"]], key=lambda x: x["id"])
        path = OVERVIEW_DIR / f"{cat['slug']}.md"
        content = category_overview_md(cat, papers)
        if not path.exists():
            created += 1
        path.write_text(content, encoding="utf-8")

    patched_entities = 0
    cat_by_group = {c["group"]: c for c in CATEGORIES}
    for p in PAPERS:
        if patch_paper_entity_related(p, cat_by_group[p["group"]]):
            patched_entities += 1

    tech = patch_tech_map()
    concept = patch_bfm_concept()
    blog = patch_wechat_blog()
    print(
        f"categories={len(CATEGORIES)} created={created} "
        f"entities_patched={patched_entities} tech_map={tech} concept={concept} blog={blog}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
