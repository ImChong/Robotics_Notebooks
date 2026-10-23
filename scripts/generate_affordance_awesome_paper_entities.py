#!/usr/bin/env python3
"""Generate index-level paper entities for hq-King/Awesome-Affordance-Learning.

Parses the upstream README tables, dedupes by arXiv / normalized title against
existing wiki pages (and across the list's own sections), and writes:
  - sources/papers/affordance_awesome_catalog.md
  - sources/papers/affordance_awesome_<slug>.md   (new papers only)
  - wiki/entities/paper-aff-<slug>.md             (new papers only)
  - wiki/overview/affordance-awesome-technology-map.md

Usage: generate_affordance_awesome_paper_entities.py <README.md>
Idempotent: existing paper-aff-* files are never overwritten.
"""

from __future__ import annotations

import re
import sys
import unicodedata
from datetime import date
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
TODAY = date.today().isoformat()
REPO_URL = "https://github.com/hq-King/Awesome-Affordance-Learning"
SURVEY = "../entities/paper-affordance-learning-embodied-ai-survey.md"
MAP_REL = "../overview/affordance-awesome-technology-map.md"
CATALOG_SRC = "sources/papers/affordance_awesome_catalog.md"
REPO_SRC = "sources/repos/awesome-affordance-learning.md"
SECTION_ORDER = ["Perception", "Reasoning", "Action", "Datasets / Benchmarks", "Related Surveys"]
SECTION_HUB = {
    "Perception": ("../tasks/manipulation.md", "../methods/vla.md"),
    "Reasoning": ("../methods/vla.md", "../tasks/manipulation.md"),
    "Action": ("../tasks/manipulation.md", "../methods/imitation-learning.md"),
    "Datasets / Benchmarks": ("../tasks/manipulation.md", "../methods/imitation-learning.md"),
    "Related Surveys": ("../tasks/manipulation.md", "../methods/vla.md"),
}
# Same paper already ingested under a non-arXiv-frontmatter page (checked by hand, title + arXiv).
MANUAL_EXISTING = {
    "2103.14127": "wiki/entities/painode-214-contactgraspnet.md",
    "1912.13470": "wiki/entities/painode-218-graspnet1billion.md",
}
ABBREV = [
    ("Affordance", "Action possibility (Gibson)", "环境/物体对 agent 提供的可交互机会"),
    ("HOI", "Human-Object Interaction", "人–物交互，常作 affordance 监督来源"),
    ("VLM", "Vision-Language Model", "视觉–语言模型，用于 affordance 推理与接地"),
    ("VLA", "Vision-Language-Action", "视觉–语言–动作策略"),
]
ARXIV_RE = re.compile(r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})", re.I)


def norm(t: str) -> str:
    return re.sub(r"[^a-z0-9]", "", t.lower())


def slugify(title: str, n: int = 56) -> str:
    s = unicodedata.normalize("NFKD", title).encode("ascii", "ignore").decode().lower()
    return (re.sub(r"[^a-z0-9]+", "-", s).strip("-") or "untitled")[:n].strip("-")


def yq(s: str) -> str:
    return re.sub(r"\s+", " ", s.replace("\\", "").replace('"', "'")).strip()


def parse(path: Path) -> list[dict]:
    sec = hdr = None
    entries: dict[str, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("### "):
            sec = next((s for s in SECTION_ORDER if s.lower().split()[0] in line.lower()), None)
            hdr = None
        elif line.startswith("| Venue/Date"):
            hdr = [c.strip() for c in line.strip().strip("|").split("|")]
        elif line.startswith("| ") and "badge/Paper" in line and hdr and sec:
            c = [x.strip() for x in line.strip().strip("|").split("|")]
            links = re.findall(r"\]\((http[^)]+)\)", c[3])
            url = links[-1] if links else ""
            m = ARXIV_RE.search(url)
            attrs = [(h, v) for h, v in zip(hdr[4:], c[4:]) if v and v != "-"]
            e: dict[str, Any] = {
                "title": c[2],
                "venue": c[0],
                "short": c[1].strip("`").strip(),
                "url": url,
                "arxiv": m.group(1) if m else None,
            }
            key = e["arxiv"] or norm(e["title"])
            cur = entries.setdefault(key, {**e, "groups": []})
            cur["groups"].append((sec, attrs))
    return list(entries.values())


def existing_index() -> tuple[dict[str, str], dict[str, str]]:
    by_arxiv: dict[str, str] = dict(MANUAL_EXISTING)
    by_title: dict[str, str] = {}
    for p in (ROOT / "wiki").rglob("*.md"):
        t = p.read_text(encoding="utf-8", errors="ignore")
        if p.name.startswith("paper-aff-"):
            continue
        fm = t.split("---", 2)[1] if t.startswith("---") else ""
        for m in re.finditer(r"(?m)^(?:arxiv|paper|papers):\s*(.*)$", fm):
            for a in re.findall(r"\d{4}\.\d{4,5}", m.group(1)):
                by_arxiv.setdefault(a, str(p.relative_to(ROOT)))
        h = re.search(r"(?m)^# (.+)$", t)
        if h:
            by_title.setdefault(norm(h.group(1)), str(p.relative_to(ROOT)))
        for b in re.findall(r"\*\*([^*]{15,200})\*\*", t[:1500]):
            by_title.setdefault(norm(b), str(p.relative_to(ROOT)))
    return by_arxiv, by_title


def primary_section(e: dict) -> str:
    return e["groups"][0][0]


def group_text(e: dict) -> str:
    parts = []
    for sec, attrs in e["groups"]:
        a = "；".join(f"{h}: {v}" for h, v in attrs)
        parts.append(f"{sec}" + (f"（{a}）" if a else ""))
    return " ｜ ".join(parts)


def paper_link(e: dict) -> str:
    return e["url"] or REPO_URL


def render_source(e: dict, wiki_rel: str) -> str:
    return f"""# {e["title"]}

> 来源归档（Awesome-Affordance-Learning 策展索引级）

- **列表：** [Awesome-Affordance-Learning]({REPO_URL})
- **分组：** {group_text(e)}
- **入库日期：** {TODAY}
- **arXiv：** {e["arxiv"] or "（无 / 非 arXiv）"}
- **出处：** {e["venue"]}
- **方法名（清单标注）：** {e["short"] or "—"}
- **论文：** <{paper_link(e)}>
- **代码：** 清单未标注（需到论文 / 项目页核实）
- **沉淀到 wiki：** [`{wiki_rel}`](../../{wiki_rel})

---

## 开源边界（步骤 2.5）

| 已发布 | 备注 |
|--------|------|
| 清单条目元数据 | 本 source 为策展摘录，非全文转存 |
| 代码/权重 | 清单未列；入库时未逐篇打开项目页核查，以原文为准 |

## 对 wiki 的映射

- 实体页：[`{wiki_rel}`](../../{wiki_rel})
- 技术地图：[`wiki/overview/affordance-awesome-technology-map.md`](../../wiki/overview/affordance-awesome-technology-map.md)
"""


def render_entity(e: dict, src_rel: str, wiki_name: str) -> str:
    sec = primary_section(e)
    hubs = SECTION_HUB[sec]
    short = e["short"] if e["short"] and e["short"] != "-" else e["title"].split(":")[0]
    short = short if len(short) <= 80 else short[:77] + "..."
    gt = group_text(e)
    summary = yq(f"{e['venue']} 的 affordance 论文，清单分组：{gt}。")[:220]
    fm = [f"project_id: aff-{slugify(e['title'])}", f'venue: "{yq(e["venue"])[:80]}"']
    doi = re.search(r"doi\.org/(10\.\S+)$", e["url"])
    if e["arxiv"]:
        fm.insert(1, f'arxiv: "{e["arxiv"]}"')
    elif doi:
        fm.insert(1, f'doi: "{doi.group(1)}"')
    elif e["url"]:
        fm.insert(1, f'url: "{e["url"]}"')
    rows = "\n".join(
        f"| 分组 {i + 1} | {s}" + (" · " + "；".join(f"{h}: {v}" for h, v in a) if a else "") + " |"
        for i, (s, a) in enumerate(e["groups"])
    )
    abbrev = "\n".join(f"| {a} | {b} | {c} |" for a, b, c in ABBREV)
    pl = paper_link(e)
    related = [SURVEY, MAP_REL, *hubs]
    return f"""---
type: entity
tags: [paper, curated-index, awesome-affordance-learning, affordance-learning, embodied-ai]
status: complete
updated: {TODAY}
{chr(10).join(fm)}
summary: "{summary}"
related:
{chr(10).join("  - " + r for r in related)}
sources:
  - ../../{src_rel}
  - ../../{CATALOG_SRC}
  - ../../{REPO_SRC}
---

# {short}

**{e["title"]}**（{e["venue"]}）收录于 [Awesome-Affordance-Learning]({REPO_URL})，属于具身 AI affordance 学习综述的配套论文清单。本页是 **清单索引**：给出它在综述分类里的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

{e["title"]}——在清单中归入 **{gt}**（清单未附方法摘要，此处不替原文概括贡献）。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
{abbrev}

## 为什么重要

- 综述把 affordance 学习拆成感知、推理、引导动作三段（见 [综述实体页]({SURVEY})），本条目是其中一段的代表文献之一。
- 可顺着 [技术地图]({MAP_REL}) 的同分组条目横向阅读同方向工作。

## 核心信息

| 字段 | 内容 |
|------|------|
| 出处 | {e["venue"]} |
| 方法名（清单标注） | {e["short"] or "—"} |
{rows}
| 论文 | <{pl}> |

## 核心机制（归纳）

清单只给出标题、出处与分类标签，**没有**方法摘要。本页不凭标题推测网络结构或训练细节；需要机制与实验请读 [原文]({pl})。

## 评测与指标

- 本页没有搬运原文的 benchmark 与实机指标；评测口径以 [原文]({pl}) 为准。
- 综述本身不提供跨论文性能排名（见 [综述实体页]({SURVEY})），不同设置下的数字不可直接比较。

## 与其他工作对比

- 本页不做与具体基线的数值对比；同分组横向对照见 [技术地图]({MAP_REL})。
- 若站内另有这篇的深读页，以那一页为准。

## 结论

**本页只提供「{short}」在 affordance 综述清单中的坐标与原文入口，不能替代原文。**

- 可确证的只有标题、出处与分类标签；方法与结果未经本页核实。
- 开源状态清单未标注，入库时未打开项目页核查，复现前请自行确认。
- 要深读：从原文入手，再回综述实体页对照 perception → reasoning → action 的位置。

## 常见误区

1. 不要把分类标签当作论文结论；标签来自清单维护者的归类。
2. 不要把本页当作方法证明；它只是清单入口。

## 关联页面

- [Affordance 学习综述]({SURVEY})
- [Awesome-Affordance-Learning 技术地图]({MAP_REL})
- [{hubs[0].split("/")[-1]}]({hubs[0]})、[{hubs[1].split("/")[-1]}]({hubs[1]})

## 参考来源

- [`{src_rel}`](../../{src_rel}) — 本条目策展摘录
- [`{CATALOG_SRC}`](../../{CATALOG_SRC}) — 列表总表
- [`{REPO_SRC}`](../../{REPO_SRC})
- 论文：<{pl}>

## 推荐继续阅读

- [Awesome-Affordance-Learning 仓库]({REPO_URL})
- [原文]({pl})
"""


def render_catalog(rows: list[dict]) -> str:
    lines = [
        "# Awesome-Affordance-Learning 论文目录",
        "",
        f"> 由 `{REPO_URL}` 解析生成；入库日 {TODAY}。同一论文在多个分组出现时合并为一行。",
        "",
        "- **技术地图：** [`wiki/overview/affordance-awesome-technology-map.md`](../../wiki/overview/affordance-awesome-technology-map.md)",
        f"- **条目数（去重后）：** {len(rows)}",
        "",
        "| 标题 | 出处 | arXiv | 分组 | wiki |",
        "|------|------|-------|------|------|",
    ]
    for r in rows:
        lines.append(
            f"| {r['title'][:80].replace('|', '/')} | {r['venue']} | {r['arxiv'] or '—'} | "
            f"{', '.join(dict.fromkeys(s for s, _ in r['groups']))} | "
            f"[`{Path(r['wiki']).name}`](../../{r['wiki']}) |"
        )
    return "\n".join(lines) + "\n"


def render_map(rows: list[dict]) -> str:
    out = []
    for sec in SECTION_ORDER:
        items = [r for r in rows if primary_section(r) == sec]
        if not items:
            continue
        out += [f"### {sec}\n", "| 出处 | 论文 |", "|------|------|"]
        for r in items:
            link = "../" + r["wiki"][len("wiki/") :]
            out.append(f"| {r['venue']} | [{r['title'][:90].replace('|', '/')}]({link}) |")
        out.append("")
    abbrev = "\n".join(f"| {a} | {b} | {c} |" for a, b, c in ABBREV)
    return f"""---
type: overview
tags: [overview, curated-index, awesome-affordance-learning, affordance-learning, technology-map]
status: complete
updated: {TODAY}
summary: "Awesome-Affordance-Learning 技术地图：清单中 {len(rows)} 篇去重论文逐条对应站内一页，按感知 / 推理 / 动作 / 数据集 / 综述分组。"
related:
  - {SURVEY}
  - ../tasks/manipulation.md
  - ../methods/vla.md
sources:
  - ../../{CATALOG_SRC}
  - ../../{REPO_SRC}
---

# Awesome-Affordance-Learning 技术地图

> 把 [Awesome-Affordance-Learning]({REPO_URL}) 清单的论文逐条拆成站内可点开的一页，按清单分组浏览。该清单是 [Affordance 学习综述]({SURVEY}) 的配套资源。

## 一句话定义

外部 Awesome 清单的站内版本：每篇论文一个节点，已有深读页的条目直接链过去，其余为索引级清单页。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
{abbrev}

## 为什么重要

- 原清单每条只有标题与链接，站内逐条成页后可检索、可互链。
- 同一论文在清单多个分组出现（如数据集 / 感知）时只保留一个节点；与站内既有页重复的条目复用既有页，不新建。

## 覆盖范围

| 项 | 值 |
|----|-----|
| 上游仓库 | <{REPO_URL}> |
| 去重后条目 | {len(rows)} |
| 目录 source | [affordance_awesome_catalog.md](../../{CATALOG_SRC}) |
| 配套综述 | [Affordance 学习综述]({SURVEY}) |

## 分组索引

{chr(10).join(out)}

## 局限与风险

- 索引级页面只含标题、出处与分类标签，**不替代**原文。
- 分组以条目首次出现的章节为准；跨分组条目的其他分组写在条目页内。
- 上游仍在更新，本页是 {TODAY} 的快照。

## 关联页面

- [Affordance 学习综述]({SURVEY})
- [manipulation](../tasks/manipulation.md)
- [vla](../methods/vla.md)

## 参考来源

- [affordance_awesome_catalog.md](../../{CATALOG_SRC})
- [awesome-affordance-learning.md](../../{REPO_SRC})
- 上游：<{REPO_URL}>

## 推荐继续阅读

- [Awesome-Affordance-Learning GitHub]({REPO_URL})
"""


def main() -> None:
    entries = parse(Path(sys.argv[1]))
    by_arxiv, by_title = existing_index()
    rows, new, reused = [], 0, 0
    used: set[str] = set()
    for e in entries:
        hit = (by_arxiv.get(e["arxiv"]) if e["arxiv"] else None) or by_title.get(norm(e["title"]))
        if hit:
            e["wiki"], reused = hit, reused + 1
        else:
            slug = slugify(e["title"])
            if slug in used:
                raise SystemExit(f"slug collision: {slug}")
            used.add(slug)
            name = f"paper-aff-{(e['arxiv'] or '').replace('.', '-') + '-' if e['arxiv'] else ''}{slug}.md"
            e["wiki"] = f"wiki/entities/{name}"
            src_rel = f"sources/papers/affordance_awesome_{slug}.md"
            if not (ROOT / src_rel).exists():
                (ROOT / src_rel).write_text(render_source(e, e["wiki"]), encoding="utf-8")
            if not (ROOT / e["wiki"]).exists():
                (ROOT / e["wiki"]).write_text(render_entity(e, src_rel, name), encoding="utf-8")
            new += 1
        rows.append(e)
    (ROOT / CATALOG_SRC).write_text(render_catalog(rows), encoding="utf-8")
    (ROOT / "wiki/overview/affordance-awesome-technology-map.md").write_text(
        render_map(rows), encoding="utf-8"
    )
    print(f"unique={len(rows)} new={new} reused={reused}")


if __name__ == "__main__":
    main()
