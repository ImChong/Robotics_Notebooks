#!/usr/bin/env python3
"""Generate index-level paper entities for RCL Awesome World-Action Models.

Parses ``data/papers.json`` (same catalog as ``docs/PAPERS.md``), dedupes by
arXiv against existing wiki pages, and writes:

  - sources/papers/rcl_awesome_wam_catalog.md
  - sources/papers/rcl_awesome_wam_{arxiv|slug}.md  (missing only)
  - wiki/entities/paper-rcl-*.md                    (missing only)
  - wiki/overview/rcl-awesome-wam-technology-map.md

Idempotent: re-running skips existing paper-rcl-* files and never creates
duplicate frontmatter arxiv IDs.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections import defaultdict
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TODAY = date.today().isoformat()
CACHE_DIR = Path("/tmp/rcl-awesome")

LIST_META = {
    "repo": "awesome-world-action-models-rcl",
    "title": "Awesome World-Action Models (RCL)",
    "url": "https://github.com/rcl-robotics/Awesome-World-Action-Models",
    "site": "https://rcl-robotics.github.io/Awesome-World-Action-Models/",
    "papers_md": "https://github.com/RCL-Robotics/Awesome-World-Action-Models/blob/main/docs/PAPERS.md",
    "entity": "wiki/entities/awesome-world-action-models-rcl.md",
    "hub_methods": [
        "../methods/generative-world-models.md",
        "../methods/vla.md",
        "../methods/model-based-rl.md",
        "../concepts/world-action-models.md",
    ],
    "hub_tasks": ["../tasks/manipulation.md", "../tasks/locomotion.md"],
    "abbrev": [
        ("WAM", "World Action Model", "世界预测与动作生成耦合"),
        ("VLA", "Vision-Language-Action", "视觉–语言–动作策略"),
        ("IDM", "Inverse Dynamics Model", "先预测未来再反推动作"),
        ("WM", "World Model", "环境前向预测模型"),
    ],
    "tag": "awesome-world-action-models-rcl",
}

MAJOR_EN = {
    "奠基性工作": "Foundational work",
    "VLA": "VLA",
    "WAM": "WAMs",
    "数据集": "Datasets",
    "评估指标（Metrics）": "Evaluation metrics",
    "评测基准与模拟器": "Benchmarks & simulators",
    "WAM Components": "Components of WAMs",
    "Related resources": "Related resources",
    None: "Major category not recorded",
}

ARXIV_RE = re.compile(
    r"(?:arxiv\.org/(?:abs|pdf)/|arxiv:\s*)(\d{4}\.\d{4,5})(?:v\d+)?",
    re.I,
)


def _slugify(title: str, max_len: int = 48) -> str:
    s = unicodedata.normalize("NFKD", title)
    s = s.encode("ascii", "ignore").decode("ascii")
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")
    if not s:
        s = "untitled"
    return s[:max_len].strip("-")


def _yaml_escape(s: str) -> str:
    s = s.replace("\\", "\\\\").replace('"', "'").replace("\n", " ").strip()
    s = re.sub(r"https?://\S+", "", s)
    s = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", s)
    s = re.sub(r"\s+", " ", s).strip(" -|")
    return s


def _yaml_list(items: list[str], indent: int = 2) -> str:
    pad = " " * indent
    return "\n".join(f"{pad}- {x}" for x in items)


def _extract_arxiv(paper: dict) -> str | None:
    pid = paper.get("id") or ""
    if re.match(r"^\d{4}\.\d{4,5}$", pid):
        return pid
    blob = json.dumps(paper, ensure_ascii=False)
    m = ARXIV_RE.search(blob)
    return m.group(1) if m else None


def _existing_arxiv_map() -> dict[str, str]:
    existing: dict[str, str] = {}
    entities = sorted((ROOT / "wiki" / "entities").glob("*.md"))
    for p in entities:
        if not p.name.startswith("paper-"):
            continue
        text = p.read_text(encoding="utf-8", errors="ignore")
        m = re.search(r'(?m)^arxiv:\s*["\']?(\d{4}\.\d{4,5})', text)
        if m:
            existing.setdefault(m.group(1), str(p.relative_to(ROOT)))
    for p in entities:
        if p.name.startswith("paper-"):
            continue
        text = p.read_text(encoding="utf-8", errors="ignore")
        m = re.search(r'(?m)^arxiv:\s*["\']?(\d{4}\.\d{4,5})', text)
        if m:
            existing.setdefault(m.group(1), str(p.relative_to(ROOT)))
    return existing


def parse_papers_json(path: Path) -> list[dict]:
    raw: list[dict] = json.loads(path.read_text(encoding="utf-8"))
    entries: list[dict] = []
    for paper in raw:
        title = (paper.get("title") or "").strip()
        if not title:
            continue
        arxiv = _extract_arxiv(paper)
        major = paper.get("majorCategory")
        section = MAJOR_EN.get(major, major or "Uncategorized")
        subs = paper.get("subcategories") or []
        sub_en = " · ".join(subs) if subs else ""
        quadrant = paper.get("quadrant") or ""
        meta_bits = " · ".join(x for x in [sub_en, quadrant] if x)
        contribution = (paper.get("contribution") or "").strip()
        if not contribution:
            contribution = (
                f"RCL Awesome WAM 清单收录（{section}）；细节以原文 PDF / 项目页为准。"
            )
        if len(contribution) > 320:
            contribution = contribution[:317].rstrip() + "..."

        code_urls = paper.get("codeUrls") or []
        code = code_urls[0] if code_urls else None
        website = paper.get("projectUrl") or None
        paper_url = paper.get("paperUrl") or paper.get("arxivUrl")
        if not paper_url and arxiv:
            paper_url = f"https://arxiv.org/abs/{arxiv}"
        venue = paper.get("venue") or ""
        if not venue and paper.get("publicationYear"):
            venue = str(paper.get("publicationYear"))

        entries.append(
            {
                "title": title,
                "section": section,
                "arxiv": arxiv,
                "paper_id": paper.get("id"),
                "publication": venue,
                "highlights": contribution,
                "code": code,
                "website": website,
                "paper_url": paper_url,
                "submitted": paper.get("submittedDate") or "",
                "meta_bits": meta_bits,
            }
        )
    return entries


def entity_filename(e: dict, idx: int) -> str:
    slug = _slugify(e["title"])
    if e["arxiv"]:
        a = e["arxiv"].replace(".", "-")
        return f"paper-rcl-{a}-{slug}.md"
    pid = (e.get("paper_id") or f"idx{idx}").replace(".", "-")
    pid = re.sub(r"[^a-zA-Z0-9-]", "-", pid)[:24]
    return f"paper-rcl-{pid}-{slug}.md"


def source_filename(e: dict, idx: int) -> str:
    slug = _slugify(e["title"], 40)
    if e["arxiv"]:
        a = e["arxiv"].replace(".", "_")
        return f"rcl_awesome_wam_{a}_{slug}.md"
    pid = re.sub(r"[^a-zA-Z0-9_]", "_", (e.get("paper_id") or f"noarxiv_{idx:04d}"))[:32]
    return f"rcl_awesome_wam_{pid}_{slug}.md"


def wiki_rel_from_root(path: str) -> str:
    assert path.startswith("wiki/")
    return "../" + path[len("wiki/") :]


def render_source(e: dict, wiki_rel: str, idx: int, total: int) -> str:
    arxiv_line = f"- **arXiv：** {e['arxiv']}" if e["arxiv"] else "- **arXiv：** （无 / 非 arXiv）"
    code_line = f"- **代码：** <{e['code']}>" if e["code"] else "- **代码：** 未在清单中标注"
    web_line = f"- **项目页：** <{e['website']}>" if e["website"] else ""
    paper_line = f"- **论文：** <{e['paper_url']}>" if e["paper_url"] else ""
    meta_line = f"- **子类 / 象限：** {e['meta_bits']}" if e.get("meta_bits") else ""
    return f"""# {e["title"]}

> 来源归档（RCL Awesome World-Action Models · ``docs/PAPERS.md`` / ``data/papers.json``）

- **列表：** [{LIST_META["title"]}]({LIST_META["url"]})
- **PAPERS.md：** <{LIST_META["papers_md"]}>
- **分组：** {e["section"]}
- **编号：** {idx:03d}/{total:03d}
- **入库日期：** {TODAY}
{arxiv_line}
- **出处：** {e["publication"] or "见清单"}
- **提交日：** {e.get("submitted") or "—"}
{paper_line}
{code_line}
{web_line}
{meta_line}
- **Contribution（清单）：** {e["highlights"]}
- **沉淀到 wiki：** [`{wiki_rel}`](../../{wiki_rel})

---

## 开源边界（步骤 2.5）

| 已发布 | 备注 |
|--------|------|
| 清单条目元数据 | 本 source 为策展摘录，非全文转存 |
| 代码/权重 | 以项目页 / GitHub 实际链接为准；清单标注见上 |

## 对 wiki 的映射

- 实体页：[`{wiki_rel}`](../../{wiki_rel})
- 列表实体：[`{LIST_META["entity"]}`](../../{LIST_META["entity"]})
"""


def render_entity(
    e: dict,
    src_rel: str,
    idx: int,
    total: int,
    tech_map_rel: str,
) -> str:
    short = e["title"].split(":")[0].strip() if ":" in e["title"] else e["title"]
    if len(short) > 80:
        short = short[:77] + "..."
    summary = _yaml_escape(e["highlights"][:220])
    tags = ["paper", "curated-index", LIST_META["tag"], "rcl-wam-catalog"]
    related = [
        wiki_rel_from_root(LIST_META["entity"]),
        tech_map_rel,
        *LIST_META["hub_methods"][:2],
        *LIST_META["hub_tasks"][:2],
    ]
    seen: set[str] = set()
    related_u = []
    for r in related:
        if r not in seen:
            seen.add(r)
            related_u.append(r)

    fm_extra = []
    if e["arxiv"]:
        fm_extra.append(f'arxiv: "{e["arxiv"]}"')
    if e["publication"]:
        fm_extra.append(f'venue: "{_yaml_escape(e["publication"][:80])}"')
    if e["code"]:
        fm_extra.append(f"code: {e['code']}")
    fm_extra_s = ("\n".join(fm_extra) + "\n") if fm_extra else ""

    abbrev_rows = "\n".join(f"| {a} | {b} | {c} |" for a, b, c in LIST_META["abbrev"][:4])
    paper_link = e["paper_url"] or (
        f"https://arxiv.org/abs/{e['arxiv']}" if e["arxiv"] else LIST_META["site"]
    )
    code_row = (
        f"\n| 代码/项目 | <{e['code']}> |"
        if e["code"]
        else ("\n| 项目页 | <" + e["website"] + "> |" if e["website"] else "")
    )
    hl = e["highlights"]
    method_name = LIST_META["hub_methods"][0].rsplit("/", 1)[-1]
    task_name = LIST_META["hub_tasks"][0].rsplit("/", 1)[-1]
    meta_row = f"\n| 子类 / 象限 | {e.get('meta_bits') or '—'} |" if e.get("meta_bits") else ""

    return f"""---
type: entity
tags: [{", ".join(tags)}]
status: complete
updated: {TODAY}
{fm_extra_s}summary: "{summary}"
related:
{_yaml_list(related_u)}
sources:
  - ../../{src_rel}
  - ../../sources/papers/rcl_awesome_wam_catalog.md
  - ../../sources/repos/{LIST_META["repo"]}.md
---

# {short}

**{e["title"]}** 收录于 [{LIST_META["title"]}]({LIST_META["url"]}) **第 {idx:03d}/{total:03d}** 篇，分组 **{e["section"]}**。本页为知识库 **策展索引级** 详情节点；方法细节与量化指标以原文 PDF / 项目页为准。

## 一句话定义

{hl}

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
{abbrev_rows}

## 为什么重要

- {hl}
- 在 [RCL Awesome WAM 技术地图]({tech_map_rel}) 中提供可点击的独立详情节点，避免清单条目无法落入知识图谱。
- 与列表实体 [{LIST_META["title"].split("(")[0].strip()}]({wiki_rel_from_root(LIST_META["entity"])}) 及站内 WAM / VLA 方法页交叉，便于从策展索引跳转到学习主线。

## 核心信息（索引级）

| 字段 | 内容 |
|------|------|
| 编号 | {idx:03d}/{total:03d} |
| 分组 | {e["section"]} |
| 出处 | {e["publication"] or "见清单 / 原文"} |
| 论文 | <{paper_link}> |{code_row}{meta_row}

## 核心机制（归纳）

### 策展导读要点

{hl}

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内 [World Action Models（WAM）](../concepts/world-action-models.md) 等概念页。

## 评测与指标（索引级）

- 本条目为 RCL Awesome **索引级** 摘录，**未搬运** 原文量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页]({paper_link}) 为准。
- 横向对照请回到 [技术地图]({tech_map_rel}) 同分组条目。

## 与其他工作对比（索引级）

- 本页 **不做** 与具体基线的逐项数值对比：索引级节点只保留清单坐标，同分组横向对照请回到 [技术地图]({tech_map_rel}) 的 **{e["section"]}** 分组逐条展开。
- 与站内 **深度论文实体** 的分界：深度页承载机构、实验表与源码运行时序；本页只承载清单 Contribution 阅读锚点。同一 arXiv 若已存在深度页，应以深度页为准。
- 与清单内相邻条目孰优孰劣，本页不下结论：清单 Contribution 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**本条目的站内价值是把「{short}」从 RCL Awesome WAM 列表提升为可链接的知识节点，并保留清单 Contribution 作为阅读锚点。**

- 起作用的是策展坐标：列表分组 **{e["section"]}** + Contribution 指出的问题设定，而不是本页自行推导的新算法结论。
- 适用边界：索引级页面不能替代 PDF；开源状态以项目页实际链接为准（清单可能滞后）。
- 若该工作成为学习主线，应再升格为深度论文实体（补机构、实验表、源码运行时序图或「不适用」说明）。

## 常见误区

1. 不要把 Awesome 条目的 Contribution 当成完整方法证明——它只是策展导读。
2. 同一 arXiv 在全库只允许一个 canonical 详情节点；若已有深度页，应以深度页为准。

## 关联页面

- 列表实体：[Awesome World-Action Models（RCL）]({wiki_rel_from_root(LIST_META["entity"])})
- 技术地图：[RCL Awesome WAM 技术地图]({tech_map_rel})
- 方法/任务：[{method_name}]({LIST_META["hub_methods"][0]})、[{task_name}]({LIST_META["hub_tasks"][0]})

## 参考来源

- [`{src_rel}`](../../{src_rel}) — 本条目策展摘录
- [`sources/papers/rcl_awesome_wam_catalog.md`](../../sources/papers/rcl_awesome_wam_catalog.md) — 列表总表
- [`sources/repos/{LIST_META["repo"]}.md`](../../sources/repos/{LIST_META["repo"]}.md)
- [`docs/PAPERS.md`]({LIST_META["papers_md"]}) — 上游论文目录
- 论文：<{paper_link}>

## 推荐继续阅读

- [{LIST_META["title"]} 仓库]({LIST_META["url"]})
- [原文]({paper_link})
"""


def render_catalog(rows: list[dict]) -> str:
    lines = [
        f"# {LIST_META['title']} 论文目录（RCL · PAPERS.md）",
        "",
        f"> 由 `{LIST_META['papers_md']}` / `data/papers.json` 解析生成；入库日 {TODAY}。",
        "",
        f"- **列表实体：** [`{LIST_META['entity']}`](../../{LIST_META['entity']})",
        "- **技术地图：** [`wiki/overview/rcl-awesome-wam-technology-map.md`](../../wiki/overview/rcl-awesome-wam-technology-map.md)",
        f"- **条目数：** {len(rows)}",
        "",
        "| # | 标题 | arXiv | 分组 | wiki |",
        "|---|------|-------|------|------|",
    ]
    for r in rows:
        aid = r["arxiv"] or "—"
        wiki = r["wiki_rel"]
        title = r["title"].replace("|", "/")[:80]
        lines.append(
            f"| {r['idx']:03d} | {title} | {aid} | {r['section'][:40]} | [`{Path(wiki).name}`](../../{wiki}) |"
        )
    lines.append("")
    return "\n".join(lines)


def render_tech_map(rows: list[dict]) -> str:
    by_sec: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_sec[r["section"]].append(r)

    abbrev_rows = "\n".join(f"| {a} | {b} | {c} |" for a, b, c in LIST_META["abbrev"][:4])
    sections_md = []
    for sec, items in by_sec.items():
        sections_md.append(f"### {sec}\n")
        sections_md.append("| # | 论文 | 详情节点 |")
        sections_md.append("|---|------|----------|")
        for r in items:
            link = wiki_rel_from_root(r["wiki_rel"])
            title = r["title"].replace("|", "/")
            sections_md.append(
                f"| {r['idx']:03d} | {title[:90]} | [{Path(r['wiki_rel']).stem}]({link}) |"
            )
        sections_md.append("")

    return f"""---
type: overview
tags: [overview, curated-index, {LIST_META["tag"]}, rcl-wam-catalog, technology-map]
status: complete
updated: {TODAY}
summary: "RCL Awesome World-Action Models 技术地图：把 PAPERS.md 里的 {len(rows)} 条文献逐条拆成站内可点开的一页，按八大类浏览。"
related:
  - {wiki_rel_from_root(LIST_META["entity"])}
  - {LIST_META["hub_methods"][0]}
  - {LIST_META["hub_tasks"][0]}
sources:
  - ../../sources/papers/rcl_awesome_wam_catalog.md
  - ../../sources/repos/{LIST_META["repo"]}.md
---

# RCL Awesome World-Action Models 技术地图

> 本页把 [Awesome World-Action Models]({LIST_META["url"]}) 的 [`docs/PAPERS.md`]({LIST_META["papers_md"]}) 清单里的论文逐条拆成站内可点开的一页，方便按 **Foundational / VLA / WAMs / Datasets / …** 分组浏览、搜索，并顺着链接读同方向的工作。

## 一句话定义

**RCL Awesome WAM 技术地图** = 外部 PAPERS.md 的站内可点开版本（564 条、按 major category 分组，一点即达论文页）。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
{abbrev_rows}

## 为什么重要

- 原清单每条只有标题 + 链接 + 子类标签；这里逐条给出一页，可检索、可顺着相关内容继续读。
- 站内已有深读页的条目直接链过去；其余给出 **清单摘要页**：Contribution、原文链接与它在清单里的位置一页可见。
- 清单共 **{len(rows)}** 条，每条都有独立 detail 节点（arXiv 去重后链到 canonical 页）。

## 覆盖范围

| 项 | 值 |
|----|-----|
| 上游仓库 | <{LIST_META["url"]}> |
| PAPERS.md | <{LIST_META["papers_md"]}> |
| 列表实体 | [Awesome World-Action Models（RCL）]({wiki_rel_from_root(LIST_META["entity"])}) |
| 目录 source | [rcl_awesome_wam_catalog.md](../../sources/papers/rcl_awesome_wam_catalog.md) |

## 分组索引

{chr(10).join(sections_md)}

## 局限与风险

- 清单摘要页只给 Contribution 要点，**不替代** 原文；要深读请从论文链接进。
- 清单里混有非 arXiv 链接（OpenReview / IEEE / DOI），这类条目按标题收录；若同一工作另有 arXiv 深度页，catalog 会链到 canonical 节点。
- 上游清单仍在更新，本页是 {TODAY} 的快照；最新条目以上游 `docs/PAPERS.md` 为准。

## 关联页面

- [Awesome World-Action Models（RCL）]({wiki_rel_from_root(LIST_META["entity"])})
- [World Action Models（WAM）](../concepts/world-action-models.md)
- [VLA](../methods/vla.md)

## 参考来源

- [rcl_awesome_wam_catalog.md](../../sources/papers/rcl_awesome_wam_catalog.md)
- [sources/repos/{LIST_META["repo"]}.md](../../sources/repos/{LIST_META["repo"]}.md)
- 上游：[docs/PAPERS.md]({LIST_META["papers_md"]})

## 推荐继续阅读

- [Awesome World-Action Models GitHub]({LIST_META["url"]})
- [Paper library（站点）]({LIST_META["site"]}papers/)
"""


def main() -> None:
    papers_path = CACHE_DIR / "papers.json"
    if not papers_path.exists():
        raise SystemExit(f"missing cached papers.json: {papers_path}")

    existing = _existing_arxiv_map()
    for p in (ROOT / "wiki" / "entities").glob("paper-rcl-*.md"):
        text = p.read_text(encoding="utf-8", errors="ignore")
        m = re.search(r'(?m)^arxiv:\s*["\']?(\d{4}\.\d{4,5})', text)
        if m:
            existing.setdefault(m.group(1), str(p.relative_to(ROOT)))

    entries = parse_papers_json(papers_path)
    entries.sort(key=lambda e: (e["section"], e["title"]))

    rows: list[dict] = []
    created_entities = 0
    created_sources = 0

    for i, e in enumerate(entries, start=1):
        created = False
        if e["arxiv"] and e["arxiv"] in existing:
            wiki_rel = existing[e["arxiv"]]
        else:
            ent_name = entity_filename(e, i)
            wiki_rel = f"wiki/entities/{ent_name}"
            src_name = source_filename(e, i)
            src_rel = f"sources/papers/{src_name}"
            ent_path = ROOT / wiki_rel
            src_path = ROOT / src_rel
            tech_map_rel = "../overview/rcl-awesome-wam-technology-map.md"

            if not src_path.exists():
                src_path.write_text(
                    render_source(e, wiki_rel, i, len(entries)),
                    encoding="utf-8",
                )
                created_sources += 1
            if not ent_path.exists():
                body = render_entity(e, src_rel, i, len(entries), tech_map_rel)
                ent_path.write_text(body, encoding="utf-8")
                created_entities += 1
                created = True
            if e["arxiv"]:
                existing[e["arxiv"]] = wiki_rel

        rows.append({**e, "idx": i, "wiki_rel": wiki_rel, "created": created})

    cat_path = ROOT / "sources/papers/rcl_awesome_wam_catalog.md"
    cat_path.write_text(render_catalog(rows), encoding="utf-8")
    map_path = ROOT / "wiki/overview/rcl-awesome-wam-technology-map.md"
    map_path.write_text(render_tech_map(rows), encoding="utf-8")

    # Upstream PAPERS.md snapshot pointer (metadata only; full table lives in catalog)
    snap = CACHE_DIR / "PAPERS.md"
    upstream_src = ROOT / "sources/papers/rcl_awesome_wam_papers_md_upstream.md"
    if snap.exists() and not upstream_src.exists():
        head = snap.read_text(encoding="utf-8", errors="ignore")[:8000]
        upstream_src.write_text(
            f"""# RCL Awesome WAM · docs/PAPERS.md（上游摘录头）

> 完整 564 条表格见上游仓库 [`docs/PAPERS.md`]({LIST_META["papers_md"]}) 与本站 [`rcl_awesome_wam_catalog.md`](rcl_awesome_wam_catalog.md)。

- **入库日期：** {TODAY}
- **条目数：** {len(entries)}

---

{head}

…（截断；完整内容请访问上游链接）
""",
            encoding="utf-8",
        )

    stats = {
        "entries": len(entries),
        "created_entities": created_entities,
        "created_sources": created_sources,
        "reused": sum(1 for r in rows if not r["created"]),
    }
    print(stats)
    (CACHE_DIR / "gen_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
