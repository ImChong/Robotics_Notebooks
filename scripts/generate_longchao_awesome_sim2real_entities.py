#!/usr/bin/env python3
"""Generate index-level paper entities for LongchaoDa/AwesomeSim2Real.

Parses the curated README (numbered-list format), dedupes by arXiv against
existing wiki pages, and writes:
  - sources/papers/lc_awesome_sim2real_catalog.md
  - sources/papers/lc_awesome_sim2real_{arxiv|slug}.md  (missing only)
  - wiki/entities/paper-as-*.md                         (missing only)
  - wiki/overview/lc-awesome-sim2real-technology-map.md

Idempotent: re-running skips existing paper-as-* files and never creates
duplicate frontmatter arxiv IDs.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
TODAY = date.today().isoformat()
CACHE = Path("/tmp/awesome-sim2real-readme.md")

LIST_KEY = "sim2real"
LIST_META: dict[str, Any] = {
    "repo": "AwesomeSim2Real",
    "title": "AwesomeSim2Real",
    "url": "https://github.com/LongchaoDa/AwesomeSim2Real",
    "entity": "wiki/entities/awesome-sim2real.md",
    "hub_methods": [
        "../methods/reinforcement-learning.md",
        "../concepts/domain-randomization.md",
        "../concepts/sim2real.md",
    ],
    "hub_tasks": ["../tasks/locomotion.md", "../tasks/manipulation.md"],
    "abbrev": [
        ("Sim2Real", "Simulation to Real", "仿真策略迁移到真机"),
        ("MDP", "Markov Decision Process", "状态–动作–转移–奖励形式化"),
        ("DR", "Domain Randomization", "域随机化"),
        ("FM", "Foundation Model", "大模型/基础模型增强迁移"),
    ],
    "tag": "awesome-sim2real",
}

ARXIV_RE = re.compile(
    r"(?:arxiv\.org/(?:abs|pdf)/|arxiv:\s*)(\d{4}\.\d{4,5})(?:v\d+)?",
    re.I,
)
NUM_ENTRY_RE = re.compile(r"^(\d+)\.\s+\*\*(.+?)\*\*")


def _slugify(title: str, max_len: int = 48) -> str:
    s = unicodedata.normalize("NFKD", title)
    s = s.encode("ascii", "ignore").decode("ascii")
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")
    return (s or "untitled")[:max_len].strip("-")


def _yaml_escape(s: str) -> str:
    s = s.replace("\\", "\\\\").replace('"', "'").replace("\n", " ").strip()
    s = re.sub(r"https?://\S+", "", s)
    s = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", s)
    s = re.sub(r"\s+", " ", s).strip(" -|")
    return s


def _yaml_list(items: list[str], indent: int = 2) -> str:
    pad = " " * indent
    return "\n".join(f"{pad}- {x}" for x in items)


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


def parse_readme(path: Path) -> list[dict]:
    lines = path.read_text(encoding="utf-8").splitlines()
    entries: list[dict] = []
    cur_h = "ROOT"
    cur_sub = ""
    i = 0
    while i < len(lines):
        line = lines[i]
        hm = re.match(r"^(#{1,4})\s+(.+)$", line)
        if hm:
            cur_h = re.sub(r"[*#]+", "", hm.group(2)).strip()
            cur_sub = ""
            i += 1
            continue
        bm = re.match(r"^\*\*(.+?)\*\*\s*$", line)
        if bm and not line.startswith("1."):
            cur_sub = bm.group(1).strip()
            i += 1
            continue
        nm = NUM_ENTRY_RE.match(line)
        if not nm:
            i += 1
            continue
        title = nm.group(1).strip()
        blob_parts = [line]
        i += 1
        while i < len(lines):
            nxt = lines[i]
            if NUM_ENTRY_RE.match(nxt) or re.match(r"^#{1,4}\s+", nxt):
                break
            if re.match(r"^\*\*(.+?)\*\*\s*$", nxt) and not nxt.startswith("1."):
                break
            if nxt.strip() == "" and i + 1 < len(lines) and NUM_ENTRY_RE.match(lines[i + 1]):
                break
            blob_parts.append(nxt)
            i += 1
        blob = " ".join(p.strip() for p in blob_parts if p.strip())
        aids = ARXIV_RE.findall(blob)
        pub = ""
        pm = re.search(r"\*\s*([^*]+)\*\.\s*([^[]+)\.\s*\d{4}", blob)
        if pm:
            pub = re.sub(r"\s+", " ", pm.group(2)).strip().rstrip(".")
        code = None
        cm = re.search(r"github\.com/([^/\)\s]+/[^/\)\s\"']+)", blob, re.I)
        if cm:
            code = f"https://github.com/{cm.group(1).rstrip('.')}"
        paper_url = None
        lm = re.search(r"\[link\]\((https?://[^)]+)\)", blob, re.I)
        if lm:
            paper_url = lm.group(1).rstrip(").,;")
        elif aids:
            paper_url = f"https://arxiv.org/abs/{aids[0]}"
        if not paper_url:
            continue
        section = cur_h if not cur_sub else f"{cur_h} / {cur_sub}"
        highlights = (
            f"LongchaoDa AwesomeSim2Real 收录；分组 {section}。"
            " 本页为策展索引级节点，细节以原文为准。"
        )
        entries.append(
            {
                "title": title,
                "section": section,
                "arxiv": aids[0] if aids else None,
                "publication": pub,
                "highlights": highlights,
                "code": code,
                "paper_url": paper_url,
                "list": LIST_KEY,
            }
        )
    return entries


def entity_filename(e: dict, idx: int) -> str:
    slug = _slugify(e["title"])
    if e["arxiv"]:
        a = e["arxiv"].replace(".", "-")
        return f"paper-as-{a}-{slug}.md"
    return f"paper-as-{idx:03d}-{slug}.md"


def source_filename(e: dict, idx: int) -> str:
    slug = _slugify(e["title"], 40)
    if e["arxiv"]:
        a = e["arxiv"].replace(".", "_")
        return f"lc_awesome_sim2real_{a}_{slug}.md"
    return f"lc_awesome_sim2real_noarxiv_{idx:03d}_{slug}.md"


def wiki_rel_from_root(path: str) -> str:
    assert path.startswith("wiki/")
    return "../" + path[len("wiki/") :]


def render_source(e: dict, wiki_rel: str, idx: int, total: int) -> str:
    arxiv_line = f"- **arXiv：** {e['arxiv']}" if e["arxiv"] else "- **arXiv：** （无 / 非 arXiv）"
    code_line = f"- **代码：** <{e['code']}>" if e["code"] else "- **代码：** 未在清单中标注"
    paper_line = f"- **论文：** <{e['paper_url']}>" if e["paper_url"] else ""
    return f"""# {e["title"]}

> 来源归档（LongchaoDa AwesomeSim2Real 策展索引级）

- **列表：** [{LIST_META["title"]}]({LIST_META["url"]})
- **分组：** {e["section"]}
- **编号：** {idx:03d}/{total:03d}
- **入库日期：** {TODAY}
{arxiv_line}
- **出处：** {e["publication"] or "见清单"}
{paper_line}
{code_line}
- **Highlights（清单）：** {e["highlights"]}
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


def render_entity(e: dict, src_rel: str, idx: int, total: int, tech_map_rel: str) -> str:
    short = e["title"].split(":")[0].strip() if ":" in e["title"] else e["title"]
    if len(short) > 80:
        short = short[:77] + "..."
    summary = _yaml_escape(e["highlights"][:220])
    tags = ["paper", "curated-index", LIST_META["tag"], "longchao-sim2real"]
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
        f"https://arxiv.org/abs/{e['arxiv']}" if e["arxiv"] else LIST_META["url"]
    )
    code_row = f"\n| 代码/项目 | <{e['code']}> |" if e["code"] else ""
    hl = e["highlights"]
    method_name = LIST_META["hub_methods"][0].rsplit("/", 1)[-1]
    task_name = LIST_META["hub_tasks"][0].rsplit("/", 1)[-1]

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
  - ../../sources/papers/lc_awesome_sim2real_catalog.md
  - ../../sources/repos/awesome-sim2real.md
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
- 在 [{LIST_META["title"]} 技术地图]({tech_map_rel}) 中提供可点击的独立详情节点，避免清单条目无法落入知识图谱。
- 与列表实体 [{LIST_META["title"]}]({wiki_rel_from_root(LIST_META["entity"])}) 及站内 Sim2Real 方法/任务页交叉，便于从策展索引跳转到学习主线。

## 核心信息（索引级）

| 字段 | 内容 |
|------|------|
| 编号 | {idx:03d}/{total:03d} |
| 分组 | {e["section"]} |
| 出处 | {e["publication"] or "见清单 / 原文"} |
| 论文 | <{paper_link}> |{code_row}

## 核心机制（归纳）

### 策展导读要点

{hl}

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标（索引级）

- 本条目为 Awesome 策展 **索引级** 摘录，**未搬运** 原文量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页]({paper_link}) 为准。
- 横向对照请回到 [技术地图]({tech_map_rel}) 同分组条目。

## 与其他工作对比（索引级）

- 本页 **不做** 与具体基线的逐项数值对比：索引级节点只保留清单坐标，同分组横向对照请回到 [技术地图]({tech_map_rel}) 的 **{e["section"]}** 分组逐条展开。
- 与站内 **深度论文实体** 的分界：深度页承载机构、实验表与源码运行时序；本页只承载清单 Highlights 阅读锚点。同一 arXiv 若已存在深度页，应以深度页为准。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome 列表可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**本条目的站内价值是把「{short}」从外部 Awesome 列表提升为可链接的知识节点，并保留清单分组作为阅读锚点。**

- 可确证的是策展坐标：列表分组 **{e["section"]}**，而不是本页自行推导的新算法结论。
- 适用边界：索引级页面不能替代 PDF；开源状态以项目页实际链接为准（清单可能滞后）。
- 若该工作成为学习主线，应再升格为深度论文实体（补机构、实验表、源码运行时序图或「不适用」说明）。

## 常见误区

1. 不要把 Awesome 条目的分组标签当成完整方法证明——它只是策展导读。
2. 同一 arXiv 在全库只允许一个 canonical 详情节点；若已有深度页，应以深度页为准。

## 关联页面

- 列表实体：[{LIST_META["title"]}]({wiki_rel_from_root(LIST_META["entity"])})
- 技术地图：[{LIST_META["title"]} 技术地图]({tech_map_rel})
- 方法/任务：[{method_name}]({LIST_META["hub_methods"][0]})、[{task_name}]({LIST_META["hub_tasks"][0]})

## 参考来源

- [`{src_rel}`](../../{src_rel}) — 本条目策展摘录
- [`sources/papers/lc_awesome_sim2real_catalog.md`](../../sources/papers/lc_awesome_sim2real_catalog.md) — 列表总表
- [`sources/repos/awesome-sim2real.md`](../../sources/repos/awesome-sim2real.md)
- 论文：<{paper_link}>

## 推荐继续阅读

- [{LIST_META["title"]} 仓库]({LIST_META["url"]})
- [原文]({paper_link})
"""


def render_catalog(rows: list[dict]) -> str:
    lines = [
        f"# {LIST_META['title']} 论文目录（LongchaoDa）",
        "",
        f"> 由 `{LIST_META['url']}` 解析生成；入库日 {TODAY}。",
        "",
        f"- **列表实体：** [`{LIST_META['entity']}`](../../{LIST_META['entity']})",
        "- **技术地图：** [`wiki/overview/lc-awesome-sim2real-technology-map.md`](../../wiki/overview/lc-awesome-sim2real-technology-map.md)",
        "- **配套综述：** [arXiv:2502.13187](https://arxiv.org/abs/2502.13187v3)",
        f"- **条目数：** {len(rows)}",
        "",
        "| # | 标题 | arXiv | 分组 | wiki |",
        "|---|------|-------|------|------|",
    ]
    for r in rows:
        aid = r["arxiv"] or "—"
        wiki = r["wiki_rel"]
        lines.append(
            f"| {r['idx']:03d} | {r['title'][:80]} | {aid} | {r['section'][:40]} | [`{Path(wiki).name}`](../../{wiki}) |"
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

    new_count = sum(1 for r in rows if r.get("created"))
    reused = len(rows) - new_count

    return f"""---
type: overview
tags: [overview, curated-index, {LIST_META["tag"]}, longchao-sim2real, technology-map]
status: complete
updated: {TODAY}
summary: "{LIST_META["title"]} 技术地图：为清单内论文提供独立详情节点索引（新建 {new_count}，复用已有 {reused}）。"
related:
  - {wiki_rel_from_root(LIST_META["entity"])}
  - {LIST_META["hub_methods"][0]}
  - {LIST_META["hub_tasks"][0]}
  - ../entities/paper-survey-sim2real-rl-foundation-models.md
sources:
  - ../../sources/papers/lc_awesome_sim2real_catalog.md
  - ../../sources/repos/awesome-sim2real.md
---

# {LIST_META["title"]} 技术地图

> 本页把 [{LIST_META["title"]}]({LIST_META["url"]}) 清单中的论文条目映射为站内 **独立详情节点**（`wiki/entities/paper-as-*` 或已有 canonical 页），供图谱与 `detail.html` 检索。配套综述见 [Sim2Real RL Survey（2502.13187）](../entities/paper-survey-sim2real-rl-foundation-models.md)。

## 一句话定义

**{LIST_META["title"]} 技术地图** = LongchaoDa 维护的 Sim2Real RL 论文策展列表的站内节点化索引（按 MDP 四要素 + 领域分组浏览）。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
{abbrev_rows}

## 为什么重要

- Awesome 列表本身不是知识图谱节点；若不升格论文实体，首页/图谱无法挂上具体工作。
- 本地图 **优先复用** 库内已有 arXiv canonical 页，仅对缺失条目新建索引级 `paper-as-*` 节点。
- 统计：清单可解析条目 **{len(rows)}**（新建详情节点 **{new_count}**，复用已有 **{reused}**）。

## 覆盖范围

| 项 | 值 |
|----|-----|
| 上游仓库 | <{LIST_META["url"]}> |
| 列表实体 | [{LIST_META["title"]}]({wiki_rel_from_root(LIST_META["entity"])}) |
| 配套综述 | [paper-survey-sim2real-rl-foundation-models.md](../entities/paper-survey-sim2real-rl-foundation-models.md) |
| 目录 source | [lc_awesome_sim2real_catalog.md](../../sources/papers/lc_awesome_sim2real_catalog.md) |

## 分组索引

{chr(10).join(sections_md)}

## 局限与风险

- 索引级节点保留清单分组，**不替代** 深度论文页；主线工作应继续升格。
- 清单含非 arXiv 链接（IEEE / ResearchGate）；无 arXiv 条目以标题 slug 建节点，后续若补 arXiv 需合并去重。
- 上游更新后需重跑 `python3 scripts/generate_longchao_awesome_sim2real_entities.py` 再 `make ci-preflight`。

## 关联页面

- [{LIST_META["title"]}（列表实体）]({wiki_rel_from_root(LIST_META["entity"])})
- [Sim2Real RL Survey（2502.13187）](../entities/paper-survey-sim2real-rl-foundation-models.md)
- [{LIST_META["hub_methods"][0].split("/")[-1]}]({LIST_META["hub_methods"][0]})
- [{LIST_META["hub_tasks"][0].split("/")[-1]}]({LIST_META["hub_tasks"][0]})

## 参考来源

- [lc_awesome_sim2real_catalog.md](../../sources/papers/lc_awesome_sim2real_catalog.md)
- [sources/repos/awesome-sim2real.md](../../sources/repos/awesome-sim2real.md)
- 上游：<{LIST_META["url"]}>

## 推荐继续阅读

- [{LIST_META["title"]} GitHub]({LIST_META["url"]})
- [A Survey of Sim-to-Real Methods in RL（arXiv:2502.13187v3）](https://arxiv.org/abs/2502.13187v3)
"""


def main() -> None:
    if not CACHE.exists():
        raise SystemExit(f"missing cached README: {CACHE}")

    existing = _existing_arxiv_map()
    for p in (ROOT / "wiki" / "entities").glob("paper-as-*.md"):
        text = p.read_text(encoding="utf-8", errors="ignore")
        m = re.search(r'(?m)^arxiv:\s*["\']?(\d{4}\.\d{4,5})', text)
        if m:
            existing.setdefault(m.group(1), str(p.relative_to(ROOT)))

    entries = parse_readme(CACHE)
    entries.sort(key=lambda e: (e["section"], e["title"]))

    rows: list[dict] = []
    created_entities = 0
    created_sources = 0
    tech_map_rel = "../overview/lc-awesome-sim2real-technology-map.md"

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

            if not src_path.exists():
                src_path.write_text(render_source(e, wiki_rel, i, len(entries)), encoding="utf-8")
                created_sources += 1
            if not ent_path.exists():
                body = render_entity(e, src_rel, i, len(entries), tech_map_rel)
                ent_path.write_text(body, encoding="utf-8")
                created_entities += 1
                created = True
            if e["arxiv"]:
                existing[e["arxiv"]] = wiki_rel

        rows.append({**e, "idx": i, "wiki_rel": wiki_rel, "created": created})

    cat_path = ROOT / "sources/papers/lc_awesome_sim2real_catalog.md"
    cat_path.write_text(render_catalog(rows), encoding="utf-8")
    map_path = ROOT / "wiki/overview/lc-awesome-sim2real-technology-map.md"
    map_path.write_text(render_tech_map(rows), encoding="utf-8")

    stats = {
        "entries": len(entries),
        "created_entities": created_entities,
        "created_sources": created_sources,
        "reused": sum(1 for r in rows if not r["created"]),
    }
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
