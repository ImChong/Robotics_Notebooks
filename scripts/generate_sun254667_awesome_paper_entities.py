#!/usr/bin/env python3
"""Generate index-level paper entities for sun254667 awesome lists.

Parses the four curated READMEs, dedupes by arXiv against existing wiki pages,
and writes:
  - sources/papers/sun_awesome_{list}_catalog.md
  - sources/papers/sun_awesome_{list}_{arxiv|slug}.md  (missing only)
  - wiki/entities/paper-sa-*.md                         (missing only)
  - wiki/overview/sun-awesome-{list}-technology-map.md

Idempotent: re-running skips existing paper-sa-* files and never creates
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
CACHE_DIR = Path("/tmp/sun-awesome")

LISTS = {
    "wm": {
        "repo": "awesome-world-models",
        "title": "Awesome World Models",
        "url": "https://github.com/sun254667/awesome-world-models",
        "entity": "wiki/entities/awesome-world-models.md",
        "hub_methods": [
            "../methods/generative-world-models.md",
            "../methods/model-based-rl.md",
            "../methods/vla.md",
        ],
        "hub_tasks": ["../tasks/manipulation.md", "../tasks/locomotion.md"],
        "abbrev": [
            ("WM", "World Model", "环境前向预测模型"),
            ("WAM", "World Action Model", "世界预测与动作联合建模"),
            ("VLA", "Vision-Language-Action", "视觉–语言–动作策略"),
            ("MBRL", "Model-Based RL", "基于模型的强化学习"),
        ],
        "tag": "awesome-world-models",
    },
    "ego": {
        "repo": "awesome-egocentric-vision",
        "title": "Awesome Egocentric Vision",
        "url": "https://github.com/sun254667/awesome-egocentric-vision",
        "entity": "wiki/entities/awesome-egocentric-vision.md",
        "hub_methods": [
            "../methods/vla.md",
            "../methods/imitation-learning.md",
            "../methods/generative-world-models.md",
        ],
        "hub_tasks": ["../tasks/manipulation.md", "../tasks/teleoperation.md"],
        "abbrev": [
            ("Ego", "Egocentric Vision", "第一人称可穿戴视角感知"),
            ("HOI", "Hand–Object Interaction", "手–物交互理解"),
            ("VLA", "Vision-Language-Action", "视觉–语言–动作策略"),
            ("VLM", "Vision-Language Model", "视觉–语言模型"),
        ],
        "tag": "awesome-egocentric-vision",
    },
    "touch": {
        "repo": "awesome-touch",
        "title": "Awesome Touch",
        "url": "https://github.com/sun254667/awesome-touch",
        "entity": "wiki/entities/awesome-touch.md",
        "hub_methods": [
            "../methods/vla.md",
            "../methods/imitation-learning.md",
            "../methods/generative-world-models.md",
        ],
        "hub_tasks": ["../tasks/manipulation.md", "../tasks/bimanual-manipulation.md"],
        "abbrev": [
            ("VTLA", "Vision-Tactile-Language-Action", "视–触–语言–动作策略"),
            ("WM", "World Model", "视触觉前向预测"),
            ("WAM", "World Action Model", "世界–动作联合建模"),
            ("Sim2Real", "Simulation to Real", "仿真到真机迁移"),
        ],
        "tag": "awesome-touch",
    },
    "r2s2r": {
        "repo": "Awesome-Real2Sim2Real",
        "title": "Awesome-Real2Sim2Real",
        "url": "https://github.com/sun254667/Awesome-Real2Sim2Real",
        "entity": "wiki/entities/awesome-real2sim2real.md",
        "hub_methods": [
            "../methods/reinforcement-learning.md",
            "../methods/crisp-real2sim.md",
            "../methods/vla.md",
        ],
        "hub_tasks": ["../tasks/locomotion.md", "../tasks/manipulation.md"],
        "abbrev": [
            ("Sim2Real", "Simulation to Real", "仿真策略迁移到真机"),
            ("Real2Sim", "Real to Simulation", "真机数据重建/校准仿真"),
            ("R2S2R", "Real2Sim2Real", "真机→仿真→真机闭环"),
            ("DR", "Domain Randomization", "域随机化"),
        ],
        "tag": "awesome-real2sim2real",
    },
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
    # Prevent truncated markdown / URLs from poisoning YAML double-quoted scalars
    s = re.sub(r"https?://\S+", "", s)
    s = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", s)
    s = re.sub(r"\s+", " ", s).strip(" -|")
    return s


def _yaml_list(items: list[str], indent: int = 2) -> str:
    pad = " " * indent
    return "\n".join(f"{pad}- {x}" for x in items)


def _existing_arxiv_map() -> dict[str, str]:
    """arxiv_id -> canonical entity path.

    Prefer ``wiki/entities/paper-*.md``, then any other ``wiki/entities/*.md``
    with a frontmatter ``arxiv:`` field (e.g. ``cosmos-3.md``). Ignore body
    mentions on concept/overview pages.
    """
    existing: dict[str, str] = {}
    entities = sorted((ROOT / "wiki" / "entities").glob("*.md"))
    # Pass 1: paper-* first so they win on collision
    for p in entities:
        if not p.name.startswith("paper-"):
            continue
        text = p.read_text(encoding="utf-8", errors="ignore")
        m = re.search(r'(?m)^arxiv:\s*["\']?(\d{4}\.\d{4,5})', text)
        if m:
            existing.setdefault(m.group(1), str(p.relative_to(ROOT)))
    # Pass 2: other entity pages with frontmatter arxiv
    for p in entities:
        if p.name.startswith("paper-"):
            continue
        text = p.read_text(encoding="utf-8", errors="ignore")
        m = re.search(r'(?m)^arxiv:\s*["\']?(\d{4}\.\d{4,5})', text)
        if m:
            existing.setdefault(m.group(1), str(p.relative_to(ROOT)))
    return existing


def parse_readme(path: Path, list_key: str) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    entries: list[dict] = []
    cur_sec = "ROOT"
    i = 0
    while i < len(lines):
        line = lines[i]
        hm = re.match(r"^(#{2,4})\s+(.+)$", line)
        if hm:
            cur_sec = re.sub(r"[^\w\s\-&/]+", "", hm.group(2)).strip()
            i += 1
            continue
        tm = re.match(r"^[\-\+]\s+\*\*(.+?)\*\*\s*$", line)
        if not tm:
            i += 1
            continue
        title = tm.group(1).strip()
        block: list[str] = []
        i += 1
        while i < len(lines):
            nxt = lines[i]
            if re.match(r"^(#{2,4})\s+", nxt) or re.match(r"^[\-\+]\s+\*\*", nxt):
                break
            if nxt.startswith("|") and not block:
                break
            block.append(nxt)
            i += 1
            if len(block) > 14:
                break
        blob = "\n".join(block)
        aids = ARXIV_RE.findall(blob)
        pub = None
        pm = re.search(r"(?m)^\s*[-*]?\s*Publication:\s*(.+)$", blob)
        if pm:
            pub = pm.group(1).strip()
        highlights = None
        # Only the Highlights line — never swallow following Paper Link / Code lines
        hm2 = re.search(r"(?m)^\s*[-*]?\s*Highlights:\s*(.+)$", blob)
        if hm2:
            highlights = re.sub(r"\s+", " ", hm2.group(1)).strip()
            # Drop trailing markdown links accidentally glued on the same line
            highlights = re.sub(r"\s*[-|]?\s*Paper Link:.*$", "", highlights).strip()
            highlights = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", highlights).strip()
            if len(highlights) > 320:
                highlights = highlights[:317].rstrip() + "..."
        code = None
        cm = re.search(r"\[Code\]\((https?://[^)]+)\)", blob)
        if cm:
            code = cm.group(1).rstrip(").,;")
        web = None
        wm = re.search(
            r"\[(?:Website|Project|Official Website|OpenReview|IEEE Xplore|DOI)\]\((https?://[^)]+)\)",
            blob,
            re.I,
        )
        if wm:
            web = wm.group(1).rstrip(").,;")
        paper_url = None
        # prefer arxiv abs
        if aids:
            paper_url = f"https://arxiv.org/abs/{aids[0]}"
        else:
            pl = re.search(r"(?m)^\s*[-*]?\s*Paper Link:\s*(.+)$", blob)
            if pl:
                lm = re.search(r"\((https?://[^)]+)\)", pl.group(1))
                if lm:
                    paper_url = lm.group(1).rstrip(").,;")
        if not (aids or paper_url or "Paper Link" in blob):
            continue
        entries.append(
            {
                "title": title,
                "section": cur_sec,
                "arxiv": aids[0] if aids else None,
                "publication": pub or "",
                "highlights": highlights or "策展清单收录条目；细节以原文 PDF / 项目页为准。",
                "code": code,
                "website": web,
                "paper_url": paper_url,
                "list": list_key,
            }
        )
    return entries


def entity_filename(e: dict, idx: int) -> str:
    slug = _slugify(e["title"])
    if e["arxiv"]:
        a = e["arxiv"].replace(".", "-")
        return f"paper-sa-{a}-{slug}.md"
    return f"paper-sa-{e['list']}-{idx:03d}-{slug}.md"


def source_filename(e: dict, idx: int) -> str:
    slug = _slugify(e["title"], 40)
    if e["arxiv"]:
        a = e["arxiv"].replace(".", "_")
        return f"sun_awesome_{e['list']}_{a}_{slug}.md"
    return f"sun_awesome_{e['list']}_noarxiv_{idx:03d}_{slug}.md"


def wiki_rel_from_root(path: str) -> str:
    """wiki/entities/foo.md -> ../entities/foo.md"""
    assert path.startswith("wiki/")
    return "../" + path[len("wiki/") :]


def render_source(e: dict, wiki_rel: str, list_meta: dict, idx: int, total: int) -> str:
    arxiv_line = f"- **arXiv：** {e['arxiv']}" if e["arxiv"] else "- **arXiv：** （无 / 非 arXiv）"
    code_line = f"- **代码：** <{e['code']}>" if e["code"] else "- **代码：** 未在清单中标注"
    web_line = f"- **项目页：** <{e['website']}>" if e["website"] else ""
    paper_line = f"- **论文：** <{e['paper_url']}>" if e["paper_url"] else ""
    return f"""# {e["title"]}

> 来源归档（sun254667 Awesome 策展索引级）

- **列表：** [{list_meta["title"]}]({list_meta["url"]})
- **分组：** {e["section"]}
- **编号：** {idx:03d}/{total:03d}
- **入库日期：** {TODAY}
{arxiv_line}
- **出处：** {e["publication"] or "见清单"}
{paper_line}
{code_line}
{web_line}
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
- 列表实体：[`{list_meta["entity"]}`](../../{list_meta["entity"]})
"""


def _repo_source_name(list_key: str, list_meta: dict) -> str:
    if list_key == "r2s2r":
        return "awesome-real2sim2real"
    return list_meta["repo"]


def render_entity(
    e: dict,
    src_rel: str,
    list_meta: dict,
    idx: int,
    total: int,
    tech_map_rel: str,
) -> str:
    short = e["title"].split(":")[0].strip() if ":" in e["title"] else e["title"]
    if len(short) > 80:
        short = short[:77] + "..."
    summary = _yaml_escape(e["highlights"][:220])
    tags = ["paper", "curated-index", list_meta["tag"], f"sun254667-{e['list']}"]
    related = [
        wiki_rel_from_root(list_meta["entity"]),
        tech_map_rel,
        *list_meta["hub_methods"][:2],
        *list_meta["hub_tasks"][:2],
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

    abbrev_rows = "\n".join(f"| {a} | {b} | {c} |" for a, b, c in list_meta["abbrev"][:4])
    paper_link = e["paper_url"] or (
        f"https://arxiv.org/abs/{e['arxiv']}" if e["arxiv"] else list_meta["url"]
    )
    code_row = (
        f"\n| 代码/项目 | <{e['code']}> |"
        if e["code"]
        else ("\n| 项目页 | <" + e["website"] + "> |" if e["website"] else "")
    )
    hl = e["highlights"]
    repo_src = _repo_source_name(e["list"], list_meta)
    method_name = list_meta["hub_methods"][0].rsplit("/", 1)[-1]
    task_name = list_meta["hub_tasks"][0].rsplit("/", 1)[-1]

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
  - ../../sources/papers/sun_awesome_{e["list"]}_catalog.md
  - ../../sources/repos/{repo_src}.md
---

# {short}

**{e["title"]}** 收录于 [{list_meta["title"]}]({list_meta["url"]}) **第 {idx:03d}/{total:03d}** 篇，分组 **{e["section"]}**。本页为知识库 **策展索引级** 详情节点；方法细节与量化指标以原文 PDF / 项目页为准。

## 一句话定义

{hl}

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
{abbrev_rows}

## 为什么重要

- {hl}
- 在 [{list_meta["title"]} 技术地图]({tech_map_rel}) 中提供可点击的独立详情节点，避免清单条目无法落入知识图谱。
- 与列表实体 [{list_meta["title"]}]({wiki_rel_from_root(list_meta["entity"])}) 及站内方法/任务页交叉，便于从策展索引跳转到学习主线。

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
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**本条目的站内价值是把「{short}」从外部 Awesome 列表提升为可链接的知识节点，并保留清单 Highlights 作为阅读锚点。**

- 起作用的是策展坐标：列表分组 **{e["section"]}** + Highlights 指出的问题设定，而不是本页自行推导的新算法结论。
- 适用边界：索引级页面不能替代 PDF；开源状态以项目页实际链接为准（清单可能滞后）。
- 若该工作成为学习主线，应再升格为深度论文实体（补机构、实验表、源码运行时序图或「不适用」说明）。

## 常见误区

1. 不要把 Awesome 条目的 Highlights 当成完整方法证明——它只是策展导读。
2. 同一 arXiv 在全库只允许一个 canonical 详情节点；若已有深度页，应以深度页为准。

## 关联页面

- 列表实体：[{list_meta["title"]}]({wiki_rel_from_root(list_meta["entity"])})
- 技术地图：[{list_meta["title"]} 技术地图]({tech_map_rel})
- 方法/任务：[{method_name}]({list_meta["hub_methods"][0]})、[{task_name}]({list_meta["hub_tasks"][0]})

## 参考来源

- [`{src_rel}`](../../{src_rel}) — 本条目策展摘录
- [`sources/papers/sun_awesome_{e["list"]}_catalog.md`](../../sources/papers/sun_awesome_{e["list"]}_catalog.md) — 列表总表
- [`sources/repos/{repo_src}.md`](../../sources/repos/{repo_src}.md)
- 论文：<{paper_link}>

## 推荐继续阅读

- [{list_meta["title"]} 仓库]({list_meta["url"]})
- [原文]({paper_link})
"""


def render_catalog(list_key: str, list_meta: dict, rows: list[dict]) -> str:
    lines = [
        f"# {list_meta['title']} 论文目录（sun254667）",
        "",
        f"> 由 `{list_meta['url']}` 解析生成；入库日 {TODAY}。",
        "",
        f"- **列表实体：** [`{list_meta['entity']}`](../../{list_meta['entity']})",
        f"- **技术地图：** [`wiki/overview/sun-awesome-{list_key}-technology-map.md`](../../wiki/overview/sun-awesome-{list_key}-technology-map.md)",
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


def render_tech_map(list_key: str, list_meta: dict, rows: list[dict]) -> str:
    by_sec: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_sec[r["section"]].append(r)

    abbrev_rows = "\n".join(f"| {a} | {b} | {c} |" for a, b, c in list_meta["abbrev"][:4])
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
tags: [overview, curated-index, {list_meta["tag"]}, sun254667, technology-map]
status: complete
updated: {TODAY}
summary: "{list_meta["title"]} 技术地图：为清单内论文提供独立详情节点索引（新建 {new_count}，复用已有 {reused}）。"
related:
  - {wiki_rel_from_root(list_meta["entity"])}
  - {list_meta["hub_methods"][0]}
  - {list_meta["hub_tasks"][0]}
sources:
  - ../../sources/papers/sun_awesome_{list_key}_catalog.md
  - ../../sources/repos/{"awesome-real2sim2real" if list_key == "r2s2r" else list_meta["repo"]}.md
---

# {list_meta["title"]} 技术地图

> 本页把 [{list_meta["title"]}]({list_meta["url"]}) 清单中的论文条目映射为站内 **独立详情节点**（`wiki/entities/paper-sa-*` 或已有 canonical 页），供图谱与 `detail.html` 检索。

## 一句话定义

**{list_meta["title"]} 技术地图** = 外部 Awesome 策展列表的站内节点化索引（按清单分组浏览，一点即达论文实体页）。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
{abbrev_rows}

## 为什么重要

- Awesome 列表本身不是知识图谱节点；若不升格论文实体，首页/图谱无法挂上具体工作。
- 本地图 **优先复用** 库内已有 arXiv canonical 页，仅对缺失条目新建索引级 `paper-sa-*` 节点。
- 统计：清单可解析条目 **{len(rows)}**（新建详情节点 **{new_count}**，复用已有 **{reused}**）。

## 覆盖范围

| 项 | 值 |
|----|-----|
| 上游仓库 | <{list_meta["url"]}> |
| 列表实体 | [{list_meta["title"]}]({wiki_rel_from_root(list_meta["entity"])}) |
| 目录 source | [sun_awesome_{list_key}_catalog.md](../../sources/papers/sun_awesome_{list_key}_catalog.md) |

## 分组索引

{chr(10).join(sections_md)}

## 局限与风险

- 索引级节点保留清单 Highlights，**不替代** 深度论文页；主线工作应继续升格。
- 清单可能含非 arXiv 链接（OpenReview / IEEE）；无 arXiv 条目以标题 slug 建节点，后续若补 arXiv 需合并去重。
- 上游更新后需重跑 `python3 scripts/generate_sun254667_awesome_paper_entities.py` 再 `make ci-preflight`。

## 关联页面

- [{list_meta["title"]}（列表实体）]({wiki_rel_from_root(list_meta["entity"])})
- [{list_meta["hub_methods"][0].split("/")[-1]}]({list_meta["hub_methods"][0]})
- [{list_meta["hub_tasks"][0].split("/")[-1]}]({list_meta["hub_tasks"][0]})

## 参考来源

- [sun_awesome_{list_key}_catalog.md](../../sources/papers/sun_awesome_{list_key}_catalog.md)
- [sources/repos/{"awesome-real2sim2real" if list_key == "r2s2r" else list_meta["repo"]}.md](../../sources/repos/{"awesome-real2sim2real" if list_key == "r2s2r" else list_meta["repo"]}.md)
- 上游：<{list_meta["url"]}>

## 推荐继续阅读

- [{list_meta["title"]} GitHub]({list_meta["url"]})
"""


def main() -> None:
    existing = _existing_arxiv_map()
    # Also treat already-generated paper-sa pages as existing
    for p in (ROOT / "wiki" / "entities").glob("paper-sa-*.md"):
        text = p.read_text(encoding="utf-8", errors="ignore")
        m = re.search(r'(?m)^arxiv:\s*["\']?(\d{4}\.\d{4,5})', text)
        if m:
            existing.setdefault(m.group(1), str(p.relative_to(ROOT)))

    stats = {}
    for list_key, list_meta in LISTS.items():
        readme = CACHE_DIR / f"{list_meta['repo']}.md"
        if not readme.exists():
            raise SystemExit(f"missing cached README: {readme}")
        entries = parse_readme(readme, list_key)
        # stable order: section then title
        entries.sort(key=lambda e: (e["section"], e["title"]))

        rows: list[dict] = []
        created_entities = 0
        created_sources = 0
        # For new entities, allocate indices in list order
        for i, e in enumerate(entries, start=1):
            created = False
            if e["arxiv"] and e["arxiv"] in existing:
                wiki_rel = existing[e["arxiv"]]
            else:
                # create new
                ent_name = entity_filename(e, i)
                wiki_rel = f"wiki/entities/{ent_name}"
                src_name = source_filename(e, i)
                src_rel = f"sources/papers/{src_name}"
                ent_path = ROOT / wiki_rel
                src_path = ROOT / src_rel
                tech_map_rel = f"../overview/sun-awesome-{list_key}-technology-map.md"

                if not src_path.exists():
                    src_path.write_text(
                        render_source(e, wiki_rel, list_meta, i, len(entries)),
                        encoding="utf-8",
                    )
                    created_sources += 1
                if not ent_path.exists():
                    body = render_entity(e, src_rel, list_meta, i, len(entries), tech_map_rel)
                    ent_path.write_text(body, encoding="utf-8")
                    created_entities += 1
                    created = True
                if e["arxiv"]:
                    existing[e["arxiv"]] = wiki_rel
            rows.append(
                {
                    **e,
                    "idx": i,
                    "wiki_rel": wiki_rel,
                    "created": created,
                }
            )

        # catalogs + tech maps always rewrite
        cat_path = ROOT / f"sources/papers/sun_awesome_{list_key}_catalog.md"
        cat_path.write_text(render_catalog(list_key, list_meta, rows), encoding="utf-8")
        map_path = ROOT / f"wiki/overview/sun-awesome-{list_key}-technology-map.md"
        map_path.write_text(render_tech_map(list_key, list_meta, rows), encoding="utf-8")

        stats[list_key] = {
            "entries": len(entries),
            "created_entities": created_entities,
            "created_sources": created_sources,
            "reused": sum(1 for r in rows if not r["created"]),
        }
        print(list_key, stats[list_key])

    (CACHE_DIR / "gen_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print("done", stats)


if __name__ == "__main__":
    main()
