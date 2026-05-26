#!/usr/bin/env python3
"""Generate sources/papers and wiki/entities for 42+19 humanoid stack survey papers."""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPERS_DIR = ROOT / "sources" / "papers"
ENTITIES_DIR = ROOT / "wiki" / "entities"
RAW_HRL = ROOT / "sources/raw/wechat_humanoid_rl_42_survey_2026-05-26.md"
RAW_AMP = ROOT / "sources/raw/wechat_humanoid_amp_19_survey_2026-05-26.md"
TODAY = date.today().isoformat()
WECHAT_HRL = "wechat_embodied_ai_lab_humanoid_rl_motion_survey.md"
WECHAT_AMP = "wechat_embodied_ai_lab_humanoid_amp_motion_prior_survey.md"
HRL_CATALOG = "humanoid_rl_stack_42_catalog.md"
AMP_CATALOG = "humanoid_amp_survey_19_catalog.md"

HRL_LAYER = {
    "data": "01 数据 · 重定向 · 遥操作",
    "tracking-control": "02 参考跟踪 · 通用控制",
    "perception": "03 感知式高动态运动",
    "task-world": "04 视觉闭环 · 任务接口 · 世界模型",
    "contact-safety": "05 接触 · 柔顺 · 安全恢复",
}

AMP_GROUP = {
    "prior-component": "01 分布约束与先验组件化",
    "locomotion": "02 人形走跑",
    "multi-skill": "03 多技能与自适应",
    "interaction": "04 交互与长时程",
}

# Optional cross-links to existing method/entity pages (not skip — still create paper-hrl-stack nodes)
HRL_WIKI_HINTS: dict[int, list[str]] = {
    1: ["wiki/methods/motion-retargeting-gmr.md"],
    2: ["wiki/methods/neural-motion-retargeting-nmr.md"],
    11: ["wiki/methods/deepmimic.md"],
    13: ["wiki/methods/any2track.md"],
    15: ["wiki/methods/beyondmimic.md"],
    17: ["wiki/methods/sonic-motion-tracking.md"],
    18: ["wiki/methods/ams.md"],
    19: ["wiki/entities/paper-behavior-foundation-model-humanoid.md"],
    28: ["wiki/entities/paper-viral-humanoid-visual-sim2real.md"],
    29: ["wiki/entities/paper-doorman-opening-sim2real-door.md"],
    34: ["wiki/entities/gr00t-wholebodycontrol.md"],
    37: ["wiki/methods/gentlehumanoid-motion-tracking.md"],
}

AMP_WIKI_HINTS: dict[int, list[str]] = {
    1: ["wiki/methods/amp-reward.md", "wiki/methods/add.md"],
    2: ["wiki/methods/add.md"],
    3: ["wiki/methods/smp.md"],
    4: ["wiki/entities/kimodo.md"],
    5: ["wiki/methods/motionbricks.md"],
    10: ["wiki/entities/paper-unified-walk-run-recovery-sdamp.md", "wiki/entities/amp-mjlab.md"],
    18: ["wiki/entities/project-instinct.md"],
    19: ["wiki/entities/project-instinct.md"],
}


def _layer(num: int) -> str:
    if num <= 10:
        return "data"
    if num <= 21:
        return "tracking-control"
    if num <= 27:
        return "perception"
    if num <= 35:
        return "task-world"
    return "contact-safety"


def _amp_group(num: int) -> str:
    if num <= 5:
        return "prior-component"
    if num <= 10:
        return "locomotion"
    if num <= 12:
        return "multi-skill"
    return "interaction"


def _slugify(title: str, max_len: int = 48) -> str:
    s = title.lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s[:max_len].rstrip("_")


def _venue(link: str) -> str:
    if "arxiv.org" in link:
        return "arXiv"
    if "openreview.net" in link:
        return "OpenReview"
    if "github.com" in link:
        return "project"
    return "curated"


def _short_title(title: str) -> str:
    if ":" in title:
        return title.split(":", 1)[0].strip()
    return title.strip()


def _parse_hrl(text: str) -> list[dict]:
    parts = re.split(r"(?=^### \d+)", text, flags=re.MULTILINE)
    out: list[dict] = []
    for sec in parts:
        hm = re.match(r"^### (\d+)([^:]+)[:：]\s*([^\n]+)", sec)
        if not hm:
            continue
        num = int(hm.group(1))
        tm = re.search(r"📄\s*\*\*论文标题\*\*[：:]\s*([^\n]+)", sec)
        if not tm:
            continue
        lm = re.search(r"🔗\s*\*\*(?:项目|论文)链接\*\*[：:]\s*(https?://[^\s\n]+)", sec)
        im = re.search(r"🏫\s*\*\*机构\*\*[：:]\s*([^\n]+)", sec)
        note = ""
        if im:
            after = sec.split(im.group(0), 1)[-1].strip().split("\n\n")[0]
            note = re.sub(r"!\[[^\]]*\]\([^)]+\)[^\n]*", "", after)
            note = re.sub(r"\s+", " ", note).strip()[:300]
        title = tm.group(1).strip()
        link = lm.group(1).strip() if lm else ""
        layer = _layer(num)
        slug = f"humanoid_rl_stack_{num:02d}_{_slugify(title)}"
        out.append(
            {
                "survey": "hrl",
                "num": num,
                "slug": slug,
                "title": title,
                "link": link,
                "inst": im.group(1).strip() if im else "",
                "note": note or hm.group(3).strip(),
                "layer": layer,
                "wiki": HRL_WIKI_HINTS.get(num, []),
            }
        )
    return sorted(out, key=lambda x: x["num"])


def _parse_amp(text: str) -> list[dict]:
    parts = re.split(r"(?=^论文\s+\d+)", text, flags=re.MULTILINE)
    out: list[dict] = []
    for sec in parts:
        hm = re.match(r"^论文\s+(\d+)\s*\n", sec)
        if not hm:
            continue
        num = int(hm.group(1))
        tm = re.search(r"📄\s*\*\*论文标题\*\*[：:]\s*([^\n]+)", sec)
        if not tm:
            continue
        lm = re.search(r"🔗\s*\*\*论文链接\*\*[：:]\s*(https?://[^\s\n]+)", sec)
        im = re.search(r"🏫\s*\*\*机构\*\*[：:]\s*([^\n]+)", sec)
        note = ""
        if im:
            after = sec.split(im.group(0), 1)[-1].strip().split("\n\n")[0]
            note = re.sub(r"!\[[^\]]*\]\([^)]+\)[^\n]*", "", after)
            note = re.sub(r"\s+", " ", note).strip()[:300]
        title = tm.group(1).strip()
        link = lm.group(1).strip() if lm else ""
        group = _amp_group(num)
        slug = f"humanoid_amp_survey_{num:02d}_{_slugify(title)}"
        out.append(
            {
                "survey": "amp",
                "num": num,
                "slug": slug,
                "title": title,
                "link": link,
                "inst": im.group(1).strip() if im else "",
                "note": note,
                "group": group,
                "wiki": AMP_WIKI_HINTS.get(num, []),
            }
        )
    return sorted(out, key=lambda x: x["num"])


def _yaml_list(items: list[str], indent: int = 0) -> str:
    pad = " " * indent
    return "\n".join(f"{pad}- {x}" for x in items)


def _wiki_to_entity_rel(wiki_path: str) -> str:
    assert wiki_path.startswith("wiki/")
    return "../" + wiki_path[len("wiki/") :]


def _related_hrl(p: dict) -> list[str]:
    out = [
        "../overview/humanoid-rl-motion-control-body-system-stack.md",
        "../overview/humanoid-amp-motion-prior-survey.md",
    ]
    for w in p.get("wiki", []):
        rel = _wiki_to_entity_rel(w)
        if rel not in out:
            out.append(rel)
    return out[:12]


def _related_amp(p: dict) -> list[str]:
    out = [
        "../overview/humanoid-amp-motion-prior-survey.md",
        "../overview/humanoid-rl-motion-control-body-system-stack.md",
    ]
    for w in p.get("wiki", []):
        rel = _wiki_to_entity_rel(w)
        if rel not in out:
            out.append(rel)
    return out[:12]


def hrl_source_md(p: dict) -> str:
    wiki_lines = "\n".join(
        f"  - [{w.split('/')[-1].replace('.md', '')}](../../{w})" for w in p["wiki"]
    )
    link_line = f"- **项目/论文链接：** <{p['link']}>\n" if p["link"] else ""
    return f"""# {p["title"]}

> 来源归档（ingest · 人形 RL 身体系统栈 42 篇 · 第 {p["num"]:02d}/42）

- **标题：** {p["title"]}
- **类型：** paper
- **系统栈层：** {HRL_LAYER[p["layer"]]}
- **机构：** {p["inst"]}
{link_line}- **索引来源：** [具身智能研究室 · 42 篇 RL 运动控制长文](../blogs/{WECHAT_HRL})（<https://mp.weixin.qq.com/s/hz9JXtJeUPRfUGzfD-pZuA>）
- **原始抓取：** [wechat_humanoid_rl_42_survey_2026-05-26.md](../raw/wechat_humanoid_rl_42_survey_2026-05-26.md)（Agent Reach + Camoufox）
- **入库日期：** {TODAY}
- **一句话说明：** {p["note"]}

## 核心摘录（策展，非全文）

- **在身体系统栈中的位置：** {HRL_LAYER[p["layer"]]}，编号 **{p["num"]:02d}/42**。
- **公众号导读要点：** {p["note"]}
- **读者动作：** 方法细节以论文 PDF / 项目页为准；总框架见 [人形 RL 身体系统栈](../../wiki/overview/humanoid-rl-motion-control-body-system-stack.md)。

## 对 wiki 的映射

{wiki_lines or "  - （待交叉链接）"}

## 参考来源（原始）

- 微信公众号编译：[{WECHAT_HRL}](../blogs/{WECHAT_HRL})
- 姊妹篇 AMP 专题：[{WECHAT_AMP}](../blogs/{WECHAT_AMP})
"""


def amp_source_md(p: dict) -> str:
    wiki_lines = "\n".join(
        f"  - [{w.split('/')[-1].replace('.md', '')}](../../{w})" for w in p["wiki"]
    )
    link_line = f"- **论文链接：** <{p['link']}>\n" if p["link"] else ""
    return f"""# {p["title"]}

> 来源归档（ingest · 人形 AMP 运动先验 19 篇 · 第 {p["num"]:02d}/19）

- **标题：** {p["title"]}
- **类型：** paper
- **AMP 叙事段：** {AMP_GROUP[p["group"]]}
- **机构：** {p["inst"]}
{link_line}- **索引来源：** [具身智能研究室 · AMP 专题长文](../blogs/{WECHAT_AMP})（<https://mp.weixin.qq.com/s/YZsm3855iP3TNTTt1aou7w>）
- **原始抓取：** [wechat_humanoid_amp_19_survey_2026-05-26.md](../raw/wechat_humanoid_amp_19_survey_2026-05-26.md)（Agent Reach + Camoufox）
- **入库日期：** {TODAY}
- **一句话说明：** {p["note"]}

## 核心摘录（策展，非全文）

- **在 AMP 四段地图中的位置：** {AMP_GROUP[p["group"]]}，编号 **{p["num"]:02d}/19**。
- **公众号导读要点：** {p["note"]}

## 对 wiki 的映射

{wiki_lines or "  - （待交叉链接）"}

## 参考来源（原始）

- 微信公众号编译：[{WECHAT_AMP}](../blogs/{WECHAT_AMP})
- 姊妹篇 42 篇栈：[{WECHAT_HRL}](../blogs/{WECHAT_HRL})
"""


def hrl_entity_md(p: dict) -> str:
    source_rel = f"../../sources/papers/{p['slug']}.md"
    related = _related_hrl(p)
    short = _short_title(p["title"])
    summary = p["note"].replace('"', "'")[:200]
    link = p["link"] or "（见项目页 / 论文标题检索）"
    venue = _venue(p["link"])
    link_row = f"| 链接 | <{link}> |" if p["link"] else "| 链接 | 见论文标题检索 |"
    return f"""---
type: entity
tags: [paper, humanoid, rl, motion-control, body-system-stack]
status: complete
updated: {TODAY}
summary: "{summary}"
related:
{_yaml_list(related, 2)}
sources:
  - {source_rel}
  - ../../sources/papers/{HRL_CATALOG}
  - ../../sources/blogs/{WECHAT_HRL}
---

# {short}

**{short}** 收录于 [具身智能研究室 · 42 篇 humanoid RL 运动控制长文](https://mp.weixin.qq.com/s/hz9JXtJeUPRfUGzfD-pZuA) **第 {p["num"]:02d}/42** 篇，归类为 **{HRL_LAYER[p["layer"]]}**。本页为知识库 **策展摘要**；方法细节以论文 PDF 与项目页为准。

## 为什么重要

- {p["note"]}
- 在 [人形 RL 身体系统栈](../overview/humanoid-rl-motion-control-body-system-stack.md) 的八层框架中，属于 **{HRL_LAYER[p["layer"]]}** 簇。

## 核心信息（索引级）

| 字段 | 内容 |
|------|------|
| 编号 | {p["num"]:02d}/42 |
| 系统栈层 | {HRL_LAYER[p["layer"]]} |
| 机构 | {p["inst"]} |
| 出处 | {venue} |
{link_row}

## 与其他页面的关系

- 总框架：[humanoid-rl-motion-control-body-system-stack.md](../overview/humanoid-rl-motion-control-body-system-stack.md)
- AMP 姊妹篇：[humanoid-amp-motion-prior-survey.md](../overview/humanoid-amp-motion-prior-survey.md)
- 原始 source：[{p["slug"]}.md](../../sources/papers/{p["slug"]}.md)

## 参考来源

- [{p["slug"]}.md](../../sources/papers/{p["slug"]}.md) — 42 篇栈策展摘录
- [{HRL_CATALOG}](../../sources/papers/{HRL_CATALOG}) — 总表
- [{WECHAT_HRL}](../../sources/blogs/{WECHAT_HRL}) — 微信公众号编译导读
- 原始抓取：[wechat_humanoid_rl_42_survey_2026-05-26.md](../../sources/raw/wechat_humanoid_rl_42_survey_2026-05-26.md)

## 推荐继续阅读

- [42 篇 RL 运动控制（微信公众号）](https://mp.weixin.qq.com/s/hz9JXtJeUPRfUGzfD-pZuA)
- [19 篇 AMP 运动先验姊妹篇](https://mp.weixin.qq.com/s/YZsm3855iP3TNTTt1aou7w)
"""


def amp_entity_md(p: dict) -> str:
    source_rel = f"../../sources/papers/{p['slug']}.md"
    related = _related_amp(p)
    short = _short_title(p["title"])
    summary = p["note"].replace('"', "'")[:200] if p["note"] else short
    link = p["link"] or "（见论文标题检索）"
    venue = _venue(p["link"])
    link_row = f"| 链接 | <{link}> |" if p["link"] else "| 链接 | 见论文标题检索 |"
    return f"""---
type: entity
tags: [paper, humanoid, amp, motion-prior, adversarial-imitation]
status: complete
updated: {TODAY}
summary: "{summary}"
related:
{_yaml_list(related, 2)}
sources:
  - {source_rel}
  - ../../sources/papers/{AMP_CATALOG}
  - ../../sources/blogs/{WECHAT_AMP}
---

# {short}

**{short}** 收录于 [具身智能研究室 · AMP 运动先验专题](https://mp.weixin.qq.com/s/YZsm3855iP3TNTTt1aou7w) **第 {p["num"]:02d}/19** 篇，归类为 **{AMP_GROUP[p["group"]]}**。本页为知识库 **策展摘要**。

## 为什么重要

- {p["note"] or "见 AMP 四段论文地图与 [amp-reward](../methods/amp-reward.md) 方法页。"}
- 在 [人形 AMP 运动先验综述](../overview/humanoid-amp-motion-prior-survey.md) 中与 mimic / 身体系统栈分工对照阅读。

## 核心信息（索引级）

| 字段 | 内容 |
|------|------|
| 编号 | {p["num"]:02d}/19 |
| 叙事段 | {AMP_GROUP[p["group"]]} |
| 机构 | {p["inst"]} |
| 出处 | {venue} |
{link_row}

## 与其他页面的关系

- AMP 综述：[humanoid-amp-motion-prior-survey.md](../overview/humanoid-amp-motion-prior-survey.md)
- 身体系统栈：[humanoid-rl-motion-control-body-system-stack.md](../overview/humanoid-rl-motion-control-body-system-stack.md)
- 原始 source：[{p["slug"]}.md](../../sources/papers/{p["slug"]}.md)

## 参考来源

- [{p["slug"]}.md](../../sources/papers/{p["slug"]}.md)
- [{AMP_CATALOG}](../../sources/papers/{AMP_CATALOG})
- [{WECHAT_AMP}](../../sources/blogs/{WECHAT_AMP})
- 原始抓取：[wechat_humanoid_amp_19_survey_2026-05-26.md](../../sources/raw/wechat_humanoid_amp_19_survey_2026-05-26.md)

## 推荐继续阅读

- [AMP 专题长文（微信公众号）](https://mp.weixin.qq.com/s/YZsm3855iP3TNTTt1aou7w)
- [42 篇 RL 身体系统栈姊妹篇](https://mp.weixin.qq.com/s/hz9JXtJeUPRfUGzfD-pZuA)
"""


def entity_filename(p: dict) -> str:
    short = _slugify(_short_title(p["title"]), max_len=36)
    if p["survey"] == "hrl":
        return f"paper-hrl-stack-{p['num']:02d}-{short}.md"
    return f"paper-amp-survey-{p['num']:02d}-{short}.md"


def hrl_catalog_md(papers: list[dict]) -> str:
    lines = [
        "# 人形 RL 身体系统栈：42 篇论文 source 索引",
        "",
        "> 来源归档（catalog）",
        "",
        f"- **微信公众号：** [wechat_embodied_ai_lab_humanoid_rl_motion_survey.md](../blogs/{WECHAT_HRL})（<https://mp.weixin.qq.com/s/hz9JXtJeUPRfUGzfD-pZuA>）",
        "- **wiki 总览：** [humanoid-rl-motion-control-body-system-stack.md](../../wiki/overview/humanoid-rl-motion-control-body-system-stack.md)",
        f"- **入库日期：** {TODAY}",
        "- **一句话说明：** 将具身智能研究室 42 篇 humanoid RL 运动控制论文分别落成独立 `sources/papers/humanoid_rl_stack_*` 与 `wiki/entities/paper-hrl-stack-*` 节点。",
        "",
        "## 论文（42）",
        "",
        "| # | 系统栈层 | Source | Wiki 实体 |",
        "|---|----------|--------|-----------|",
    ]
    for p in papers:
        ent = entity_filename(p)
        lines.append(
            f"| {p['num']:02d} | {HRL_LAYER[p['layer']]} | [{p['slug']}.md]({p['slug']}.md) | [../../wiki/entities/{ent}](../../wiki/entities/{ent}) |"
        )
    lines.extend(
        [
            "",
            "## 参考来源",
            "",
            f"- [{WECHAT_HRL}](../blogs/{WECHAT_HRL})",
            "- [wechat_humanoid_rl_42_survey_2026-05-26.md](../raw/wechat_humanoid_rl_42_survey_2026-05-26.md)",
        ]
    )
    return "\n".join(lines)


def amp_catalog_md(papers: list[dict]) -> str:
    lines = [
        "# 人形 AMP 运动先验：19 篇论文 source 索引",
        "",
        "> 来源归档（catalog）",
        "",
        f"- **微信公众号：** [wechat_embodied_ai_lab_humanoid_amp_motion_prior_survey.md](../blogs/{WECHAT_AMP})（<https://mp.weixin.qq.com/s/YZsm3855iP3TNTTt1aou7w>）",
        "- **wiki 总览：** [humanoid-amp-motion-prior-survey.md](../../wiki/overview/humanoid-amp-motion-prior-survey.md)",
        f"- **入库日期：** {TODAY}",
        "- **一句话说明：** 将 AMP 专题 19 篇论文分别落成独立 source 与 `wiki/entities/paper-amp-survey-*` 节点。",
        "",
        "## 论文（19）",
        "",
        "| # | 叙事段 | Source | Wiki 实体 |",
        "|---|--------|--------|-----------|",
    ]
    for p in papers:
        ent = entity_filename(p)
        lines.append(
            f"| {p['num']:02d} | {AMP_GROUP[p['group']]} | [{p['slug']}.md]({p['slug']}.md) | [../../wiki/entities/{ent}](../../wiki/entities/{ent}) |"
        )
    lines.extend(
        [
            "",
            "## 参考来源",
            "",
            f"- [{WECHAT_AMP}](../blogs/{WECHAT_AMP})",
            "- [wechat_humanoid_amp_19_survey_2026-05-26.md](../raw/wechat_humanoid_amp_19_survey_2026-05-26.md)",
        ]
    )
    return "\n".join(lines)


def overview_entity_index(papers: list[dict], survey: str) -> str:
    lines = [
        "",
        "## Wiki 实体索引（站内详情页）",
        "",
        f"> 以下 {len(papers)} 篇均已升格为 `wiki/entities/` 详情页（可搜索、进图谱）。",
        "",
        "| # | 论文 | 实体页 |",
        "|---|------|--------|",
    ]
    for p in papers:
        ent = entity_filename(p)
        short = _short_title(p["title"])
        lines.append(f"| {p['num']:02d} | {short} | [{ent}](../entities/{ent}) |")
    return "\n".join(lines)


def patch_overview(path: Path, marker: str, section: str) -> None:
    text = path.read_text(encoding="utf-8")
    if marker in text:
        # replace existing section
        pattern = re.compile(
            r"\n## Wiki 实体索引.*?(?=\n## |\Z)",
            re.DOTALL,
        )
        text = pattern.sub(section, text, count=1)
    else:
        # insert before ## 局限
        insert_at = text.find("\n## 局限")
        if insert_at == -1:
            insert_at = text.find("\n## 参考来源")
        if insert_at == -1:
            text += section
        else:
            text = text[:insert_at] + section + text[insert_at:]
    path.write_text(text, encoding="utf-8")


def main() -> None:
    if not RAW_HRL.exists() or not RAW_AMP.exists():
        raise SystemExit(f"Missing raw wechat files: {RAW_HRL} / {RAW_AMP}")

    hrl = _parse_hrl(RAW_HRL.read_text(encoding="utf-8"))
    amp = _parse_amp(RAW_AMP.read_text(encoding="utf-8"))
    if len(hrl) != 42:
        raise SystemExit(f"Expected 42 HRL papers, got {len(hrl)}: {[p['num'] for p in hrl]}")
    if len(amp) != 19:
        raise SystemExit(f"Expected 19 AMP papers, got {len(amp)}")

    PAPERS_DIR.mkdir(parents=True, exist_ok=True)
    ENTITIES_DIR.mkdir(parents=True, exist_ok=True)
    created_src = created_ent = 0

    for p in hrl:
        src = PAPERS_DIR / f"{p['slug']}.md"
        ent = ENTITIES_DIR / entity_filename(p)
        src.write_text(hrl_source_md(p), encoding="utf-8")
        ent.write_text(hrl_entity_md(p), encoding="utf-8")
        created_src += 1
        created_ent += 1

    for p in amp:
        src = PAPERS_DIR / f"{p['slug']}.md"
        ent = ENTITIES_DIR / entity_filename(p)
        src.write_text(amp_source_md(p), encoding="utf-8")
        ent.write_text(amp_entity_md(p), encoding="utf-8")
        created_src += 1
        created_ent += 1

    (PAPERS_DIR / HRL_CATALOG).write_text(hrl_catalog_md(hrl), encoding="utf-8")
    (PAPERS_DIR / AMP_CATALOG).write_text(amp_catalog_md(amp), encoding="utf-8")

    hrl_overview = ROOT / "wiki/overview/humanoid-rl-motion-control-body-system-stack.md"
    amp_overview = ROOT / "wiki/overview/humanoid-amp-motion-prior-survey.md"
    patch_overview(hrl_overview, "Wiki 实体索引", overview_entity_index(hrl, "hrl"))
    patch_overview(amp_overview, "Wiki 实体索引", overview_entity_index(amp, "amp"))

    # Update limitation bullets
    for path in (hrl_overview, amp_overview):
        text = path.read_text(encoding="utf-8")
        text = text.replace(
            "「仅在源文中提及」一栏的工作",
            "部分工作另有 `wiki/methods/` 方法深读页",
        )
        text = re.sub(
            r"- 表格中「仅在源文中提及」[^\\n]+\\n",
            "- 42/19 篇均已各有 `wiki/entities/paper-hrl-stack-*` / `paper-amp-survey-*` 索引节点；方法细节仍以 arXiv / 项目页为准。\\n",
            text,
            count=1,
        )
        path.write_text(text, encoding="utf-8")

    print(f"hrl={len(hrl)} amp={len(amp)} sources={created_src} entities={created_ent}")


if __name__ == "__main__":
    main()
