#!/usr/bin/env python3
"""Bootstrap 161 paper-loco-manip-161-* entities + sources from WeChat raw."""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "sources/raw/wechat_humanoid_loco_manip_161_2026-06-26"


def _resolve_raw_md() -> Path:
    if RAW.exists():
        files = list(RAW.glob("*.md"))
        if files:
            return files[0]
    return RAW / "重磅整理！161篇论文带你看人形机器人移动操作的十个方向和技术版图全景.md"


RAW_MD = _resolve_raw_md()

CATS = {
    1: ("01", "运控基座与通用全身跟踪", "motion-base-wbt"),
    2: ("02", "上半身中心控制与移动操作接口", "upper-body-interface"),
    3: ("03", "视觉感知驱动的人形移动操作", "visuomotor"),
    4: ("04", "生成式运动、语言控制与轨迹规划", "generative-language-trajectory"),
    5: ("05", "动捕、人类视频与交互动作规划", "mocap-human-video"),
    6: ("06", "特殊任务、接触规划与视觉闭环", "contact-tasks"),
    7: ("07", "数据采集与遥操作系统", "data-teleop"),
    8: ("08", "硬件平台、感知配置与部署扩展", "hardware-deployment"),
    9: ("09", "人形 VLA、世界模型与通用操作", "vla-world-models"),
    10: ("10", "从人类第一视角视频学习", "ego-video"),
}

TODAY = date.today().isoformat()

# Prior catalog cross-links (wiki path without .md) for related pages
PRIOR_WIKI: dict[int, str] = {}

# Loco-Manip 161 槽位 → 已有 canonical 实体（跳过独立 stub 生成）
CANONICAL_ENTITY_BY_NUM: dict[int, str] = {
    19: "paper-sonic",
    103: "paper-sonic",
    110: "paper-hrl-stack-06-hdmi",
    148: "paper-hrl-stack-34-gr00t_n1",
}


def slugify(text: str, num: int) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "-", text).strip("-").lower()
    s = re.sub(r"-+", "-", s)
    if len(s) < 3 or not re.search(r"[a-z]", s):
        s = f"n{num:03d}"
    return s[:48].strip("-")


def parse_papers(text: str) -> list[dict]:
    papers: list[dict] = []
    for part in re.split(r"^## ", text, flags=re.M)[1:]:
        lines = part.strip().splitlines()
        cat_m = re.match(r"(\d+)(.+)", lines[0].strip())
        if not cat_m:
            continue
        cat_num = int(cat_m.group(1))
        cat_name = cat_m.group(2).strip()
        for block in re.split(r"^### ", part, flags=re.M)[1:]:
            blines = block.strip().splitlines()
            header = blines[0].strip()
            hm = re.match(r"(\d{3})\s+(.+?)(?:｜|\|)", header) or re.match(
                r"(\d{3})\s+(.+)", header
            )
            if not hm:
                continue
            num = int(hm.group(1))
            short = hm.group(2).strip()
            title = inst = link = date_s = summary = ""
            for line in blines:
                if "**原文题目：**" in line:
                    title = line.split("**原文题目：**", 1)[1].strip()
                elif "**机构：**" in line:
                    inst = line.split("**机构：**", 1)[1].strip()
                elif "**项目链接：**" in line:
                    link = line.split("**项目链接：**", 1)[1].strip()
                elif "**发表日期：**" in line:
                    date_s = line.split("**发表日期：**", 1)[1].strip()
                elif line.startswith("**算法实现总结：**"):
                    summary = line.replace("**算法实现总结：**", "").strip()
            slug = slugify(short.split("｜")[0].split("|")[0], num)
            papers.append(
                {
                    "num": num,
                    "short": short,
                    "title": title or short,
                    "inst": inst,
                    "link": link,
                    "date": date_s,
                    "summary": summary,
                    "cat_num": cat_num,
                    "cat_name": cat_name,
                    "slug": slug,
                }
            )
    return papers


def load_prior_wiki_from_catalog() -> dict[int, str]:
    catalog = ROOT / "sources/papers/humanoid_loco_manip_161_catalog.md"
    if not catalog.exists():
        return {}
    text = catalog.read_text(encoding="utf-8")
    out: dict[int, str] = {}
    for line in text.splitlines():
        m = re.match(r"^\| (\d{3}) \| .+ \| \[(.+)\]\(\.\./\.\./(wiki/.+?)\.md\) \|", line)
        if m:
            out[int(m.group(1))] = m.group(3)
    return out


def write_source(p: dict, entity_name: str) -> Path:
    cat_id, cat_label, _ = CATS[p["cat_num"]]
    path = ROOT / "sources/papers" / f"loco_manip_161_survey_{p['num']:03d}_{p['slug']}.md"
    body = f"""# {p["short"]}

> 来源归档（ingest · 人形 Loco-Manip 161 篇长文 第 {p["num"]:03d}/161）

- **标题：** {p["title"]}
- **类型：** paper
- **Loco-Manip 161 分类：** {cat_id} {cat_label}
- **机构：** {p["inst"] or "（见原文）"}
- **项目页：** {p["link"] or "（见原文）"}
- **发表日期：** {p["date"] or "（见原文）"}
- **入库日期：** {TODAY}
- **一句话说明：** {p["summary"] or p["short"]}

## 核心摘录（策展，非全文）

- **在 161 篇地图中的位置：** {cat_id} {cat_label}，编号 **{p["num"]:03d}/161**。
- **算法实现总结（公众号）：** {p["summary"] or "（见 raw 正文）"}

## 对 wiki 的映射

- [{entity_name}](../../wiki/entities/{entity_name}.md)
- [loco-manip-161-category-{p["cat_num"]:02d}-{CATS[p["cat_num"]][2]}](../../wiki/overview/loco-manip-161-category-{p["cat_num"]:02d}-{CATS[p["cat_num"]][2]}.md)

## 参考来源（原始）

- 微信公众号编译：[wechat_embodied_ai_lab_humanoid_loco_manip_161_survey.md](../blogs/wechat_embodied_ai_lab_humanoid_loco_manip_161_survey.md)
- 原始抓取：[wechat_humanoid_loco_manip_161_2026-06-26.md](../raw/wechat_humanoid_loco_manip_161_2026-06-26.md)
"""
    path.write_text(body, encoding="utf-8")
    return path


def write_entity(p: dict, prior: str | None) -> str:
    entity_name = f"paper-loco-manip-161-{p['num']:03d}-{p['slug']}"
    cat_id, cat_label, cat_slug = CATS[p["cat_num"]]
    source_rel = f"../../sources/papers/loco_manip_161_survey_{p['num']:03d}_{p['slug']}.md"
    one_liner = p["summary"] or p["title"]
    related = [
        "  - ../overview/humanoid-loco-manip-161-papers-technology-map.md",
        f"  - ../overview/loco-manip-161-category-{p['cat_num']:02d}-{cat_slug}.md",
        "  - ../tasks/loco-manipulation.md",
    ]
    if prior and prior != f"wiki/entities/{entity_name}":
        rel = prior.replace("wiki/", "../")
        related.append(f"  - {rel}.md")

    related_yaml = "\n".join(related)
    body = f"""---
type: entity
tags: [paper, loco-manipulation, loco-manip-161-survey, humanoid]
status: complete
updated: {TODAY}
venue: curated
summary: "{one_liner[:200].replace('"', "'")}"
related:
{related_yaml}
sources:
  - {source_rel}
  - ../../sources/blogs/wechat_embodied_ai_lab_humanoid_loco_manip_161_survey.md
  - ../../sources/papers/humanoid_loco_manip_161_catalog.md
---

# {p["short"].split("｜")[0].split("|")[0].strip()}

**{p["short"].split("｜")[0].split("|")[0].strip()}** 收录于 [具身智能研究室 · 人形 Loco-Manip 161 篇长文](https://mp.weixin.qq.com/s/pACh9EhsISiyPGdiiR0C3A) **第 {p["num"]:03d}/161** 篇，归类为 **{cat_id} {cat_label}**。

## 一句话定义

{one_liner}

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Loco-Manip | Loco-Manipulation | 行走与操作动力学耦合的全身任务 |
| WBC | Whole-Body Control | 协调全身关节满足多任务/约束的控制层 |
| VLA | Vision-Language-Action | 视觉-语言-动作多模态策略 |

## 为什么重要

- {one_liner}
- 人形 Loco-Manip 161 篇 **#{p["num"]:03d}/161** · {cat_label}。

## 核心信息（索引级）

| 字段 | 内容 |
|------|------|
| 编号 | {p["num"]:03d}/161 |
| 分组 | {cat_id} {cat_label} |
| 原文题目 | {p["title"] or "（见项目页）"} |
| 机构 | {p["inst"] or "（见原文）"} |
| 发表日期 | {p["date"] or "（见原文）"} |
| 论文/项目 | {p["link"] or "（见原文）"} |

## 核心机制（归纳）

### 策展导读要点

{one_liner}

## 常见误区

1. 161 篇策展条目提供 **地图坐标**；量化 benchmark 与实机指标以原文 PDF / 项目页为准。
2. Loco-manip 单篇工作不自动解决 **底层 WBC 鲁棒性**；须与运控/接触控制对照。

## 与其他页面的关系

- 技术地图：[humanoid-loco-manip-161-papers-technology-map.md](../overview/humanoid-loco-manip-161-papers-technology-map.md)
- 分类 hub：[loco-manip-161-category-{p["cat_num"]:02d}-{cat_slug}.md](../overview/loco-manip-161-category-{p["cat_num"]:02d}-{cat_slug}.md)
- 原始 source：[loco_manip_161_survey_{p["num"]:03d}_{p["slug"]}.md](../../sources/papers/loco_manip_161_survey_{p["num"]:03d}_{p["slug"]}.md)

## 参考来源

- [loco_manip_161_survey_{p["num"]:03d}_{p["slug"]}.md](../../sources/papers/loco_manip_161_survey_{p["num"]:03d}_{p["slug"]}.md) — 161 篇策展摘录
- [humanoid_loco_manip_161_catalog.md](../../sources/papers/humanoid_loco_manip_161_catalog.md)
- [wechat_embodied_ai_lab_humanoid_loco_manip_161_survey.md](../../sources/blogs/wechat_embodied_ai_lab_humanoid_loco_manip_161_survey.md)

## 推荐继续阅读

- [Loco-Manipulation 任务页](../tasks/loco-manipulation.md)
"""
    if prior and prior != f"wiki/entities/{entity_name}":
        stem = Path(prior).stem
        body += f"- 同题深读/既有实体：[{stem}](../{prior.replace('wiki/', '')}.md)\n"

    out = ROOT / "wiki/entities" / f"{entity_name}.md"
    out.write_text(body, encoding="utf-8")
    return entity_name


def write_catalog(papers: list[dict]) -> None:
    lines = [
        "# 人形 Loco-Manip 161 篇论文 source 索引\n\n",
        "> 来源归档（catalog）\n\n",
        "- **微信公众号导读：** [wechat_embodied_ai_lab_humanoid_loco_manip_161_survey.md](../blogs/wechat_embodied_ai_lab_humanoid_loco_manip_161_survey.md)\n",
        "- **wiki 技术地图：** [humanoid-loco-manip-161-papers-technology-map.md](../../wiki/overview/humanoid-loco-manip-161-papers-technology-map.md)\n",
        "- **原始链接：** <https://mp.weixin.qq.com/s/pACh9EhsISiyPGdiiR0C3A>\n",
        "- **入库日期：** 2026-06-26\n",
        "- **一句话说明：** 具身智能研究室 161 篇人形移动操作论文 **十类** 策展索引；**每篇** 对应独立实体 `wiki/entities/paper-loco-manip-161-{NNN}-*.md`。\n\n",
    ]
    for cn in range(1, 11):
        _, cname, _ = CATS[cn]
        desc = {
            1: "底层身体控制、运动跟踪、抗扰动与通用动作执行",
            2: "手臂、躯干、根部和末端执行器之间的协同控制",
            3: "视觉完成目标定位、场景理解和操作闭环",
            4: "从语言、目标或条件输入生成全身动作和轨迹",
            5: "人类动作数据转成机器人可用的运动和交互先验",
            6: "开门、推物、搬运、触碰等具体接触任务",
            7: "训练数据如何高效采集",
            8: "本体、传感器和真实部署系统",
            9: "视觉、语言、动作和世界建模接到执行层",
            10: "人类 egocentric 视频学习操作经验和行为先验",
        }[cn]
        lines.append(f"## {cn:02d}. {cname}\n\n> {desc}\n\n")
        lines.append("| # | 工作 | Wiki 实体 |\n|---|------|-----------|\n")
        for p in papers:
            if p["cat_num"] != cn:
                continue
            en = p["entity"]
            lines.append(
                f"| {p['num']:03d} | {p['short']} | [{en}](../../wiki/entities/{en}.md) |\n"
            )
        lines.append("\n")
    (ROOT / "sources/papers/humanoid_loco_manip_161_catalog.md").write_text(
        "".join(lines), encoding="utf-8"
    )


def write_category_pages(papers: list[dict]) -> None:
    abbr = """| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Loco-Manip | Loco-Manipulation | 行走与操作动力学耦合的全身任务 |
| WBC | Whole-Body Control | 协调全身关节满足多任务/约束的控制层 |
| VLA | Vision-Language-Action | 视觉-语言-动作多模态策略 |
| WM | World Model | 学习环境动态以供想象/规划的世界模型 |
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略 |"""
    for cn in range(1, 11):
        _, cname, cat_slug = CATS[cn]
        desc = {
            1: "底层身体控制、运动跟踪、抗扰动与通用动作执行",
            2: "手臂、躯干、根部和末端执行器之间的协同控制",
            3: "视觉完成目标定位、场景理解和操作闭环",
            4: "从语言、目标或条件输入生成全身动作和轨迹",
            5: "人类动作数据转成机器人可用的运动和交互先验",
            6: "开门、推物、搬运、触碰等具体接触任务",
            7: "训练数据如何高效采集",
            8: "本体、传感器和真实部署系统",
            9: "视觉、语言、动作和世界建模接到执行层",
            10: "人类 egocentric 视频学习操作经验和行为先验",
        }[cn]
        group = [p for p in papers if p["cat_num"] == cn]
        fname = f"loco-manip-161-category-{cn:02d}-{cat_slug}.md"
        lines = [
            f"""---
type: overview
tags: [loco-manipulation, humanoid, category-hub, survey]
status: complete
updated: {TODAY}
summary: "人形 Loco-Manip 161 篇 · {cn:02d} {cname}（{len(group)} 篇）— {desc}。"
related:
  - ./humanoid-loco-manip-161-papers-technology-map.md
sources:
  - ../../sources/blogs/wechat_embodied_ai_lab_humanoid_loco_manip_161_survey.md
  - ../../sources/papers/humanoid_loco_manip_161_catalog.md
---

# Loco-Manip 161 分类 {cn:02d}：{cname}

> **图谱分类节点**：**{cn:02d} {cname}**；总地图见 [人形 Loco-Manip 161 篇技术地图](./humanoid-loco-manip-161-papers-technology-map.md)。

## 英文缩写速查

{abbr}

## 核心问题

{desc}

## 本组论文（{len(group)} 篇）

| # | 工作 | Wiki 实体 |
|---|------|-----------|
"""
        ]
        for p in group:
            lines.append(
                f"| {p['num']:03d} | {p['short']} | [{p['entity']}](../entities/{p['entity']}.md) |\n"
            )
        lines.append(
            """
## 关联页面

- [人形 Loco-Manip 161 篇技术地图](./humanoid-loco-manip-161-papers-technology-map.md)
- [Loco-Manipulation 任务页](../tasks/loco-manipulation.md)

## 参考来源

- [wechat_embodied_ai_lab_humanoid_loco_manip_161_survey.md](../../sources/blogs/wechat_embodied_ai_lab_humanoid_loco_manip_161_survey.md)
- [humanoid_loco_manip_161_catalog.md](../../sources/papers/humanoid_loco_manip_161_catalog.md)

## 推荐继续阅读

- [运动小脑 64 篇技术地图](./humanoid-motion-cerebellum-technology-map.md)
"""
        )
        (ROOT / "wiki/overview" / fname).write_text("".join(lines), encoding="utf-8")


def update_parent_map(papers: list[dict]) -> None:
    path = ROOT / "wiki/overview/humanoid-loco-manip-161-papers-technology-map.md"
    text = path.read_text(encoding="utf-8")
    text = text.replace(
        "**节点复用：** 与姊妹篇重叠的论文 **链接到既有实体**（约 94/161 已挂接）；其余见 [catalog](../../sources/papers/humanoid_loco_manip_161_catalog.md)，后续按需升格。",
        "**独立节点：** 161 篇 **各建** `paper-loco-manip-161-{NNN}-*` 实体；与 `paper-hrl-stack-*` / `paper-notebook-*` 等同题者以 `related` 互链，见 [catalog](../../sources/papers/humanoid_loco_manip_161_catalog.md)。",
    )
    # add entity list in speed table - replace 七项目速查 section... actually parent has 十组分类 only
    path.write_text(text, encoding="utf-8")


def update_blog() -> None:
    path = ROOT / "sources/blogs/wechat_embodied_ai_lab_humanoid_loco_manip_161_survey.md"
    text = path.read_text(encoding="utf-8")
    text = text.replace(
        "全量索引：[humanoid_loco_manip_161_catalog.md](../../sources/papers/humanoid_loco_manip_161_catalog.md)（**94/161** 已挂接既有 wiki 实体，其余 catalog only）",
        "全量索引：[humanoid_loco_manip_161_catalog.md](../../sources/papers/humanoid_loco_manip_161_catalog.md)（**161/161** 各建 `paper-loco-manip-161-{NNN}-*` 独立实体）",
    )
    text = text.replace(
        "- 项目实体：`wiki/entities/agibot-world-2026.md`、`genie-sim-3.md`、`ge-sim-2.md`（已有）、`go-2.md`、`agibot-bfm-2.md`、`agibot-agile.md`、`genie-studio-agent.md`",
        "- 论文实体：`wiki/entities/paper-loco-manip-161-001-*.md` … `paper-loco-manip-161-161-*.md`（161 个独立节点）",
    )
    path.write_text(text, encoding="utf-8")


def main() -> None:
    text = RAW_MD.read_text(encoding="utf-8")
    papers = parse_papers(text)
    prior_map = load_prior_wiki_from_catalog()
    assert len(papers) == 161, len(papers)

    for p in papers:
        canonical = CANONICAL_ENTITY_BY_NUM.get(p["num"])
        if canonical:
            p["entity"] = canonical
            write_source(p, canonical)
            continue
        prior = prior_map.get(p["num"])
        en = write_entity(p, prior)
        write_source(p, en)
        p["entity"] = en

    write_catalog(papers)
    write_category_pages(papers)
    update_parent_map(papers)
    update_blog()
    print(f"Created {len(papers)} entities + sources")


if __name__ == "__main__":
    main()
