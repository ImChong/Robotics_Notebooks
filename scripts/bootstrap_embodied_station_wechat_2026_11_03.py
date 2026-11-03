#!/usr/bin/env python3
"""Bootstrap ingest: 具身智能小站 2026-11-03（49 篇）公众号盘点。

- 抓取方式：`scripts/fetch_wechat_article.py`（Chromium + 微信 UA）；本脚本直接解析抓取的纯文本。
- 49 条论文 → 49 个独立 `wiki/entities/paper-*.md` 节点：新建 44，复用 5（arXiv 已在库内）。
- 开源结论（步骤 2.5）：GitHub 仓库用 `git ls-remote` 核实可达；项目页用 curl 读取并扫描
  GitHub 链接 / "soon" 字样（入库日 2026-11-03）。OPEN_CHECK 为当日核查结果。
- 用法：python3 scripts/bootstrap_embodied_station_wechat_2026_11_03.py <抓取的 txt>
"""

from __future__ import annotations

import re
import sys
from datetime import date
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
TODAY = date.today().isoformat()

ARTICLES: dict[str, dict[str, Any]] = {
    "a1": {
        "key": "49_papers_sacq_action_aware_quantization_2026-11-03",
        "title": "机器人不仅要完成任务，还要知道哪里出了错",
        "url": "https://mp.weixin.qq.com/s/KBuNOCHIWGw905j4IpOhZw",
        "date": "2026-11-03",
        "n": 49,
        "map": "embodied-station-2026-11-03-49-papers-technology-map.md",
        "map_title": "动作感知量化、失败定位与可验证自主：具身智能小站 11-03 期 49 篇阅读坐标",
        "map_summary": "具身智能小站 2026-11-03 盘点：SACQ 动作感知超低比特 VLA 量化（深读）、可达性采样极限、膝关节外骨骼，以及策略部署、评测数据、人形灵巧控制、导航与优化等 49 篇；49/49 独立节点（新建 44 / 复用 5）。",
        "view": "本期主线是让机器人不只完成任务、还要知道哪里出了错：SACQ 把量化目标对齐最终动作误差，另有失败模式定位、操作质量与仿真接触伪影审计、可验证自主等工作；其余覆盖策略部署、评测数据、人形与灵巧控制、导航与优化。深读 SACQ，跟进可达性采样极限与仿真驱动膝关节外骨骼。",
    },
}


def blog_name(a: str) -> str:
    return f"wechat_embodied_station_{ARTICLES[a]['key']}.md"


M = "../methods/"
C = "../concepts/"
T = "../tasks/"
VLA = M + "vla.md"
RL = M + "reinforcement-learning.md"
IL = M + "imitation-learning.md"
DP = M + "diffusion-policy.md"
GWM = M + "generative-world-models.md"
WAM = C + "world-action-models.md"
SIMEVAL = C + "simulation-evaluation-infrastructure.md"
S2R = C + "sim2real.md"
TACT = C + "tactile-sensing.md"
DEX = C + "dexterous-kinematics.md"
CONTACT = C + "contact-rich-manipulation.md"
RETARGET = C + "motion-retargeting.md"
LLMCTRL = C + "llm-robotics-control-interfaces.md"
DATA4 = C + "embodied-data-collection-four-layers-taxonomy.md"
MANIP = T + "manipulation.md"
BIMAN = T + "bimanual-manipulation.md"
TELEOP = T + "teleoperation.md"
LOCO = T + "locomotion.md"
VLN = T + "vision-language-navigation.md"

OPEN = "已开源"
PART = "部分开源"
TBD = "待发布"
NOLINK = "项目页未列 GitHub"

# 已在库内的 arXiv → 复用节点（不新建，仅补来源与地图关联）
REUSE: dict[str, str] = {
    "2610.40207": "paper-roboatom-atomic-action-supervision",
    "2610.40349": "paper-touchbridge-contact-aware-dexterous-human-interaction",
    "2610.40559": "paper-shape-servoing-deformable-objects-tutorial-part1",
    "2610.40364": "paper-unsupervised-environment-design-quadruped-benchmark",
    "2610.40248": "paper-llm-guided-planar-loco-manipulation-quadruped",
}

# 公众号序号 → 分组
GROUPS: dict[str, list[int]] = {
    "策略学习与部署": [1, 5, 9, 21, 34, 15, 10, 29, 36, 4, 28],
    "评测、数据与基准": [7, 12, 19, 23, 24, 32, 43, 47, 30, 11],
    "人形与灵巧控制": [6, 16, 27, 31, 37, 39, 18, 41, 26],
    "世界模型、记忆与交互": [17, 20, 22, 25],
    "四足与导航": [44, 45, 46, 14],
    "规划、优化与感知基础": [2, 8, 13, 35, 38, 48, 49, 33, 40],
    "外骨骼与教程": [3, 42],
}
GROUP_OF = {n: g for g, ns in GROUPS.items() for n in ns}
GROUP_TAGS = {
    "策略学习与部署": ["vla", "policy-deployment"],
    "评测、数据与基准": ["benchmark", "evaluation"],
    "人形与灵巧控制": ["humanoid", "dexterous"],
    "世界模型、记忆与交互": ["world-model", "memory"],
    "四足与导航": ["quadruped", "navigation"],
    "规划、优化与感知基础": ["planning", "optimization"],
    "外骨骼与教程": ["exoskeleton", "tutorial"],
}
GROUP_RELATED = {
    "策略学习与部署": [VLA, IL, RL],
    "评测、数据与基准": [SIMEVAL, VLA, MANIP],
    "人形与灵巧控制": [LOCO, RETARGET, DEX],
    "世界模型、记忆与交互": [WAM, GWM, VLA],
    "四足与导航": [LOCO, VLN, RL],
    "规划、优化与感知基础": [MANIP, LOCO, S2R],
    "外骨骼与教程": [MANIP, LOCO],
}
EXTRA_RELATED = {
    1: [S2R],
    10: [LLMCTRL],
    25: [LLMCTRL],
    28: [LLMCTRL],
    18: [TACT],
    23: [TACT],
    41: [TACT],
    26: [CONTACT],
    39: [CONTACT],
    47: [BIMAN],
    32: [S2R],
}

# 公众号标题无短名（或冒号过远）时的手工短名
SHORT_OVERRIDE = {
    2: "Sampling-Based Reachability Limits",
    3: "Knee Exoskeleton Sim-Derived Torque",
    6: "∇Mimic",
    11: "Gym-Anything for Robotics",
    13: "Differentiable ILC Black-Box Manipulators",
    14: "Decentralized Collision Verification Swarms",
    15: "Validation Loss vs Real-Robot Success",
    21: "Editing the Goal",
    22: "Knowing Where to Look",
    32: "Simulator Contact Artifacts Audit",
    35: "Contact-Based Million-Robot Planning",
    42: "Shape Servoing Tutorial Part I",
    44: "UED Quadruped Benchmark",
    45: "LLM-Guided Planar Loco-Manipulation",
    46: "Dont Risk It",
    48: "VI RGB-D SLAM with Encoders",
    49: "GPU Incremental EDT",
}

# 与库内既有（不同论文）节点同名 / 非 ASCII 的 slug
SLUG_OVERRIDE = {6: "nablamimic", 46: "dont-risk-it-crowd-navigation"}

# 公众号「资源」链接按代码仓库对待（本期无）
RESOURCE_AS_CODE: set[int] = set()

# 步骤 2.5 核查（2026-11-03）：repo = git ls-remote 结果；site = 项目页 curl 扫描。
# 未出现的论文：有 code 且 ls-remote 成功 → 已开源。
OPEN_CHECK: dict[int, tuple[str, str]] = {
    13: (NOLINK, "项目页 HTTP 200，未见 GitHub 链接或 coming soon 字样。"),
    15: (NOLINK, "公众号项目页入口为 Hugging Face 数据集页（HTTP 200），未见代码仓库。"),
    16: (NOLINK, "项目页 HTTP 200，未见 GitHub 链接或 coming soon 字样。"),
    17: (NOLINK, "项目页 HTTP 200，未见 GitHub 链接或 coming soon 字样。"),
    19: (NOLINK, "项目页 HTTP 200，未见 GitHub 链接或 coming soon 字样。"),
    21: (NOLINK, "项目页为 doubleblind 匿名页面（HTTP 200），未见代码仓库。"),
    22: (NOLINK, "项目页 HTTP 200，未见 GitHub 链接或 coming soon 字样。"),
    24: (NOLINK, "项目页 HTTP 200，未见 GitHub 链接或 coming soon 字样。"),
    27: (
        TBD,
        "项目页 HTTP 200，资源项标注 Available soon（curl 扫描）；页面仅链接 EGalahad/mimic-lite（相关库，未核对是否即本文代码）。",
    ),
    31: (OPEN, "公众号未给代码链接；项目页链接的 wendell0218/Movens-Zero 经 git ls-remote 可达。"),
    34: (
        OPEN,
        "公众号未给代码链接；项目页链接的 D-Robotics-AI-Lab/RTPolicy 经 git ls-remote 可达。",
    ),
    36: (TBD, "项目页 HTTP 200，Code Repository 标 coming soon（curl 扫描）。"),
    37: (NOLINK, "项目页为 anonymous.4open.science 匿名链接（HTTP 200），未见 GitHub 链接。"),
    41: (NOLINK, "项目页 HTTP 200，未见 GitHub 链接或 coming soon 字样。"),
    14: (OPEN, "仓库 unl-nimbus-lab/arl-swarm（分支 verifier_2025）经 git ls-remote 可达。"),
    47: (
        PART,
        "公众号代码入口为 clothdojo/clothdojo.github.io（项目页站点仓库，git ls-remote 可达），不一定含完整训练 / 评测代码，内容未核对。",
    ),
    48: (
        PART,
        "公众号代码入口为 leavesnight/ORB_SLAM2（git ls-remote 可达），是否为论文官方实现未核对。",
    ),
}


def _sentences(lines: list[str]) -> list[str]:
    out: list[str] = []
    for l in lines:
        l = re.sub(r"^(一句话判断|看点|结果|方法亮点)[：:]", "", l.lstrip("• ").strip())
        out += [s.strip() + "。" for s in re.split(r"。", l) if s.strip()]
    return out


def parse(txt: str) -> list[dict]:
    lines = [l.strip() for l in txt[txt.index("先看这一篇") :].split("\n")]
    blocks: list[dict] = []
    for l in lines:
        m = re.match(r"^(\d{1,2})([^\d\s].+)$", l)
        if m:
            blocks.append(dict(n=int(m.group(1)), head=m.group(2), lines=[]))
        elif blocks and l:
            blocks[-1]["lines"].append(l)
    skip = re.compile(
        r"^(另外 \d+ 篇|\d+ 篇一眼|下面每篇|关注本号|整理不易|具身智能小站|Like|Share|Boost|Comment|​)"
    )
    out = []
    for b in blocks:
        ls = b["lines"]
        links: dict[str, str] = {}
        body = []
        for l in ls[1:]:
            mm = re.match(r"^(论文|代码|项目页|资源)：(\S+)$", l)
            if mm:
                links[mm.group(1)] = mm.group(2)
            elif not skip.match(l) and l != "你可以带走的两点":
                body.append(l)
        out.append(dict(n=b["n"], head=b["head"], title=ls[0], body=body, links=links))
    return out


def _abbr(short: str, title: str) -> list[tuple[str, str, str]]:
    rows = [(short, title, "本文方法 / 基准的简称")]
    for key, full, zh in [
        ("Vision-Language-Action", "Vision-Language-Action", "视觉–语言–动作策略"),
        ("VLA", "Vision-Language-Action", "视觉–语言–动作策略"),
        ("VLM", "Vision-Language Model", "视觉语言模型"),
        ("UMI", "Universal Manipulation Interface", "手持通用操作采集接口"),
        ("WAM", "World Action Model", "世界–动作联合模型"),
        ("CSI", "Channel State Information", "WiFi 信道状态信息"),
        ("SysML", "Systems Modeling Language", "系统建模语言"),
        ("3D", "3D", "三维"),
    ]:
        if key in title and zh not in [r[2] for r in rows]:
            rows.append((key if key != "Vision-Language-Action" else "VLA", full, zh))
    return rows


def build_papers(txt: str) -> list[dict]:
    papers = []
    for o in parse(txt):
        n = o["n"]
        ax = o["links"]["论文"].rsplit("/", 1)[-1]
        title = o["title"]
        if n in SHORT_OVERRIDE:
            short = SHORT_OVERRIDE[n]
        else:
            idx = title.find(":")
            if not 0 < idx <= 32:
                raise SystemExit(f"需手工短名: {n} {title}")
            short = title[:idx]
        g = GROUP_OF[n]
        if ax in REUSE:
            papers.append(dict(n=n, a="a1", g=g, short=short, arxiv=ax, open=OPEN, reuse=True))
            continue
        slug = SLUG_OVERRIDE.get(n) or re.sub(r"[^a-z0-9]+", "-", short.lower()).strip("-")
        code = o["links"].get("代码")
        project = o["links"].get("项目页")
        resource = o["links"].get("资源")
        if n in RESOURCE_AS_CODE and not code:
            code, resource = resource, None
        if code and "github.com" not in code:
            # 非 GitHub 的「代码」入口（官网 / 匿名链接）：无项目页时当项目页，否则只在核查说明里提
            project = project or code
            code = None
        S = _sentences(o["body"])
        if n in OPEN_CHECK:
            op, note = OPEN_CHECK[n]
        elif code:
            op, note = (
                OPEN,
                "仓库可达（git ls-remote）。" + ("项目页 HTTP 200。" if project else ""),
            )
        else:
            op, note = NOLINK, "项目页 HTTP 200，未见 GitHub 链接或 coming soon 字样。"
        if code and "gitlab.com/-/snippets" in code:
            code_field = code
        else:
            code_field = code
        papers.append(
            dict(
                n=n,
                a="a1",
                g=g,
                slug=slug,
                short=short,
                arxiv=ax,
                title=title,
                code=code_field,
                project=project,
                resource=resource,
                open=op,
                open_note=note,
                tags=["paper"] + GROUP_TAGS[g],
                one=S[0],
                why=o["head"] + "。" + (S[1] if len(S) > 1 else ""),
                mech=S[0],
                metrics=("公众号转述：" + S[-1])
                if len(S) > 1
                else "公众号未给具体数字，本页不引用。",
                concl=o["head"] + "：" + (S[-1] if len(S) > 1 else S[0]),
                related=GROUP_RELATED[g] + EXTRA_RELATED.get(n, []),
                abbr=_abbr(short, title),
            )
        )
    return papers


def _yaml_list(items: list[str], indent: int = 2) -> str:
    pad = " " * indent
    return "\n".join(f"{pad}- {x}" for x in items)


def _ax_file(p: dict) -> str:
    return f"{p['slug'].replace('-', '_')}_arxiv_{p['arxiv'].replace('.', '_')}.md"


def _repo_file(p: dict) -> str:
    return f"{p['slug'].replace('-', '_')}.md"


def _site_file(p: dict) -> str:
    return f"{p['slug']}.md"


def _paper_source(p: dict) -> str:
    a = ARTICLES[p["a"]]
    lines = [
        f"# {p['short']}（arXiv:{p['arxiv']}）",
        "",
        "> 来源归档（paper）",
        "",
        f"- **标题：** {p['title']}",
        "- **类型：** paper",
        f"- **arXiv：** <https://arxiv.org/abs/{p['arxiv']}>",
        f"- **PDF：** <https://arxiv.org/pdf/{p['arxiv']}>",
    ]
    if p.get("project"):
        lines.append(f"- **项目页：** <{p['project']}>")
    if p.get("code"):
        lines.append(f"- **代码：** <{p['code']}>")
    if p.get("resource"):
        lines.append(f"- **资源：** <{p['resource']}>")
    lines += [
        f"- **入库日期：** {TODAY}",
        f"- **线索来源：** [{a['title']}](../blogs/{blog_name(p['a'])})（具身智能小站，{a['date']}）",
        f"- **一句话说明：** {p['one']}",
        "",
        "## 开源状态",
        "",
        f"- **{p['open']}**（步骤 2.5 核查，{TODAY}）：{p['open_note']}",
        "",
        "## 核心摘录（公众号转述，未核验原文表格）",
        "",
        f"- {p['why']}",
        f"- {p['mech']}",
        f"- 指标：{p['metrics']}",
        "",
        "## 对 wiki 的映射",
        "",
        f"- [paper-{p['slug']}](../../wiki/entities/paper-{p['slug']}.md)",
        f"- [{a['map_title']}](../../wiki/overview/{a['map']})",
    ]
    return "\n".join(lines) + "\n"


def _repo(p: dict) -> str:
    return f"""# {p["short"]} 官方仓库

> 来源归档（repo）

- **标题：** {p["title"]}
- **类型：** repo
- **链接：** {p["code"]}
- **arXiv：** <https://arxiv.org/abs/{p["arxiv"]}>
- **入库日期：** {TODAY}
- **一句话说明：** {p["one"]}
- **沉淀到 wiki：** [`wiki/entities/paper-{p["slug"]}.md`](../../wiki/entities/paper-{p["slug"]}.md)

## 开源状态

- **{p["open"]}**：{p["open_note"]} 仓库内容以 README 与 release 为准（本次仅用 `git ls-remote` 核实可达，未逐个阅读 README）。
"""


def _site(p: dict) -> str:
    code = f"- **代码：** <{p['code']}>\n" if p.get("code") else ""
    return f"""# {p["short"]} 项目页

> 来源归档（site）

- **标题：** {p["title"]}
- **类型：** site
- **链接：** {p["project"]}
- **arXiv：** <https://arxiv.org/abs/{p["arxiv"]}>
{code}- **入库日期：** {TODAY}
- **一句话说明：** {p["one"]}
- **沉淀到 wiki：** [`wiki/entities/paper-{p["slug"]}.md`](../../wiki/entities/paper-{p["slug"]}.md)

## 开源状态

- **{p["open"]}**（步骤 2.5 核查，{TODAY}）：{p["open_note"]}
"""


def _seq(p: dict) -> str:
    if p["open"] in (OPEN, PART) and p.get("code"):
        repo = p["code"].split("github.com/")[-1]
        return f"""## 源码运行时序图

以下为按仓库入口（`{repo}`）推断的通用运行链路；入库日仅核实仓库可达，未逐文件阅读，具体脚本名以 README 为准。

```mermaid
sequenceDiagram
  participant U as 用户
  participant R as {repo}
  participant E as 环境 / 数据
  participant M as 模型 / 策略
  U->>R: clone + 安装依赖（README）
  U->>E: 准备基准 / 数据集或真机接口
  U->>R: 训练或加载预训练权重
  R->>M: 构建 {p["short"]} 模型
  U->>R: 运行评测脚本
  R->>E: 重置环境、取观测
  E-->>M: 观测
  M-->>E: 动作 / 预测
  R-->>U: 指标与日志
```
"""
    if p["open"] == PART:
        return f"""## 源码运行时序图

**部分开源** — {p["open_note"]} 待可运行代码公开后补图。
"""
    return f"""## 源码运行时序图

**不适用（{p["open"]}）** — {p["open_note"]} 不编造运行脚本；代码公开后再补。
"""


def _entity(p: dict) -> str:
    ax = p["arxiv"]
    slug = p["slug"]
    a = ARTICLES[p["a"]]
    srcs = [f"../../sources/papers/{_ax_file(p)}"]
    if p.get("code"):
        srcs.append(f"../../sources/repos/{_repo_file(p)}")
    if p.get("project"):
        srcs.append(f"../../sources/sites/{_site_file(p)}")
    srcs.append(f"../../sources/blogs/{blog_name(p['a'])}")
    fm_extra = ""
    if p.get("project"):
        fm_extra += f"project: {p['project']}\n"
    if p.get("code"):
        fm_extra += f"code: {p['code']}\n"
    abbrev = "\n".join(f"| {x} | {y} | {z} |" for x, y, z in p["abbr"])
    links = []
    if p.get("project"):
        links.append(f"[项目页]({p['project']})")
    if p.get("code"):
        links.append(f"[代码]({p['code']})")
    if p.get("resource"):
        links.append(f"[资源]({p['resource']})")
    link_txt = ("，" + "，".join(links)) if links else ""
    related = p["related"] + [f"../overview/{a['map']}"]
    rel_lines = "\n".join(f"- [{r.split('/')[-1].replace('.md', '')}]({r})" for r in related)
    reads = [f"- [arXiv PDF](https://arxiv.org/pdf/{ax})"]
    if p.get("project"):
        reads.append(f"- [项目页]({p['project']})")
    if p.get("code"):
        reads.append(f"- [{p['code']}]({p['code']})")
    if p.get("resource"):
        reads.append(f"- [{p['resource']}]({p['resource']})")
    summary = f"{p['short']}（arXiv:{ax}）：{p['one']}".replace('"', "'")
    return f"""---
type: entity
project_id: {slug}
tags:
{_yaml_list(p["tags"], 2)}
status: complete
updated: {TODAY}
arxiv: "{ax}"
{fm_extra}related:
{_yaml_list(related, 2)}
sources:
{_yaml_list(srcs, 2)}
summary: "{summary}"
---

# {p["short"]}（arXiv:{ax}）

**{p["short"]}**（*{p["title"]}*，[arXiv:{ax}](https://arxiv.org/abs/{ax}){link_txt}）来自 [具身智能小站 {a["date"]} 盘点](../../sources/blogs/{blog_name(p["a"])})，属「{p["g"]}」组。

## 一句话定义

**{p["one"]}**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
{abbrev}

## 为什么重要

- {p["why"]}
- 开源结论：**{p["open"]}**（步骤 2.5，{TODAY}）。{p["open_note"]}
- 同期同组工作见 [{a["map_title"]}](../overview/{a["map"]})。

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [{ax}](https://arxiv.org/abs/{ax}) |
| **开源** | **{p["open"]}** |
| **要点** | {p["mech"]} |
| **文内指标** | {p["metrics"]} |

{_seq(p)}
## 实验与评测

- {p["metrics"]}
- **读法：** 本页为索引级摘要，依据公众号转述的官方摘要与仓库信息整理；逐项数字与 baseline 以原文 PDF 为准。

## 与其他工作对比

- 横向索引见 [{a["map_title"]}](../overview/{a["map"]})；同 arXiv 不重复造页。

## 结论

**{p["concl"]}**

1. 开源边界：**{p["open"]}** — {p["open_note"]}
2. 核心机制：{p["mech"]}
3. 复现与选型前核对任务协议、硬件与基准口径，勿直接横比公众号摘录数字。

## 关联页面

{rel_lines}

## 参考来源

- [{_ax_file(p)}](../../sources/papers/{_ax_file(p)})
- [{blog_name(p["a"])}](../../sources/blogs/{blog_name(p["a"])})
- [arXiv:{ax}](https://arxiv.org/abs/{ax})

## 推荐继续阅读

{chr(10).join(reads)}
"""


def _real(a: str) -> list[dict]:
    return [p for p in PAPERS if p["a"] == a]


def _wiki_of(p: dict) -> str:
    return REUSE[p["arxiv"]] if p.get("reuse") else f"paper-{p['slug']}"


def _rows(a: str) -> list[tuple[str, str, str, str, str, str, bool]]:
    rows = []
    for p in _real(a):
        w = _wiki_of(p)
        reuse = bool(p.get("reuse"))
        tag = "（**复用**）" if reuse else ""
        tag2 = "（复用）" if reuse else ""
        rows.append(
            (
                p["g"],
                p["short"],
                p["arxiv"],
                p["open"],
                f"[{w}](../../wiki/entities/{w}.md){tag}",
                f"[{w}](../entities/{w}.md){tag2}",
                reuse,
            )
        )
    return rows


def _blog(a: str) -> str:
    A = ARTICLES[a]
    rows = _rows(a)
    new = sum(1 for r in rows if not r[6])
    body = "\n".join(
        f"| {i:02d} | {r[0]} | {r[1]} | [{r[2]}](https://arxiv.org/abs/{r[2]}) | **{r[3]}** | {r[4]} |"
        for i, r in enumerate(rows, 1)
    )
    return f"""# {A["title"]}

> 来源归档（blog / 微信公众号）

- **标题：** {A["title"]}
- **类型：** blog
- **作者：** 具身智能小站（微信公众号）
- **原始链接：** {A["url"]}
- **发表日期：** {A["date"]}
- **入库日期：** {TODAY}
- **抓取方式：** `scripts/fetch_wechat_article.py`（Chromium + 微信 UA）
- **原始抓取落盘：** [`sources/raw/wechat_embodied_station_{A["key"]}.md`](../raw/wechat_embodied_station_{A["key"]}.md)
- **一句话说明：** {A["view"]} 共 {A["n"]} 篇；**{A["n"]}/{A["n"]} 独立详情节点**（本 ingest **新建 {new}**、**复用 {len(rows) - new}**）。

## {A["n"]} 篇 → 本库节点

| # | 组 | 论文 | arXiv | 开源结论 | wiki |
|---|----|------|-------|----------|------|
{body}

## 对 wiki 的映射

- **{A["n"]}/{A["n"]} 独立详情节点**；同 arXiv 不重复造页（复用节点已补本文来源与地图关联）。
- 阅读坐标：[{A["map_title"]}](../../wiki/overview/{A["map"]})
- 开源结论口径：GitHub 仓库用 `git ls-remote` 核实可达；项目页用 curl 读取扫描 GitHub 链接与 "coming soon"；「项目页未列 GitHub」表示入库日页面无代码链接且未见开源承诺字样。

## 当前提炼状态

- [x] 公众号正文抓取（{TODAY}）
- [x] {A["n"]} 篇独立节点（{new} 新建 / {len(rows) - new} 复用）
- [x] 项目页 / 仓库开源状态核查（步骤 2.5）
- [ ] 原文实验表格核验（节点内数字仅为公众号转述）
"""


def _map(a: str) -> str:
    A = ARTICLES[a]
    rows = _rows(a)
    groups: dict[str, list] = {}
    for r in rows:
        groups.setdefault(r[0], []).append(r)
    sections = []
    for g, rs in groups.items():
        tbl = "\n".join(f"| {r[1]} | {r[5]} | {r[3]} |" for r in rs)
        sections.append(
            f"### {g}（{len(rs)} 篇）\n\n| 论文 | 节点 | 开源 |\n|------|------|------|\n{tbl}\n"
        )
    related = [f"../entities/{_wiki_of(p)}.md" for p in _real(a)[:3]] + [VLA, WAM]
    flow_nodes = "\n".join(
        f'  G{i}["{g}（{len(rs)}）"] --> GOAL' for i, (g, rs) in enumerate(groups.items())
    )
    return f"""---
type: overview
tags: [overview, survey, embodied-ai, vla, world-model, benchmark, technology-map, wechat]
status: complete
updated: {TODAY}
related:
{_yaml_list(related, 2)}
sources:
  - ../../sources/blogs/{blog_name(a)}
summary: "{A["map_summary"]}"
---

# {A["map_title"]}

> **本页定位**：为 [具身智能小站 · {A["title"]}]({A["url"]})（{A["date"]}）提供按研究方向组织的阅读坐标；{A["n"]} 篇全部有独立详情节点。

## 一句话观点

**{A["view"]}**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| WAM | World Action Model | 世界–动作联合模型 |
| VLM | Vision-Language Model | 视觉语言模型 |

## 为什么单独做这张地图

- 公众号一次列出 {A["n"]} 篇（1 深读 + 2 跟进 + 46 扫读）、方向跨度大，需要横切面索引；每篇只保留一个判断，细节在各节点。
- 节点内的数字为公众号转述，未核验原文表格；开源结论为入库日核查结果，见各节点「为什么重要」。

## 流程总览

```mermaid
flowchart TB
{flow_nodes}
  GOAL[可部署具身系统]
```

## 分组索引

{chr(10).join(sections)}
## 关联页面

- [VLA](../methods/vla.md)
- [World Action Models](../concepts/world-action-models.md)

## 参考来源

- [{blog_name(a)}](../../sources/blogs/{blog_name(a)})

## 推荐继续阅读

- [{rows[0][1]}]({rows[0][5].split("](")[1].split(")")[0]})
- [{rows[1][1]}]({rows[1][5].split("](")[1].split(")")[0]})
"""


def _patch_reuse(a: str) -> None:
    """复用节点：补本期博客来源与地图关联（幂等）。"""
    blog = f"  - ../../sources/blogs/{blog_name(a)}\n"
    mapl = f"  - ../overview/{ARTICLES[a]['map']}\n"
    for w in REUSE.values():
        path = ROOT / "wiki/entities" / f"{w}.md"
        text = path.read_text(encoding="utf-8")
        head, sep, body = text.partition("\n---\n")
        if not sep:
            raise SystemExit(f"frontmatter 缺失: {path}")
        lines = head.split("\n")

        def _append(key: str, item: str) -> None:
            if item.strip() in (l.strip() for l in lines):
                return
            try:
                i = next(i for i, l in enumerate(lines) if l.startswith(f"{key}:"))
            except StopIteration:
                lines.append(f"{key}:")
                lines.append(item.rstrip("\n"))
                return
            if lines[i].strip() != f"{key}:":
                raise SystemExit(f"{path}: {key} 为行内写法，需手工补")
            j = i + 1
            while j < len(lines) and lines[j].startswith("  "):
                j += 1
            lines.insert(j, item.rstrip("\n"))

        _append("sources", blog)
        _append("related", mapl)
        lines = [f"updated: {TODAY}" if l.startswith("updated:") else l for l in lines]
        path.write_text("\n".join(lines) + sep + body, encoding="utf-8")


def main() -> None:
    raw = Path(sys.argv[1])
    txt = raw.read_text(encoding="utf-8")
    papers = build_papers(txt)
    global PAPERS
    PAPERS = papers
    if len(papers) != 49:
        raise SystemExit(f"应为 49 篇，实际 {len(papers)}")
    new = [p for p in PAPERS if not p.get("reuse")]
    seen: set[str] = set()
    slugs: set[str] = set()
    for p in PAPERS:
        if p["arxiv"] in seen:
            raise SystemExit(f"重复 arXiv: {p['arxiv']}")
        seen.add(p["arxiv"])
        if p.get("reuse"):
            if not (ROOT / "wiki/entities" / f"{REUSE[p['arxiv']]}.md").exists():
                raise SystemExit(f"复用节点不存在: {p['arxiv']}")
            continue
        if p["slug"] in slugs:
            raise SystemExit(f"slug 重复: {p['slug']}")
        slugs.add(p["slug"])
        ent = ROOT / "wiki/entities" / f"paper-{p['slug']}.md"
        if ent.exists():
            raise SystemExit(f"节点已存在: {ent}")
        for rel in p["related"]:
            if not (ROOT / "wiki/entities" / rel).resolve().exists():
                raise SystemExit(f"关联页不存在: {rel} ({p['slug']})")
    for a, A in ARTICLES.items():
        dst = ROOT / "sources/raw" / f"wechat_embodied_station_{A['key']}.md"
        dst.write_text(
            f"# {A['title']}\n\n> 原始抓取（{TODAY}，`scripts/fetch_wechat_article.py`），仅供溯源；来源：{A['url']}\n\n"
            + txt,
            encoding="utf-8",
        )
        (ROOT / "sources/blogs" / blog_name(a)).write_text(_blog(a), encoding="utf-8")
        (ROOT / "wiki/overview" / str(A["map"])).write_text(_map(a), encoding="utf-8")
        _patch_reuse(a)
    for p in new:
        (ROOT / "sources/papers" / _ax_file(p)).write_text(_paper_source(p), encoding="utf-8")
        (ROOT / "wiki/entities" / f"paper-{p['slug']}.md").write_text(_entity(p), encoding="utf-8")
        if p.get("code"):
            (ROOT / "sources/repos" / _repo_file(p)).write_text(_repo(p), encoding="utf-8")
        if p.get("project"):
            (ROOT / "sources/sites" / _site_file(p)).write_text(_site(p), encoding="utf-8")
    print(f"Wrote 1 blog, 1 map, {len(new)} new entities, patched {len(REUSE)} reuse nodes")


PAPERS: list[dict] = []

if __name__ == "__main__":
    main()
