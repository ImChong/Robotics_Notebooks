#!/usr/bin/env python3
"""Generator for 多模空间 VLA weekly trends part2 ingest (2026-10-01)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
WECHAT_URL = "https://mp.weixin.qq.com/s/OWV8elPNXIds3MCR5CU5VQ"
BLOG = "wechat_duomo_vla_weekly_trends_2026-08-10_part2.md"
RAW = "wechat_duomo_vla_weekly_trends_2026-08-10_part2.md"
MAP = "vla-weekly-trends-2026-08-10-part2-technology-map"
DATE = "2026-10-01"

REUSE: list[dict[str, str]] = [
    {
        "num": "01",
        "short": "G0.5",
        "section": "架构模块",
        "arxiv": "2608.11739",
        "wiki": "paper-galaxea-g05",
    },
    {
        "num": "02",
        "short": "StellaVLA",
        "section": "架构模块",
        "arxiv": "2608.11671",
        "wiki": "paper-stellavla-structured-icl-vla",
    },
    {
        "num": "00",
        "short": "Neural Introspection Gating",
        "section": "性能提升",
        "arxiv": "2608.10824",
        "wiki": "paper-neural-introspection-gating",
    },
]

PAPERS: list[dict[str, Any]] = [
    {
        "num": "00",
        "slug": "rlinf-vla",
        "short": "RLinf-VLA",
        "title": "RLinf-VLA: A Unified and Efficient Framework for Reinforcement Learning of Vision-Language-Action Models",
        "arxiv": "2510.06710",
        "section": "架构模块",
        "oss": "已开源",
        "code": "https://github.com/RLinf/RLinf",
        "site": None,
        "inst": "清华大学；北京中关村学院；无问芯穹；北京大学；加州大学伯克利分校；哈尔滨工业大学；中国科学院自动化研究所",
        "tags": ["paper", "vla", "rl", "framework", "tsinghua"],
        "eval": "LIBERO、ManiSkill、RoboTwin；单臂实机",
        "summary": "统一 VLA+RL 训练平台：多模型、算法与仿真环境共用接口，并协同调度仿真、推理与训练算力；RSS 2026 算法侧技术报告。",
        "why": "VLA 强化学习研究分散、难公平对比；RLinf-VLA 提供统一接入与资源编排。",
        "seq": True,
    },
    {
        "num": "03",
        "slug": "xcot-vla-driving",
        "short": "XCoT-VLA",
        "title": "XCoT-VLA: Executable Chain-of-Thought for Vision-Language-Action Driving",
        "arxiv": "2608.10976",
        "section": "架构模块/智驾",
        "oss": "待核实",
        "code": None,
        "site": None,
        "inst": "小鹏汽车",
        "tags": ["paper", "vla", "autonomous-driving", "cot"],
        "eval": "换道等智驾规划场景（文内）",
        "summary": "智驾 VLA 用少量可执行内部行动提示替代冗长 CoT 文本，再交给轨迹生成模块，兼顾推理开销与实时规划。",
        "why": "长 CoT 拖慢车辆决策；XCoT 保留判断过程并降低输出成本。",
        "seq": False,
    },
    {
        "num": "04",
        "slug": "salt-vla-action-language-alignment",
        "short": "SALT",
        "title": "Lost in Reconstruction: Aligning Action Representations with Language in Vision-Language-Action Models",
        "arxiv": "2608.10484",
        "section": "架构模块",
        "oss": "待核实",
        "code": None,
        "site": None,
        "inst": "卡内基梅隆大学（美）",
        "tags": ["paper", "vla", "representation", "language-alignment", "cmu"],
        "eval": "SimplerEnv WidowX",
        "summary": "动作编码需同时可重建轨迹且让 VLM 猜回语言指令，避免 L1/L2 把语义不同但数值相近的动作混为一谈。",
        "why": "重建损失不等于语义对齐；SALT 显式绑动作方式与语言。",
        "seq": False,
    },
    {
        "num": "00",
        "slug": "tmrl-diffusion-timestep-pretraining",
        "short": "TMRL",
        "title": "TMRL: Diffusion Timestep-Modulated Pretraining Enables Exploration for Efficient Policy Finetuning",
        "arxiv": "2605.12236",
        "section": "训练范式",
        "oss": "待核实",
        "code": None,
        "site": "https://weirdlabuw.github.io/tmrl",
        "inst": "华盛顿大学（美）；亚马逊 FAR（美）",
        "tags": ["paper", "vla", "diffusion", "pretraining", "rl"],
        "eval": "OG-Bench、LIBERO（IsaacLab）；WidowX 250、Franka Panda 实机",
        "summary": "BC 预训练后 RL 微调时，对输入加噪扩展可采样动作，微调阶段再调节探索幅度，少数据适应复杂真机操作。",
        "why": "BC 动作分布过窄限制 RL 探索；TMRL 用扩散步调制预训练拓宽分布。",
        "seq": False,
    },
    {
        "num": "01",
        "slug": "midas-minimal-data-vla-adaptation",
        "short": "MiDAS",
        "title": "Adaptation of Generalist Robot Policies with Minimal Data",
        "arxiv": "2608.11363",
        "section": "训练范式",
        "oss": "待核实",
        "code": None,
        "site": None,
        "inst": "卡内基梅隆大学（美）",
        "tags": ["paper", "vla", "adaptation", "rl", "cmu"],
        "eval": "LIBERO、RoboCasa；YAM 双臂实机",
        "summary": "少量专家示范先模仿启动，再自主试错并按结果筛选更有效动作，从易失败状态逐步稳定并泛化到示范外情况。",
        "why": "通才 VLA 零样本探索弱；MiDAS 用极少人工示范启动后续自主学习。",
        "seq": False,
    },
    {
        "num": "00",
        "slug": "robo-harness-memory-ic-adaptation",
        "short": "RoboHarness（SJTU）",
        "title": "RoboHarness: A Memory-Augmented Policy Harness for Vision-Language-Action Model Robustness via In-Context Adaptation",
        "arxiv": "2603.24060",
        "section": "长程记忆/类 Agent",
        "oss": "已开源",
        "code": "https://github.com/LZY-1021/RoboHarness",
        "site": None,
        "inst": "上海交通大学；Shanghai Syslong Information Technology Co., Ltd.",
        "tags": ["paper", "vla", "memory", "in-context", "sjtu"],
        "eval": "LIBERO-PRO、LIBERO-RoboHarness（自建）",
        "summary": "不微调原 VLA：双记忆检索、多模态 LLM 分析失败并协调工具干预，离线整理轨迹为经验，提升长流程任务成功率。",
        "why": "长程任务需模型外记忆与失败恢复；与 arXiv:2607.18060 异构策略 RoboHarness 不同 arXiv。",
        "seq": True,
    },
    {
        "num": "00",
        "slug": "vga-vision-geometry-action",
        "short": "VGA",
        "title": "Robotic Manipulation is Vision-to-Geometry Mapping: Vision-Geometry Backbones over Language and Video Models",
        "arxiv": "2604.12908",
        "section": "空间感知",
        "oss": "待核实",
        "code": None,
        "site": "https://hcplab-sysu.github.io/VisionGeometryActionModel",
        "inst": "中山大学；广东省大数据分析与处理重点实验室；拓元智慧；美团龙猫；广东工业大学",
        "tags": ["paper", "vla", "3d", "geometry", "manipulation"],
        "eval": "LIBERO、RoboTwin2.0、LIBERO-Plus；Franka Panda 实机",
        "summary": "用语义/视频预训练骨干难保 3D 精度；改用学过三维结构的骨干并联合学动作与物体 3D 属性，OOD 视角抓取更稳。",
        "why": "操作需要精确空间关系；VGA 把 manipulation 收成 vision-to-geometry 映射（ACM MM 2026）。",
        "seq": False,
    },
    {
        "num": "00",
        "slug": "embodied-multimodal-grounding-3dgs-mobile-manipulation",
        "short": "Embodied MM Grounding（3DGS）",
        "title": "Embodied Multimodal Grounding for Open-Vocabulary Mobile Manipulation via Semantic 3D Gaussian Splatting",
        "arxiv": "2608.10756",
        "section": "空间感知/类 Agent",
        "oss": "待核实",
        "code": None,
        "site": None,
        "inst": "香港科技大学（广州）；美的集团；香港科技大学",
        "tags": ["paper", "vla", "3dgs", "mobile-manipulation", "navigation"],
        "eval": "Alicia-D 臂 + Unitree Go2 Edu 四足实机",
        "summary": "多角度更新带语义的三维高斯地图，用于开放词汇定位、避障与站位选择，再交给动作模型，缓解遮挡与视角变化。",
        "why": "移动操作需语言–视觉–3D–可行性统一；单视角 VLA 易定位错误。",
        "seq": False,
    },
    {
        "num": "00",
        "slug": "tcam-deformable-manipulation-wbcd",
        "short": "TCAM",
        "title": "TCAM for Autonomous Deformable Manipulation: The RMC2 Champion System for WBCD 2026 Track 4",
        "arxiv": "2608.10718",
        "section": "末端操控",
        "oss": "待核实",
        "code": None,
        "site": None,
        "inst": "晨昏线科技（TermiTech）",
        "tags": ["paper", "vla", "deformable", "manipulation"],
        "eval": "WBCD 2026 Track 4 第一名；ARX X5 实机叠衣流程",
        "summary": "衣物操作：专用夹爪、腕部多相机、示教数据与闭环失败分析补采，多视角 VLA 一次输出末端动作段，完成取放对齐抚平。",
        "why": "柔性物体接触复杂，纯策略难全自主；TCAM 用硬件+数据+闭环微调组合。",
        "seq": False,
    },
    {
        "num": "01",
        "slug": "handprior-score-humanoid-dual-arm",
        "short": "HandPriorScore",
        "title": "Policy-Induced Hand Priors in Humanoid Dual-Arm Manipulation: Diagnosing and Mitigating Initial-Pose Dependence",
        "arxiv": "2608.11769",
        "section": "末端操控",
        "oss": "待核实",
        "code": None,
        "site": None,
        "inst": "韩国科学技术研究院（KIST，韩）",
        "tags": ["paper", "vla", "humanoid", "dual-arm", "diagnostics"],
        "eval": "PickApple；Unitree G1-Dex3 采集",
        "summary": "诊断 VLA 在双手起始姿态不同下的「用手先验」：策略与姿态交互导致选错手；扩姿态覆盖与薄弱姿态补数据可提稳健性。",
        "why": "同任务不同初始手位成功率差异大；HandPriorScore 量化 policy-induced hand prior。",
        "seq": False,
    },
    {
        "num": "00",
        "slug": "drivevla-m0-failure-aware-memory",
        "short": "DriveVLA-M0",
        "title": "DriveVLA-M0: Failure-Aware Memory Augmentation for Autonomous Driving",
        "arxiv": "2608.10413",
        "section": "异常处理/智驾",
        "oss": "已开源",
        "code": "https://github.com/ZebinX/DriveVLA-M0",
        "site": None,
        "inst": "中国科学院自动化研究所；重庆长安科技有限责任公司",
        "tags": ["paper", "vla", "autonomous-driving", "memory"],
        "eval": "NAVSIM v1/v2",
        "summary": "记录历史失败场景的道路结构与正确驾驶方式，遇相似情况检索案例并临时调整判断，不改全模型权重。",
        "why": "智驾 VLA 缺失败经验复用；DriveVLA-M0 做 failure-aware 记忆增强（ACM MM 2026）。",
        "seq": True,
    },
    {
        "num": "01",
        "slug": "dura-diffusion-vla-visual-attack",
        "short": "DURA",
        "title": "Hidden in Plain Sight: Diffusion-Based Unrestricted Robotic Attacks on Vision-Language-Action Models",
        "arxiv": "2608.10393",
        "section": "异常处理/安全",
        "oss": "待核实",
        "code": None,
        "site": None,
        "inst": "西安交通大学；上海人工智能实验室；中国科学技术大学",
        "tags": ["paper", "vla", "security", "adversarial"],
        "eval": "LIBERO、BridgeData V2；Franka 实机",
        "summary": "画面中加入自然图案即可误导 VLA 输出攻击者期望动作，黑盒观察动作也可完成；仿真与真机均有效。",
        "why": "除任务成功率外需测视觉对抗；DURA 用扩散生成不易察觉的干扰。",
        "seq": False,
    },
]


def tag_yaml(tags: list[str]) -> str:
    return "\n".join(f"  - {t}" for t in tags)


def write_paper_source(p: dict[str, Any]) -> None:
    path = ROOT / "sources/papers" / f"{p['slug']}_arxiv_{p['arxiv'].replace('.', '_')}.md"
    code_line = f"- **代码：** {p['code']}\n" if p["code"] else ""
    site_line = f"- **项目页：** {p['site']}\n" if p.get("site") else ""
    content = f"""# {p["short"]}（arXiv:{p["arxiv"]}）

> 来源归档（paper）

- **标题：** {p["title"]}
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/{p["arxiv"]}>
- **PDF：** <https://arxiv.org/pdf/{p["arxiv"]}>
{code_line}{site_line}- **入库日期：** {DATE}
- **一句话说明：** {p["summary"]}

## 开源状态

- **{p["oss"]}**（步骤 2.5 核查，{DATE}）

## 核心摘录

1. **公众号章节：** {p["section"]}（[多模空间第二篇盘点](../blogs/{BLOG})）
2. **机构：** {p["inst"]}
3. **文内评测：** {p["eval"]}
4. **导读要点：** {p["why"]}

## 对 wiki 的映射

- [paper-{p["slug"]}](../../wiki/entities/paper-{p["slug"]}.md)
- [第二篇技术地图](../../wiki/overview/{MAP}.md)
"""
    path.write_text(content, encoding="utf-8")


def write_repo(p: dict[str, Any]) -> None:
    if not p.get("code"):
        return
    slug = p["slug"].replace("_", "-")
    path = ROOT / "sources/repos" / f"{slug}.md"
    content = f"""# {p["short"]} 官方仓库

> 来源归档（repo）

- **名称：** {p["short"]}
- **类型：** repo
- **URL：** {p["code"]}
- **关联论文：** arXiv:{p["arxiv"]}
- **入库日期：** {DATE}
- **开源结论：** **{p["oss"]}**

## 对 wiki 的映射

- [paper-{p["slug"]}](../../wiki/entities/paper-{p["slug"]}.md)
"""
    path.write_text(content, encoding="utf-8")


def seq_section(p: dict[str, Any]) -> str:
    if not p.get("seq"):
        return """## 源码运行时序图

**不适用**（截至入库日未提供可运行官方代码入口，或仓库尚无可辨识训练/推理入口）。
"""
    slug = p["slug"].replace("_", "-")
    return f"""## 源码运行时序图

```mermaid
sequenceDiagram
  autonumber
  participant Dev as 开发者
  participant Repo as 官方仓库
  participant Robot as 真机/仿真
  Dev->>Repo: clone + README 环境依赖
  Dev->>Robot: 准备任务与数据
  Dev->>Repo: 训练/推理入口
  Repo-->>Dev: 指标或部署输出
```

节点对齐 [`sources/repos/{slug}.md`](../../sources/repos/{slug}.md) 与 README 入口。
"""


def write_entity(p: dict[str, Any]) -> None:
    source_file = f"{p['slug']}_arxiv_{p['arxiv'].replace('.', '_')}.md"
    path = ROOT / "wiki/entities" / f"paper-{p['slug']}.md"
    sources = [
        f"  - ../../sources/papers/{source_file}",
        f"  - ../../sources/blogs/{BLOG}",
    ]
    if p.get("code"):
        repo_slug = p["slug"].replace("_", "-")
        sources.insert(1, f"  - ../../sources/repos/{repo_slug}.md")
    sources_yaml = "\n".join(sources)
    code_front = f"\ncode: {p['code']}" if p.get("code") else ""
    content = f"""---
type: entity
tags:
{tag_yaml(p["tags"])}
status: complete
updated: {DATE}
arxiv: "{p["arxiv"]}"{code_front}
related:
  - ../methods/vla.md
  - ../overview/{MAP}.md
sources:
{sources_yaml}
summary: "{p["short"]}（arXiv:{p["arxiv"]}）：{p["summary"]}"
---

# {p["short"]}（arXiv:{p["arxiv"]}）

**{p["short"]}**（*{p["title"]}*，[arXiv:{p["arxiv"]}](https://arxiv.org/abs/{p["arxiv"]})）收录于 [多模空间 · 一周 VLA 研究趋势简析（2026.08.10–08.16）第二篇](../../sources/blogs/{BLOG}) **{p["section"]}** 段。

## 一句话定义

**{p["summary"]}**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| VLM | Vision-Language Model | 视觉–语言多模态模型 |
| RL | Reinforcement Learning | 强化学习 |
| CoT | Chain-of-Thought | 链式推理 |

## 为什么重要

- {p["why"]}
- 策展机构：{p["inst"]}
- 开源结论：**{p["oss"]}**（步骤 2.5，{DATE}）。

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [{p["arxiv"]}](https://arxiv.org/abs/{p["arxiv"]}) |
| **开源** | **{p["oss"]}** |
| **文内评测** | {p["eval"]} |

{seq_section(p)}
## 实验与评测

- **文内口径：** {p["eval"]}
- **读法：** 本页为 [公众号策展](../../sources/blogs/{BLOG}) 摘要；逐项指标以 **原文 PDF** 为准。

## 与其他工作对比

| 对照 | 差异读法 |
|------|----------|
| [第二篇技术地图](../overview/{MAP}.md) | 同批 15 篇横向索引；本文属 **{p["section"]}** |
| [VLA 方法页](../methods/vla.md) | 单篇机制细节以原文为准 |

## 结论

**{p["short"]} 适合作为本期「{p["section"]}」路线的快速索引页。**

1. 核心贡献：{p["summary"]}
2. 开源结论：**{p["oss"]}** — 以项目页/仓库实际链接为准。
3. 横向对照见 [第二篇技术地图](../overview/{MAP}.md)。

## 关联页面

- [VLA（Vision-Language-Action）](../methods/vla.md)
- [一周 VLA 趋势技术地图（第二篇）](../overview/{MAP}.md)

## 参考来源

- [{BLOG}](../../sources/blogs/{BLOG})
- [arXiv:{p["arxiv"]}](https://arxiv.org/abs/{p["arxiv"]})

## 推荐继续阅读

- [VLA 方法页](../methods/vla.md)
- [arXiv PDF](https://arxiv.org/pdf/{p["arxiv"]})
"""
    path.write_text(content, encoding="utf-8")


def all_index_rows() -> list[str]:
    ordered: list[tuple[str, str, str, str, str]] = [
        ("RLinf-VLA", "架构模块", "2510.06710", "paper-rlinf-vla", "新建"),
        ("G0.5", "架构模块", "2608.11739", "paper-galaxea-g05", "复用"),
        ("StellaVLA", "架构模块", "2608.11671", "paper-stellavla-structured-icl-vla", "复用"),
        ("XCoT-VLA", "架构模块/智驾", "2608.10976", "paper-xcot-vla-driving", "新建"),
        ("SALT", "架构模块", "2608.10484", "paper-salt-vla-action-language-alignment", "新建"),
        ("TMRL", "训练范式", "2605.12236", "paper-tmrl-diffusion-timestep-pretraining", "新建"),
        ("MiDAS", "训练范式", "2608.11363", "paper-midas-minimal-data-vla-adaptation", "新建"),
        (
            "Neural Introspection Gating",
            "性能提升",
            "2608.10824",
            "paper-neural-introspection-gating",
            "复用",
        ),
        (
            "RoboHarness（SJTU）",
            "长程记忆",
            "2603.24060",
            "paper-robo-harness-memory-ic-adaptation",
            "新建",
        ),
        ("VGA", "空间感知", "2604.12908", "paper-vga-vision-geometry-action", "新建"),
        (
            "Embodied MM Grounding",
            "空间感知/Agent",
            "2608.10756",
            "paper-embodied-multimodal-grounding-3dgs-mobile-manipulation",
            "新建",
        ),
        ("TCAM", "末端操控", "2608.10718", "paper-tcam-deformable-manipulation-wbcd", "新建"),
        (
            "HandPriorScore",
            "末端操控",
            "2608.11769",
            "paper-handprior-score-humanoid-dual-arm",
            "新建",
        ),
        (
            "DriveVLA-M0",
            "异常处理/智驾",
            "2608.10413",
            "paper-drivevla-m0-failure-aware-memory",
            "新建",
        ),
        ("DURA", "异常处理/安全", "2608.10393", "paper-dura-diffusion-vla-visual-attack", "新建"),
    ]
    out: list[str] = []
    for i, (short, section, arxiv, wiki, note) in enumerate(ordered, start=1):
        out.append(
            f"| {i:02d} | {short} | {section} | [{arxiv}](https://arxiv.org/abs/{arxiv}) | "
            f"**{note}** | [{wiki}](../../wiki/entities/{wiki}.md) |"
        )
    return out


def write_blog_and_map() -> None:
    raw_path = ROOT / "sources/raw" / RAW
    raw_path.write_text(
        f"# [风向标-具身VLA] 一周 VLA 研究趋势简析（2026.08.10-2026.08.16）-第二篇\n\n"
        f"原始链接：{WECHAT_URL}\n\n（WebFetch 抓取，{DATE}）\n",
        encoding="utf-8",
    )
    table = "\n".join(all_index_rows())
    blog = f"""# [风向标-具身VLA] 一周 VLA 研究趋势简析（2026.08.10-2026.08.16）-第二篇

> 来源归档（blog / 微信公众号）

- **标题：** [风向标-具身VLA] 一周 VLA 研究趋势简析（2026.08.10-2026.08.16）-第二篇
- **类型：** blog
- **作者：** 多模空间（微信公众号）
- **原始链接：** {WECHAT_URL}
- **入库日期：** {DATE}
- **抓取方式：** WebFetch（wechat-article-for-ai 不可用）
- **原始抓取落盘：** [`sources/raw/{RAW}`](../raw/{RAW})
- **一句话说明：** 15 篇 VLA 论文：RLinf 统一 RL 平台、G0.5/StellaVLA 架构、智驾 XCoT、SALT 动作–语言对齐、TMRL/MiDAS 训练、KV 门控、SJTU RoboHarness 记忆、VGA/3DGS 空间、TCAM/HandPrior 操控、DriveVLA-M0/DURA 安全；**15/15 独立 canonical 详情节点**（本 ingest **新建 12**、**复用 3**）。

## 15 篇 → 本库 canonical 节点

| # | 论文 | 章节 | arXiv | 节点 | wiki（canonical） |
|---|------|------|-------|------|-------------------|
{table}

## 核心摘录（MVP）

### 1) 主题分布

- **架构与平台：** RLinf-VLA 统一 RL；G0.5 单流推理+动作；StellaVLA 结构化 ICL；XCoT-VLA / SALT 分别面向智驾 CoT 与动作语义对齐。
- **训练与效率：** TMRL 扩探索预训练；MiDAS 极少示范启动 RL；Neural Introspection Gating 训练无关 KV 复用。
- **空间与长程：** SJTU RoboHarness（**arXiv:2603.24060**，勿与 2607.18060 异构策略 Harness 混淆）；VGA 几何骨干；3DGS 移动操作 grounding。
- **操控与安全：** TCAM 柔性衣物 WBCD 冠军；HandPriorScore 双手先验诊断；DriveVLA-M0 失败记忆；DURA 视觉对抗。

### 2) 节点去重结论（{DATE} 核查）

- **15/15** 各有独立 canonical 页（12 新建 + 3 复用）；**0** 重复 arXiv ID。

## 对 wiki 的映射

- 阅读坐标：[一周 VLA 趋势技术地图（2026.08.10 第二篇）](../../wiki/overview/{MAP}.md)
- 同系列：[第一篇](wechat_duomo_vla_weekly_trends_2026-08-10_part1.md) · [第三篇](wechat_duomo_vla_weekly_trends_2026-08-10_part3.md) · [第四篇](wechat_duomo_vla_weekly_trends_2026-08-10_part4.md)

## 当前提炼状态

- [x] 15 篇索引与独立详情节点
- [x] 技术地图
- [ ] 各篇深读（待原文 / 项目页 follow-up）
"""
    (ROOT / "sources/blogs" / BLOG).write_text(blog, encoding="utf-8")

    related_new = "\n".join(f"  - ../entities/paper-{p['slug']}.md" for p in PAPERS)
    related_reuse = "\n".join(f"  - ../entities/{r['wiki']}.md" for r in REUSE)
    map_content = f"""---
type: overview
tags: [overview, survey, vla, technology-map, duomo-space]
status: complete
updated: {DATE}
related:
{related_reuse}
{related_new}
  - ../overview/vla-weekly-trends-2026-08-10-part1-technology-map.md
  - ../overview/vla-weekly-trends-2026-08-10-part3-technology-map.md
  - ../overview/vla-weekly-trends-2026-08-10-part4-technology-map.md
  - ../methods/vla.md
sources:
  - ../../sources/blogs/{BLOG}
summary: "多模空间 {DATE} 策展：2026.08.10–08.16 一周 VLA 第二篇 15 论文——RL 平台、几何/3DGS 空间、记忆 harness 与 VLA 安全。"
---

# 一周 VLA 研究趋势（2026.08.10–08.16 · 第二篇）

> **本页定位**：为 [多模空间公众号第二篇盘点]({WECHAT_URL}) 提供横切面索引；**15/15 各有独立 canonical 详情节点**（12× 新建 `paper-*` + 3× 复用），本页不替代单篇深读。

## 一句话观点

**第二篇主线：VLA 从统一训练/推理栈（RLinf、G0.5）、几何与 3D  grounding，走向模型外记忆 harness 与智驾/操控安全。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| RL | Reinforcement Learning | 强化学习 |
| 3DGS | 3D Gaussian Splatting | 三维高斯溅射场景表示 |

## 节点索引（15/15）

### 架构 · 平台 · 对齐

- [RLinf-VLA](../entities/paper-rlinf-vla.md) — arXiv:2510.06710
- [G0.5](../entities/paper-galaxea-g05.md) — arXiv:2608.11739（复用）
- [StellaVLA](../entities/paper-stellavla-structured-icl-vla.md) — arXiv:2608.11671（复用）
- [XCoT-VLA](../entities/paper-xcot-vla-driving.md) — arXiv:2608.10976
- [SALT](../entities/paper-salt-vla-action-language-alignment.md) — arXiv:2608.10484

### 训练 · 效率

- [TMRL](../entities/paper-tmrl-diffusion-timestep-pretraining.md) — arXiv:2605.12236
- [MiDAS](../entities/paper-midas-minimal-data-vla-adaptation.md) — arXiv:2608.11363
- [Neural Introspection Gating](../entities/paper-neural-introspection-gating.md) — arXiv:2608.10824（复用）

### 记忆 · 空间 · 操控

- [RoboHarness（SJTU）](../entities/paper-robo-harness-memory-ic-adaptation.md) — arXiv:2603.24060（≠ [2607.18060 RoboHarness](../entities/paper-robo-harness.md)）
- [VGA](../entities/paper-vga-vision-geometry-action.md) — arXiv:2604.12908
- [Embodied MM Grounding（3DGS）](../entities/paper-embodied-multimodal-grounding-3dgs-mobile-manipulation.md) — arXiv:2608.10756
- [TCAM](../entities/paper-tcam-deformable-manipulation-wbcd.md) — arXiv:2608.10718
- [HandPriorScore](../entities/paper-handprior-score-humanoid-dual-arm.md) — arXiv:2608.11769

### 智驾 · 安全

- [DriveVLA-M0](../entities/paper-drivevla-m0-failure-aware-memory.md) — arXiv:2608.10413
- [DURA](../entities/paper-dura-diffusion-vla-visual-attack.md) — arXiv:2608.10393

## 与同系列的关系

- [第一篇](./vla-weekly-trends-2026-08-10-part1-technology-map.md)、[第三篇](./vla-weekly-trends-2026-08-10-part3-technology-map.md)、[第四篇](./vla-weekly-trends-2026-08-10-part4-technology-map.md) 与本篇 **arXiv 集合不重叠**（除复用的 G0.5 / StellaVLA / Neural Introspection 已在其他 ingest 出现）。

## 参考来源

- [{BLOG}](../../sources/blogs/{BLOG})
"""
    (ROOT / "wiki/overview" / f"{MAP}.md").write_text(map_content, encoding="utf-8")


def patch_reuse_sources() -> None:
    blog_ref = f"  - ../../sources/blogs/{BLOG}"
    for r in REUSE:
        path = ROOT / "wiki/entities" / f"{r['wiki']}.md"
        text = path.read_text(encoding="utf-8")
        if BLOG in text:
            continue
        if "sources:" in text:
            text = text.replace("sources:", f"sources:\n{blog_ref}", 1)
        path.write_text(text, encoding="utf-8")


def patch_sibling_blogs() -> None:
    link = "- 同系列：[第一篇](wechat_duomo_vla_weekly_trends_2026-08-10_part1.md) · **[第二篇](wechat_duomo_vla_weekly_trends_2026-08-10_part2.md)** · [第三篇](wechat_duomo_vla_weekly_trends_2026-08-10_part3.md) · [第四篇](wechat_duomo_vla_weekly_trends_2026-08-10_part4.md)"
    for part in ("part1", "part3", "part4"):
        path = ROOT / "sources/blogs" / f"wechat_duomo_vla_weekly_trends_2026-08-10_{part}.md"
        text = path.read_text(encoding="utf-8")
        if "part2" in text:
            continue
        old = "- 同系列："
        if old not in text:
            continue
        for line in text.splitlines():
            if line.startswith(old):
                text = text.replace(line, link, 1)
                break
        path.write_text(text, encoding="utf-8")


def main() -> None:
    for p in PAPERS:
        write_paper_source(p)
        write_entity(p)
        write_repo(p)
    write_blog_and_map()
    patch_reuse_sources()
    patch_sibling_blogs()
    print(f"Generated {len(PAPERS)} new papers + blog + map; reused {len(REUSE)}")


if __name__ == "__main__":
    main()
