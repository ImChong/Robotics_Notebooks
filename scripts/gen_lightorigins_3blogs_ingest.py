#!/usr/bin/env python3
"""Generator for Light Origins 3 tech blogs ingest (2026-09-21)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
MAP = "lightorigins-3blogs-technology-map"
DATE = "2026-09-21"

BLOGS = {
    "react": {
        "file": "lightorigins_light_react_2026-09-09.md",
        "url": "https://www.lightorigins.com/blog/light-react",
        "title": "Light REACT：面向规模化部署的全身韧性智能",
        "date": "2026-09-09",
    },
    "nav": {
        "file": "lightorigins_lightnav_0_2026-09-01.md",
        "url": "https://www.lightorigins.com/blog/lightnav-0",
        "title": "LightNav-0：以规模化 Real2Sim2Real 实现零样本通用导航",
        "date": "2026-09-01",
    },
    "parkour": {
        "file": "lightorigins_lightparkour_2026-08-03.md",
        "url": "https://www.lightorigins.com/blog/lightparkour",
        "title": "LightParkour：通过 Real2Sim2Real 拓展人形机器人的跑酷技能",
        "date": "2026-08-03",
    },
}

# Papers / benchmarks / products cited — each gets independent entity
ENTITIES: list[dict[str, Any]] = [
    # --- light-react cited papers ---
    {
        "slug": "tolebi",
        "kind": "paper",
        "short": "TOLEBI",
        "title": "TOLEBI: Learning Fault-Tolerant Bipedal Locomotion via Online Status Estimation and Fallibility Rewards",
        "arxiv": "2602.05596",
        "venue": "ICRA 2026",
        "blog": "react",
        "oss": "待核实",
        "site": None,
        "tags": ["paper", "humanoid", "fault-tolerant", "locomotion", "reinforcement-learning"],
        "summary": "在线关节状态估计 + fallibility rewards 学双足容错行走；仿真注入关节锁定/掉电/扰动，TOCABI 真机验证。",
        "why": "Light REACT 韧性金字塔「硬件受损后调整移动」的对照：TOLEBI 显式估计关节状态而非仅靠交互历史 ICL。",
        "source_file": "tolebi_arxiv_2602_05596.md",
    },
    {
        "slug": "locoformer",
        "kind": "paper",
        "short": "LocoFormer",
        "title": "LocoFormer: Generalist Locomotion via Long-Context Adaptation",
        "arxiv": "2509.23745",
        "venue": "CoRL 2025",
        "blog": "react",
        "oss": "待发布",
        "site": "https://generalist-locomotion.github.io/",
        "tags": [
            "paper",
            "locomotion",
            "transformer-xl",
            "in-context-learning",
            "cross-embodiment",
        ],
        "summary": "大规模 PPO + 程序生成机器人 + Transformer-XL 跨 episode 记忆；未见形态/电机故障下 test-time 适应。",
        "why": "Light REACT 对比 Transformer 64 帧上下文时引用；LocoFormer 代表「长上下文运动适应」前序。",
        "source_file": "locoformer_corl_2025.md",
        "replaces_stub": "paper-notebook-locoformer-generalist-locomotion-via-long-contex",
    },
    {
        "slug": "humanup-getting-up",
        "kind": "paper",
        "short": "HUMANUP",
        "title": "Learning Getting-Up Policies for Real-World Humanoid Robots",
        "arxiv": "2502.12152",
        "venue": "RSS 2025",
        "blog": "react",
        "oss": "待核实",
        "site": "https://humanoid-getup.github.io/",
        "tags": ["paper", "humanoid", "fall-recovery", "getting-up", "unitree-g1"],
        "summary": "两阶段 RL：先发现起身轨迹再 refine 为可部署策略；G1 六地形俯卧/仰卧起身 78.3% 成功率。",
        "why": "Light REACT 韧性金字塔 L2「跌倒后起身恢复行走」的直接前序。",
        "source_file": "humanup_getting_up_arxiv_2502_12152.md",
        "replaces_stub": "paper-notebook-learning-getting-up-policies-for-real-world-huma",
    },
    # --- lightnav products / benchmarks ---
    {
        "slug": "lightnav-er",
        "kind": "product",
        "short": "LightNav-ER",
        "title": "LightNav-ER：具身推理中期训练模型",
        "arxiv": None,
        "venue": "Light Origins Tech Blog 2026-09-01",
        "blog": "nav",
        "oss": "未单独发布",
        "site": "https://www.lightorigins.com/blog/lightnav-0",
        "tags": ["entity", "navigation", "embodied-reasoning", "vlm", "light-origins"],
        "summary": "LightNav-0 第一阶段 ER 中期训练产物；8 项具身推理基准平均 67.4（4 第一 / 4 第二），为 SFT 提供空间先验。",
        "why": "对齐段「先理解空间再学行动」的可操作锚点；MolmoER / Gemini Robotics-ER 同路线对照。",
        "source_file": "lightnav_er_lightorigins_2026.md",
    },
    {
        "slug": "insight-bench",
        "kind": "benchmark",
        "short": "INSIGHT-Bench",
        "title": "INSIGHT-Bench：Real2Sim2Real 导航数据引擎评测集",
        "arxiv": None,
        "venue": "Light Origins Tech Blog 2026-09-01",
        "blog": "nav",
        "oss": "部分公开",
        "site": "https://www.lightorigins.com/blog/lightnav-0",
        "tags": ["benchmark", "navigation", "vln", "real2sim2real", "light-origins"],
        "summary": "1683 训练场景 / 53090 片段 + 210 评测场景 / 1097 片段；覆盖 HM3D/MP3D/InteriorGS/HabitatGS/VLNVerse 与五类指代。",
        "why": "LightNav Real2Sim2Real 引擎的可复现数据子集与评测矩阵入口。",
        "source_file": "insight_bench_lightorigins_2026.md",
    },
    # --- lightnav cited external ---
    {
        "slug": "trackvla",
        "kind": "paper",
        "short": "TrackVLA",
        "title": "TrackVLA: Embodied Visual Tracking in the Wild",
        "arxiv": "2505.23189",
        "venue": "2025",
        "blog": "nav",
        "oss": "待核实",
        "site": None,
        "tags": ["paper", "navigation", "visual-tracking", "vla", "evt-bench"],
        "summary": "提出 EVT-Bench 野外具身视觉跟踪基准；LightNav-0 第三阶段 GRPO 在线 RL 在该基准干扰跟踪上继续提升。",
        "why": "LightNav 三阶段训练 RL 阶段的评测锚点。",
        "source_file": "trackvla_arxiv_2505_23189.md",
    },
    {
        "slug": "rxr",
        "kind": "paper",
        "short": "RxR",
        "title": "Room-Across-Room: Multilingual VLN with Dense Spatiotemporal Grounding",
        "arxiv": "2010.07954",
        "venue": "EMNLP 2020",
        "blog": "nav",
        "oss": "数据集公开",
        "site": None,
        "tags": ["paper", "vln", "dataset", "multilingual", "navigation"],
        "summary": "多语言 VLN 数据集与 dense spatiotemporal grounding；LightNav-0 SFT 数据池组分之一。",
        "why": "LightNav 第二阶段对齐 SFT 的公开导航语料对照。",
        "source_file": "rxr_emnlp_2020.md",
    },
    {
        "slug": "hm3d-ovon",
        "kind": "paper",
        "short": "HM3D-OVON",
        "title": "Open-Vocabulary Object Goal Navigation with Embodied Foundation Models",
        "arxiv": "2409.01535",
        "venue": "2024",
        "blog": "nav",
        "oss": "待核实",
        "site": None,
        "tags": ["paper", "navigation", "objectnav", "open-vocabulary", "benchmark"],
        "summary": "HM3D 上开放词汇目标导航设定；LightNav-0 十项单目评测之一。",
        "why": "LightNav 跨任务泛化评测矩阵中的 ObjectNav 轴。",
        "source_file": "hm3d_ovon_2024.md",
    },
    {
        "slug": "srdf-vln-flywheel",
        "kind": "paper",
        "short": "SRDF",
        "title": "Bootstrapping Language-Guided Navigation Learning with Self-Refining Data Flywheel",
        "arxiv": "2407.07689",
        "venue": "2024",
        "blog": "nav",
        "oss": "待核实",
        "site": None,
        "tags": ["paper", "vln", "data-generation", "navigation"],
        "summary": "Self-Refining Data Flywheel 导航数据自举；LightNav SFT 数据池 4.7M 样本来源之一（博客脚注 SRDF）。",
        "why": "与 LightNav Real2Sim2Real 引擎同属「合成/自举导航数据」路线对照。",
        "source_file": "srdf_vln_flywheel_2024.md",
    },
    {
        "slug": "habitatgs",
        "kind": "dataset",
        "short": "HabitatGS",
        "title": "HabitatGS：高斯泼溅场景重建（LightNav 数据引擎来源）",
        "arxiv": None,
        "venue": "LightNav Tech Blog 引用",
        "blog": "nav",
        "oss": "待核实",
        "site": None,
        "tags": ["dataset", "navigation", "gaussian-splatting", "simulation"],
        "summary": "INSIGHT-Bench 训练片段 10.9% 来自 HabitatGS；无标注场景经 Molmo2 多视角目标检测后接入统一目标库。",
        "why": "LightNav Real2Sim2Real 场景格式之一。",
        "source_file": "habitatgs_lightnav_2026.md",
    },
    {
        "slug": "interiorgs",
        "kind": "dataset",
        "short": "InteriorGS",
        "title": "InteriorGS：室内高斯泼溅场景（LightNav 数据引擎来源）",
        "arxiv": None,
        "venue": "LightNav Tech Blog 引用",
        "blog": "nav",
        "oss": "待核实",
        "site": None,
        "tags": ["dataset", "navigation", "gaussian-splatting", "simulation"],
        "summary": "INSIGHT-Bench 训练 16.1% 来自 InteriorGS；可导入官方 3D bbox 或 Molmo2 开放词汇标注。",
        "why": "LightNav 引擎「有标注 / 无标注」双路径场景源。",
        "source_file": "interiorgs_lightnav_2026.md",
    },
    {
        "slug": "vlnverse",
        "kind": "dataset",
        "short": "VLNVerse",
        "title": "VLNVerse：VLN 场景数据源（LightNav 数据引擎）",
        "arxiv": None,
        "venue": "LightNav Tech Blog 引用",
        "blog": "nav",
        "oss": "待核实",
        "site": None,
        "tags": ["dataset", "navigation", "vln"],
        "summary": "INSIGHT-Bench 训练 8.0% 片段来源；丰富视觉外观与空间结构。",
        "why": "LightNav 多源场景覆盖矩阵中的一列。",
        "source_file": "vlnverse_lightnav_2026.md",
    },
    # embodied reasoning benchmarks (LightNav-ER eval)
    {
        "slug": "er-point-bench",
        "kind": "benchmark",
        "short": "Point-Bench",
        "title": "Point-Bench：图像点定位具身推理基准",
        "arxiv": None,
        "venue": "LightNav-ER 评测套件",
        "blog": "nav",
        "oss": "待核实",
        "site": None,
        "tags": ["benchmark", "embodied-reasoning", "pointing", "vlm"],
        "summary": "LightNav-ER 八项具身推理评测之一；点定位能力占 ER 中期训练数据 35.14%。",
        "why": "读 LightNav「空间先验 → 行动」需对齐 ER 评测定义。",
        "source_file": "er_point_bench_lightnav_2026.md",
    },
    {
        "slug": "refspatial",
        "kind": "benchmark",
        "short": "RefSpatial",
        "title": "RefSpatial：空间指代具身推理基准",
        "arxiv": None,
        "venue": "LightNav-ER 评测套件",
        "blog": "nav",
        "oss": "待核实",
        "site": None,
        "tags": ["benchmark", "embodied-reasoning", "spatial-reasoning"],
        "summary": "LightNav-ER 评测项；检验语言-空间指代与图像 grounding。",
        "why": "LightNav Point CoT 空间意图 token 的能力对照。",
        "source_file": "refspatial_lightnav_2026.md",
    },
    {
        "slug": "robospatial",
        "kind": "benchmark",
        "short": "RoboSpatial",
        "title": "RoboSpatial：机器人空间推理基准",
        "arxiv": None,
        "venue": "LightNav-ER 评测套件",
        "blog": "nav",
        "oss": "待核实",
        "site": None,
        "tags": ["benchmark", "embodied-reasoning", "spatial-reasoning"],
        "summary": "POI 与 VQA 子项分别计入 LightNav-ER 八项评测。",
        "why": "LightNav 博客脚注 [3] 明确列出。",
        "source_file": "robospatial_lightnav_2026.md",
    },
    {
        "slug": "where2place",
        "kind": "benchmark",
        "short": "Where2Place",
        "title": "Where2Place：可放置空间推理基准",
        "arxiv": None,
        "venue": "LightNav-ER 评测套件",
        "blog": "nav",
        "oss": "待核实",
        "site": None,
        "tags": ["benchmark", "embodied-reasoning", "affordance"],
        "summary": "LightNav-ER 八项评测之一；与可通行点 / 落脚点推理相关。",
        "why": "连接 ER 中期训练与 LightNav Point CoT 可通行点监督。",
        "source_file": "where2place_lightnav_2026.md",
    },
    {
        "slug": "cv-bench-embodied",
        "kind": "benchmark",
        "short": "CV-Bench",
        "title": "CV-Bench：通用视觉与抽象推理基准（具身 ER 套件）",
        "arxiv": None,
        "venue": "LightNav-ER 评测套件",
        "blog": "nav",
        "oss": "待核实",
        "site": None,
        "tags": ["benchmark", "embodied-reasoning", "visual-reasoning"],
        "summary": "LightNav-ER 评测项；占 ER 训练采样 20% 的通用视觉/抽象推理数据对应评测。",
        "why": "区分「导航 SFT」与「纯视觉推理」能力边界。",
        "source_file": "cv_bench_embodied_lightnav_2026.md",
    },
    {
        "slug": "erqa",
        "kind": "benchmark",
        "short": "ERQA",
        "title": "ERQA：具身推理问答基准",
        "arxiv": None,
        "venue": "LightNav-ER 评测套件",
        "blog": "nav",
        "oss": "待核实",
        "site": None,
        "tags": ["benchmark", "embodied-reasoning", "vqa"],
        "summary": "LightNav-ER 八项评测之一；VQA 式具身推理。",
        "why": "LightNav 第一阶段 ER 与第二阶段 VQA 保留比例（22.4%）的能力锚点。",
        "source_file": "erqa_lightnav_2026.md",
    },
    {
        "slug": "embspatial",
        "kind": "benchmark",
        "short": "EmbSpatial",
        "title": "EmbSpatial：具身空间推理基准",
        "arxiv": None,
        "venue": "LightNav-ER 评测套件",
        "blog": "nav",
        "oss": "待核实",
        "site": None,
        "tags": ["benchmark", "embodied-reasoning", "spatial-reasoning"],
        "summary": "LightNav-ER 评测套件成员；Blog 脚注 [3] 列出。",
        "why": "完整覆盖 LightNav 引用的 ER 八项基准。",
        "source_file": "embspatial_lightnav_2026.md",
    },
    # VLM / backbone cited
    {
        "slug": "molmo2-vlm",
        "kind": "model",
        "short": "Molmo2",
        "title": "Molmo2: Open Weights and Data for Vision-Language Models with Video Understanding and Grounding",
        "arxiv": "2601.03225",
        "venue": "2026",
        "blog": "nav",
        "oss": "待核实",
        "site": None,
        "tags": ["entity", "vlm", "open-weights", "video-grounding"],
        "summary": "LightNav 数据引擎用 Molmo2 在无标注 Gaussian Splatting 场景做开放词汇目标定位。",
        "why": "Real2Sim2Real 目标库构建链路的 VLM 组件。",
        "source_file": "molmo2_vlm_2026.md",
    },
    {
        "slug": "molmo-er",
        "kind": "model",
        "short": "MolmoER",
        "title": "MolmoAct2 / MolmoER：具身推理 VLM 骨干",
        "arxiv": None,
        "venue": "MolmoAct2 2026",
        "blog": "nav",
        "oss": "待核实",
        "site": None,
        "tags": ["entity", "vlm", "embodied-reasoning", "navigation"],
        "summary": "MolmoAct2 使用 MolmoER 作为 VLM 骨干做动作推理；LightNav 博客列为 LightNav-ER 同路线对照。",
        "why": "「先 ER 再 SFT」三阶段导航后训练的行业平行实现。",
        "source_file": "molmo_er_molmoact2_2026.md",
    },
    {
        "slug": "seed2-0",
        "kind": "model",
        "short": "Seed2.0",
        "title": "Seed2.0 Model Card（ByteDance Seed）",
        "arxiv": None,
        "venue": "ByteDance Seed 2026",
        "blog": "nav",
        "oss": "闭源/API",
        "site": None,
        "tags": ["entity", "vlm", "video-understanding"],
        "summary": "LightNav 引擎用 Seed2.0 生成并验证导航指令（模板/视频 VLM → 语义筛选 → 终点画面验证）。",
        "why": "Real2Sim2Real 指令标注链的外部 VLM 依赖。",
        "source_file": "seed2_0_bytedance_2026.md",
    },
    {
        "slug": "qwen3-vl",
        "kind": "model",
        "short": "Qwen3-VL",
        "title": "Qwen3-VL：视觉-语言基座（LightNav-ER 初始化）",
        "arxiv": None,
        "venue": "Alibaba Qwen 2025–2026",
        "blog": "nav",
        "oss": "部分开源",
        "site": None,
        "tags": ["entity", "vlm", "open-weights"],
        "summary": "LightNav-ER 自 Qwen3-VL 初始化；博客报告 ER 中期训练平均 +4.3 分提升。",
        "why": "LightNav 对齐段公开的技术栈锚点。",
        "source_file": "qwen3_vl_lightnav_2026.md",
    },
    # --- lightparkour hardware ---
    {
        "slug": "lightbot-0",
        "kind": "hardware",
        "short": "Lightbot 0",
        "title": "Lightbot 0：亮源新创自研人形平台",
        "arxiv": None,
        "venue": "Light Origins 2026",
        "blog": "parkour",
        "oss": "未开源",
        "site": "https://www.lightorigins.com/blog/lightparkour",
        "tags": ["entity", "humanoid", "hardware", "light-origins"],
        "summary": "90 cm / 18.9 kg / 21 DoF QDD 人形；D435 深度 + 骨盆 IMU；Jetson Orin Nano 50 Hz 机载策略。",
        "why": "LightParkour / Light-O1 真机演示统一硬件载体。",
        "source_file": "lightbot_0_lightorigins_2026.md",
    },
]

EXISTING = [
    ("BeyondMimic", "wiki/methods/beyondmimic.md", "react"),
    ("SONIC", "wiki/methods/sonic-motion-tracking.md", "react"),
    ("HoST", "wiki/entities/paper-host-humanoid-standingup.md", "react"),
    ("PPO", "wiki/methods/ppo.md", "react"),
    ("AMP", "wiki/methods/amp-reward.md", "react"),
    ("Gemini Robotics-ER", "wiki/entities/gemini-robotics.md", "nav"),
    ("LightNav-0", "wiki/entities/paper-lightnav-0.md", "nav"),
    ("Light REACT", "wiki/entities/light-react.md", "nav"),
    ("Light-Loco-Parkour", "wiki/entities/paper-light-loco-parkour.md", "parkour"),
    ("R2R", "wiki/entities/paper-vln-01-r2r.md", "nav"),
    ("ScaleVLN", "wiki/entities/paper-vln-08-scalevln.md", "nav"),
    ("VLN-CE", "wiki/entities/paper-vln-02-vln-ce.md", "nav"),
    ("DAgger", "wiki/methods/dagger.md", "parkour"),
]


def tag_yaml(tags: list[str]) -> str:
    return "\n".join(f"  - {t}" for t in tags)


def entity_filename(e: dict[str, Any]) -> str:
    if e["kind"] == "paper":
        return f"paper-{e['slug']}.md"
    return f"{e['slug']}.md"


def write_source(e: dict[str, Any]) -> None:
    path = ROOT / "sources/papers" / e["source_file"]
    arxiv_block = ""
    if e.get("arxiv"):
        arxiv_block = (
            f"- **arXiv：** <https://arxiv.org/abs/{e['arxiv']}>\n"
            f"- **PDF：** <https://arxiv.org/pdf/{e['arxiv']}>\n"
        )
    site_line = f"- **项目页/引用：** {e['site']}\n" if e.get("site") else ""
    blog = BLOGS[e["blog"]]
    content = f"""# {e["short"]}（{e.get("venue", "归档")}）

> 来源归档 — Light Origins Tech Blog ingest（{blog["title"]}）

- **标题：** {e["title"]}
- **类型：** {e["kind"]}
{arxiv_block}{site_line}- **Tech Blog：** <{blog["url"]}>
- **入库日期：** {DATE}
- **一句话说明：** {e["summary"]}

## 开源状态

- **{e["oss"]}**（步骤 2.5，{DATE}）

## 核心摘录

1. **引用上下文：** [{blog["file"]}](../blogs/{blog["file"]})
2. **导读：** {e["why"]}

## 对 wiki 的映射

- [{entity_filename(e).replace(".md", "")}](../../wiki/entities/{entity_filename(e)})
- [3 篇技术地图](../../wiki/overview/{MAP}.md)
"""
    path.write_text(content, encoding="utf-8")


def write_entity(e: dict[str, Any]) -> None:
    fname = entity_filename(e)
    path = ROOT / "wiki/entities" / fname
    blog = BLOGS[e["blog"]]
    sources = [
        f"  - ../../sources/papers/{e['source_file']}",
        f"  - ../../sources/blogs/{blog['file']}",
    ]
    arxiv_front = f'\narxiv: "{e["arxiv"]}"' if e.get("arxiv") else ""
    is_paper = e["kind"] == "paper"
    conclusion = ""
    if is_paper:
        conclusion = f"""
## 结论

**{e["short"]} 在 Light Origins 三篇 Tech Blog 引用链中承担「{e["why"][:40]}…」角色——部署前以 arXiv/项目页与开源状态为准。**

1. 开源：**{e["oss"]}**；勿凭博客脚注臆断可复现性。
2. 与 [Light REACT](./light-react.md) / [LightNav-0](./paper-lightnav-0.md) / [Light-Loco-Parkour](./paper-light-loco-parkour.md) 按能力轴交叉阅读。
3. 定量指标以原文 PDF 为准；本页为博客 ingest 级摘要。
"""
        seq = """
## 源码运行时序图

**不适用**（截至入库日无官方可运行实现，或仅有博客/论文叙述）。
"""
    else:
        conclusion = f"""
## 结论

**{e["short"]} 是 LightNav / LightParkour 管线中的关键组件——读博客数字前先对齐本页定义与开源边界。**

1. 状态：**{e["oss"]}**
2. 与机构页 [亮源新创（Light Origins）](./light-origins.md) 三段范式对照阅读。
3. 工程复现以官方后续发布为准。
"""
        seq = ""

    content = f"""---
type: entity
tags:
{tag_yaml(e["tags"])}
status: complete
updated: {DATE}{arxiv_front}
related:
  - ./light-origins.md
  - ../overview/{MAP}.md
  - ./paper-lightnav-0.md
  - ./light-react.md
  - ./paper-light-loco-parkour.md
sources:
{chr(10).join(sources)}
summary: "{e["short"]}：{e["summary"]}"
---

# {e["short"]}

**{e["short"]}**（{e["title"]}）在 [Light Origins · {blog["title"]}]({blog["url"]}) 中被引用。

## 一句话定义

**{e["summary"]}**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLM | Vision-Language Model | 视觉-语言多模态模型 |
| VLN | Vision-and-Language Navigation | 视觉-语言导航 |
| ER | Embodied Reasoning | 具身推理；LightNav 第一阶段中期训练 |
| RL | Reinforcement Learning | 强化学习 |
| R2S2R | Real-to-Sim-to-Real | 真场景→仿真合成→真机部署 |

## 为什么重要

- {e["why"]}
- 博客 ingest 独立节点（非重复 stub）；见 [3 篇技术地图](../overview/{MAP}.md)。

## 核心信息

| 项 | 内容 |
|----|------|
| **类型** | {e["kind"]} |
| **出处** | {e.get("venue", "Light Origins Blog")} |
| **开源** | **{e["oss"]}** |
{("| **arXiv** | [" + e["arxiv"] + "](https://arxiv.org/abs/" + e["arxiv"] + ") |") if e.get("arxiv") else ""}

{seq}{conclusion}
## 关联页面

- [亮源新创（Light Origins）](./light-origins.md)
- [{MAP}](../overview/{MAP}.md)
- [LightNav-0](./paper-lightnav-0.md)
- [Light REACT](./light-react.md)

## 参考来源

- [{e["source_file"]}](../../sources/papers/{e["source_file"]})
- [{blog["file"]}](../../sources/blogs/{blog["file"]})
- [Tech Blog]({blog["url"]})

## 推荐继续阅读

- [Light Origins 官网](https://www.lightorigins.com/)
- [3 篇技术地图](../overview/{MAP}.md)
"""
    path.write_text(content, encoding="utf-8")


def write_grpo_method() -> None:
    path = ROOT / "wiki/methods/grpo.md"
    if path.exists():
        return
    content = f"""---
type: method
tags: [reinforcement-learning, llm, policy-optimization, grpo]
status: complete
updated: {DATE}
related:
  - ./policy-optimization.md
  - ./reinforcement-learning.md
  - ../entities/paper-lightnav-0.md
  - ../entities/paper-prism-grpo.md
sources:
  - ../../sources/papers/deepseekmath_grpo_2024.md
  - ../../sources/blogs/{BLOGS["nav"]["file"]}
summary: "GRPO（Group Relative Policy Optimization）：DeepSeekMath 提出的组内相对优势 RL 变体；LightNav-0 第三阶段在线 RL 采用 GRPO 比较规划执行结果。"
---

# GRPO（Group Relative Policy Optimization）

**GRPO** 是 DeepSeekMath 提出的 RL 算法变体：在同一 prompt/状态下采样一组轨迹，用**组内相对回报**估计 advantage，省去独立 critic 网络。

## 一句话定义

**用一组并行 rollout 的相对排序代替 per-token critic，适合大模型 / VLA 后训练阶段的在线 RL。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| GRPO | Group Relative Policy Optimization | 组相对策略优化 |
| RL | Reinforcement Learning | 强化学习 |
| PPO | Proximal Policy Optimization | 近端策略优化；GRPO 常作为其 advantage 估计替代 |
| VLA | Vision-Language-Action | 视觉-语言-动作策略 |

## 为什么重要

- [LightNav-0](../entities/paper-lightnav-0.md) 第三阶段在仿真中用 **GRPO** 比较自主规划执行结果，EVT-Bench 跟踪 SR 从 74.4→82.6。
- 仓库另有 [Prism-GRPO](../entities/paper-prism-grpo.md)、[Temporal-GRPO](../entities/paper-temporal-grpo.md) 等变体论文页。

## 核心原理

1. 对同一上下文采样多条 rollout（规划/动作序列）。
2. 按组内回报排序或归一化得到 relative advantage。
3. 用 clipped policy gradient 更新策略（与 PPO 类似的目标，但 advantage 来自组内比较）。

## 工程实践

| 项 | 建议 |
|----|------|
| 适用 | LLM/VLM/VLA 后训练、可并行采样的仿真环境 |
| 与 PPO | 见 [policy-optimization](./policy-optimization.md)；GRPO 主要改 advantage 估计 |
| 导航实例 | LightNav-0 三阶段：SFT 后 GRPO 在线改进跟踪与恢复 |

## 局限与风险

- 组大小与采样成本 trade-off；仿真吞吐不足时 RL 阶段增益有限。
- 相对回报对 reward 标定敏感；导航任务需与 SFT 阶段接口一致（LightNav 复用 RVQ 动作词表）。

## 关联页面

- [policy-optimization](./policy-optimization.md)
- [LightNav-0](../entities/paper-lightnav-0.md)
- [Prism-GRPO](../entities/paper-prism-grpo.md)

## 参考来源

- [deepseekmath_grpo_2024.md](../../sources/papers/deepseekmath_grpo_2024.md)
- [{BLOGS["nav"]["file"]}](../../sources/blogs/{BLOGS["nav"]["file"]})

## 推荐继续阅读

- DeepSeekMath 技术报告（GRPO 原始提出）
- [LightNav-0 Tech Blog]({BLOGS["nav"]["url"]})
"""
    path.write_text(content, encoding="utf-8")
    (ROOT / "sources/papers/deepseekmath_grpo_2024.md").write_text(
        f"""# DeepSeekMath / GRPO

> 来源归档 — LightNav-0 Tech Blog 脚注

- **标题：** DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models
- **提出：** GRPO（Group Relative Policy Optimization）
- **LightNav 引用：** 第三阶段在线 RL
- **入库日期：** {DATE}

## 对 wiki 的映射

- [grpo](../../wiki/methods/grpo.md)
""",
        encoding="utf-8",
    )


def write_blogs_and_sites() -> None:
    for key, b in BLOGS.items():
        cited_new = [e for e in ENTITIES if e["blog"] == key]
        cited_exist = [x for x in EXISTING if x[2] == key]
        rows = []
        for e in cited_new:
            rows.append(
                f"| {e['short']} | 新建 | [{entity_filename(e).replace('.md', '')}](../../wiki/entities/{entity_filename(e)}) |"
            )
        for name, wiki_path, _ in cited_exist:
            rows.append(f"| {name} | 复用 | [{wiki_path}](../../{wiki_path}) |")
        table = "\n".join(rows)
        content = f"""# {b["title"]}

> 来源归档（blog / Light Origins Tech Blog）

- **标题：** {b["title"]}
- **类型：** blog
- **机构：** 亮源新创（Light Origins）
- **URL：** {b["url"]}
- **发表日期：** {b["date"]}
- **入库日期：** {DATE}
- **抓取方式：** WebFetch
- **一句话说明：** 官方 Tech Blog 正文 ingest；引用项目/论文均建独立 wiki 节点。

## 引用 → 本库节点

| 名称 | 状态 | wiki |
|------|------|------|
{table}

## 对 wiki 的映射

- [3 篇技术地图](../../wiki/overview/{MAP}.md)
- [亮源新创（Light Origins）](../../wiki/entities/light-origins.md)

## 开源核查摘要

- **LightNav-0：** 已开源 `lightorigins/LightNav-0` + HF 权重
- **Light REACT / LightParkour：** 截至 {DATE} **未列** 公开训练代码（博客/演示为主）
"""
        (ROOT / "sources/blogs" / b["file"]).write_text(content, encoding="utf-8")

    # update site archives
    react_site = ROOT / "sources/sites/light-react.md"
    text = react_site.read_text(encoding="utf-8")
    text = text.replace(
        "| 独立项目页 / Tech Blog | **未上线**（官网博客区仅有 LightNav-0、LightParkour） |",
        "| Tech Blog | **已上线** <https://www.lightorigins.com/blog/light-react> |",
    )
    text = text.replace("- **入库日期：** 2026-09-09", f"- **入库日期：** {DATE}")
    if "lightorigins_light_react" not in text:
        text += "\n- Tech Blog 归档：[lightorigins_light_react_2026-09-09](../blogs/lightorigins_light_react_2026-09-09.md)\n"
    react_site.write_text(text, encoding="utf-8")

    parkour_site = ROOT / "sources/sites/lightparkour.md"
    parkour_site.write_text(
        f"""# LightParkour Tech Blog（lightorigins.com）

- **标题：** LightParkour：通过 Real2Sim2Real 拓展人形机器人的跑酷技能
- **URL：** {BLOGS["parkour"]["url"]}
- **PDF（并行项目页）：** <https://light-loco-parkour.github.io/paper.pdf>
- **机构：** 亮源新创（Light Origins）
- **平台：** [Lightbot 0](../../wiki/entities/lightbot-0.md)
- **入库日期：** {DATE}

## 开源状态

| 资源 | 状态 |
|------|------|
| Tech Blog + 项目页 PDF | **已发布** |
| 训练代码 | **未列出** |

## 交叉链接

- 博客归档：[lightorigins_lightparkour_2026-08-03](../blogs/lightorigins_lightparkour_2026-08-03.md)
- 实体：[paper-light-loco-parkour](../../wiki/entities/paper-light-loco-parkour.md)
- 硬件：[lightbot-0](../../wiki/entities/lightbot-0.md)
""",
        encoding="utf-8",
    )

    nav_site = ROOT / "sources/sites/lightnav-0.md"
    nav_text = nav_site.read_text(encoding="utf-8")
    if "lightorigins_lightnav" not in nav_text:
        nav_text += "\n- Tech Blog 归档：[lightorigins_lightnav_0_2026-09-01](../blogs/lightorigins_lightnav_0_2026-09-01.md)\n"
    nav_site.write_text(nav_text, encoding="utf-8")


def write_technology_map() -> None:
    entity_links = "\n".join(f"  - ../entities/{entity_filename(e)}" for e in ENTITIES)
    rows = []
    for e in ENTITIES:
        rows.append(
            f"| {e['short']} | {BLOGS[e['blog']]['date'][:7]} | "
            f"[{entity_filename(e).replace('.md', '')}](../entities/{entity_filename(e)}) |"
        )
    for name, wiki_path, blog_key in EXISTING:
        rows.append(
            f"| {name} | {BLOGS.get(blog_key, {}).get('date', DATE)[:7]} | "
            f"[{wiki_path.replace('wiki/', '')}](../{wiki_path.replace('wiki/', '')}) |"
        )
    content = f"""---
type: overview
tags: [overview, light-origins, navigation, humanoid, technology-map]
status: complete
updated: {DATE}
related:
{entity_links}
  - ../entities/light-origins.md
  - ../entities/paper-lightnav-0.md
  - ../entities/light-react.md
  - ../entities/paper-light-loco-parkour.md
sources:
  - ../../sources/blogs/lightorigins_light_react_2026-09-09.md
  - ../../sources/blogs/lightorigins_lightnav_0_2026-09-01.md
  - ../../sources/blogs/lightorigins_lightparkour_2026-08-03.md
summary: "亮源新创三篇官方 Tech Blog（LightNav-0 / LightParkour / Light REACT）引用链独立节点索引。"
---

# 亮源新创三篇 Tech Blog：引用阅读坐标

> **定位：** [LightNav-0](https://www.lightorigins.com/blog/lightnav-0) · [LightParkour](https://www.lightorigins.com/blog/lightparkour) · [Light REACT](https://www.lightorigins.com/blog/light-react)

## 一句话观点

**亮源新创公开的三段范式——预训练（Light-O1）→ 对齐（LightNav-0）→ 部署（Light REACT）——在跑酷线（LightParkour）上还有并行的全身感知运动蒸馏轴；读博客时要把「引用论文/基准/硬件」拆成独立节点，避免混在机构页里。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| R2S2R | Real-to-Sim-to-Real | LightNav 数据引擎核心范式 |
| ER | Embodied Reasoning | 导航后训练第一阶段 |
| ICL | In-Context Learning | Light REACT 全身韧性上下文适应 |
| RVQ | Residual Vector Quantization | LightNav 动作 token 化 |

## 完整索引

| 名称 | 博客日期 | 详情 |
|------|----------|------|
{chr(10).join(rows)}

## 关联页面

- [亮源新创（Light Origins）](../entities/light-origins.md)
- [LightNav-0](../entities/paper-lightnav-0.md)
- [Light REACT](../entities/light-react.md)
- [Light-Loco-Parkour](../entities/paper-light-loco-parkour.md)

## 参考来源

- [lightorigins_light_react_2026-09-09.md](../../sources/blogs/lightorigins_light_react_2026-09-09.md)
- [lightorigins_lightnav_0_2026-09-01.md](../../sources/blogs/lightorigins_lightnav_0_2026-09-01.md)
- [lightorigins_lightparkour_2026-08-03.md](../../sources/blogs/lightorigins_lightparkour_2026-08-03.md)
"""
    (ROOT / "wiki/overview" / f"{MAP}.md").write_text(content, encoding="utf-8")


def update_page_aliases() -> None:
    aliases_path = ROOT / "schema/page-aliases.json"
    data = json.loads(aliases_path.read_text(encoding="utf-8"))
    for e in ENTITIES:
        stub = e.get("replaces_stub")
        if not stub:
            continue
        old_id = f"entity-{stub}"
        new_id = f"entity-paper-{e['slug']}"
        data["aliases"][old_id] = new_id
    aliases_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def remove_stubs() -> None:
    for e in ENTITIES:
        stub = e.get("replaces_stub")
        if not stub:
            continue
        path = ROOT / "wiki/entities" / f"{stub}.md"
        if path.exists():
            path.unlink()


def patch_cross_refs() -> None:
    replacements = [
        (
            "wiki/concepts/robot-in-context-learning.md",
            "paper-notebook-locoformer-generalist-locomotion-via-long-contex",
            "paper-locoformer",
        ),
        (
            "wiki/entities/skild-s1.md",
            "paper-notebook-locoformer-generalist-locomotion-via-long-contex",
            "paper-locoformer",
        ),
        (
            "wiki/entities/skild-ai.md",
            "paper-notebook-locoformer-generalist-locomotion-via-long-contex",
            "paper-locoformer",
        ),
        (
            "wiki/overview/paper-notebook-category-05-locomotion.md",
            "paper-notebook-locoformer-generalist-locomotion-via-long-contex",
            "paper-locoformer",
        ),
        (
            "wiki/overview/paper-notebook-category-04-loco-manipulation-and-wbc.md",
            "paper-notebook-learning-getting-up-policies-for-real-world-huma",
            "paper-humanup-getting-up",
        ),
    ]
    for rel, old, new in replacements:
        path = ROOT / rel
        if path.exists():
            path.write_text(path.read_text(encoding="utf-8").replace(old, new), encoding="utf-8")


def patch_main_entities() -> None:
    # light-react: add blog source + update status
    lr = ROOT / "wiki/entities/light-react.md"
    t = lr.read_text(encoding="utf-8")
    if "lightorigins_light_react" not in t:
        t = t.replace(
            "  - ../../sources/blogs/wechat_lightorigins_light_react_2026-09-09.md",
            "  - ../../sources/blogs/wechat_lightorigins_light_react_2026-09-09.md\n"
            "  - ../../sources/blogs/lightorigins_light_react_2026-09-09.md",
        )
    t = t.replace(
        "> **落地状态（2026-09-09）：** 仅微信公众号发布与文内仿真/真机演示叙述；**无 arXiv、无独立项目页、无公开代码**。",
        "> **落地状态（2026-09-21）：** 官方 [Tech Blog](https://www.lightorigins.com/blog/light-react) 已上线（韧性金字塔 + 三阶段 RL/蒸馏/偏好对齐 + Transformer ICL 真机演示）；**仍无 arXiv / 公开代码**。",
    )
    t = t.replace(
        "**Light REACT**（**REsilient humAnoid ConTrol**，亮源新创 **2026-09-09** [官方发布](https://mp.weixin.qq.com/s/Xfps8-XAv3u1S--EpS5Ezw)）",
        "**Light REACT**（**REsilient humAnoid ConTrol**，亮源新创 **2026-09-09** [Tech Blog](https://www.lightorigins.com/blog/light-react)，[微信发布](https://mp.weixin.qq.com/s/Xfps8-XAv3u1S--EpS5Ezw)）",
    )
    if "paper-tolebi" not in t:
        t = t.replace(
            "  - ./paper-light-loco-parkour.md",
            "  - ./paper-tolebi.md\n  - ./paper-locoformer.md\n  - ./paper-humanup-getting-up.md\n  - ./paper-light-loco-parkour.md",
        )
    lr.write_text(t, encoding="utf-8")

    ln = ROOT / "wiki/entities/paper-lightnav-0.md"
    tn = ln.read_text(encoding="utf-8")
    if "lightorigins_lightnav" not in tn:
        tn = tn.replace(
            "  - ../../sources/sites/lightnav-0.md",
            "  - ../../sources/blogs/lightorigins_lightnav_0_2026-09-01.md\n  - ../../sources/sites/lightnav-0.md",
        )
    if "lightnav-er" not in tn:
        tn = tn.replace(
            "  - ./light-origins.md",
            "  - ./light-origins.md\n  - ./lightnav-er.md\n  - ./insight-bench.md\n  - ../overview/lightorigins-3blogs-technology-map.md",
        )
    ln.write_text(tn, encoding="utf-8")

    lp = ROOT / "wiki/entities/paper-light-loco-parkour.md"
    tp = lp.read_text(encoding="utf-8")
    if "lightorigins.com/blog/lightparkour" not in tp:
        tp = tp.replace(
            "[项目页](https://light-loco-parkour.github.io/)",
            "[Tech Blog](https://www.lightorigins.com/blog/lightparkour)，[项目页](https://light-loco-parkour.github.io/)",
        )
    if "lightbot-0" not in tp:
        tp = tp.replace("  - ./paper-hrl-stack-22", "  - ./lightbot-0.md\n  - ./paper-hrl-stack-22")
    if "lightorigins_lightparkour" not in tp:
        tp = tp.replace(
            "  - ../../sources/sites/light-loco-parkour-github-io.md",
            "  - ../../sources/blogs/lightorigins_lightparkour_2026-08-03.md\n  - ../../sources/sites/lightparkour.md\n  - ../../sources/sites/light-loco-parkour-github-io.md",
        )
    lp.write_text(tp, encoding="utf-8")

    lo = ROOT / "wiki/entities/light-origins.md"
    to = lo.read_text(encoding="utf-8")
    if MAP not in to:
        to = to.replace(
            "  - ../../sources/sites/lightorigins-about.md",
            f"  - ../overview/{MAP}.md\n  - ../../sources/sites/lightorigins-about.md",
        )
    lo.write_text(to, encoding="utf-8")


def append_log() -> None:
    entry = f"""
## [{DATE}] ingest | sources/blogs/lightorigins_light_{"{react,nav,parkour}"} — 亮源新创三篇官方 Tech Blog；引用论文/基准/硬件独立节点 + 合并 stub 别名

- **意图：** 用户指定 ingest light-react / lightnav-0 / lightparkour 三篇 Tech Blog，并要求引用项独立非重复节点。
- **开源结论：** LightNav-0 **已开源**；Light REACT / LightParkour **未开源**；Lightbot 0 为自研硬件叙事。
- **关键页：** [lightorigins-3blogs-technology-map](wiki/overview/lightorigins-3blogs-technology-map.md)；升级 LocoFormer / HUMANUP stub → 完整实体；新建 LightNav-ER、INSIGHT-Bench、TOLEBI、TrackVLA 等节点。
"""
    log = ROOT / "log.md"
    log.write_text(entry + log.read_text(encoding="utf-8"), encoding="utf-8")


def main() -> None:
    for e in ENTITIES:
        write_source(e)
        write_entity(e)
    write_grpo_method()
    write_blogs_and_sites()
    write_technology_map()
    update_page_aliases()
    remove_stubs()
    patch_cross_refs()
    patch_main_entities()
    append_log()
    print(f"Generated {len(ENTITIES)} entities + 3 blogs + map + grpo")


if __name__ == "__main__":
    main()
