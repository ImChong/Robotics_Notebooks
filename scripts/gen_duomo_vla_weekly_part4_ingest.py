#!/usr/bin/env python3
"""Generator for 多模空间 VLA weekly trends part4 ingest (2026-09-27)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
WECHAT_URL = "https://mp.weixin.qq.com/s/Eae0_0kuz-Hz8mmbpK3qBA"
BLOG = "wechat_duomo_vla_weekly_trends_2026-08-10_part4.md"
RAW = "wechat_duomo_vla_weekly_trends_2026-08-10_part4.md"
MAP = "vla-weekly-trends-2026-08-10-part4-technology-map"
DATE = "2026-09-27"

# wiki path without paper- prefix for reuse rows
REUSE: list[dict[str, str]] = [
    {
        "num": "00",
        "short": "GigaBrain-0.7",
        "section": "架构模块",
        "arxiv": "2608.15875",
        "wiki": "paper-gigabrain-0-7",
    },
    {
        "num": "03",
        "short": "ReflexVLA",
        "section": "架构模块",
        "arxiv": "2608.14379",
        "wiki": "paper-reflexvla",
    },
    {
        "num": "01",
        "short": "ViTaR",
        "section": "架构模块",
        "arxiv": "2608.15816",
        "wiki": "paper-vitar",
    },
    {
        "num": "00",
        "short": "StructRL",
        "section": "训练范式",
        "arxiv": "2608.15139",
        "wiki": "paper-structrl",
    },
]

PAPERS: list[dict[str, Any]] = [
    {
        "num": "01",
        "slug": "robo-dopamine-2",
        "short": "Robo-Dopamine 2.0",
        "title": "Robo-Dopamine 2.0: History-Conditioned and OOD-Aware Process Reward Modeling for Robotic Manipulation",
        "arxiv": "2608.15680",
        "section": "架构模块",
        "oss": "待核实",
        "code": None,
        "site": None,
        "inst": "北京大学；EvoPhys AI；北京邮电大学；腾讯；中国人民大学；中山大学",
        "tags": ["paper", "vla", "process-reward", "rl", "manipulation"],
        "eval": "RoboTwin；双 Franka 实机",
        "summary": "历史条件 + OOD 感知的视觉过程奖励：对照参考过程判断继续/变化/失败/恢复，先学顺序再学进度并用异常轨迹校验，辅助 RL 改进 VLA。",
        "why": "稀疏成功奖励与只看前后帧的 PRM 难区分正常变化与失败；2.0 显式引入历史与参考过程对齐。",
        "seq": False,
    },
    {
        "num": "02",
        "slug": "core-vla-counterfactual-realignment",
        "short": "CoRe",
        "title": "Imagining Recovery: Inference-Time Counterfactual Realignment for Vision-Language-Action Models",
        "arxiv": "2608.14822",
        "section": "架构模块",
        "oss": "待核实",
        "code": None,
        "site": None,
        "inst": "凯斯西储大学（美）",
        "tags": ["paper", "vla", "recovery", "inference-time", "manipulation"],
        "eval": "LangSwitch、LIBERO-Long；UFACTORY xArm6 实机",
        "summary": "推理期反事实重对齐：跑偏后用合成画面在内部设想恢复路径，小步把状态接回再让原 VLA 继续，无需失败数据或重训。",
        "why": "目标/布局突变时 VLA 易偏离；CoRe 把试错留在想象空间，保留已完成进度。",
        "seq": False,
    },
    {
        "num": "00",
        "slug": "forceu-vla",
        "short": "ForceU-VLA",
        "title": "ForceU-VLA: A Force-Aware Vision-Language-Action Model for Embodied Ultrasound Scanning",
        "arxiv": "2608.15009",
        "section": "架构模块/医疗",
        "oss": "已开源",
        "code": "https://github.com/VMVLab/ForceU-VLA",
        "site": None,
        "inst": "中国海洋大学；山东大学；合肥工业大学",
        "tags": ["paper", "vla", "medical", "force", "ultrasound"],
        "eval": "UR7e + RGB + 超声图像实机",
        "summary": "超声扫描 VLA 联合超声图像、力反馈与动作，协同融合并随扫描阶段自适应多模态权重，改善接触稳定与压力调节。",
        "why": "医疗超声需力–视联合建模与阶段识别；ForceU-VLA 针对探头–组织接触闭环。",
        "seq": True,
    },
    {
        "num": "00",
        "slug": "ecovla",
        "short": "EcoVLA",
        "title": "EcoVLA: Energy-Efficient Device-Edge Co-Inference for Vision-Language-Action Models under Real-Time Constraints",
        "arxiv": "2608.15502",
        "section": "性能提升",
        "oss": "待核实",
        "code": None,
        "site": None,
        "inst": "北京航空航天大学；北京工业大学",
        "tags": ["paper", "vla", "edge", "efficiency", "deployment"],
        "eval": "Jetson AGX Orin + RTX 4090 边端协同实机",
        "summary": "端–边协同推理：按段拆分 VLA 计算并在网络波动时动态分工，压缩传输量以兼顾实时与能耗。",
        "why": "端侧算力/电量有限、纯边缘又受延迟抖动；EcoVLA 做运行时切分调度。",
        "seq": False,
    },
    {
        "num": "01",
        "slug": "specvla",
        "short": "SpecVLA",
        "title": "Algorithm-Architecture Co-Design for Efficient VLA Inference via Speculative Inference and Verification",
        "arxiv": "2608.15636",
        "section": "性能提升",
        "oss": "待核实",
        "code": None,
        "site": None,
        "inst": "上海交通大学；KAUST（沙特）",
        "tags": ["paper", "vla", "inference", "speculative", "deployment"],
        "eval": "LIBERO、ManiSkill",
        "summary": "推测–验证共设计：低影响阶段 speculative 长序列、关键阶段轻量验证，配合残差建模与混合精度及 GPU/专用硬件并行。",
        "why": "VLA 逐步解码延迟高；SpecVLA 按交互重要性动态安排推理深度（MICRO 2026 模板）。",
        "seq": False,
    },
    {
        "num": "00",
        "slug": "vla-bit-flip-attacks-int8",
        "short": "VLA Bit-Flip Attacks",
        "title": "Bit-Flip Attacks on Vision-Language-Action Models: Action-Decoding Architecture Shapes the Vulnerability",
        "arxiv": "2608.15475",
        "section": "安全防御",
        "oss": "待核实",
        "code": None,
        "site": None,
        "inst": "香港科技大学；阿德莱德大学；佐治亚理工；北卡罗莱纳大学教堂山；中国石油大学（华东）",
        "tags": ["paper", "vla", "security", "quantization", "deployment"],
        "eval": "LIBERO、SimplerEnv；双臂实机",
        "summary": "量化 VLA 的 INT8 权重易受 Rowhammer 类比特翻转；关键位集中在动作解码层，少量翻转即可让闭环任务失效。",
        "why": "部署侧需把权重完整性纳入威胁模型；动作头架构决定脆弱性分布。",
        "seq": False,
    },
    {
        "num": "01",
        "slug": "phaselora",
        "short": "PhaseLoRA",
        "title": "PhaseLoRA: Control-Regime-Conditioned Low-Rank Adaptation for Continuous-Action Vision-Language-Action Policies",
        "arxiv": "2608.15285",
        "section": "训练范式",
        "oss": "已开源",
        "code": "https://github.com/Grinffin/PhaseLoRA",
        "site": None,
        "inst": "清华大学",
        "tags": ["paper", "vla", "lora", "manipulation", "fine-tuning"],
        "eval": "LIBERO；AgileX PiPER 实机",
        "summary": "按操作阶段（精细控制倾向 + 事件强度弱监督）动态调整 action expert 内 LoRA 方向，骨干大多冻结。",
        "why": "抓取–接触–移动需不同控制律；统一 LoRA 难以分阶段适配。",
        "seq": True,
    },
    {
        "num": "00",
        "slug": "pace-phase-progress-vla",
        "short": "PACE（VLA 长程信用）",
        "title": "PACE: Phase-Progress-Aware Credit for Long-Horizon Embodied Manipulation",
        "arxiv": "2608.15026",
        "section": "长程记忆",
        "oss": "待核实",
        "code": None,
        "site": None,
        "inst": "大连理工大学；吉林大学",
        "tags": ["paper", "vla", "long-horizon", "credit-assignment", "manipulation"],
        "eval": "LIBERO-Long；AgileX PiPER 双臂实机",
        "summary": "融合局部视觉–动作变化识别阶段与进度，修正剩余代价并为每步生成信用；先保护预训练再高信用增强推理。",
        "why": "长任务仅末端成功信号无法告诉机器人哪步做对；PACE 做阶段–进度感知信用。",
        "seq": False,
    },
    {
        "num": "01",
        "slug": "remember-smarter-vla-memory",
        "short": "Remember Smarter（RS）",
        "title": "Remember Smarter: Visual History Compressor and Hyperbolic Experience Space for Robotic Memory",
        "arxiv": "2608.15269",
        "section": "长程记忆",
        "oss": "待核实",
        "code": None,
        "site": None,
        "inst": "西安电子科技大学",
        "tags": ["paper", "vla", "memory", "long-horizon", "manipulation"],
        "eval": "LIBERO、LIBERO-Plus；单臂实机",
        "summary": "近期多视角压缩为短记录 + 双曲空间经验库后台检索转提示，扩大可用历史而不拖慢 chunk 推理。",
        "why": "长程 VLA 需近期连续性与成功经验复用；直接堆历史会膨胀上下文。",
        "seq": False,
    },
    {
        "num": "00",
        "slug": "evoscene-vla",
        "short": "EvoScene-VLA",
        "title": "EvoScene-VLA: Evolving Scene Beliefs Inside the Action Decoder for Chunked Robot Control",
        "arxiv": "2605.21862",
        "section": "长程记忆",
        "oss": "待核实",
        "code": None,
        "site": None,
        "inst": "澳大利亚国立大学；昆士兰大学；北京师范大学",
        "tags": ["paper", "vla", "scene-belief", "chunked-control", "manipulation"],
        "eval": "LIBERO、RoboTwin；Galaxea R1-Lite 实机",
        "summary": "动作块间保留可更新场景状态，VLM 融合新观测与动作形成的先验，解码器同时输出动作与紧凑场景更新（v2 2026-08-15）。",
        "why": "单帧 VLA 忽略自身动作引起的场景变化；EvoScene 在解码器内演化 scene belief。",
        "seq": False,
    },
]


def tag_yaml(tags: list[str]) -> str:
    return "\n".join(f"  - {t}" for t in tags)


def write_paper_source(p: dict[str, Any]) -> None:
    path = ROOT / "sources/papers" / f"{p['slug']}_arxiv_{p['arxiv'].replace('.', '_')}.md"
    code_line = f"- **代码：** {p['code']}\n" if p["code"] else ""
    content = f"""# {p["short"]}（arXiv:{p["arxiv"]}）

> 来源归档（paper）

- **标题：** {p["title"]}
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/{p["arxiv"]}>
- **PDF：** <https://arxiv.org/pdf/{p["arxiv"]}>
{code_line}- **入库日期：** {DATE}
- **一句话说明：** {p["summary"]}

## 开源状态

- **{p["oss"]}**（步骤 2.5 核查，{DATE}）

## 核心摘录

1. **公众号章节：** {p["section"]}（[多模空间第四篇盘点](../blogs/{BLOG})）
2. **机构：** {p["inst"]}
3. **文内评测：** {p["eval"]}
4. **导读要点：** {p["why"]}

## 对 wiki 的映射

- [paper-{p["slug"]}](../../wiki/entities/paper-{p["slug"]}.md)
- [第四篇技术地图](../../wiki/overview/{MAP}.md)
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
  participant Robot as 真机/数据
  Dev->>Repo: clone + README 环境依赖
  Dev->>Robot: 准备传感器与任务数据
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

**{p["short"]}**（*{p["title"]}*，[arXiv:{p["arxiv"]}](https://arxiv.org/abs/{p["arxiv"]})）收录于 [多模空间 · 一周 VLA 研究趋势简析（2026.08.10–08.16）第四篇](../../sources/blogs/{BLOG}) **{p["section"]}** 段。

## 一句话定义

**{p["summary"]}**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| VLM | Vision-Language Model | 视觉–语言多模态模型 |
| RL | Reinforcement Learning | 强化学习 |
| PRM | Process Reward Model | 过程/进度奖励模型 |
| OOD | Out-of-Distribution | 分布外场景或轨迹 |

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
| [第四篇技术地图](../overview/{MAP}.md) | 同批 14 篇横向索引；本文属 **{p["section"]}** |
| [VLA 方法页](../methods/vla.md) | 单篇机制细节以原文为准 |

## 结论

**{p["short"]} 适合作为本期「{p["section"]}」路线的快速索引页。**

1. 核心贡献：{p["summary"]}
2. 开源结论：**{p["oss"]}** — 以项目页/仓库实际链接为准。
3. 横向对照见 [第四篇技术地图](../overview/{MAP}.md)。

## 关联页面

- [VLA（Vision-Language-Action）](../methods/vla.md)
- [一周 VLA 趋势技术地图（第四篇）](../overview/{MAP}.md)

## 参考来源

- [{BLOG}](../../sources/blogs/{BLOG})
- [arXiv:{p["arxiv"]}](https://arxiv.org/abs/{p["arxiv"]})

## 推荐继续阅读

- [VLA 方法页](../methods/vla.md)
- [arXiv PDF](https://arxiv.org/pdf/{p["arxiv"]})
"""
    path.write_text(content, encoding="utf-8")


def all_index_rows() -> list[str]:
    """Preserve公众号正文顺序（非 arXiv 排序）。"""
    ordered: list[tuple[str, str, str, str]] = [
        ("GigaBrain-0.7", "架构模块", "2608.15875", "paper-gigabrain-0-7"),
        ("Robo-Dopamine 2.0", "架构模块", "2608.15680", "paper-robo-dopamine-2"),
        ("CoRe", "架构模块", "2608.14822", "paper-core-vla-counterfactual-realignment"),
        ("ReflexVLA", "架构模块", "2608.14379", "paper-reflexvla"),
        ("ForceU-VLA", "架构模块/医疗", "2608.15009", "paper-forceu-vla"),
        ("ViTaR", "架构模块", "2608.15816", "paper-vitar"),
        ("EcoVLA", "性能提升", "2608.15502", "paper-ecovla"),
        ("SpecVLA", "性能提升", "2608.15636", "paper-specvla"),
        ("VLA Bit-Flip Attacks", "安全防御", "2608.15475", "paper-vla-bit-flip-attacks-int8"),
        ("StructRL", "训练范式", "2608.15139", "paper-structrl"),
        ("PhaseLoRA", "训练范式", "2608.15285", "paper-phaselora"),
        ("PACE（VLA 长程信用）", "长程记忆", "2608.15026", "paper-pace-phase-progress-vla"),
        ("Remember Smarter（RS）", "长程记忆", "2608.15269", "paper-remember-smarter-vla-memory"),
        ("EvoScene-VLA", "长程记忆", "2605.21862", "paper-evoscene-vla"),
    ]
    out: list[str] = []
    for i, (short, section, arxiv, wiki) in enumerate(ordered, start=1):
        out.append(
            f"| {i:02d} | {short} | {section} | [{arxiv}](https://arxiv.org/abs/{arxiv}) | "
            f"[{wiki}](../../wiki/entities/{wiki}.md) |"
        )
    return out


def write_blog_and_map() -> None:
    raw_path = ROOT / "sources/raw" / RAW
    raw_path.write_text(
        f"# [风向标-具身VLA] 一周 VLA 研究趋势简析（2026.08.10-2026.08.16）-第四篇\n\n"
        f"原始链接：{WECHAT_URL}\n\n（WebFetch 抓取，{DATE}）\n",
        encoding="utf-8",
    )
    table = "\n".join(all_index_rows())
    blog = f"""# [风向标-具身VLA] 一周 VLA 研究趋势简析（2026.08.10-2026.08.16）-第四篇

> 来源归档（blog / 微信公众号）

- **标题：** [风向标-具身VLA] 一周 VLA 研究趋势简析（2026.08.10-2026.08.16）-第四篇-顺祝大家双节快乐
- **类型：** blog
- **作者：** 多模空间（微信公众号）
- **原始链接：** {WECHAT_URL}
- **入库日期：** {DATE}
- **抓取方式：** WebFetch（wechat-article-for-ai 不可用）
- **原始抓取落盘：** [`sources/raw/{RAW}`](../raw/{RAW})
- **一句话说明：** 14 篇 VLA 论文：三系统基础模型、过程奖励、推理期恢复、动态低延迟、力觉超声、端边协同、推测推理、比特翻转安全、flow RL 探索、阶段 LoRA、长程信用与记忆；**14/14 独立 canonical 详情节点**（本 ingest **新建 10**、**复用 4**）。

## 14 篇 → 本库 canonical 节点

| # | 论文 | 章节 | arXiv | wiki（canonical） |
|---|------|------|-------|-------------------|
{table}

## 核心摘录（MVP）

### 1) 主题分布

- **架构与恢复：** GigaBrain-0.7 三系统扩展；Robo-Dopamine 2.0 过程奖励；CoRe 推理期反事实重对齐；Reflex 动态操纵；ForceU-VLA / ViTaR 模态扩展。
- **部署效率与安全：** EcoVLA 端–边协同；SpecVLA 推测–验证；INT8 比特翻转威胁模型。
- **训练与长程：** StructRL flow 动作空间探索；PhaseLoRA 阶段 LoRA；PACE / RS / EvoScene 长程信用与场景信念。

### 2) 节点去重结论（{DATE} 核查）

- **14/14** 各有独立 canonical 页（10 新建 `paper-*` + 4 复用）；**0** 重复 arXiv ID。

## 对 wiki 的映射

- 阅读坐标：[一周 VLA 趋势技术地图（2026.08.10 第四篇）](../../wiki/overview/{MAP}.md)
- 同系列：[第一篇](wechat_duomo_vla_weekly_trends_2026-08-10_part1.md) · [第三篇](wechat_duomo_vla_weekly_trends_2026-08-10_part3.md)

## 当前提炼状态

- [x] 14 篇索引与独立详情节点
- [x] 技术地图
- [ ] 各篇深读（待原文 / 项目页 follow-up）
"""
    (ROOT / "sources/blogs" / BLOG).write_text(blog, encoding="utf-8")

    related_new = "\n".join(f"  - ../entities/paper-{p['slug']}.md" for p in PAPERS)
    related_reuse = "\n".join(f"  - ../entities/{r['wiki']}.md" for r in REUSE)
    map_sections = """
### 架构 · 恢复 · 模态

- [GigaBrain-0.7](../entities/paper-gigabrain-0-7.md) — arXiv:2608.15875（复用）
- [Robo-Dopamine 2.0](../entities/paper-robo-dopamine-2.md) — arXiv:2608.15680
- [CoRe](../entities/paper-core-vla-counterfactual-realignment.md) — arXiv:2608.14822
- [ReflexVLA](../entities/paper-reflexvla.md) — arXiv:2608.14379（复用）
- [ForceU-VLA](../entities/paper-forceu-vla.md) — arXiv:2608.15009
- [ViTaR](../entities/paper-vitar.md) — arXiv:2608.15816（复用）

### 部署 · 安全

- [EcoVLA](../entities/paper-ecovla.md) — arXiv:2608.15502
- [SpecVLA](../entities/paper-specvla.md) — arXiv:2608.15636
- [VLA Bit-Flip Attacks](../entities/paper-vla-bit-flip-attacks-int8.md) — arXiv:2608.15475

### 训练 · 长程

- [StructRL](../entities/paper-structrl.md) — arXiv:2608.15139（复用）
- [PhaseLoRA](../entities/paper-phaselora.md) — arXiv:2608.15285
- [PACE（VLA 长程信用）](../entities/paper-pace-phase-progress-vla.md) — arXiv:2608.15026
- [Remember Smarter](../entities/paper-remember-smarter-vla-memory.md) — arXiv:2608.15269
- [EvoScene-VLA](../entities/paper-evoscene-vla.md) — arXiv:2605.21862
"""
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
  - ../methods/vla.md
sources:
  - ../../sources/blogs/{BLOG}
summary: "多模空间 {DATE} 策展：2026.08.10–08.16 一周 VLA 第四篇 14 论文——恢复、低延迟、端边协同、安全与长程信用。"
---

# 一周 VLA 研究趋势（2026.08.10–08.16 · 第四篇）

> **本页定位**：为 [多模空间公众号第四篇盘点]({WECHAT_URL}) 提供横切面索引；**14/14 各有独立 canonical 详情节点**（10× 新建 `paper-*` + 4× 复用），本页不替代单篇深读。

## 一句话观点

**第四篇主线：VLA 从「能做完静态任务」走向「可恢复、可部署、可长程」——过程奖励、推理期重对齐、端边/推测加速、量化安全与 scene belief 记忆。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| PRM | Process Reward Model | 过程/进度奖励 |
| LoRA | Low-Rank Adaptation | 低秩适配 |

## 节点索引（14/14）
{map_sections}

## 与同系列的关系

- [第一篇技术地图](./vla-weekly-trends-2026-08-10-part1-technology-map.md)（21 篇）、[第三篇](./vla-weekly-trends-2026-08-10-part3-technology-map.md)（12 篇）与本篇 **arXiv 不重叠**（除复用的 GigaBrain / Reflex / ViTaR / StructRL 已在其他 ingest 出现）。

## 参考来源

- [{BLOG}](../../sources/blogs/{BLOG})
"""
    (ROOT / "wiki/overview" / f"{MAP}.md").write_text(map_content, encoding="utf-8")


def patch_reuse_sources() -> None:
    """Append part4 blog to reused entity frontmatter sources if missing."""
    blog_ref = f"  - ../../sources/blogs/{BLOG}"
    for r in REUSE:
        path = ROOT / "wiki/entities" / f"{r['wiki']}.md"
        text = path.read_text(encoding="utf-8")
        if BLOG in text:
            continue
        if "sources:" in text:
            text = text.replace("sources:", f"sources:\n{blog_ref}", 1)
        path.write_text(text, encoding="utf-8")


def main() -> None:
    for p in PAPERS:
        write_paper_source(p)
        write_entity(p)
        write_repo(p)
    write_blog_and_map()
    patch_reuse_sources()
    print(f"Generated {len(PAPERS)} new papers + blog + map; reused {len(REUSE)}")


if __name__ == "__main__":
    main()
