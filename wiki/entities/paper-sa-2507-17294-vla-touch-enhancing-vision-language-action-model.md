---
type: entity
tags: [paper, curated-index, awesome-touch, sun254667-touch]
status: complete
updated: 2026-10-06
arxiv: "2507.17294"
venue: "arXiv 2025"
summary: "Enhances VLA with tactile sensing without fine-tuning the base VLA; leverages pretrained tactile-language model for semantic feedback and diffusion-based controller for action refinement."
related:
  - ../entities/awesome-touch.md
  - ../overview/sun-awesome-touch-technology-map.md
  - ../methods/vla.md
  - ../methods/imitation-learning.md
  - ../tasks/manipulation.md
  - ../tasks/bimanual-manipulation.md
sources:
  - ../../sources/papers/sun_awesome_touch_2507_17294_vla-touch-enhancing-vision-language-acti.md
  - ../../sources/papers/sun_awesome_touch_catalog.md
  - ../../sources/repos/awesome-touch.md
---

# VLA-Touch

**VLA-Touch: Enhancing Vision-Language-Action Models with Dual-Level Tactile Feedback** 收录于 [Awesome Touch](https://github.com/sun254667/awesome-touch) **第 014/065** 篇，分组 **21 Unified Vision-Tactile-Language-Action Frameworks**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

Enhances VLA with tactile sensing without fine-tuning the base VLA; leverages pretrained tactile-language model for semantic feedback and diffusion-based controller for action refinement.

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VTLA | Vision-Tactile-Language-Action | 视–触–语言–动作策略 |
| WM | World Model | 视触觉前向预测 |
| WAM | World Action Model | 世界–动作联合建模 |
| Sim2Real | Simulation to Real | 仿真到真机迁移 |

## 为什么重要

- Enhances VLA with tactile sensing without fine-tuning the base VLA; leverages pretrained tactile-language model for semantic feedback and diffusion-based controller for action refinement.
- 想横向对照同一分组的其他工作，可以从 [Awesome Touch 技术地图](../overview/sun-awesome-touch-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome Touch](../entities/awesome-touch.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TB
    G["任务目标 g + 场景图像 o_t"] --> PL["VLM 任务规划器（GPT-4o）<br/>输出单句原子操作指令 I_k"]
    PL --> VLA["基础 VLA（RDT）<br/>按视觉观察 s_t 与指令 I_k 生成动作块 a_t:t+Ta"]
    VLA --> SEG["滑动窗口：每次取未重叠的 T_r 步 a_t:t+Tr"]
    OBS["当前观察：RGB + 本体状态 s_t<br/>触觉力信号 m_t = (X, Y, M)<br/>由 marker tracking 估计"] --> REF
    SEG --> REF["触觉增强控制器 π_I（BRIDGeR 插值扩散）<br/>以 VLA 动作为源分布，输出精修动作"]
    REF --> EXE["执行精修动作并更新观察"]
    EXE --> MORE{"动作块是否已全部精修？"}
    MORE -- 否 --> SEG
    MORE -- 是 --> DONE{"指令 I_k 完成或达最大步数？<br/>（执行失败则退出触发重规划）"}
    DONE -- 否 --> VLA
    DONE -- 是 --> TAC["采集原始触觉输入 o^m_t<br/>（GelSight，6 帧序列）"]
    TAC --> OCT["触觉语言模型 Octopi<br/>转为粗糙度 / 硬度等语言描述 L^m_t"]
    OCT --> PL
```

图依据：论文 §3.1–§3.2 / Fig.1、Algorithm 1。

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 014/065 |
| 分组 | 21 Unified Vision-Tactile-Language-Action Frameworks |
| 出处 | arXiv 2025 |
| 论文 | <https://arxiv.org/abs/2507.17294> |

## 核心机制（归纳）

### 策展导读要点

Enhances VLA with tactile sensing without fine-tuning the base VLA; leverages pretrained tactile-language model for semantic feedback and diffusion-based controller for action refinement.

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2507.17294) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-touch-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-touch-technology-map.md) 的 **21 Unified Vision-Tactile-Language-Action Frameworks** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「VLA-Touch」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

- 可确证的只有清单坐标：分组 **21 Unified Vision-Tactile-Language-Action Frameworks**，以及 Highlights 点出的问题设定；本页不自行推导新结论。
- 适用边界：本页不能替代原文 PDF；开源状态以项目页实际链接为准（清单可能滞后）。
- 要深读这篇，建议直接从原文入手，再回到下方关联的方法 / 任务页对照。

## 常见误区

1. 不要把 Awesome 条目的 Highlights 当成完整方法证明——它只是策展导读。
2. 若站内已有这篇的深读页，以那一页为准——本页只是清单入口，不含实验数据。

## 关联页面

- 列表实体：[Awesome Touch](../entities/awesome-touch.md)
- 技术地图：[Awesome Touch 技术地图](../overview/sun-awesome-touch-technology-map.md)
- 方法/任务：[vla.md](../methods/vla.md)、[manipulation.md](../tasks/manipulation.md)

## 参考来源

- [`sources/papers/sun_awesome_touch_2507_17294_vla-touch-enhancing-vision-language-acti.md`](../../sources/papers/sun_awesome_touch_2507_17294_vla-touch-enhancing-vision-language-acti.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_touch_catalog.md`](../../sources/papers/sun_awesome_touch_catalog.md) — 列表总表
- [`sources/repos/awesome-touch.md`](../../sources/repos/awesome-touch.md)
- 论文：<https://arxiv.org/abs/2507.17294>
- [arXiv PDF](https://arxiv.org/pdf/2507.17294)：已通读方法部分（实验与附录以略读为准）：§3 Methodology（§3.1 触觉辅助任务规划、§3.2 触觉精修 VLA 策略、Algorithm 1），§4 实验设定略读

## 推荐继续阅读

- [Awesome Touch 仓库](https://github.com/sun254667/awesome-touch)
- [原文](https://arxiv.org/abs/2507.17294)
