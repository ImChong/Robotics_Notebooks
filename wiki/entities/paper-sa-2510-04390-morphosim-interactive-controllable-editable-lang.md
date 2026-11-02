---
type: entity
tags: [paper, curated-index, awesome-world-models, sun254667-wm]
status: complete
updated: 2026-10-06
arxiv: "2510.04390"
venue: "arXiv 2025"
code: https://github.com/eric-ai-lab/Morph4D
summary: "Interactive, controllable, editable language-guided 4D world simulator."
related:
  - ../entities/awesome-world-models.md
  - ../overview/sun-awesome-wm-technology-map.md
  - ../methods/generative-world-models.md
  - ../methods/model-based-rl.md
  - ../tasks/manipulation.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/papers/sun_awesome_wm_2510_04390_morphosim-interactive-controllable-edita.md
  - ../../sources/papers/sun_awesome_wm_catalog.md
  - ../../sources/repos/awesome-world-models.md
---

# MorphoSim

**MorphoSim: Interactive, Controllable, Editable Language-guided 4D World Simulator** 收录于 [Awesome World Models](https://github.com/sun254667/awesome-world-models) **第 099/571** 篇，分组 **43 3D & 4D World Models General Purpose**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

Interactive, controllable, editable language-guided 4D world simulator.

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WM | World Model | 环境前向预测模型 |
| WAM | World Action Model | 世界预测与动作联合建模 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| MBRL | Model-Based RL | 基于模型的强化学习 |

## 为什么重要

- Interactive, controllable, editable language-guided 4D world simulator.
- 想横向对照同一分组的其他工作，可以从 [Awesome World Models 技术地图](../overview/sun-awesome-wm-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome World Models](../entities/awesome-world-models.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
    L["自然语言指令 L"] --> P["命令参数化器：LLM agent 解析指令，得到执行计划 P = (M, Q)，M 为 GEN 或 EDIT"]
    P -->|"M = GEN"| G1["LLM 把运动方向解析为轨迹 T 与逐帧 bounding box，快速运动对应更大的框间距"]
    G1 --> G2["场景生成：文本到视频扩散，Dynamic Control 偏置交叉注意力，最小化能量函数并更新 latent（无需训练）"]
    G2 --> G3["重建阶段：基于 MoSca 的动态 3D 高斯，并蒸馏 2D 基础模型特征，得到 4D 场景"]
    G3 --> OUT1["生成的 4D 场景"]
    P -->|"M = EDIT"| E1["Static Edit：LLM 迭代调整阈值，按高斯语义特征与查询特征的相似度筛选目标高斯"]
    G3 --> E1
    E1 --> E2["执行颜色编辑、物体移除或物体提取，同一参数应用到所有帧"]
    E2 --> OUT2["编辑后的 4D 场景"]
```

图依据：论文 §III（Fig.2）、§III-B、§III-C。

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 099/571 |
| 分组 | 43 3D & 4D World Models General Purpose |
| 出处 | arXiv 2025 |
| 论文 | <https://arxiv.org/abs/2510.04390> |
| 代码/项目 | <https://github.com/eric-ai-lab/Morph4D> |

## 核心机制（归纳）

### 策展导读要点

Interactive, controllable, editable language-guided 4D world simulator.

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2510.04390) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 的 **43 3D & 4D World Models General Purpose** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「MorphoSim」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

- 可确证的只有清单坐标：分组 **43 3D & 4D World Models General Purpose**，以及 Highlights 点出的问题设定；本页不自行推导新结论。
- 适用边界：本页不能替代原文 PDF；开源状态以项目页实际链接为准（清单可能滞后）。
- 要深读这篇，建议直接从原文入手，再回到下方关联的方法 / 任务页对照。

## 常见误区

1. 不要把 Awesome 条目的 Highlights 当成完整方法证明——它只是策展导读。
2. 若站内已有这篇的深读页，以那一页为准——本页只是清单入口，不含实验数据。

## 关联页面

- 列表实体：[Awesome World Models](../entities/awesome-world-models.md)
- 技术地图：[Awesome World Models 技术地图](../overview/sun-awesome-wm-technology-map.md)
- 方法/任务：[generative-world-models.md](../methods/generative-world-models.md)、[manipulation.md](../tasks/manipulation.md)

## 参考来源

- [`sources/papers/sun_awesome_wm_2510_04390_morphosim-interactive-controllable-edita.md`](../../sources/papers/sun_awesome_wm_2510_04390_morphosim-interactive-controllable-edita.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_wm_catalog.md`](../../sources/papers/sun_awesome_wm_catalog.md) — 列表总表
- [`sources/repos/awesome-world-models.md`](../../sources/repos/awesome-world-models.md)
- 论文：<https://arxiv.org/abs/2510.04390>
- [arXiv PDF](https://arxiv.org/pdf/2510.04390)：已通读方法部分（实验与附录以略读为准）：§III MorphoSim（命令参数化器、场景生成、场景编辑模块，含 Fig.2）；§IV 实验仅略读

## 推荐继续阅读

- [Awesome World Models 仓库](https://github.com/sun254667/awesome-world-models)
- [原文](https://arxiv.org/abs/2510.04390)
