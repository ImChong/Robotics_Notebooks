---
type: entity
tags: [paper, curated-index, awesome-world-models, sun254667-wm]
status: complete
updated: 2026-10-06
arxiv: "2605.14398"
venue: "arXiv 2026"
summary: "Builds physics‑based world models through executable simulation code, with a framework coordinating planning, code generation, visual review, and physics analysis agents."
related:
  - ../entities/awesome-world-models.md
  - ../overview/sun-awesome-wm-technology-map.md
  - ../methods/generative-world-models.md
  - ../methods/model-based-rl.md
  - ../tasks/manipulation.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/papers/sun_awesome_wm_2605_14398_coding-agent-is-good-as-world-simulator.md
  - ../../sources/papers/sun_awesome_wm_catalog.md
  - ../../sources/repos/awesome-world-models.md
---

# Coding Agent Is Good As World Simulator

**Coding Agent Is Good As World Simulator** 收录于 [Awesome World Models](https://github.com/sun254667/awesome-world-models) **第 054/571** 篇，分组 **116 Code Agents & Simulation Construction**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

Builds physics‑based world models through executable simulation code, with a framework coordinating planning, code generation, visual review, and physics analysis agents.

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WM | World Model | 环境前向预测模型 |
| WAM | World Action Model | 世界预测与动作联合建模 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| MBRL | Model-Based RL | 基于模型的强化学习 |

## 为什么重要

- Builds physics‑based world models through executable simulation code, with a framework coordinating planning, code generation, visual review, and physics analysis agents.
- 想横向对照同一分组的其他工作，可以从 [Awesome World Models 技术地图](../overview/sun-awesome-wm-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome World Models](../entities/awesome-world-models.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
  U["用户 prompt\n（可选参考图）"] --> P["Plan Agent\n结构化场景计划（13 节 schema）"]
  P --> A["资产抽取与生成\n目录 mesh / 原生车辆机器人 / 程序化基元 / 生成边界"]
  P --> T["拓扑谓词\n按依赖顺序推导位姿"]
  P --> UC{"用户确认 / 修改计划"}
  UC --> C["Code Agent\n生成独立 PyChrono 程序"]
  SK["Skill Library"] --> C
  RAG["RAG 检索\n源码 / 文档 / demo / 论坛"] --> C
  A --> AC["统一资产目录\n离线生成：参考图 → SAM 3D 重建"]
  AC --> C
  C --> L["静态 linter"]
  L -->|未通过| C
  L --> E["分阶段执行\n物理 pass → 首帧 pass → 完整录制"]
  E --> V["Visual-Analysis Agent\n描述渲染视频"]
  E --> N["确定性物理检查\n轨迹异常扫描"]
  V --> R["Review Agent\n三路证据交叉验证"]
  N --> R
  R -->|缺陷报告：一次针对性补丁| C
  R -->|通过| G["验收门\n剥离插桩 + 校验和"]
  R -->|超出修复上限| B["blocked"]
  G --> O["可复现的仿真程序"]
```

图依据：论文 §2.1–2.4 / Fig.2、Fig.4、Fig.5（论文方法名为 ChronoAgentic）

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 054/571 |
| 分组 | 116 Code Agents & Simulation Construction |
| 出处 | arXiv 2026 |
| 论文 | <https://arxiv.org/abs/2605.14398> |

## 核心机制（归纳）

### 策展导读要点

Builds physics‑based world models through executable simulation code, with a framework coordinating planning, code generation, visual review, and physics analysis agents.

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2605.14398) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 的 **116 Code Agents & Simulation Construction** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「Coding Agent Is Good As World Simulator」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

- 可确证的只有清单坐标：分组 **116 Code Agents & Simulation Construction**，以及 Highlights 点出的问题设定；本页不自行推导新结论。
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

- [`sources/papers/sun_awesome_wm_2605_14398_coding-agent-is-good-as-world-simulator.md`](../../sources/papers/sun_awesome_wm_2605_14398_coding-agent-is-good-as-world-simulator.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_wm_catalog.md`](../../sources/papers/sun_awesome_wm_catalog.md) — 列表总表
- [`sources/repos/awesome-world-models.md`](../../sources/repos/awesome-world-models.md)
- 论文：<https://arxiv.org/abs/2605.14398>
- [arXiv PDF](https://arxiv.org/pdf/2605.14398)：已通读方法部分（实验与附录以略读为准）：摘要、§1、§2.1–2.4、§3 实验设定、附录 Fig.4/Fig.5 图注

## 推荐继续阅读

- [Awesome World Models 仓库](https://github.com/sun254667/awesome-world-models)
- [原文](https://arxiv.org/abs/2605.14398)
