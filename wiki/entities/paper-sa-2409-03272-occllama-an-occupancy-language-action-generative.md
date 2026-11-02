---
type: entity
tags: [paper, curated-index, awesome-world-models, sun254667-wm]
status: complete
updated: 2026-10-06
arxiv: "2409.03272"
venue: "arXiv 2024"
summary: "Occupancy-language-action generative world model."
related:
  - ../entities/awesome-world-models.md
  - ../overview/sun-awesome-wm-technology-map.md
  - ../methods/generative-world-models.md
  - ../methods/model-based-rl.md
  - ../tasks/manipulation.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/papers/sun_awesome_wm_2409_03272_occllama-an-occupancy-language-action-ge.md
  - ../../sources/papers/sun_awesome_wm_catalog.md
  - ../../sources/repos/awesome-world-models.md
---

# OccLLaMA

**OccLLaMA: An Occupancy-Language-Action Generative World Model for Autonomous Driving** 收录于 [Awesome World Models](https://github.com/sun254667/awesome-world-models) **第 464/571** 篇，分组 **82 Autonomous Driving**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

Occupancy-language-action generative world model.

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WM | World Model | 环境前向预测模型 |
| WAM | World Action Model | 世界预测与动作联合建模 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| MBRL | Model-Based RL | 基于模型的强化学习 |

## 为什么重要

- Occupancy-language-action generative world model.
- 想横向对照同一分组的其他工作，可以从 [Awesome World Models 技术地图](../overview/sun-awesome-wm-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome World Models](../entities/awesome-world-models.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
    A["占据栅格 x (H×W×D)<br/>语义 occupancy"] --> B["稀疏编码：丢弃 air 体素<br/>→ 按 BEV 柱的伪点云 (高度 d, 语义 l)"]
    B --> C["Pillars 嵌入 + Swin Transformer<br/>→ BEV 特征 z"]
    C --> D["向量量化 (码本 Z)<br/>→ 离散场景 token"]
    D --> E["解码器：voxel head 给占据 mask<br/>class head 仅监督被占据体素语义"]
    D --> F["统一词表 V：场景 token + 语言 token<br/>+ 动作 token(waypoint 分箱) + 功能 token"]
    G["语言输入 / 历史动作"] --> F
    F --> H["LLaMA 生成式世界模型<br/>next token 预测 + next scene 预测<br/>(场景查询 + 空间注意力，一次前向预测整幅场景)"]
    H --> I["输出：未来占据 / 场景描述与问答 / 规划动作"]
    J["训练阶段 1：场景 tokenizer<br/>(之后冻结)"] -.-> D
    K["阶段 2：占据-语言-动作预训练<br/>世界模型目标 + 场景 caption 目标"] -.-> H
    L["阶段 3：指令微调 (LoRA)"] -.-> H
```

图依据：论文 §3（Fig.2 框架）、§3.2 场景 tokenizer、§3.3 生成式世界模型 / Algorithm 1、§3.4 三阶段训练。

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 464/571 |
| 分组 | 82 Autonomous Driving |
| 出处 | arXiv 2024 |
| 论文 | <https://arxiv.org/abs/2409.03272> |

## 核心机制（归纳）

### 策展导读要点

Occupancy-language-action generative world model.

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2409.03272) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 的 **82 Autonomous Driving** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「OccLLaMA」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

- 可确证的只有清单坐标：分组 **82 Autonomous Driving**，以及 Highlights 点出的问题设定；本页不自行推导新结论。
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

- [`sources/papers/sun_awesome_wm_2409_03272_occllama-an-occupancy-language-action-ge.md`](../../sources/papers/sun_awesome_wm_2409_03272_occllama-an-occupancy-language-action-ge.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_wm_catalog.md`](../../sources/papers/sun_awesome_wm_catalog.md) — 列表总表
- [`sources/repos/awesome-world-models.md`](../../sources/repos/awesome-world-models.md)
- 论文：<https://arxiv.org/abs/2409.03272>
- [arXiv PDF](https://arxiv.org/pdf/2409.03272)：已通读方法部分（实验与附录以略读为准）：§3 方法（概述、3.2 场景 tokenizer、3.3 生成式世界模型、3.4 三阶段训练）、§4 实验设定与实现细节

## 推荐继续阅读

- [Awesome World Models 仓库](https://github.com/sun254667/awesome-world-models)
- [原文](https://arxiv.org/abs/2409.03272)
