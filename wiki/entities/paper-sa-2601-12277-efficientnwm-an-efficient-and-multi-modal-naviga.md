---
type: entity
tags: [paper, curated-index, awesome-world-models, sun254667-wm]
status: complete
updated: 2026-10-06
arxiv: "2601.12277"
venue: "arXiv 2026"
code: https://github.com/robotnav-bot/NOW
summary: "Efficient and multi-modal navigation system with a one-step world model."
related:
  - ../entities/awesome-world-models.md
  - ../overview/sun-awesome-wm-technology-map.md
  - ../methods/generative-world-models.md
  - ../methods/model-based-rl.md
  - ../tasks/manipulation.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/papers/sun_awesome_wm_2601_12277_efficientnwm-an-efficient-and-multi-moda.md
  - ../../sources/papers/sun_awesome_wm_catalog.md
  - ../../sources/repos/awesome-world-models.md
---

# EfficientNWM

**EfficientNWM: An Efficient and Multi-Modal Navigation System with One-Step World Model** 收录于 [Awesome World Models](https://github.com/sun254667/awesome-world-models) **第 346/571** 篇，分组 **813 Navigation**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

Efficient and multi-modal navigation system with a one-step world model.

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WM | World Model | 环境前向预测模型 |
| WAM | World Action Model | 世界预测与动作联合建模 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| MBRL | Model-Based RL | 基于模型的强化学习 |

## 为什么重要

- Efficient and multi-modal navigation system with a one-step world model.
- 想横向对照同一分组的其他工作，可以从 [Awesome World Models 技术地图](../overview/sun-awesome-wm-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome World Models](../entities/awesome-world-models.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart LR
    subgraph WM["单步导航世界模型（3D U-Net）"]
        direction TB
        OBS["当前观测 s_τ"] --> VAE["预训练 VAE 编码到 latent"]
        ACT["动作 a_τ：N 个相对路点 (x,y,φ)"] --> MLP["4 层 MLP 动作嵌入"]
        VAE --> UNET["3D U-Net：CNN + Spatial Attn + Temporal Attn"]
        MLP -->|"cross-attention"| UNET
        UNET --> GEN["shortcut 目标：一步生成 11 帧未来 latent"]
        GEN --> DEC["VAE 解码为未来图像"]
    end
    subgraph PL["基于 CEM 的规划"]
        direction TB
        INIT["anchor 初始化：固定线/角速度候选轨迹"] --> CAND["32 个候选动作"]
        CAND --> ROLL["世界模型预测未来图像"]
        ROLL --> SCORE["按目标打分：LPIPS / SigLIP / 距离+Depth-Anything 深度项"]
        SCORE --> CEM["CEM 选 top-5 并迭代 5 次"]
        CEM -->|"更新采样分布"| CAND
        CEM --> BEST["输出最优动作"]
    end
    GOAL["多模态目标：图像 / 语言 / 点目标"] --> SCORE
    WM -.->|"作为前向模型"| ROLL
```

图依据：论文 §3.1–3.4 / Fig.2 / Fig.3；§5.1 的打分函数与 CEM 设置（32 样本、top-5、5 次迭代）

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 346/571 |
| 分组 | 813 Navigation |
| 出处 | arXiv 2026 |
| 论文 | <https://arxiv.org/abs/2601.12277> |
| 代码/项目 | <https://github.com/robotnav-bot/NOW> |

## 核心机制（归纳）

### 策展导读要点

Efficient and multi-modal navigation system with a one-step world model.

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2601.12277) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 的 **813 Navigation** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「EfficientNWM」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

- 可确证的只有清单坐标：分组 **813 Navigation**，以及 Highlights 点出的问题设定；本页不自行推导新结论。
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

- [`sources/papers/sun_awesome_wm_2601_12277_efficientnwm-an-efficient-and-multi-moda.md`](../../sources/papers/sun_awesome_wm_2601_12277_efficientnwm-an-efficient-and-multi-moda.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_wm_catalog.md`](../../sources/papers/sun_awesome_wm_catalog.md) — 列表总表
- [`sources/repos/awesome-world-models.md`](../../sources/repos/awesome-world-models.md)
- 论文：<https://arxiv.org/abs/2601.12277>
- [arXiv PDF](https://arxiv.org/pdf/2601.12277)：已通读方法部分（实验与附录以略读为准）：§1–§5（含 3.1 单步世界模型、3.2 架构、3.3 训练、3.4 规划、5.1 导航评测设定）

## 推荐继续阅读

- [Awesome World Models 仓库](https://github.com/sun254667/awesome-world-models)
- [原文](https://arxiv.org/abs/2601.12277)
