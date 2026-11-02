---
type: entity
tags: [paper, curated-index, awesome-world-models, sun254667-wm]
status: complete
updated: 2026-10-06
arxiv: "2505.20425"
venue: "arXiv 2025"
summary: "One-shot visual imitation for unseen tasks using world-model-guided trajectory generation."
related:
  - ../entities/awesome-world-models.md
  - ../overview/sun-awesome-wm-technology-map.md
  - ../methods/generative-world-models.md
  - ../methods/model-based-rl.md
  - ../tasks/manipulation.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/papers/sun_awesome_wm_2505_20425_osvi-wm-one-shot-visual-imitation-for-un.md
  - ../../sources/papers/sun_awesome_wm_catalog.md
  - ../../sources/repos/awesome-world-models.md
---

# OSVI-WM

**OSVI-WM: One-Shot Visual Imitation for Unseen Tasks using World-Model-Guided Trajectory Generation** 收录于 [Awesome World Models](https://github.com/sun254667/awesome-world-models) **第 313/571** 篇，分组 **812 Manipulation**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

One-shot visual imitation for unseen tasks using world-model-guided trajectory generation.

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WM | World Model | 环境前向预测模型 |
| WAM | World Action Model | 世界预测与动作联合建模 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| MBRL | Model-Based RL | 基于模型的强化学习 |

## 为什么重要

- One-shot visual imitation for unseen tasks using world-model-guided trajectory generation.
- 想横向对照同一分组的其他工作，可以从 [Awesome World Models 技术地图](../overview/sun-awesome-wm-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome World Models](../entities/awesome-world-models.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
    A["专家演示帧 E_1..E_N<br/>(下采样到 N=10)"] --> B["共享 ResNet-18 编码器<br/>展平为潜状态 e_1..e_N"]
    C["智能体初始观测 R_1"] --> B2["同一编码器得 r_1"]
    subgraph WM["World-Model-Guided Trajectory Generation"]
        P["动作模型 π (因果 Transformer)<br/>a_i = π(r_i..r_1, e_N..e_1)"]
        W["世界模型 W (因果 Transformer)<br/>r_{i+1} = W(r_i, a_i, ..., r_1, a_1)"]
        P --> W
        W -->|"递归 i = 1..M-1"| P
    end
    B --> P
    B2 --> P
    W --> T["预测的未来潜状态 r_2..r_M"]
    T --> U["空间池化(2D softmax 期望坐标)<br/>+ 时间维注意力池化 → 2F 维向量"]
    U --> V["两层 MLP 预测 5 个带属性路径点<br/>末端 3D 位置 + 夹爪开/合"]
    V --> X["相机到机器人坐标变换<br/>逆运动学路径点控制器执行"]
    X -.->|"可选: 执行失败则重新观测<br/>结合原演示重规划路径点"| C
    T -.->|"训练: L1 世界模型损失 L_wm<br/>(真值潜状态 stop-gradient)"| L["总损失 L = L_sdtw + α(τ) L_wm"]
    V -.->|"训练: soft-DTW 路径点损失 L_sdtw"| L
```

图依据：论文 §3 Our Method（§3.1 World-Model-Guided Trajectory Generation、§3.2 Predicting Waypoints、§3.3 Training）/ Fig.2–3。

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 313/571 |
| 分组 | 812 Manipulation |
| 出处 | arXiv 2025 |
| 论文 | <https://arxiv.org/abs/2505.20425> |

## 核心机制（归纳）

### 策展导读要点

One-shot visual imitation for unseen tasks using world-model-guided trajectory generation.

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2505.20425) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 的 **812 Manipulation** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「OSVI-WM」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

- 可确证的只有清单坐标：分组 **812 Manipulation**，以及 Highlights 点出的问题设定；本页不自行推导新结论。
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

- [`sources/papers/sun_awesome_wm_2505_20425_osvi-wm-one-shot-visual-imitation-for-un.md`](../../sources/papers/sun_awesome_wm_2505_20425_osvi-wm-one-shot-visual-imitation-for-un.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_wm_catalog.md`](../../sources/papers/sun_awesome_wm_catalog.md) — 列表总表
- [`sources/repos/awesome-world-models.md`](../../sources/repos/awesome-world-models.md)
- 论文：<https://arxiv.org/abs/2505.20425>
- [arXiv PDF](https://arxiv.org/pdf/2505.20425)：已通读方法部分（实验与附录以略读为准）：§3.1–3.3

## 推荐继续阅读

- [Awesome World Models 仓库](https://github.com/sun254667/awesome-world-models)
- [原文](https://arxiv.org/abs/2505.20425)
