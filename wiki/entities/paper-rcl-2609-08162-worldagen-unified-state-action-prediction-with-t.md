---
type: entity
tags: [paper, curated-index, awesome-world-action-models-rcl, rcl-wam-catalog]
status: complete
updated: 2026-10-06
arxiv: "2609.08162"
venue: "2026"
summary: "WorldAgen shares a Transformer between action prediction and future-observation prediction, then adapts the shared representation using exploratory target-environment transitions. Simulated manipulation improves after ob"
related:
  - paper-rcl-wam-robot-learning-control-survey.md
  - ../overview/rcl-awesome-wam-technology-map.md
  - ../methods/generative-world-models.md
  - ../methods/vla.md
  - ../tasks/manipulation.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/papers/rcl_awesome_wam_2609_08162_worldagen-unified-state-action-predictio.md
  - ../../sources/papers/rcl_awesome_wam_catalog.md
  - ../../sources/repos/awesome-world-action-models-rcl.md
---

# WorldAgen

**WorldAgen: Unified State-Action Prediction with Test-Time World Model Training** 收录于 [Awesome World-Action Models (RCL)](https://github.com/rcl-robotics/Awesome-World-Action-Models) **第 555/564** 篇，分组 **WAMs**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

WorldAgen shares a Transformer between action prediction and future-observation prediction, then adapts the shared representation using exploratory target-environment transitions. Simulated manipulation improves after observation-only test-time training (TTT), while implementation ambiguities and unreported uncertai...

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WAM | World Action Model | 世界预测与动作生成耦合 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| IDM | Inverse Dynamics Model | 先预测未来再反推动作 |
| WM | World Model | 环境前向预测模型 |

## 为什么重要

- WorldAgen shares a Transformer between action prediction and future-observation prediction, then adapts the shared representation using exploratory target-environment transitions. Simulated manipulation improves after observation-only test-time training (TTT), while implementation ambiguities and unreported uncertai...
- 想横向对照同一分组的其他工作，可以从 [RCL Awesome WAM 技术地图](../overview/rcl-awesome-wam-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome World-Action Models](paper-rcl-wam-robot-learning-control-survey.md) 与站内 WAM / VLA 方法页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
    subgraph PRE["预训练（机器人示范轨迹，teacher forcing）"]
        P1["轨迹切分与动作分块<br/>轨迹单元 U_i = (G_i, O_i, S_i, A_i, A_p, O_p)"] --> P2["共享 Transformer 骨干 + Mixed Unidirectional Attention Mask<br/>（局部 mask + 全局 mask，严格因果）"]
        P2 --> P3["Agent model 头：预测动作块 Â_i（条件含指令 G_i）"]
        P2 --> P4["World model 头：预测下一观测块 Ô_i（任务无关，看不到 G_i）"]
        P3 --> P5["联合损失 L = L_a + λ·L_o"]
        P4 --> P5
    end
    subgraph INF["两步推理（对每个轨迹单元 i）"]
        I1["Agent model 步：零初始化动作占位 A_p 输入，预测 Â_i 并覆盖占位"] --> I2["World model 步：基于 Â_i 预测下一观测块 Ô_i"]
        I2 --> I3["执行动作，环境产生下一单元 U_i+1"]
        I3 -.-> I1
    end
    subgraph TTT["测试时训练 TTT（部署到新环境）"]
        T1["阶段 1：随机探索<br/>记录 (O_i, S_i, A_i)，指令统一换成 no-lang token"] --> T2["阶段 2：仅用观测重建损失 L_o<br/>对共享骨干做 LoRA 微调，策略头冻结"]
    end
    PRE --> INF
    PRE --> TTT
    T2 -.->|"更新后的世界模型表征"| INF
```

图依据：论文 Method 各小节（Task Formulation、Joint State–Action Prediction、Mixed Unidirectional Attention Mask、Two-Step Inference、Test-Time Training）/ Fig.1–3

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 555/564 |
| 分组 | WAMs |
| 出处 | 2026 |
| 论文 | <https://arxiv.org/abs/2609.08162> |
| 子类 / 象限 | 联合视频动作建模 · 泛化与动作对齐 · Q1 · One Model × 联合预测 |

## 核心机制（归纳）

### 策展导读要点

WorldAgen shares a Transformer between action prediction and future-observation prediction, then adapts the shared representation using exploratory target-environment transitions. Simulated manipulation improves after observation-only test-time training (TTT), while implementation ambiguities and unreported uncertai...

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内 [World Action Models（WAM）](../concepts/world-action-models.md) 等概念页。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2609.08162) 为准。
- 横向对照请回到 [技术地图](../overview/rcl-awesome-wam-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/rcl-awesome-wam-technology-map.md) 的 **WAMs** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：清单 Contribution 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「WorldAgen」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

- 可确证的只有清单坐标：分组 **WAMs**，以及 Contribution 点出的问题设定；本页不自行推导新结论。
- 适用边界：本页不能替代原文 PDF；开源状态以项目页实际链接为准（清单可能滞后）。
- 要深读这篇，建议直接从原文入手，再回到下方关联的方法 / 任务页对照。

## 常见误区

1. 不要把 Awesome 条目的 Contribution 当成完整方法证明——它只是策展导读。
2. 若站内已有这篇的深读页，以那一页为准——本页只是清单入口，不含实验数据。

## 关联页面

- 列表实体：[Awesome World-Action Models（RCL）](paper-rcl-wam-robot-learning-control-survey.md)
- 技术地图：[RCL Awesome WAM 技术地图](../overview/rcl-awesome-wam-technology-map.md)
- 方法/任务：[generative-world-models.md](../methods/generative-world-models.md)、[manipulation.md](../tasks/manipulation.md)

## 参考来源

- [`sources/papers/rcl_awesome_wam_2609_08162_worldagen-unified-state-action-predictio.md`](../../sources/papers/rcl_awesome_wam_2609_08162_worldagen-unified-state-action-predictio.md) — 本条目策展摘录
- [`sources/papers/rcl_awesome_wam_catalog.md`](../../sources/papers/rcl_awesome_wam_catalog.md) — 列表总表
- [`sources/repos/awesome-world-action-models-rcl.md`](../../sources/repos/awesome-world-action-models-rcl.md)
- [`docs/PAPERS.md`](https://github.com/RCL-Robotics/Awesome-World-Action-Models/blob/main/docs/PAPERS.md) — 上游论文目录
- 论文：<https://arxiv.org/abs/2609.08162>
- [arXiv PDF](https://arxiv.org/pdf/2609.08162)：已通读方法部分（实验与附录以略读为准）：摘要、Introduction、Method 全部小节、Experiments 的 Implementation Details；结果表略读

## 推荐继续阅读

- [Awesome World-Action Models (RCL) 仓库](https://github.com/rcl-robotics/Awesome-World-Action-Models)
- [原文](https://arxiv.org/abs/2609.08162)
