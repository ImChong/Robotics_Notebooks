---
type: entity
tags: [paper, curated-index, awesome-touch, sun254667-touch]
status: complete
updated: 2026-10-06
arxiv: "2512.23864"
venue: "arXiv 2025"
summary: "Hierarchical perception scheme with macro/local/micro vision; Hierarchical Spatial Alignment (HSA) loss aligns tactile tokens with spatial counterparts; tactile world model predicts future tactile signals; achieves up to"
related:
  - ../entities/awesome-touch.md
  - ../overview/sun-awesome-touch-technology-map.md
  - ../methods/vla.md
  - ../methods/imitation-learning.md
  - ../tasks/manipulation.md
  - ../tasks/bimanual-manipulation.md
sources:
  - ../../sources/papers/sun_awesome_touch_2512_23864_dreamtacvla-learning-to-feel-the-future.md
  - ../../sources/papers/sun_awesome_touch_catalog.md
  - ../../sources/repos/awesome-touch.md
---

# DreamTacVLA

**DreamTacVLA: Learning to Feel the Future for Contact-Rich Manipulation** 收录于 [Awesome Touch](https://github.com/sun254667/awesome-touch) **第 018/065** 篇，分组 **22 Tactile-Enhanced VLA with Foundation Models**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

Hierarchical perception scheme with macro/local/micro vision; Hierarchical Spatial Alignment (HSA) loss aligns tactile tokens with spatial counterparts; tactile world model predicts future tactile signals; achieves up to 95% success.

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VTLA | Vision-Tactile-Language-Action | 视–触–语言–动作策略 |
| WM | World Model | 视触觉前向预测 |
| WAM | World Action Model | 世界–动作联合建模 |
| Sim2Real | Simulation to Real | 仿真到真机迁移 |

## 为什么重要

- Hierarchical perception scheme with macro/local/micro vision; Hierarchical Spatial Alignment (HSA) loss aligns tactile tokens with spatial counterparts; tactile world model predicts future tactile signals; achieves up to 95% success.
- 想横向对照同一分组的其他工作，可以从 [Awesome Touch 技术地图](../overview/sun-awesome-touch-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome Touch](../entities/awesome-touch.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
    IN["输入：第三人称图像 + 腕部图像<br/>触觉图像 + 语言指令 + 机器人状态"] --> ENC["多模态编码器 E_ψ<br/>CLIP ViT + 状态 MLP"]
    subgraph S1["Stage 1：草稿动作"]
        ENC --> HSA["HSA 层次空间对齐损失<br/>触觉对齐腕部/第三人称对应区域（InfoNCE）"]
        HSA --> HA["对齐表示 H_align"]
        HA --> POL1["策略 π_θ（H_null 占位）<br/>行为克隆 L1 损失"]
        POL1 --> AD["草稿动作 a_draft"]
    end
    WM["触觉世界模型 W_φ<br/>V-JEPA 2，无标注触觉序列自监督预训练并冻结"] --> Z["当前触觉嵌入 z_τ"]
    subgraph S2["Stage 2：Think-Dream-Act"]
        AD --> FM["DREAM：预测 MLP F_η"]
        Z --> FM
        FM --> HD["想象的未来触觉潜变量 H_dream"]
        HA --> POL2["ACT：策略 π_θ 再次推理"]
        HD --> POL2
        POL2 --> AF["最终动作 a_final"]
    end
    WM -. "监督 L_W（stop-gradient）" .-> FM
```

图依据：论文 §3 方法（Fig. 2、§3.1 Stage 1、§3.2 Stage 2）。

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 018/065 |
| 分组 | 22 Tactile-Enhanced VLA with Foundation Models |
| 出处 | arXiv 2025 |
| 论文 | <https://arxiv.org/abs/2512.23864> |

## 核心机制（归纳）

### 策展导读要点

Hierarchical perception scheme with macro/local/micro vision; Hierarchical Spatial Alignment (HSA) loss aligns tactile tokens with spatial counterparts; tactile world model predicts future tactile signals; achieves up to 95% success.

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2512.23864) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-touch-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-touch-technology-map.md) 的 **22 Tactile-Enhanced VLA with Foundation Models** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「DreamTacVLA」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

- 可确证的只有清单坐标：分组 **22 Tactile-Enhanced VLA with Foundation Models**，以及 Highlights 点出的问题设定；本页不自行推导新结论。
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

- [`sources/papers/sun_awesome_touch_2512_23864_dreamtacvla-learning-to-feel-the-future.md`](../../sources/papers/sun_awesome_touch_2512_23864_dreamtacvla-learning-to-feel-the-future.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_touch_catalog.md`](../../sources/papers/sun_awesome_touch_catalog.md) — 列表总表
- [`sources/repos/awesome-touch.md`](../../sources/repos/awesome-touch.md)
- 论文：<https://arxiv.org/abs/2512.23864>
- [arXiv PDF](https://arxiv.org/pdf/2512.23864)：已通读方法部分（实验与附录以略读为准）：摘要、§3 方法（§3.1–3.2）

## 推荐继续阅读

- [Awesome Touch 仓库](https://github.com/sun254667/awesome-touch)
- [原文](https://arxiv.org/abs/2512.23864)
