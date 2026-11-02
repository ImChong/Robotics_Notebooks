---
type: entity
tags: [paper, curated-index, awesome-touch, sun254667-touch]
status: complete
updated: 2026-10-06
arxiv: "2606.26663"
venue: "arXiv 2026"
summary: "Introduces Tactile Asymmetric Attention Mechanism (TAAM) with VideoClean mask to prevent tactile pollution; improves mean success rate by 38.9% overall and 86% on contact-rich tasks."
related:
  - ../entities/awesome-touch.md
  - ../overview/sun-awesome-touch-technology-map.md
  - ../methods/vla.md
  - ../methods/imitation-learning.md
  - ../tasks/manipulation.md
  - ../tasks/bimanual-manipulation.md
sources:
  - ../../sources/papers/sun_awesome_touch_2606_26663_tactile-wam-touch-aware-world-action-mod.md
  - ../../sources/papers/sun_awesome_touch_catalog.md
  - ../../sources/repos/awesome-touch.md
---

# Tactile-WAM

**Tactile-WAM: Touch-Aware World Action Model with Tactile Asymmetric Attention** 收录于 [Awesome Touch](https://github.com/sun254667/awesome-touch) **第 056/065** 篇，分组 **Tactile World Action Models WAM**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

Introduces Tactile Asymmetric Attention Mechanism (TAAM) with VideoClean mask to prevent tactile pollution; improves mean success rate by 38.9% overall and 86% on contact-rich tasks.

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VTLA | Vision-Tactile-Language-Action | 视–触–语言–动作策略 |
| WM | World Model | 视触觉前向预测 |
| WAM | World Action Model | 世界–动作联合建模 |
| Sim2Real | Simulation to Real | 仿真到真机迁移 |

## 为什么重要

- Introduces Tactile Asymmetric Attention Mechanism (TAAM) with VideoClean mask to prevent tactile pollution; improves mean success rate by 38.9% overall and 86% on contact-rich tasks.
- 想横向对照同一分组的其他工作，可以从 [Awesome Touch 技术地图](../overview/sun-awesome-touch-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome Touch](../entities/awesome-touch.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
    RGB["RGB 历史观测"] --> VAE
    TAC["左 / 右触觉图像 → 横向拼接 mosaic"] --> VAE
    VAE["共享冻结 Wan2.2 VAE：视觉潜变量与触觉潜变量（各加模态嵌入）"]
    LANG["语言指令（UMT5-XXL，经交叉注意注入）"] --> DIT
    ST["本体感觉状态 + 带噪动作"] --> DIT
    VAE --> DIT
    subgraph DIT["语言条件 Wan DiT：联合预测未来视觉潜变量、未来触觉潜变量、动作块"]
        direction TB
        VC["VideoClean 掩码：阻断视觉 query 访问触觉 key（保护视觉先验）"]
        TB["Touch-aware 注意力偏置：按因果块把 action query 导向触觉 key"]
    end
    TAC --> PHI["可微形变估计器 Φ（无可学习参数）：由相邻触觉帧得 6 维 touch-aware proxy"]
    PHI -->|"接触强度 sc → 偏置 bc"| TB
    DIT --> OUTV["未来视频帧"]
    DIT --> OUTT["未来触觉潜变量 → 冻结解码器 → 触觉图像"]
    DIT --> OUTA["动作块"]
    OUTT -.->|"训练：proxy 损失监督"| PHI
```

图依据：论文 §4 Method（架构概览、Tactile Asymmetric Attention Mask、Touch-aware Proxy）/ Fig. 4、附录 B

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 056/065 |
| 分组 | Tactile World Action Models WAM |
| 出处 | arXiv 2026 |
| 论文 | <https://arxiv.org/abs/2606.26663> |

## 核心机制（归纳）

### 策展导读要点

Introduces Tactile Asymmetric Attention Mechanism (TAAM) with VideoClean mask to prevent tactile pollution; improves mean success rate by 38.9% overall and 86% on contact-rich tasks.

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2606.26663) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-touch-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-touch-technology-map.md) 的 **Tactile World Action Models WAM** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「Tactile-WAM」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

- 可确证的只有清单坐标：分组 **Tactile World Action Models WAM**，以及 Highlights 点出的问题设定；本页不自行推导新结论。
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

- [`sources/papers/sun_awesome_touch_2606_26663_tactile-wam-touch-aware-world-action-mod.md`](../../sources/papers/sun_awesome_touch_2606_26663_tactile-wam-touch-aware-world-action-mod.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_touch_catalog.md`](../../sources/papers/sun_awesome_touch_catalog.md) — 列表总表
- [`sources/repos/awesome-touch.md`](../../sources/repos/awesome-touch.md)
- 论文：<https://arxiv.org/abs/2606.26663>
- [arXiv PDF](https://arxiv.org/pdf/2606.26663)：已通读方法部分（实验与附录以略读为准）：§4 方法（架构概览、联合视觉-触觉-动作建模、非对称注意力掩码、touch-aware 偏置与 proxy）、附录 B 模型细节

## 推荐继续阅读

- [Awesome Touch 仓库](https://github.com/sun254667/awesome-touch)
- [原文](https://arxiv.org/abs/2606.26663)
