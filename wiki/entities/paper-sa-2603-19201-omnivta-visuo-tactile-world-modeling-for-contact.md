---
type: entity
tags: [paper, curated-index, awesome-touch, sun254667-touch]
status: complete
updated: 2026-10-06
arxiv: "2603.19201"
venue: "arXiv 2026"
summary: "Large-scale OmniViTac dataset with 21,000+ tasks and 86 objects; world-model-based framework with self-supervised tactile encoder, two-stream visuo-tactile world model, contact-aware fusion policy, and 60Hz reflexive con"
related:
  - ../entities/awesome-touch.md
  - ../overview/sun-awesome-touch-technology-map.md
  - ../methods/vla.md
  - ../methods/imitation-learning.md
  - ../tasks/manipulation.md
  - ../tasks/bimanual-manipulation.md
  - ./paper-feelworld.md
sources:
  - ../../sources/papers/sun_awesome_touch_2603_19201_omnivta-visuo-tactile-world-modeling-for.md
  - ../../sources/papers/sun_awesome_touch_catalog.md
  - ../../sources/repos/awesome-touch.md
---

# OmniVTA

**OmniVTA: Visuo-Tactile World Modeling for Contact-Rich Robotic Manipulation** 收录于 [Awesome Touch](https://github.com/sun254667/awesome-touch) **第 061/065** 篇，分组 **Visuo-Tactile World Models**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

Large-scale OmniViTac dataset with 21,000+ tasks and 86 objects; world-model-based framework with self-supervised tactile encoder, two-stream visuo-tactile world model, contact-aware fusion policy, and 60Hz reflexive controller.

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VTLA | Vision-Tactile-Language-Action | 视–触–语言–动作策略 |
| WM | World Model | 视触觉前向预测 |
| WAM | World Action Model | 世界–动作联合建模 |
| Sim2Real | Simulation to Real | 仿真到真机迁移 |

## 为什么重要

- Large-scale OmniViTac dataset with 21,000+ tasks and 86 objects; world-model-based framework with self-supervised tactile encoder, two-stream visuo-tactile world model, contact-aware fusion policy, and 60Hz reflexive controller.
- 想横向对照同一分组的其他工作，可以从 [Awesome Touch 技术地图](../overview/sun-awesome-touch-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome Touch](../entities/awesome-touch.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
    T["3D marker 位移触觉输入"] --> V1["TactileVAE<br/>时空编码器 + 隐式形变解码器"]
    V1 --> W
    subgraph SLOW["Slow Policy（约 4 Hz）"]
        W["Visuo-Tactile World Model<br/>双流时空 DiT：视觉流 + 触觉流"]
        W --> L["Latent Tactile Differential Encoder<br/>当前触觉 / 预测触觉 / 二者之差"]
        L --> G["接触概率预测 → 门控<br/>自适应加权视觉与触觉特征"]
        G --> P["Visuo-Tactile Diffusion Policy<br/>生成 action chunk"]
    end
    I["图像 15 Hz、触觉 60 Hz、本体感知"] --> W
    I --> P
    L -. "预测触觉 latent" .-> R
    subgraph FAST["Fast Policy（60 Hz）"]
        R["Reflexive Latent Tactile Controller<br/>MLP 输出修正动作"]
    end
    F["单帧触觉反馈"] --> R
    P --> S["加权求和：慢速规划动作 + 快速修正动作"]
    R --> S
    S --> X["执行"]
```

图依据：论文 §IV-A–E / Fig.5–8（世界模型条件含 action 的 2D 投影等细节未画出）

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 061/065 |
| 分组 | Visuo-Tactile World Models |
| 出处 | arXiv 2026 |
| 论文 | <https://arxiv.org/abs/2603.19201> |

## 核心机制（归纳）

### 策展导读要点

Large-scale OmniViTac dataset with 21,000+ tasks and 86 objects; world-model-based framework with self-supervised tactile encoder, two-stream visuo-tactile world model, contact-aware fusion policy, and 60Hz reflexive controller.

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2603.19201) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-touch-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-touch-technology-map.md) 的 **Visuo-Tactile World Models** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「OmniVTA」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

- 可确证的只有清单坐标：分组 **Visuo-Tactile World Models**，以及 Highlights 点出的问题设定；本页不自行推导新结论。
- 适用边界：本页不能替代原文 PDF；开源状态以项目页实际链接为准（清单可能滞后）。
- 要深读这篇，建议直接从原文入手，再回到下方关联的方法 / 任务页对照。

## 常见误区

1. 不要把 Awesome 条目的 Highlights 当成完整方法证明——它只是策展导读。
2. 若站内已有这篇的深读页，以那一页为准——本页只是清单入口，不含实验数据。

## 关联页面

- 列表实体：[Awesome Touch](../entities/awesome-touch.md)
- 技术地图：[Awesome Touch 技术地图](../overview/sun-awesome-touch-technology-map.md)
- 方法/任务：[vla.md](../methods/vla.md)、[manipulation.md](../tasks/manipulation.md)
- 同族深读：[FeelWorld（分层视触觉 WM + CEM）](./paper-feelworld.md)（arXiv:2607.24267v1）

## 参考来源

- [`sources/papers/sun_awesome_touch_2603_19201_omnivta-visuo-tactile-world-modeling-for.md`](../../sources/papers/sun_awesome_touch_2603_19201_omnivta-visuo-tactile-world-modeling-for.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_touch_catalog.md`](../../sources/papers/sun_awesome_touch_catalog.md) — 列表总表
- [`sources/repos/awesome-touch.md`](../../sources/repos/awesome-touch.md)
- 论文：<https://arxiv.org/abs/2603.19201>
- [arXiv PDF](https://arxiv.org/pdf/2603.19201)：已通读方法部分（实验与附录以略读为准）：§IV 方法全部小节（系统概览、TactileVAE、世界模型、融合策略、RLTC）；OmniViTac 数据集（§III）与实验仅略读

## 推荐继续阅读

- [Awesome Touch 仓库](https://github.com/sun254667/awesome-touch)
- [原文](https://arxiv.org/abs/2603.19201)
