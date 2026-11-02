---
type: entity
tags: [paper, curated-index, awesome-touch, sun254667-touch]
status: complete
updated: 2026-10-06
arxiv: "2606.11184"
venue: "arXiv 2026"
summary: "Lightweight force-conditioned tactile foresight framework; TacForceWM predicts short-horizon tactile latent dynamics from dual-finger tactile observations conditioned on wrist force/torque; enables proactive contact reas"
related:
  - ../entities/awesome-touch.md
  - ../overview/sun-awesome-touch-technology-map.md
  - ../methods/vla.md
  - ../methods/imitation-learning.md
  - ../tasks/manipulation.md
  - ../tasks/bimanual-manipulation.md
sources:
  - ../../sources/papers/sun_awesome_touch_2606_11184_tacforesight-force-guided-tactile-world.md
  - ../../sources/papers/sun_awesome_touch_catalog.md
  - ../../sources/repos/awesome-touch.md
---

# TacForeSight

**TacForeSight: Force-Guided Tactile World Model for Contact-Rich Manipulation** 收录于 [Awesome Touch](https://github.com/sun254667/awesome-touch) **第 062/065** 篇，分组 **Visuo-Tactile World Models**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

Lightweight force-conditioned tactile foresight framework; TacForceWM predicts short-horizon tactile latent dynamics from dual-finger tactile observations conditioned on wrist force/torque; enables proactive contact reasoning with real-time inference.

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VTLA | Vision-Tactile-Language-Action | 视–触–语言–动作策略 |
| WM | World Model | 视触觉前向预测 |
| WAM | World Action Model | 世界–动作联合建模 |
| Sim2Real | Simulation to Real | 仿真到真机迁移 |

## 为什么重要

- Lightweight force-conditioned tactile foresight framework; TacForceWM predicts short-horizon tactile latent dynamics from dual-finger tactile observations conditioned on wrist force/torque; enables proactive contact reasoning with real-time inference.
- 想横向对照同一分组的其他工作，可以从 [Awesome Touch 技术地图](../overview/sun-awesome-touch-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome Touch](../entities/awesome-touch.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
    subgraph ST1["阶段1：力条件触觉世界模型（预训练）"]
        T["双指触觉场（3D 标记位移）"] --> TK["触觉 tokenizer：共享空间编码器 + 位置/手指 ID 嵌入 + Transformer，[CLS] → 触觉隐变量 z_t"]
        FT["腕部六轴力/力矩序列（高频）"] --> FE["力编码器：膨胀因果 1D 卷积 + 因果下采样 → 力条件 c_t"]
        TK --> LD["隐空间动力学预测器 T_ψ（AdaLN 注入力条件）"]
        FE --> LD
        LD --> ZF["预测未来触觉隐变量块 Ẑ（偏移 Δ）"]
        ZF --> LW["损失：预测 MSE + 一阶时间差分项 + SIGReg 正则"]
    end
    subgraph ST2["阶段2：预测式触觉条件策略"]
        IMG["RGB → 冻结 DINOv2-small → 视觉特征 h_img"]
        PRO["本体历史 → MLP → h_s"]
        TK --> CA
        ZF --> CA["当前-未来触觉交互：当前隐变量作 Query，预测隐变量作 Key/Value，残差得 H_tac，时间平均 → h_tac"]
        CA --> GT["触觉引导门控 α = σ(MLP(h_tac))，按通道融合：h_vt = (1-α)h_img + α·h_tac"]
        IMG --> GT
        GT --> CD["与 h_s 拼接，条件编码器 → 全局条件 y_t"]
        PRO --> CD
        CD --> AH["条件 flow-matching 动作头（时序 U-Net）→ 动作块 A_t"]
    end
```

图依据：论文 §III-A / §III-B / Fig.2

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 062/065 |
| 分组 | Visuo-Tactile World Models |
| 出处 | arXiv 2026 |
| 论文 | <https://arxiv.org/abs/2606.11184> |

## 核心机制（归纳）

### 策展导读要点

Lightweight force-conditioned tactile foresight framework; TacForceWM predicts short-horizon tactile latent dynamics from dual-finger tactile observations conditioned on wrist force/torque; enables proactive contact reasoning with real-time inference.

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2606.11184) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-touch-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-touch-technology-map.md) 的 **Visuo-Tactile World Models** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「TacForeSight」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

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

## 参考来源

- [`sources/papers/sun_awesome_touch_2606_11184_tacforesight-force-guided-tactile-world.md`](../../sources/papers/sun_awesome_touch_2606_11184_tacforesight-force-guided-tactile-world.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_touch_catalog.md`](../../sources/papers/sun_awesome_touch_catalog.md) — 列表总表
- [`sources/repos/awesome-touch.md`](../../sources/repos/awesome-touch.md)
- 论文：<https://arxiv.org/abs/2606.11184>
- [arXiv PDF](https://arxiv.org/pdf/2606.11184)：已通读方法部分（实验与附录以略读为准）：§III Method（§III-A 力条件触觉世界模型、§III-B 预测式触觉条件策略）

## 推荐继续阅读

- [Awesome Touch 仓库](https://github.com/sun254667/awesome-touch)
- [原文](https://arxiv.org/abs/2606.11184)
