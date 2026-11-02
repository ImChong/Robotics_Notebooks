---
type: entity
tags: [paper, curated-index, awesome-world-models, sun254667-wm]
status: complete
updated: 2026-10-06
arxiv: "2504.13065"
venue: "arXiv 2025"
code: https://github.com/LeapLabTHU/EchoWorld
summary: "Learning motion-aware world models for echocardiography probe guidance."
related:
  - ../entities/awesome-world-models.md
  - ../overview/sun-awesome-wm-technology-map.md
  - ../methods/generative-world-models.md
  - ../methods/model-based-rl.md
  - ../tasks/manipulation.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/papers/sun_awesome_wm_2504_13065_echoworld-learning-motion-aware-world-mo.md
  - ../../sources/papers/sun_awesome_wm_catalog.md
  - ../../sources/repos/awesome-world-models.md
---

# EchoWorld

**EchoWorld: Learning Motion-Aware World Models for Echocardiography Probe Guidance** 收录于 [Awesome World Models](https://github.com/sun254667/awesome-world-models) **第 520/571** 篇，分组 **84 Scientific Discovery**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

Learning motion-aware world models for echocardiography probe guidance.

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WM | World Model | 环境前向预测模型 |
| WAM | World Action Model | 世界预测与动作联合建模 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| MBRL | Model-Based RL | 基于模型的强化学习 |

## 为什么重要

- Learning motion-aware world models for echocardiography probe guidance.
- 想横向对照同一分组的其他工作，可以从 [Awesome World Models 技术地图](../overview/sun-awesome-wm-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome World Models](../entities/awesome-world-models.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
    DATA["示教数据：机械臂持探头的心脏超声扫描<br/>每个 scan 为 (图像 I_t, 探头位姿 p_t) 序列"] --> PRE
    subgraph PRE["阶段一：预训练心脏世界模型 (§4.1)"]
        SP["空间世界建模：遮挡图像块<br/>context encoder 处理可见块，predictor 以 mask token 预测被遮区域特征 (L_spatial)"]
        MO["运动世界建模：同一 scan 的 I_a、I_b + 相对运动 p_a→b<br/>运动编码器 A_ψ 得 z，predictor 预测目标平均池化特征 (InfoNCE, L_motion)"]
        TG["target encoder 为 context encoder 的 EMA"]
        SP --> JOINT["L_total = L_spatial + λ·L_motion (λ=0.1)"]
        MO --> JOINT
        TG -.-> SP
        TG -.-> MO
    end
    PRE --> FT
    subgraph FT["阶段二：运动感知探头引导微调 (§4.2)"]
        HIST["历史视觉-运动序列 {I_ti, p_ti}"] --> FEAT["预训练视觉编码器与运动编码器提取特征"]
        FEAT --> MAA["motion-aware attention：<br/>相对位姿差 z_i→j 嵌入 key / value"]
        MAA --> HEAD["预测头回归到目标平面的探头运动 â_t<br/>L_guide = ‖a_t − â_t‖₁"]
    end
```

图依据：论文 §3.2、§4.1–4.2 / Fig.2、Fig.5 / Eq.1–12

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 520/571 |
| 分组 | 84 Scientific Discovery |
| 出处 | arXiv 2025 |
| 论文 | <https://arxiv.org/abs/2504.13065> |
| 代码/项目 | <https://github.com/LeapLabTHU/EchoWorld> |

## 核心机制（归纳）

### 策展导读要点

Learning motion-aware world models for echocardiography probe guidance.

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2504.13065) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 的 **84 Scientific Discovery** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「EchoWorld」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

- 可确证的只有清单坐标：分组 **84 Scientific Discovery**，以及 Highlights 点出的问题设定；本页不自行推导新结论。
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

- [`sources/papers/sun_awesome_wm_2504_13065_echoworld-learning-motion-aware-world-mo.md`](../../sources/papers/sun_awesome_wm_2504_13065_echoworld-learning-motion-aware-world-mo.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_wm_catalog.md`](../../sources/papers/sun_awesome_wm_catalog.md) — 列表总表
- [`sources/repos/awesome-world-models.md`](../../sources/repos/awesome-world-models.md)
- 论文：<https://arxiv.org/abs/2504.13065>
- [arXiv PDF](https://arxiv.org/pdf/2504.13065)：已通读方法部分（实验与附录以略读为准）：摘要、§1、§3（背景与数据）、§4 EchoWorld（4.1–4.2）、§5.2 评测协议开头

## 推荐继续阅读

- [Awesome World Models 仓库](https://github.com/sun254667/awesome-world-models)
- [原文](https://arxiv.org/abs/2504.13065)
