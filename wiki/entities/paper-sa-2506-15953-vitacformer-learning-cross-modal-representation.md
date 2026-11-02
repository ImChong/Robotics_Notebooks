---
type: entity
tags: [paper, curated-index, awesome-touch, sun254667-touch]
status: complete
updated: 2026-10-06
arxiv: "2506.15953"
venue: "RSS 2026"
summary: "Cross-modal representation learning for visuo-tactile dexterous manipulation."
related:
  - ../entities/awesome-touch.md
  - ../overview/sun-awesome-touch-technology-map.md
  - ../methods/vla.md
  - ../methods/imitation-learning.md
  - ../tasks/manipulation.md
  - ../tasks/bimanual-manipulation.md
sources:
  - ../../sources/papers/sun_awesome_touch_2506_15953_vitacformer-learning-cross-modal-represe.md
  - ../../sources/papers/sun_awesome_touch_catalog.md
  - ../../sources/repos/awesome-touch.md
---

# ViTacFormer

**ViTacFormer: Learning Cross-Modal Representation for Visuo-Tactile Dexterous Manipulation** 收录于 [Awesome Touch](https://github.com/sun254667/awesome-touch) **第 030/065** 篇，分组 **51 Imitation Learning with Tactile Feedback**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

Cross-modal representation learning for visuo-tactile dexterous manipulation.

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VTLA | Vision-Tactile-Language-Action | 视–触–语言–动作策略 |
| WM | World Model | 视触觉前向预测 |
| WAM | World Action Model | 世界–动作联合建模 |
| Sim2Real | Simulation to Real | 仿真到真机迁移 |

## 为什么重要

- Cross-modal representation learning for visuo-tactile dexterous manipulation.
- 想横向对照同一分组的其他工作，可以从 [Awesome Touch 技术地图](../overview/sun-awesome-touch-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome Touch](../entities/awesome-touch.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
    subgraph TRAIN["训练期（CVAE 风格）"]
        PA["本体感觉（关节）+ 专家动作序列"] --> ENC["Transformer 编码器"] --> Z["风格变量 z<br/>推理时置零"]
    end
    subgraph OBS["多模态观测"]
        V["视觉：ZED Mini 立体 + 两个腕部鱼眼相机"]
        T["触觉：10 指尖 3 轴力/力矩"]
        J["本体感觉：双臂/灵巧手/颈部关节"]
    end
    V --> CA1["跨注意力融合（第 1 次）<br/>视觉查询触觉、触觉查询视觉，拼接为隐状态"]
    T --> CA1
    CA1 --> FT["自回归预测未来触觉 token<br/>输入：z + 本体感觉 + 视触觉观测"]
    Z --> FT
    J --> FT
    FT --> CA2["拼接预测触觉与当前输入 token<br/>跨注意力融合（第 2 次）"]
    CA2 --> DEC["Transformer 编码器-解码器 → 动作序列（100 帧）"]
    DEC --> EXE["时间平滑后 10 Hz 执行"]
    FT -. "训练监督：L_tactile" .-> LOSS["总损失 = KL + 关节 L1 + 触觉 L1 + 末端位姿 L_arm"]
    DEC -. "训练监督" .-> LOSS
    FT -. "前 75% epoch 用真值触觉 token，后 25% 改用预测值" .-> CUR["两阶段课程（Scheduled Sampling 思路）"]
```

图依据：论文 §IV-A / Fig.3（跨注意力融合）、§IV-B（触觉预测与两阶段课程）、§IV-C / Fig.2 与式 (1)(2)（CVAE 架构与损失）、§IV-D 实现细节。

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 030/065 |
| 分组 | 51 Imitation Learning with Tactile Feedback |
| 出处 | RSS 2026 |
| 论文 | <https://arxiv.org/abs/2506.15953> |

## 核心机制（归纳）

### 策展导读要点

Cross-modal representation learning for visuo-tactile dexterous manipulation.

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2506.15953) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-touch-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-touch-technology-map.md) 的 **51 Imitation Learning with Tactile Feedback** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「ViTacFormer」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

- 可确证的只有清单坐标：分组 **51 Imitation Learning with Tactile Feedback**，以及 Highlights 点出的问题设定；本页不自行推导新结论。
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

- [`sources/papers/sun_awesome_touch_2506_15953_vitacformer-learning-cross-modal-represe.md`](../../sources/papers/sun_awesome_touch_2506_15953_vitacformer-learning-cross-modal-represe.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_touch_catalog.md`](../../sources/papers/sun_awesome_touch_catalog.md) — 列表总表
- [`sources/repos/awesome-touch.md`](../../sources/repos/awesome-touch.md)
- 论文：<https://arxiv.org/abs/2506.15953>
- [arXiv PDF](https://arxiv.org/pdf/2506.15953)：已通读方法部分（实验与附录以略读为准）：§III 问题与硬件、§IV-A/B/C/D 方法与实现细节、附录 A–D 输入与推理细节。

## 推荐继续阅读

- [Awesome Touch 仓库](https://github.com/sun254667/awesome-touch)
- [原文](https://arxiv.org/abs/2506.15953)
