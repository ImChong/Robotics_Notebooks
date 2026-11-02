---
type: entity
tags: [paper, curated-index, awesome-touch, sun254667-touch]
status: complete
updated: 2026-10-06
arxiv: "2603.01700"
venue: "arXiv 2026"
summary: "Asynchronous architecture: tactile encoder (System 1) performs continuous streaming inference at 100Hz; VLA planner (System 2) queries compressed tactile history on-demand."
related:
  - ../entities/awesome-touch.md
  - ../overview/sun-awesome-touch-technology-map.md
  - ../methods/vla.md
  - ../methods/imitation-learning.md
  - ../tasks/manipulation.md
  - ../tasks/bimanual-manipulation.md
sources:
  - ../../sources/papers/sun_awesome_touch_2603_01700_tacmamba-a-tactile-history-compression-a.md
  - ../../sources/papers/sun_awesome_touch_catalog.md
  - ../../sources/repos/awesome-touch.md
---

# TacMamba

**TacMamba: A Tactile History Compression Adapter Bridging Fast Reflexes and Slow VLA Reasoning** 收录于 [Awesome Touch](https://github.com/sun254667/awesome-touch) **第 011/065** 篇，分组 **21 Unified Vision-Tactile-Language-Action Frameworks**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

Asynchronous architecture: tactile encoder (System 1) performs continuous streaming inference at 100Hz; VLA planner (System 2) queries compressed tactile history on-demand.

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VTLA | Vision-Tactile-Language-Action | 视–触–语言–动作策略 |
| WM | World Model | 视触觉前向预测 |
| WAM | World Action Model | 世界–动作联合建模 |
| Sim2Real | Simulation to Real | 仿真到真机迁移 |

## 为什么重要

- Asynchronous architecture: tactile encoder (System 1) performs continuous streaming inference at 100Hz; VLA planner (System 2) queries compressed tactile history on-demand.
- 想横向对照同一分组的其他工作，可以从 [Awesome Touch 技术地图](../overview/sun-awesome-touch-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome Touch](../entities/awesome-touch.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
    A["指尖 1D 力传感器, 100 Hz 触觉流"] --> B["RevIN + Channel Independence"]
    B --> C["堆叠 Mamba block: 选择性 SSM, 参数 Δ / B / C 随输入变化"]
    C --> D["压缩隐状态 ht, 每个新读数 O(1) 更新"]
    subgraph S1["Stage 1: 时序判别自监督预训练"]
        P1["按触觉梯度幅值自动切分子任务阶段"] --> P2["构造状态对 (hi, hj), 关系均衡采样"]
        P2 --> P3["判别器分类 Early / Simultaneous / Late, 交叉熵损失"]
    end
    D --> P2
    D --> S2["Stage 2: 冻结 Mamba 编码器, 把 ht 投影为 soft prompt z_tac"]
    S2 --> V["低频 (约 1 Hz) VLA 推理, 异步注入触觉 soft prompt"]
    P1 --> S3["Phase-Uniform Sampling: 各子任务阶段在 batch 中等权"]
    S3 --> V
```

图依据：论文 §III、§IV-A–B（Fig. 1–3）。

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 011/065 |
| 分组 | 21 Unified Vision-Tactile-Language-Action Frameworks |
| 出处 | arXiv 2026 |
| 论文 | <https://arxiv.org/abs/2603.01700> |

## 核心机制（归纳）

### 策展导读要点

Asynchronous architecture: tactile encoder (System 1) performs continuous streaming inference at 100Hz; VLA planner (System 2) queries compressed tactile history on-demand.

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2603.01700) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-touch-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-touch-technology-map.md) 的 **21 Unified Vision-Tactile-Language-Action Frameworks** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「TacMamba」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

- 可确证的只有清单坐标：分组 **21 Unified Vision-Tactile-Language-Action Frameworks**，以及 Highlights 点出的问题设定；本页不自行推导新结论。
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

- [`sources/papers/sun_awesome_touch_2603_01700_tacmamba-a-tactile-history-compression-a.md`](../../sources/papers/sun_awesome_touch_2603_01700_tacmamba-a-tactile-history-compression-a.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_touch_catalog.md`](../../sources/papers/sun_awesome_touch_catalog.md) — 列表总表
- [`sources/repos/awesome-touch.md`](../../sources/repos/awesome-touch.md)
- 论文：<https://arxiv.org/abs/2603.01700>
- [arXiv PDF](https://arxiv.org/pdf/2603.01700)：已通读方法部分（实验与附录以略读为准）：§III 硬件、§IV 方法、§V-A 实验设定

## 推荐继续阅读

- [Awesome Touch 仓库](https://github.com/sun254667/awesome-touch)
- [原文](https://arxiv.org/abs/2603.01700)
