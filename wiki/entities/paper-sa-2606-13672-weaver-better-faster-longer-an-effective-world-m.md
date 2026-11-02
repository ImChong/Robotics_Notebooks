---
type: entity
tags: [paper, curated-index, awesome-world-models, sun254667-wm]
status: complete
updated: 2026-10-06
arxiv: "2606.13672"
venue: "arXiv 2026"
summary: "Multi-view WM with flow-matching loss, SOTA on robotic manipulation, satisfying fidelity, consistency, and efficiency."
related:
  - ../entities/awesome-world-models.md
  - ../overview/sun-awesome-wm-technology-map.md
  - ../methods/generative-world-models.md
  - ../methods/model-based-rl.md
  - ../tasks/manipulation.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/papers/sun_awesome_wm_2606_13672_weaver-better-faster-longer-an-effective.md
  - ../../sources/papers/sun_awesome_wm_catalog.md
  - ../../sources/repos/awesome-world-models.md
---

# WEAVER

**WEAVER: Better, Faster, Longer: An Effective World Model for Robotic Manipulation** 收录于 [Awesome World Models](https://github.com/sun254667/awesome-world-models) **第 230/571** 篇，分组 **63 World Models for VLA Training & Evaluation Combining Paradigms**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

Multi-view WM with flow-matching loss, SOTA on robotic manipulation, satisfying fidelity, consistency, and efficiency.

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WM | World Model | 环境前向预测模型 |
| WAM | World Action Model | 世界预测与动作联合建模 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| MBRL | Model-Based RL | 基于模型的强化学习 |

## 为什么重要

- Multi-view WM with flow-matching loss, SOTA on robotic manipulation, satisfying fidelity, consistency, and efficiency.
- 想横向对照同一分组的其他工作，可以从 [Awesome World Models 技术地图](../overview/sun-awesome-wm-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome World Models](../entities/awesome-world-models.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
    OBS["观测 o_t：多视角 RGB 图像 + 本体状态 q_t"] --> ENC["预训练编码器（SD3 VAE）：图像 patch token + 本体 token → 隐状态 z_t"]
    ENC --> MEM["稀疏记忆（每 k 帧）+ 短期历史（最近帧）"]
    POL["基础策略 π0.5：采样 h 步动作块 a_t"] --> WM
    MEM --> WM["隐空间动力学模型 f_φ：2D Transformer（空间注意力 + 因果时间注意力），flow-matching 训练"]
    WM --> ZF["预测未来隐变量 ẑ"]
    ZF --> DEC["预训练解码器：未来多视角图像与本体状态 ô"]
    DEC --> POL
    ZF --> RH["隐空间奖励头 R 与 critic V"]
    RH --> EV["策略评估：真实动作轨迹开环执行，记录预测奖励"]
    RH --> IM["策略改进：B 条 rollout 的 H 步优势，最优者超过阈值才蒸馏进基础策略"]
    RH --> TP["测试时规划：采样 B 个动作块，执行优势最高者（best-of-N）"]
```

图依据：论文 §3 / §3.1-§3.4 / Fig.2

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 230/571 |
| 分组 | 63 World Models for VLA Training & Evaluation Combining Paradigms |
| 出处 | arXiv 2026 |
| 论文 | <https://arxiv.org/abs/2606.13672> |

## 核心机制（归纳）

### 策展导读要点

Multi-view WM with flow-matching loss, SOTA on robotic manipulation, satisfying fidelity, consistency, and efficiency.

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2606.13672) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 的 **63 World Models for VLA Training & Evaluation Combining Paradigms** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「WEAVER」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

- 可确证的只有清单坐标：分组 **63 World Models for VLA Training & Evaluation Combining Paradigms**，以及 Highlights 点出的问题设定；本页不自行推导新结论。
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

- [`sources/papers/sun_awesome_wm_2606_13672_weaver-better-faster-longer-an-effective.md`](../../sources/papers/sun_awesome_wm_2606_13672_weaver-better-faster-longer-an-effective.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_wm_catalog.md`](../../sources/papers/sun_awesome_wm_catalog.md) — 列表总表
- [`sources/repos/awesome-world-models.md`](../../sources/repos/awesome-world-models.md)
- 论文：<https://arxiv.org/abs/2606.13672>
- [arXiv PDF](https://arxiv.org/pdf/2606.13672)：已通读方法部分（实验与附录以略读为准）：§1 引言、§3 WEAVER（§3.1-§3.4）、§4 实验设置开头

## 推荐继续阅读

- [Awesome World Models 仓库](https://github.com/sun254667/awesome-world-models)
- [原文](https://arxiv.org/abs/2606.13672)
