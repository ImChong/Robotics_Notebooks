---
type: entity
tags: [paper, curated-index, awesome-world-models, sun254667-wm]
status: complete
updated: 2026-10-06
arxiv: "2509.18428"
venue: "arXiv 2025"
code: https://github.com/baheytharwat/lawm
summary: "Latent action pretraining through world modeling."
related:
  - ../entities/awesome-world-models.md
  - ../overview/sun-awesome-wm-technology-map.md
  - ../methods/generative-world-models.md
  - ../methods/model-based-rl.md
  - ../tasks/manipulation.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/papers/sun_awesome_wm_2509_18428_lawm-latent-action-pretraining-through-w.md
  - ../../sources/papers/sun_awesome_wm_catalog.md
  - ../../sources/repos/awesome-world-models.md
---

# LAWM

**LAWM: Latent Action Pretraining Through World Modeling** 收录于 [Awesome World Models](https://github.com/sun254667/awesome-world-models) **第 198/571** 篇，分组 **62 Joint WAM & Unified VLA-World Models**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

Latent action pretraining through world modeling.

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WM | World Model | 环境前向预测模型 |
| WAM | World Action Model | 世界预测与动作联合建模 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| MBRL | Model-Based RL | 基于模型的强化学习 |

## 为什么重要

- Latent action pretraining through world modeling.
- 想横向对照同一分组的其他工作，可以从 [Awesome World Models 技术地图](../overview/sun-awesome-wm-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome World Models](../entities/awesome-world-models.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
  subgraph PRE["阶段一：Latent Action Pretraining（无动作标签，自监督）"]
    V["无标注视频：机器人录像 / 人类操作视频"] --> X["当前帧 x_t"]
    LANG["语言指令 c"] --> IL
    X --> IL["模仿学习模型 IL_θ（实现中为 BAKU 或 Diffusion Policy）"]
    IL --> Z["预测 n 个潜在动作 z_{t:t+n-1}"]
    X --> WM
    Z --> WM["世界模型 WM_φ（DreamerV3 RSSM）从 x_t 起逐步 rollout"]
    WM --> XH["预测未来 n 帧"]
    XH --> LOSS["与真实视频帧比较：MSE + β·KL 正则"]
    V --> LOSS
    LOSS -.联合优化.-> IL
    LOSS -.联合优化.-> WM
  end
  IL --> PRIOR["保留预训练后的 IL_θ，丢弃世界模型"]
  subgraph FT["阶段二：Action Finetuning（有动作标签）"]
    PRIOR --> RE["仅重新初始化最后一层 action latent 层以匹配真实动作维度"]
    DEMO["带真值动作的下游示范（图像、指令、机器人状态）"] --> SUP["监督微调：动作预测损失"]
    RE --> SUP
    SUP --> OUT["输出真值动作序列 a_{t:t+n-1}"]
  end
```

图依据：论文 §3.1 / §3.2 与 Fig. 2。

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 198/571 |
| 分组 | 62 Joint WAM & Unified VLA-World Models |
| 出处 | arXiv 2025 |
| 论文 | <https://arxiv.org/abs/2509.18428> |
| 代码/项目 | <https://github.com/baheytharwat/lawm> |

## 核心机制（归纳）

### 策展导读要点

Latent action pretraining through world modeling.

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2509.18428) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 的 **62 Joint WAM & Unified VLA-World Models** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「LAWM」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

- 可确证的只有清单坐标：分组 **62 Joint WAM & Unified VLA-World Models**，以及 Highlights 点出的问题设定；本页不自行推导新结论。
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

- [`sources/papers/sun_awesome_wm_2509_18428_lawm-latent-action-pretraining-through-w.md`](../../sources/papers/sun_awesome_wm_2509_18428_lawm-latent-action-pretraining-through-w.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_wm_catalog.md`](../../sources/papers/sun_awesome_wm_catalog.md) — 列表总表
- [`sources/repos/awesome-world-models.md`](../../sources/repos/awesome-world-models.md)
- 论文：<https://arxiv.org/abs/2509.18428>
- [arXiv PDF](https://arxiv.org/pdf/2509.18428)：已通读方法部分（§1 引言、§3 方法）；实验（§4–§5）略读：实验设定与主要结果。

## 推荐继续阅读

- [Awesome World Models 仓库](https://github.com/sun254667/awesome-world-models)
- [原文](https://arxiv.org/abs/2509.18428)
