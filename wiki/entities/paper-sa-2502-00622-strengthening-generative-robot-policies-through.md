---
type: entity
tags: [paper, curated-index, awesome-world-models, sun254667-wm]
status: complete
updated: 2026-10-06
arxiv: "2502.00622"
venue: "arXiv 2025"
summary: "Strengthens generative robot policies through predictive world modeling."
related:
  - ../entities/awesome-world-models.md
  - ../overview/sun-awesome-wm-technology-map.md
  - ../methods/generative-world-models.md
  - ../methods/model-based-rl.md
  - ../tasks/manipulation.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/papers/sun_awesome_wm_2502_00622_strengthening-generative-robot-policies.md
  - ../../sources/papers/sun_awesome_wm_catalog.md
  - ../../sources/repos/awesome-world-models.md
---

# Strengthening Generative Robot Policies through Predictive World Modeling

**Strengthening Generative Robot Policies through Predictive World Modeling** 收录于 [Awesome World Models](https://github.com/sun254667/awesome-world-models) **第 334/571** 篇，分组 **812 Manipulation**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

Strengthens generative robot policies through predictive world modeling.（arXiv 当前版本题为《Inference-Time Enhancement of Generative Robot Policies via Predictive World Modeling》；本页沿用清单所列题名）

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WM | World Model | 环境前向预测模型 |
| WAM | World Action Model | 世界预测与动作联合建模 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| MBRL | Model-Based RL | 基于模型的强化学习 |

## 为什么重要

- Strengthens generative robot policies through predictive world modeling.
- 想横向对照同一分组的其他工作，可以从 [Awesome World Models 技术地图](../overview/sun-awesome-wm-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome World Models](../entities/awesome-world-models.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
    DEMO["专家示教数据 D_expert<br/>滑动窗口切成 状态-动作片段"] --> POL["生成式策略训练<br/>Diffusion Policy（DDPM）<br/>输入 I_t，输出动作块 a_t:t+T"]
    DEMO --> WMD
    EXP["探索数据 D_explore<br/>人为随机扰动系统，不求完成任务"] --> WMD["世界模型数据 D_W<br/>= D_expert ∪ D_explore"]
    WMD --> WM["预测式世界模型 W<br/>输入 I_t 与 a_t:t+T，预测 I_t+1:t+T+1<br/>视觉版：条件扩散，单步图像预测器递归调用<br/>两阶段训练，推理时固定噪声"]
    POL --> PROP["在线：从策略采样 K 个动作提议"]
    PROP --> RANK["GPC-RANK<br/>W 预测各提议的未来，奖励模型 R 打分，选最高者"]
    PROP --> OPT["GPC-OPT<br/>以提议为热启动，对 R(W(·)) 做 M 步梯度上升"]
    WM --> RANK
    WM --> OPT
    RANK --> ACT["执行所选动作块，receding horizon"]
    OPT --> ACT
    RW["奖励模型 R：训练的小网络，或 VLM 直接挑选"] --> RANK
    RW --> OPT
    ACT -.->|"新观测"| PROP
```

图依据：论文 §III（Overview of Generative Predictive Control）、§IV（World Model Learning）与 Fig. 2、Alg. 1。GPC-RANK+OPT 为两者的组合（对 K 个提议各做优化后选最高者）。

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 334/571 |
| 分组 | 812 Manipulation |
| 出处 | arXiv 2025 |
| 论文 | <https://arxiv.org/abs/2502.00622> |
| 项目页 | <https://computationalrobotics.seas.harvard.edu/GPC> |

## 核心机制（归纳）

### 策展导读要点

Strengthens generative robot policies through predictive world modeling.

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2502.00622) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 的 **812 Manipulation** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「Strengthening Generative Robot Policies through Predictive World Modeling」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

- 可确证的只有清单坐标：分组 **812 Manipulation**，以及 Highlights 点出的问题设定；本页不自行推导新结论。
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

- [`sources/papers/sun_awesome_wm_2502_00622_strengthening-generative-robot-policies.md`](../../sources/papers/sun_awesome_wm_2502_00622_strengthening-generative-robot-policies.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_wm_catalog.md`](../../sources/papers/sun_awesome_wm_catalog.md) — 列表总表
- [`sources/repos/awesome-world-models.md`](../../sources/repos/awesome-world-models.md)
- 论文：<https://arxiv.org/abs/2502.00622>
- [arXiv PDF](https://arxiv.org/pdf/2502.00622)：已通读方法部分（实验与附录以略读为准）：§III Overview of Generative Predictive Control、§IV World Model Learning。注：当前 arXiv 版标题为 Inference-Time Enhancement of Generative Robot Policies via Predictive World Modeling，与页面标题措辞不同，编号一致
- 核对（2026-11-02）：查 arXiv 摘要页版本史：v1（2025-02-02）题 Strengthening Generative Robot Policies through Predictive World Modeling，v4（2026-03-11，现行）改题；本页题名对应 v1；已补题名变化说明；本次只核对版本史与题名，流程图及方法细节未重读，沿用此前核对。

## 推荐继续阅读

- [Awesome World Models 仓库](https://github.com/sun254667/awesome-world-models)
- [原文](https://arxiv.org/abs/2502.00622)
