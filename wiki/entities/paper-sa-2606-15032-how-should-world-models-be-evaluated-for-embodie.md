---
type: entity
tags: [paper, curated-index, awesome-world-models, sun254667-wm]
status: complete
updated: 2026-10-06
arxiv: "2606.15032"
venue: "arXiv 2026"
summary: "Proposes an L0-L7 evaluation ladder spanning from visual plausibility to policy optimization utility, emphasizing intervention action fidelity and closed-loop rollout effectiveness."
related:
  - ../entities/awesome-world-models.md
  - ../overview/sun-awesome-wm-technology-map.md
  - ../methods/generative-world-models.md
  - ../methods/model-based-rl.md
  - ../tasks/manipulation.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/papers/sun_awesome_wm_2606_15032_how-should-world-models-be-evaluated-for.md
  - ../../sources/papers/sun_awesome_wm_catalog.md
  - ../../sources/repos/awesome-world-models.md
---

# How Should World Models Be Evaluated for Embodied Decision-Making? A Decision...

**How Should World Models Be Evaluated for Embodied Decision-Making? A Decision-Making-Centric Position** 收录于 [Awesome World Models](https://github.com/sun254667/awesome-world-models) **第 537/571** 篇，分组 **Surveys & Overviews**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

Proposes an L0-L7 evaluation ladder spanning from visual plausibility to policy optimization utility, emphasizing intervention action fidelity and closed-loop rollout effectiveness.

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WM | World Model | 环境前向预测模型 |
| WAM | World Action Model | 世界预测与动作联合建模 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| MBRL | Model-Based RL | 基于模型的强化学习 |

## 为什么重要

- Proposes an L0-L7 evaluation ladder spanning from visual plausibility to policy optimization utility, emphasizing intervention action fidelity and closed-loop rollout effectiveness.
- 想横向对照同一分组的其他工作，可以从 [Awesome World Models 技术地图](../overview/sun-awesome-wm-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome World Models](../entities/awesome-world-models.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
  Q["核心问题：对用于 embodied decision-making 的 world model，应测什么？<br/>论点：评测应关注 interventional reasoning、policy evaluation、planning、policy optimization，而非仅视频逼真度"] --> S["§3 What Is Being Evaluated?<br/>按论文实际评测内容梳理近期文献；3.6 The world-model claim"]
  S --> M["诊断：claim/evidence mismatch<br/>较低层证据被用来支撑更强的决策层主张"]
  S --> L["§4 World-Model Evaluation Ladder L0–L7"]
  L --> L1["L0–L3 偏产物质量：L0 Visual plausibility、L1 Logged-future prediction、L2 Semantic alignment、L3 Physical plausibility"]
  L --> L2["L4–L7 偏决策效用：L4 Action controllability / interventional fidelity、L5 Reward, value, and outcome fidelity、L6 Policy evaluation and ranking、L7 Policy optimization / planning utility"]
  M --> F["§5 Decision-Making-Centric Evaluation Framework<br/>先声明 decision contract"]
  L1 --> F
  L2 --> F
  F --> P["§6 Benchmark Protocol"]
  P --> P1["Step 0 声明 world-model contract；Step 1 构造 policy 与 intervention splits；Step 2 L0–L3 仅作 diagnostics"]
  P1 --> P2["Step 3 interventional action fidelity；Step 4 closed-loop fixed-policy rollouts；Step 5 固定预算下的 policy optimization"]
  P2 --> P3["Step 6 adversarial exploitability 与 uncertainty；Step 7 hidden tasks、held-out policies、统计报告；Step 8 报告 decision-utility profile"]
  P3 --> R["real-robot 的 minimal feasible variant"]
  R --> D["§7 Discussion and Conclusion<br/>7.1 为何区分重要；7.2 反对意见与适用范围；7.3 局限与开放问题"]
```

图依据：立场论证与评测框架图，论文摘要与 §1 贡献列表、§3.6、§4（Table 7 的 L0–L7 阶梯）、§5、§6（Step 0–8）、§7 小节标题；本图只给论证结构与协议步骤，不含论文的统计结论。

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 537/571 |
| 分组 | Surveys & Overviews |
| 出处 | arXiv 2026 |
| 论文 | <https://arxiv.org/abs/2606.15032> |

## 核心机制（归纳）

### 策展导读要点

Proposes an L0-L7 evaluation ladder spanning from visual plausibility to policy optimization utility, emphasizing intervention action fidelity and closed-loop rollout effectiveness.

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2606.15032) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 的 **Surveys & Overviews** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「How Should World Models Be Evaluated for Embodied Decision-Making? A Decision...」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

- 可确证的只有清单坐标：分组 **Surveys & Overviews**，以及 Highlights 点出的问题设定；本页不自行推导新结论。
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

- [`sources/papers/sun_awesome_wm_2606_15032_how-should-world-models-be-evaluated-for.md`](../../sources/papers/sun_awesome_wm_2606_15032_how-should-world-models-be-evaluated-for.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_wm_catalog.md`](../../sources/papers/sun_awesome_wm_catalog.md) — 列表总表
- [`sources/repos/awesome-world-models.md`](../../sources/repos/awesome-world-models.md)
- 论文：<https://arxiv.org/abs/2606.15032>
- [arXiv PDF](https://arxiv.org/pdf/2606.15032)：已通读摘要、§1、§2 开头、§4 的 L0–L7 定义与 Table 7、§5 开头、§6 的 Step 0–8 标题与部分正文、§7 小节标题；§3 文献综述表与附录以略读为准。

## 推荐继续阅读

- [Awesome World Models 仓库](https://github.com/sun254667/awesome-world-models)
- [原文](https://arxiv.org/abs/2606.15032)
