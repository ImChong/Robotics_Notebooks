---
type: entity
tags: [paper, curated-index, awesome-sim2real, longchao-sim2real]
status: complete
updated: 2026-10-06
arxiv: "2310.18847"
summary: "LongchaoDa AwesomeSim2Real 收录；分组 Observation / Sensor Fusion。 本页是清单索引，细节以原文为准。"
related:
  - ../entities/awesome-sim2real.md
  - ../overview/lc-awesome-sim2real-technology-map.md
  - ../concepts/sim2real.md
  - ../methods/reinforcement-learning.md
  - ../tasks/locomotion.md
  - ../tasks/manipulation.md
sources:
  - ../../sources/papers/lc_awesome_sim2real_2310_18847_world-model-based-sim2real-transfer-for.md
  - ../../sources/papers/lc_awesome_sim2real_catalog.md
  - ../../sources/repos/awesome-sim2real.md
---

# World model based sim2real transfer for visual navigation

**World model based sim2real transfer for visual navigation** 收录于 [AwesomeSim2Real](https://github.com/LongchaoDa/AwesomeSim2Real) **第 064/139** 篇，分组 **Observation / Sensor Fusion**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

LongchaoDa AwesomeSim2Real 收录；分组 Observation / Sensor Fusion。 本页是清单索引，细节以原文为准。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Sim2Real | Simulation to Real | 仿真策略迁移到真机 |
| MDP | Markov Decision Process | 状态–动作–转移–奖励形式化 |
| DR | Domain Randomization | 域随机化 |
| FM | Foundation Model | 大模型/基础模型增强迁移 |

## 为什么重要

- LongchaoDa AwesomeSim2Real 收录；分组 Observation / Sensor Fusion。 本页是清单索引，细节以原文为准。
- 想横向对照同一分组的其他工作，可以从 [AwesomeSim2Real 技术地图](../overview/lc-awesome-sim2real-technology-map.md) 逐条展开。
- 顺着列表实体 [AwesomeSim2Real](../entities/awesome-sim2real.md) 与站内 Sim2Real 方法/任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
    SIM["CARLA 仿真（差速机器人，天气随机化 + 非玩家交通）→ FPV-BEV 图像序列"] --> PTRAIN["训练感知模型：ResNet-50（ImageNet 初始化）FPV → 嵌入 z_t，损失 L_CR = 重建损失 L_R + β·InfoNCE"]
    PTRAIN --> BEVDEC["BEV decoder 由 z_t 重建 BEV 图"]
    SIM --> MTRAIN["训练 Memory 模型：LSTM + MDN，输入 (z, a) 历史序列"]
    OBS["RGB 观测 o_t（真实机器人，冻结模型零样本部署）"] --> RES["ResNet-50 编码 → z_t"]
    RES --> ASC["ASC：在 1439 个 BEV anchor 嵌入中取最近邻，替换 z_t 得 z̄_t"]
    ASC --> TSC["TSC：置信度 τ_t ≥ ρ 则采用 z̄_t，否则采用 LSTM 上一步预测"]
    TSC --> LSTM["LSTM 记忆模型：结合上一动作 a_t-1 与历史，输出 ẑ_t（亦用于观测延迟时插值）"]
    LSTM --> POL["控制策略（RL 训练）→ 动作 a_t（油门、转向）"]
    LSTM --> BEVDEC
    BEVDEC --> PLAN["BEV 重建用于规划导航"]
    PLAN --> ACTOUT["动作指令 a_t"]
    POL --> ACTOUT
```

图依据：论文 §III（Fig.2 感知模型训练、Fig.3 系统总览、Fig.4 鲁棒性模块 TSC / ASC，Eq.4–8）、§IV-A/B（平台与数据）。

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 064/139 |
| 分组 | Observation / Sensor Fusion |
| 出处 | 见清单 / 原文 |
| 论文 | <https://arxiv.org/abs/2310.18847> |

## 核心机制（归纳）

### 策展导读要点

LongchaoDa AwesomeSim2Real 收录；分组 Observation / Sensor Fusion。 本页是清单索引，细节以原文为准。

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2310.18847) 为准。
- 横向对照请回到 [技术地图](../overview/lc-awesome-sim2real-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/lc-awesome-sim2real-technology-map.md) 的 **Observation / Sensor Fusion** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome 列表可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「World model based sim2real transfer for visual navigation」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

- 可确证的只有清单坐标：分组 **Observation / Sensor Fusion**；本页不自行推导新结论。
- 适用边界：本页不能替代原文 PDF；开源状态以项目页实际链接为准（清单可能滞后）。
- 要深读这篇，建议直接从原文入手，再回到下方关联的方法 / 任务页对照。

## 常见误区

1. 不要把 Awesome 条目的分组标签当成完整方法证明——它只是策展导读。
2. 若站内已有这篇的深读页，以那一页为准——本页只是清单入口，不含实验数据。

## 关联页面

- 列表实体：[AwesomeSim2Real](../entities/awesome-sim2real.md)
- 技术地图：[AwesomeSim2Real 技术地图](../overview/lc-awesome-sim2real-technology-map.md)
- 方法/任务：[sim2real.md](../concepts/sim2real.md)、[locomotion.md](../tasks/locomotion.md)

## 参考来源

- [`sources/papers/lc_awesome_sim2real_2310_18847_world-model-based-sim2real-transfer-for.md`](../../sources/papers/lc_awesome_sim2real_2310_18847_world-model-based-sim2real-transfer-for.md) — 本条目策展摘录
- [`sources/papers/lc_awesome_sim2real_catalog.md`](../../sources/papers/lc_awesome_sim2real_catalog.md) — 列表总表
- [`sources/repos/awesome-sim2real.md`](../../sources/repos/awesome-sim2real.md)
- 论文：<https://arxiv.org/abs/2310.18847>
- [arXiv PDF](https://arxiv.org/pdf/2310.18847)：已通读方法部分（实验与附录以略读为准）：§I 引言、§III 方法（感知模型、Memory 模型、TSC / ASC）、§IV-A/B 平台与数据

## 推荐继续阅读

- [AwesomeSim2Real 仓库](https://github.com/LongchaoDa/AwesomeSim2Real)
- [原文](https://arxiv.org/abs/2310.18847)
