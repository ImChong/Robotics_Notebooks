---
type: entity
tags: [paper, curated-index, awesome-sim2real, longchao-sim2real]
status: complete
updated: 2026-10-06
arxiv: "1910.04700"
venue: "International Conference on Robotics and Automation (ICRA)"
summary: "LongchaoDa AwesomeSim2Real 收录；分组 Robotics Environments。 本页是清单索引，细节以原文为准。"
related:
  - ../entities/awesome-sim2real.md
  - ../overview/lc-awesome-sim2real-technology-map.md
  - ../concepts/sim2real.md
  - ../methods/reinforcement-learning.md
  - ../tasks/locomotion.md
  - ../tasks/manipulation.md
sources:
  - ../../sources/papers/lc_awesome_sim2real_1910_04700_assistive-gym-a-physics-simulation-frame.md
  - ../../sources/papers/lc_awesome_sim2real_catalog.md
  - ../../sources/repos/awesome-sim2real.md
---

# Assistive Gym

**Assistive Gym: A Physics Simulation Framework for Assistive Robotics** 收录于 [AwesomeSim2Real](https://github.com/LongchaoDa/AwesomeSim2Real) **第 091/139** 篇，分组 **Robotics Environments**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

LongchaoDa AwesomeSim2Real 收录；分组 Robotics Environments。 本页是清单索引，细节以原文为准。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Sim2Real | Simulation to Real | 仿真策略迁移到真机 |
| MDP | Markov Decision Process | 状态–动作–转移–奖励形式化 |
| DR | Domain Randomization | 域随机化 |
| FM | Foundation Model | 大模型/基础模型增强迁移 |

## 为什么重要

- LongchaoDa AwesomeSim2Real 收录；分组 Robotics Environments。 本页是清单索引，细节以原文为准。
- 想横向对照同一分组的其他工作，可以从 [AwesomeSim2Real 技术地图](../overview/lc-awesome-sim2real-technology-map.md) 逐条展开。
- 顺着列表实体 [AwesomeSim2Real](../entities/awesome-sim2real.md) 与站内 Sim2Real 方法/任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
  subgraph setup["环境创建（PyBullet）"]
    HUMAN["人体模型\n男 / 女，40 个可控关节\n随机施加震颤 / 关节无力 / 活动范围受限之一"]
    ROBOT["机器人：PR2 / Jaco / Baxter / Sawyer\n基座位姿采样 100 个，按 IK 与 JLWKI 选择"]
    TASK["六类任务环境\n抓痒 / 擦浴 / 喝水 / 喂食 / 穿衣 / 手臂操作"]
  end
  HUMAN --> TASK
  ROBOT --> TASK
  TASK --> OBS["观测：末端位姿、关节位置、末端受力、人体关键关节位置"]
  OBS --> POLICY["PPO 策略\n2 层 64 节点 tanh"]
  POLICY --> ACT["动作：关节位置增量 ΔP"]
  ACT --> STEP["仿真一步\n人体手臂姿态经关节限位网络检查\n无效则回退到上一步构型"]
  STEP --> REWARD["奖励 r(s) = rR(s) + rH(s)\n任务成功奖励 + 人体偏好惩罚"]
  REWARD --> POLICY
  STEP --> OBS
  POLICY -.->|"协同优化：人体策略与机器人策略共享 r(s) 同时训练"| HPOL["人体策略（手臂 / 头部）"]
  HPOL -.-> STEP
```

图依据：论文 §III-A 至 §III-E、§IV（策略学习与控制）、§V-B（协同优化）。

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 091/139 |
| 分组 | Robotics Environments |
| 出处 | International Conference on Robotics and Automation (ICRA) |
| 论文 | <https://arxiv.org/abs/1910.04700> |

## 核心机制（归纳）

### 策展导读要点

LongchaoDa AwesomeSim2Real 收录；分组 Robotics Environments。 本页是清单索引，细节以原文为准。

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/1910.04700) 为准。
- 横向对照请回到 [技术地图](../overview/lc-awesome-sim2real-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/lc-awesome-sim2real-technology-map.md) 的 **Robotics Environments** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome 列表可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「Assistive Gym」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

- 可确证的只有清单坐标：分组 **Robotics Environments**；本页不自行推导新结论。
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

- [`sources/papers/lc_awesome_sim2real_1910_04700_assistive-gym-a-physics-simulation-frame.md`](../../sources/papers/lc_awesome_sim2real_1910_04700_assistive-gym-a-physics-simulation-frame.md) — 本条目策展摘录
- [`sources/papers/lc_awesome_sim2real_catalog.md`](../../sources/papers/lc_awesome_sim2real_catalog.md) — 列表总表
- [`sources/repos/awesome-sim2real.md`](../../sources/repos/awesome-sim2real.md)
- 论文：<https://arxiv.org/abs/1910.04700>
- [arXiv PDF](https://arxiv.org/pdf/1910.04700)：已通读方法部分（实验与附录以略读为准）：§III Assistive Gym（人体与机器人、任务、关节限位、基座位姿、人体偏好）、§IV 策略学习与控制、§V 评测设置

## 推荐继续阅读

- [AwesomeSim2Real 仓库](https://github.com/LongchaoDa/AwesomeSim2Real)
- [原文](https://arxiv.org/abs/1910.04700)
