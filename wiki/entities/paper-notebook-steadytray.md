---

type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-stub, unitree]
status: stub
updated: 2026-06-26
arxiv: "2603.10306"
related:
  - ../overview/paper-notebook-category-04-loco-manipulation-and-wbc.md
  - ../overview/humanoid-paper-notebooks-index.md
sources:
  - ../../sources/papers/humanoid_pnb_steadytray.md
summary: "SteadyTray 把\"端托盘 + 走路\"这件高耦合的活，显式拆成两层 RL：底层用一个稳健的人形行走策略当老师，上层挂一个残差模块专门抵消步态引起的末端抖动；通过四阶段课程（预训练 → 托盘微调 → 残差教师 → 学生蒸馏），在 Unitree G1 上做到 96.9% 速度跟踪成功率 / 74.5% 抗扰鲁棒性，并且零样本 sim-to-real 落地真机。"
---

# SteadyTray

**SteadyTray: Learning Object Balancing Tasks in Humanoid Tray Transport via Residual Reinforcement Learning** 收录于 [Robot Learning Paper Notebooks](https://imchong.github.io/Robot_Learning_Paper_Notebooks/index.html)（分类：04_Loco-Manipulation_and_WBC）。本页为 **索引级实体**，链向深读笔记与原始论文；详细机制待从笔记消化后补充。

## 一句话定义

SteadyTray 把"端托盘 + 走路"这件高耦合的活，显式拆成两层 RL：底层用一个稳健的人形行走策略当老师，上层挂一个残差模块专门抵消步态引起的末端抖动；通过四阶段课程（预训练 → 托盘微调 → 残差教师 → 学生蒸馏），在 Unitree G1 上做到 96.9% 速度跟踪成功率 / 74.5% 抗扰鲁棒性，并且零样本 sim-to-real 落地真机。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略 |
| WBC | Whole-Body Control | 协调全身关节满足多任务/约束的控制基础设施 |
| Sim2Real | Simulation to Real | 把仿真中学到的策略迁移落地真机的工程主线 |

## 为什么重要

- 列入 Paper Notebooks 策展清单，便于与全库 [机器人学习论文笔记总索引](../overview/humanoid-paper-notebooks-index.md) 及分类父节点交叉检索。
- 深读笔记提供比摘要更贴近实现的阅读路径，适合作为后续 ingest 深化起点。

## 核心信息

| 字段 | 内容 |
|------|------|
| 分类 | 04_Loco-Manipulation_and_WBC |
| 深读笔记 | <https://imchong.github.io/Robot_Learning_Paper_Notebooks/papers/04_Loco-Manipulation_and_WBC/SteadyTray__Learning_Object_Balancing_Tasks_in_Humanoid_Tray_Transport_via_Resid/SteadyTray__Learning_Object_Balancing_Tasks_in_Humanoid_Tray_Transport_via_Resid.html> |
| arXiv | <https://arxiv.org/abs/2603.10306> |

## 实验与评测

- 本页为 **策展索引级** 摘要；量化 benchmark、消融与实机指标以 **深读笔记与论文 PDF** 为准（链接见 [参考来源](#参考来源)）。

## 结论

**SteadyTray 的取舍是「不重训整套全身策略，而是在稳健行走策略之上挂一层残差」：把托盘平衡当作对既有步态的扰动补偿问题，而不是一个新的全身控制目标。**

- 真正起作用的是 **分层 + 残差**：底层行走策略负责移动鲁棒性，上层残差模块只负责抵消步态引起的末端抖动，四阶段课程（预训练 → 托盘微调 → 残差教师 → 学生蒸馏）把这条链路训得可蒸馏、可部署。
- 本页给出的关键量化是 **96.9% 速度跟踪成功率 / 74.5% 抗扰鲁棒性**，且在 Unitree G1 上 **零样本 sim-to-real**——说明残差层没有把策略过拟合到仿真。
- 适用边界：面向「端托盘 + 行走」这类 **末端稳定性与步态强耦合** 的任务；本页未涉及托盘之外的负载形态或更一般的 loco-manipulation。
- 本页为 **索引级实体**，消融与完整实机指标以深读笔记与论文 PDF 为准，不宜直接引用本页数字做横向对比。

## 与其他页面的关系

- 分类父节点：[paper-notebook-category-04-loco-manipulation-and-wbc](../overview/paper-notebook-category-04-loco-manipulation-and-wbc.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)

## 参考来源

- [humanoid_pnb_steadytray.md](../../sources/papers/humanoid_pnb_steadytray.md)
- 深读笔记：<https://imchong.github.io/Robot_Learning_Paper_Notebooks/papers/04_Loco-Manipulation_and_WBC/SteadyTray__Learning_Object_Balancing_Tasks_in_Humanoid_Tray_Transport_via_Resid/SteadyTray__Learning_Object_Balancing_Tasks_in_Humanoid_Tray_Transport_via_Resid.html>
- 论文：<https://arxiv.org/abs/2603.10306>

## 推荐继续阅读

- [机器人论文阅读笔记：SteadyTray](https://imchong.github.io/Robot_Learning_Paper_Notebooks/papers/04_Loco-Manipulation_and_WBC/SteadyTray__Learning_Object_Balancing_Tasks_in_Humanoid_Tray_Transport_via_Resid/SteadyTray__Learning_Object_Balancing_Tasks_in_Humanoid_Tray_Transport_via_Resid.html)
