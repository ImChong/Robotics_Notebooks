---
type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-stub]
status: stub
updated: 2026-06-26
arxiv: "2507.10105"
related:
  - ../overview/paper-notebook-category-09-state-estimation.md
  - ../overview/humanoid-paper-notebooks-index.md
sources:
  - ../../sources/papers/humanoid_pnb_physics-informed-neural-networks-with-ukf-for-se.md
summary: "要让一台没有关节力矩传感器的人形机器人也能做力矩控制，就必须把\"力矩\"从其他传感器里估出来；论文的做法是：先用 PINN 把谐波减速器最难刻画的非线性摩擦学下来，再把 PINN 的摩擦估计当作 UKF 的一个测量量喂进去，最终在 ergoCub 真机平衡实验上让腿部 6 个关节的力矩跟踪 RMSE 落到 0.08–1.41 Nm，整体优于工业界默认基线 RNEA。"
---

# Physics-Informed Neural Networks with Unscented Kalman Filter for Sensorless Joint Torque Estimation in Humanoid Robots

**Physics-Informed Neural Networks with Unscented Kalman Filter for Sensorless Joint Torque Estimation in Humanoid Robots** 收录于 [Humanoid Robot Learning Paper Notebooks](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/index.html)（分类：09_State_Estimation）。本页为 **索引级实体**，链向深读笔记与原始论文；详细机制待从笔记消化后补充。

## 一句话定义

要让一台没有关节力矩传感器的人形机器人也能做力矩控制，就必须把"力矩"从其他传感器里估出来；论文的做法是：先用 PINN 把谐波减速器最难刻画的非线性摩擦学下来，再把 PINN 的摩擦估计当作 UKF 的一个测量量喂进去，最终在 ergoCub 真机平衡实验上让腿部 6 个关节的力矩跟踪 RMSE 落到 0.08–1.41 Nm，整体优于工业界默认基线 RNEA。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略 |
| WBC | Whole-Body Control | 协调全身关节满足多任务/约束的控制基础设施 |
| Sim2Real | Simulation to Real | 把仿真中学到的策略迁移落地真机的工程主线 |

## 为什么重要

- 列入 Paper Notebooks 策展清单，便于与全库 [人形论文笔记总索引](../overview/humanoid-paper-notebooks-index.md) 及分类父节点交叉检索。
- 深读笔记提供比摘要更贴近实现的阅读路径，适合作为后续 ingest 深化起点。

## 核心信息

| 字段 | 内容 |
|------|------|
| 分类 | 09_State_Estimation |
| 深读笔记 | <https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/09_State_Estimation/Physics-Informed_Neural_Networks_with_UKF_for_Sensorless_Joint_Torque_Estimation/Physics-Informed_Neural_Networks_with_UKF_for_Sensorless_Joint_Torque_Estimation.html> |
| arXiv | <https://arxiv.org/abs/2507.10105> |

## 实验与评测

- 本页为 **策展索引级** 摘要；量化 benchmark、消融与实机指标以 **深读笔记与论文 PDF** 为准（链接见 [参考来源](#参考来源)）。

## 与其他页面的关系

- 分类父节点：[paper-notebook-category-09-state-estimation](../overview/paper-notebook-category-09-state-estimation.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)

## 参考来源

- [humanoid_pnb_physics-informed-neural-networks-with-ukf-for-se.md](../../sources/papers/humanoid_pnb_physics-informed-neural-networks-with-ukf-for-se.md)
- 深读笔记：<https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/09_State_Estimation/Physics-Informed_Neural_Networks_with_UKF_for_Sensorless_Joint_Torque_Estimation/Physics-Informed_Neural_Networks_with_UKF_for_Sensorless_Joint_Torque_Estimation.html>
- 论文：<https://arxiv.org/abs/2507.10105>

## 推荐继续阅读

- [机器人论文阅读笔记：Physics-Informed Neural Networks with Unscented Kalman Filter for Sensorless Joint Torque Estimation in Humanoid Robots](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/09_State_Estimation/Physics-Informed_Neural_Networks_with_UKF_for_Sensorless_Joint_Torque_Estimation/Physics-Informed_Neural_Networks_with_UKF_for_Sensorless_Joint_Torque_Estimation.html)
