---
type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-stub]
status: stub
updated: 2026-06-07
arxiv: "2511.16306"
related:
  - ../overview/paper-notebook-category-09-state-estimation.md
  - ../overview/humanoid-paper-notebooks-index.md
sources:
  - ../../sources/papers/humanoid_pnb_inekformer.md
summary: "InEKFormer 把经典 不变扩展卡尔曼滤波（InEKF） 的几何结构保留下来，但让 Transformer 从一段「状态 / 观测残差」的历史里隐式输出噪声相关的修正量，从而绕开「手调噪声协方差 Q/R」这件让所有滤波工程师头大的活；在 RH5 真机数据上跟 InEKF / KalmanNet 两条基线对照，验证了 Transformer 在人形高维状态估计里的可行性，同时也点出了「自回归训练不鲁棒就会爆」的现实问题。"
---

# InEKFormer

**InEKFormer: A Hybrid State Estimator for Humanoid Robots** 收录于 [Humanoid Robot Learning Paper Notebooks](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/index.html)（分类：09_State_Estimation）。本页为 **索引级实体**，链向深读笔记与原始论文；详细机制待从笔记消化后补充。

## 一句话定义

InEKFormer 把经典 不变扩展卡尔曼滤波（InEKF） 的几何结构保留下来，但让 Transformer 从一段「状态 / 观测残差」的历史里隐式输出噪声相关的修正量，从而绕开「手调噪声协方差 Q/R」这件让所有滤波工程师头大的活；在 RH5 真机数据上跟 InEKF / KalmanNet 两条基线对照，验证了 Transformer 在人形高维状态估计里的可行性，同时也点出了「自回归训练不鲁棒就会爆」的现实问题。

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
| 深读笔记 | <https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/09_State_Estimation/InEKFormer__A_Hybrid_State_Estimator_for_Humanoid_Robots/InEKFormer__A_Hybrid_State_Estimator_for_Humanoid_Robots.html> |
| arXiv | <https://arxiv.org/abs/2511.16306> |

## 实验与评测

- 本页为 **策展索引级** 摘要；量化 benchmark、消融与实机指标以 **深读笔记与论文 PDF** 为准（链接见 [参考来源](#参考来源)）。

## 与其他页面的关系

- 分类父节点：[paper-notebook-category-09-state-estimation](../overview/paper-notebook-category-09-state-estimation.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)

## 参考来源

- [humanoid_pnb_inekformer.md](../../sources/papers/humanoid_pnb_inekformer.md)
- 深读笔记：<https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/09_State_Estimation/InEKFormer__A_Hybrid_State_Estimator_for_Humanoid_Robots/InEKFormer__A_Hybrid_State_Estimator_for_Humanoid_Robots.html>
- 论文：<https://arxiv.org/abs/2511.16306>

## 推荐继续阅读

- [机器人论文阅读笔记：InEKFormer](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/09_State_Estimation/InEKFormer__A_Hybrid_State_Estimator_for_Humanoid_Robots/InEKFormer__A_Hybrid_State_Estimator_for_Humanoid_Robots.html)
