---
type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-stub]
status: stub
updated: 2026-06-07
arxiv: "2402.19469"
related:
  - ../overview/paper-notebook-category-03-high-impact-selection.md
  - ../overview/humanoid-paper-notebooks-index.md
sources:
  - ../../sources/papers/humanoid_pnb_humanoid-locomotion-as-next-token-prediction.md
summary: "把真实人形 locomotion 写成「下一词预测」：用 因果 Transformer 对 传感–动作 token 序列 做自回归拟合，模态对齐 地预测下一 token；对缺动作的轨迹用 可学习 mask token 统一格式，从而吃进 RL 策略轨迹、MPC 观测、动捕与 YouTube 人体视频。仅用约 27 小时量级行走数据 训练即可 零样本 在旧金山多路面部署，并能泛化到如 后退行走 等训练外指令。"
---

# Humanoid Locomotion as Next Token Prediction

**Humanoid Locomotion as Next Token Prediction** 收录于 [Humanoid Robot Learning Paper Notebooks](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/index.html)（分类：03_High_Impact_Selection）。本页为 **索引级实体**，链向深读笔记与原始论文；详细机制待从笔记消化后补充。

## 一句话定义

把真实人形 locomotion 写成「下一词预测」：用 因果 Transformer 对 传感–动作 token 序列 做自回归拟合，模态对齐 地预测下一 token；对缺动作的轨迹用 可学习 mask token 统一格式，从而吃进 RL 策略轨迹、MPC 观测、动捕与 YouTube 人体视频。仅用约 27 小时量级行走数据 训练即可 零样本 在旧金山多路面部署，并能泛化到如 后退行走 等训练外指令。

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
| 分类 | 03_High_Impact_Selection |
| 深读笔记 | <https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/03_High_Impact_Selection/Humanoid_Locomotion_as_Next_Token_Prediction/Humanoid_Locomotion_as_Next_Token_Prediction.html> |
| arXiv | <https://arxiv.org/abs/2402.19469> |

## 实验与评测

- 本页为 **策展索引级** 摘要；量化 benchmark、消融与实机指标以 **深读笔记与论文 PDF** 为准（链接见 [参考来源](#参考来源)）。

## 与其他页面的关系

- 分类父节点：[paper-notebook-category-03-high-impact-selection](../overview/paper-notebook-category-03-high-impact-selection.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)

## 参考来源

- [humanoid_pnb_humanoid-locomotion-as-next-token-prediction.md](../../sources/papers/humanoid_pnb_humanoid-locomotion-as-next-token-prediction.md)
- 深读笔记：<https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/03_High_Impact_Selection/Humanoid_Locomotion_as_Next_Token_Prediction/Humanoid_Locomotion_as_Next_Token_Prediction.html>
- 论文：<https://arxiv.org/abs/2402.19469>

## 推荐继续阅读

- [机器人论文阅读笔记：Humanoid Locomotion as Next Token Prediction](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/03_High_Impact_Selection/Humanoid_Locomotion_as_Next_Token_Prediction/Humanoid_Locomotion_as_Next_Token_Prediction.html)
