---
type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-stub]
status: stub
updated: 2026-06-07
venue: "SIGGRAPH Asia 2025"
code: https://github.com/gouruiyu/ControlOperators
related:
  - ../overview/paper-notebook-category-14-human-motion.md
  - ../overview/humanoid-paper-notebooks-index.md
sources:
  - ../../sources/papers/humanoid_pnb_control-operators-for-interactive-character-anim.md
summary: "把「控制输入 → 神经网络」这件原本需要 ML 专家手工设计的事，拆解成一组有语义、可组合的「控制算子（Control Operator）」：每个算子对设计师来说是一个直观概念（\"沿这条轨迹走\"\"朝这个目标看\"\"按摇杆方向/速度移动\"\"在某时刻到达某位置\"），对网络来说则对应一段固定的编码结构。把若干算子拼起来，非技术用户就能自己训练出带多技能、多控制模式的学习型角色控制器——本文在 Learned Motion Matching 变体 和一个新的流匹配（flow-matching）自回归模型上都做了演示，并通过工业界从业者的用户研究验证其易用性。"
---

# Control Operators for Interactive Character Animation

**Control Operators for Interactive Character Animation** 收录于 [Humanoid Robot Learning Paper Notebooks](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/index.html)（分类：14_Human_Motion）。本页为 **索引级实体**，链向深读笔记与原始论文；详细机制待从笔记消化后补充。

## 一句话定义

把「控制输入 → 神经网络」这件原本需要 ML 专家手工设计的事，拆解成一组有语义、可组合的「控制算子（Control Operator）」：每个算子对设计师来说是一个直观概念（"沿这条轨迹走""朝这个目标看""按摇杆方向/速度移动""在某时刻到达某位置"），对网络来说则对应一段固定的编码结构。把若干算子拼起来，非技术用户就能自己训练出带多技能、多控制模式的学习型角色控制器——本文在 Learned Motion Matching 变体 和一个新的流匹配（flow-matching）自回归模型上都做了演示，并通过工业界从业者的用户研究验证其易用性。

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
| 分类 | 14_Human_Motion |
| 深读笔记 | <https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/14_Human_Motion/Control_Operators_for_Interactive_Character_Animation/Control_Operators_for_Interactive_Character_Animation.html> |

## 实验与评测

- 本页为 **策展索引级** 摘要；量化 benchmark、消融与实机指标以 **深读笔记与论文 PDF** 为准（链接见 [参考来源](#参考来源)）。

## 与其他页面的关系

- 分类父节点：[paper-notebook-category-14-human-motion](../overview/paper-notebook-category-14-human-motion.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)

## 参考来源

- [humanoid_pnb_control-operators-for-interactive-character-anim.md](../../sources/papers/humanoid_pnb_control-operators-for-interactive-character-anim.md)
- 深读笔记：<https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/14_Human_Motion/Control_Operators_for_Interactive_Character_Animation/Control_Operators_for_Interactive_Character_Animation.html>

## 推荐继续阅读

- [机器人论文阅读笔记：Control Operators for Interactive Character Animation](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/14_Human_Motion/Control_Operators_for_Interactive_Character_Animation/Control_Operators_for_Interactive_Character_Animation.html)
