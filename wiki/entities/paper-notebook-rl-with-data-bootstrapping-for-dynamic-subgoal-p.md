---
type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-stub]
status: stub
updated: 2026-07-10
arxiv: "2506.02206"
related:
  - ../overview/paper-notebook-category-08-navigation.md
  - ../overview/humanoid-paper-notebooks-index.md
sources:
  - ../../sources/papers/humanoid_pnb_rl-with-data-bootstrapping-for-dynamic-subgoal-p.md
summary: "安全、实时导航是人形应用的基础，但现有双足导航框架常难以平衡计算效率与稳定行走所需的精度。本文提出一个分层框架，持续生成动态子目标引导机器人穿越杂乱环境：高层 RL 规划器在机器人中心坐标系里选子目标，低层基于 MPC 的规划器产出鲁棒行走步态去到达这些子目标。为加速并稳定训练，引入一种数据自举（data bootstrapping）技术——借基于模型的导航方法生成多样、信息丰富的数据集。在 Agility Digit 人形上、多种随机障碍场景仿真验证：相比原模型法与其它学习法，成功率与适应性显著提升。"
---

# Reinforcement Learning with Data Bootstrapping for Dynamic Subgoal Pursuit in Humanoid Robot Navigation

**Reinforcement Learning with Data Bootstrapping for Dynamic Subgoal Pursuit in Humanoid Robot Navigation** 收录于 [Humanoid Robot Learning Paper Notebooks](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/index.html)（分类：08_Navigation），深读笔记已完成。本页为 **深读笔记索引实体**，正文要点编译自笔记；细节以笔记页与论文 PDF 为准。

## 一句话定义

安全、实时导航是人形应用的基础，但现有双足导航框架常难以平衡计算效率与稳定行走所需的精度。本文提出一个分层框架，持续生成动态子目标引导机器人穿越杂乱环境：高层 RL 规划器在机器人中心坐标系里选子目标，低层基于 MPC 的规划器产出鲁棒行走步态去到达这些子目标。为加速并稳定训练，引入一种数据自举（data bootstrapping）技术——借基于模型的导航方法生成多样、信息丰富的数据集。在 Agility Digit 人形上、多种随机障碍场景仿真验证：相比原模型法与其它学习法，成功率与适应性显著提升。

## 英文缩写速查

| 缩写 | 含义 |
|---|---|
| Dynamic Subgoal | 动态子目标，随情境持续更新的中间目标 |
| Hierarchical | 分层，高层规划 + 低层控制 |
| MPC | Model Predictive Control，模型预测控制 |
| Data Bootstrapping | 数据自举，用模型法生成训练数据 |
| Robot-Centric Frame | 机器人中心坐标系 |
| Digit | Agility Robotics 的双足人形 |

## 为什么重要

- **RL+MPC 分层**是导航兼顾"智能选路"与"稳定执行"的务实组合；
- **数据自举**用现成模型法解决 RL 冷启动，是低成本提效手段；
- **动态子目标**比一次性全局规划更适应杂乱动态环境；
- 与 NavDP、社交导航等共同丰富人形导航的方法谱。

## 解决什么问题

双足导航要**安全 + 实时**，但： - 难**平衡计算效率与稳定行走精度**； - 杂乱环境需**动态**调整路径； - RL 直接训练**慢且不稳**。

论文要：一个**高效、稳定、适应杂乱环境**的分层导航框架。

## 核心机制

1. **分层 RL+MPC 导航**：高层动态子目标 + 低层鲁棒步态；
2. **机器人中心子目标**：贴合双足局部决策；
3. **数据自举**：借模型法生成数据，加速稳定 RL 训练；
4. **Digit 验证**：杂乱随机障碍场景成功率/适应性优于模型法与其它学习法。

方法拆解（深读笔记小节）：分层：RL 子目标 + MPC 步态；数据自举（加速稳定训练）；评测；🧭 整体流程（mermaid）。

## 核心信息

| 字段 | 内容 |
|------|------|
| 分类 | 08_Navigation |
| 深读笔记 | <https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/08_Navigation/RL_with_Data_Bootstrapping_for_Dynamic_Subgoal_Pursuit_in_Humanoid_Navigation/RL_with_Data_Bootstrapping_for_Dynamic_Subgoal_Pursuit_in_Humanoid_Navigation.html> |
| arXiv | <https://arxiv.org/abs/2506.02206> |
| 作者 | Chengyang Peng、Zhihao Zhang、Shiting Gong、Sankalp Agrawal、Keith A. Redmill、Ayonga Hereid（OSU） |
| 发表 | 2025 年 6 月 |
| 笔记阅读日期 | 2026-06-21 |

## 实验与评测

- 本页为 **深读笔记编译** 的索引级摘要；量化 benchmark、消融与实机指标以 **深读笔记与论文 PDF** 为准（链接见 [参考来源](#参考来源)）。

## 与其他页面的关系

- 分类父节点：[paper-notebook-category-08-navigation](../overview/paper-notebook-category-08-navigation.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)

## 参考来源

- [humanoid_pnb_rl-with-data-bootstrapping-for-dynamic-subgoal-p.md](../../sources/papers/humanoid_pnb_rl-with-data-bootstrapping-for-dynamic-subgoal-p.md)
- 深读笔记：<https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/08_Navigation/RL_with_Data_Bootstrapping_for_Dynamic_Subgoal_Pursuit_in_Humanoid_Navigation/RL_with_Data_Bootstrapping_for_Dynamic_Subgoal_Pursuit_in_Humanoid_Navigation.html>
- 论文：<https://arxiv.org/abs/2506.02206>

## 推荐继续阅读

- [机器人论文阅读笔记：Reinforcement Learning with Data Bootstrapping for Dynamic Subgoal Pursuit in Humanoid Robot Navigation](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/08_Navigation/RL_with_Data_Bootstrapping_for_Dynamic_Subgoal_Pursuit_in_Humanoid_Navigation/RL_with_Data_Bootstrapping_for_Dynamic_Subgoal_Pursuit_in_Humanoid_Navigation.html)
