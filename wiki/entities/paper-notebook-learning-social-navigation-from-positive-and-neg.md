---
type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-stub]
status: stub
updated: 2026-07-10
arxiv: "2510.12215"
related:
  - ../overview/paper-notebook-category-08-navigation.md
  - ../overview/humanoid-paper-notebooks-index.md
sources:
  - ../../sources/papers/humanoid_pnb_learning-social-navigation-from-positive-and-neg.md
summary: "移动机器人在动态人群环境导航，需要策略既能适应多样人类行为、又遵守安全约束。本文从正示范与负示范（positive and negative demonstrations）学一个密度型奖励（density-based reward），并叠加基于规则的目标（避障、到达目标）。一个基于采样的前瞻控制器（sampling-based lookahead controller）产出既安全又自适应的监督动作，再蒸馏成一个紧凑学生策略，可实时运行并给出不确定性估计。在合成与电梯共乘（elevator co-boarding）仿真中，成功率与时间效率一致优于基线；真人参与的真实实验验证了可部署性。"
---

# Learning Social Navigation from Positive and Negative Demonstrations and Rule-Based Specifications

**Learning Social Navigation from Positive and Negative Demonstrations and Rule-Based Specifications** 收录于 [Humanoid Robot Learning Paper Notebooks](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/index.html)（分类：08_Navigation），深读笔记已完成。本页为 **深读笔记索引实体**，正文要点编译自笔记；细节以笔记页与论文 PDF 为准。

## 一句话定义

移动机器人在动态人群环境导航，需要策略既能适应多样人类行为、又遵守安全约束。本文从正示范与负示范（positive and negative demonstrations）学一个密度型奖励（density-based reward），并叠加基于规则的目标（避障、到达目标）。一个基于采样的前瞻控制器（sampling-based lookahead controller）产出既安全又自适应的监督动作，再蒸馏成一个紧凑学生策略，可实时运行并给出不确定性估计。在合成与电梯共乘（elevator co-boarding）仿真中，成功率与时间效率一致优于基线；真人参与的真实实验验证了可部署性。

## 英文缩写速查

| 缩写 | 含义 |
|---|---|
| Social Navigation | 社交导航，在人群中礼貌且安全地移动 |
| Positive/Negative Demo | 正/负示范，期望与不期望的行为样本 |
| Density-Based Reward | 密度型奖励，按示范分布密度塑形 |
| Rule-Based Spec | 规则规范，显式安全/目标约束 |
| Lookahead Controller | 前瞻控制器，基于采样向前看若干步 |
| Distillation | 蒸馏，把监督动作压进实时学生策略 |

## 为什么重要

- **负示范是被低估的监督信号**：显式告诉策略"别这样"，对安全攸关的社交导航尤为有用；
- **学习 + 规则**的混合是安全导航的务实范式，呼应 SafeFlow 的"生成 + 门控"；
- **前瞻 + 蒸馏**兼顾质量与实时，适合算力受限的机器人；
- 对人形而言，社交导航是进入人类空间（电梯、走廊）的关键能力。

## 解决什么问题

人群环境导航的核心张力： - **适应性**：要顺应多样、动态的人类行为； - **合规性**：要遵守**安全约束**（不撞人、保持礼貌距离）。

纯学习易违规，纯规则不够灵活。论文要：把**示范学习**与**规则约束**结合，得到既安全又自适应、且能实时跑的社交导航策略。

## 核心机制

1. **正负示范的密度型奖励**：同时利用好/坏样本塑形社交行为；
2. **示范 + 规则融合**：叠加避障/到达的规则目标，兼顾适应与安全；
3. **前瞻控制器 + 蒸馏**：产出安全自适应监督动作并压成实时学生策略（带不确定性）；
4. **仿真 + 真人验证**：电梯共乘等场景成功率/时效双升。

方法拆解（深读笔记小节）：正负示范 → 密度型奖励；叠加规则目标；采样前瞻控制器 → 学生蒸馏；评测；🧭 整体流程（mermaid）。

## 核心信息

| 字段 | 内容 |
|------|------|
| 分类 | 08_Navigation |
| 深读笔记 | <https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/08_Navigation/Learning_Social_Navigation_from_Positive_and_Negative_Demonstrations_and_Rule-Based_Specifications/Learning_Social_Navigation_from_Positive_and_Negative_Demonstrations_and_Rule-Based_Specifications.html> |
| arXiv | <https://arxiv.org/abs/2510.12215> |
| 作者 | Chanwoo Kim 等（共 12 位作者） |
| 发表 | 2025 年 10 月 |
| 笔记阅读日期 | 2026-06-21 |

## 实验与评测

- 本页为 **深读笔记编译** 的索引级摘要；量化 benchmark、消融与实机指标以 **深读笔记与论文 PDF** 为准（链接见 [参考来源](#参考来源)）。

## 与其他页面的关系

- 分类父节点：[paper-notebook-category-08-navigation](../overview/paper-notebook-category-08-navigation.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)

## 参考来源

- [humanoid_pnb_learning-social-navigation-from-positive-and-neg.md](../../sources/papers/humanoid_pnb_learning-social-navigation-from-positive-and-neg.md)
- 深读笔记：<https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/08_Navigation/Learning_Social_Navigation_from_Positive_and_Negative_Demonstrations_and_Rule-Based_Specifications/Learning_Social_Navigation_from_Positive_and_Negative_Demonstrations_and_Rule-Based_Specifications.html>
- 论文：<https://arxiv.org/abs/2510.12215>

## 推荐继续阅读

- [机器人论文阅读笔记：Learning Social Navigation from Positive and Negative Demonstrations and Rule-Based Specifications](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/08_Navigation/Learning_Social_Navigation_from_Positive_and_Negative_Demonstrations_and_Rule-Based_Specifications/Learning_Social_Navigation_from_Positive_and_Negative_Demonstrations_and_Rule-Based_Specifications.html)
