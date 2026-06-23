---

type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-stub, unitree]
status: stub
updated: 2026-06-07
arxiv: "2602.05310"
related:
  - ../overview/paper-notebook-category-04-loco-manipulation-and-wbc.md
  - ../overview/humanoid-paper-notebooks-index.md
sources:
  - ../../sources/papers/humanoid_pnb_learning-soccer-skills-for-humanoid-robots.md
summary: "PAiD 把\"踢球\"这件事拆成 运动跟踪 → 感知融合 → 物理对齐迁移 三个递进阶段，先在干净的运动学世界里学会拟人的踢球姿态，再加入第一视角感知泛化到任意球位与滚动球，最后通过接触动力学对齐和感知噪声建模完成 Unitree G1 真机部署，实现 91.3% 踢球成功率。"
---

# Learning Soccer Skills for Humanoid Robots

**Learning Soccer Skills for Humanoid Robots: A Progressive Perception-Action Framework** 收录于 [Humanoid Robot Learning Paper Notebooks](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/index.html)（分类：04_Loco-Manipulation_and_WBC）。本页为 **索引级实体**，链向深读笔记与原始论文；详细机制待从笔记消化后补充。

## 一句话定义

PAiD 把"踢球"这件事拆成 运动跟踪 → 感知融合 → 物理对齐迁移 三个递进阶段，先在干净的运动学世界里学会拟人的踢球姿态，再加入第一视角感知泛化到任意球位与滚动球，最后通过接触动力学对齐和感知噪声建模完成 Unitree G1 真机部署，实现 91.3% 踢球成功率。

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
| 分类 | 04_Loco-Manipulation_and_WBC |
| 深读笔记 | <https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/04_Loco-Manipulation_and_WBC/Learning_Soccer_Skills_for_Humanoid_Robots____A_Progressive_Perception-Action_Fr/Learning_Soccer_Skills_for_Humanoid_Robots____A_Progressive_Perception-Action_Fr.html> |
| arXiv | <https://arxiv.org/abs/2602.05310> |

## 实验与评测

- 本页为 **策展索引级** 摘要；量化 benchmark、消融与实机指标以 **深读笔记与论文 PDF** 为准（链接见 [参考来源](#参考来源)）。

## 与其他页面的关系

- 分类父节点：[paper-notebook-category-04-loco-manipulation-and-wbc](../overview/paper-notebook-category-04-loco-manipulation-and-wbc.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)

## 参考来源

- [humanoid_pnb_learning-soccer-skills-for-humanoid-robots.md](../../sources/papers/humanoid_pnb_learning-soccer-skills-for-humanoid-robots.md)
- 深读笔记：<https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/04_Loco-Manipulation_and_WBC/Learning_Soccer_Skills_for_Humanoid_Robots____A_Progressive_Perception-Action_Fr/Learning_Soccer_Skills_for_Humanoid_Robots____A_Progressive_Perception-Action_Fr.html>
- 论文：<https://arxiv.org/abs/2602.05310>

## 推荐继续阅读

- [机器人论文阅读笔记：Learning Soccer Skills for Humanoid Robots](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/04_Loco-Manipulation_and_WBC/Learning_Soccer_Skills_for_Humanoid_Robots____A_Progressive_Perception-Action_Fr/Learning_Soccer_Skills_for_Humanoid_Robots____A_Progressive_Perception-Action_Fr.html)
