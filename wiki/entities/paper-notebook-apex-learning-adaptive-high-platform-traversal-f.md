---
type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-stub]
status: stub
updated: 2026-06-07
arxiv: "2602.11143"
related:
  - ../overview/paper-notebook-category-05-locomotion.md
  - ../overview/humanoid-paper-notebooks-index.md
sources:
  - ../../sources/papers/humanoid_pnb_apex-learning-adaptive-high-platform-traversal-f.md
summary: "把\"高平台穿越\"从起跳式改成攀爬式：用 6 个 *(climb-up · climb-down · walk · crawl · stand-up · lie-down)* 子技能 + 一个把\"局部 LiDAR 几何\"映射到全身动作的单一策略，配上棘轮式进度奖励（ratchet progress reward）只允许\"已最佳进度\"被记账、抑制反复试探的伪进步——结果是在 G1 上零样本翻越 0.8 m 高 ≈ 腿长 114% 的台子，全程不靠跳跃。"
---

# APEX

**APEX: Learning Adaptive High-Platform Traversal for Humanoid Robots** 收录于 [Humanoid Robot Learning Paper Notebooks](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/index.html)（分类：05_Locomotion）。本页为 **索引级实体**，链向深读笔记与原始论文；详细机制待从笔记消化后补充。

## 一句话定义

把"高平台穿越"从起跳式改成攀爬式：用 6 个 *(climb-up · climb-down · walk · crawl · stand-up · lie-down)* 子技能 + 一个把"局部 LiDAR 几何"映射到全身动作的单一策略，配上棘轮式进度奖励（ratchet progress reward）只允许"已最佳进度"被记账、抑制反复试探的伪进步——结果是在 G1 上零样本翻越 0.8 m 高 ≈ 腿长 114% 的台子，全程不靠跳跃。

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
| 分类 | 05_Locomotion |
| 深读笔记 | <https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/05_Locomotion/APEX_Learning_Adaptive_High-Platform_Traversal_for_Humanoid_Robots/APEX_Learning_Adaptive_High-Platform_Traversal_for_Humanoid_Robots.html> |
| arXiv | <https://arxiv.org/abs/2602.11143> |

## 实验与评测

- 本页为 **策展索引级** 摘要；量化 benchmark、消融与实机指标以 **深读笔记与论文 PDF** 为准（链接见 [参考来源](#参考来源)）。

## 与其他页面的关系

- 分类父节点：[paper-notebook-category-05-locomotion](../overview/paper-notebook-category-05-locomotion.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)

## 参考来源

- [humanoid_pnb_apex-learning-adaptive-high-platform-traversal-f.md](../../sources/papers/humanoid_pnb_apex-learning-adaptive-high-platform-traversal-f.md)
- 深读笔记：<https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/05_Locomotion/APEX_Learning_Adaptive_High-Platform_Traversal_for_Humanoid_Robots/APEX_Learning_Adaptive_High-Platform_Traversal_for_Humanoid_Robots.html>
- 论文：<https://arxiv.org/abs/2602.11143>

## 推荐继续阅读

- [机器人论文阅读笔记：APEX](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/05_Locomotion/APEX_Learning_Adaptive_High-Platform_Traversal_for_Humanoid_Robots/APEX_Learning_Adaptive_High-Platform_Traversal_for_Humanoid_Robots.html)
