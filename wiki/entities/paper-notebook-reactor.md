---
type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-stub]
status: stub
updated: 2026-06-11
arxiv: "2605.06593v1"
related:
  - ../overview/paper-notebook-category-02-motion-retargeting.md
  - ../overview/humanoid-paper-notebooks-index.md
sources:
  - ../../sources/papers/humanoid_pnb_reactor.md
summary: "针对传统几何重定向需要 手工接触模板 / 大量调参、且仍产生 脚滑、自碰、动力学不可行 等问题，ReActor 提出 物理感知、RL 内嵌的双层框架：用户只给 稀疏语义刚体对应 与名义姿态对齐，系统自动搜索一组 有界偏移参数 把源运动 $\mathbf{m}_t$ 映到参数化参考 $\mathbf{g}_t(\mathbf{p})$；下层策略在仿真里跟踪 $\mathbf{g}_t$，上层最小化 $\mathbf{g}$ 与仿真 rollout 状态 $\mathbf{s}_t$ 的误差。论文在 两台人形 + 四足 上展示跨大差异本体的重定向，并分析近似梯度与泛化行为。"
---

# ReActor

**ReActor: Reinforcement Learning for Physics-Aware Motion Retargeting** 收录于 [Humanoid Robot Learning Paper Notebooks](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/index.html)（分类：02_Motion_Retargeting）。本页为 **索引级实体**，链向深读笔记与原始论文；详细机制待从笔记消化后补充。

## 一句话定义

针对传统几何重定向需要 手工接触模板 / 大量调参、且仍产生 脚滑、自碰、动力学不可行 等问题，ReActor 提出 物理感知、RL 内嵌的双层框架：用户只给 稀疏语义刚体对应 与名义姿态对齐，系统自动搜索一组 有界偏移参数 把源运动 $\mathbf{m}_t$ 映到参数化参考 $\mathbf{g}_t(\mathbf{p})$；下层策略在仿真里跟踪 $\mathbf{g}_t$，上层最小化 $\mathbf{g}$ 与仿真 rollout 状态 $\mathbf{s}_t$ 的误差。论文在 两台人形 + 四足 上展示跨大差异本体的重定向，并分析近似梯度与泛化行为。

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
| 分类 | 02_Motion_Retargeting |
| 深读笔记 | <https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/02_Motion_Retargeting/ReActor__Reinforcement_Learning_for_Physics-Aware_Motion_Retargeting/ReActor__Reinforcement_Learning_for_Physics-Aware_Motion_Retargeting.html> |
| arXiv | <https://arxiv.org/abs/2605.06593v1> |

## 实验与评测

- 本页为 **策展索引级** 摘要；量化 benchmark、消融与实机指标以 **深读笔记与论文 PDF** 为准（链接见 [参考来源](#参考来源)）。

## 与其他页面的关系

- 分类父节点：[paper-notebook-category-02-motion-retargeting](../overview/paper-notebook-category-02-motion-retargeting.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)

## 参考来源

- [humanoid_pnb_reactor.md](../../sources/papers/humanoid_pnb_reactor.md)
- 深读笔记：<https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/02_Motion_Retargeting/ReActor__Reinforcement_Learning_for_Physics-Aware_Motion_Retargeting/ReActor__Reinforcement_Learning_for_Physics-Aware_Motion_Retargeting.html>
- 论文：<https://arxiv.org/abs/2605.06593v1>

## 推荐继续阅读

- [机器人论文阅读笔记：ReActor](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/02_Motion_Retargeting/ReActor__Reinforcement_Learning_for_Physics-Aware_Motion_Retargeting/ReActor__Reinforcement_Learning_for_Physics-Aware_Motion_Retargeting.html)
