---
type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-stub]
status: stub
updated: 2026-06-26
arxiv: "2207.06780"
related:
  - ../overview/paper-notebook-category-09-state-estimation.md
  - ../overview/humanoid-paper-notebooks-index.md
sources:
  - ../../sources/papers/humanoid_pnb_an-empirical-evaluation-of-four-off-the-shelf-pr.md
summary: "不是又一篇新算法，而是一篇「实测对比」基准论文：作者用同一只「手提四传感器架」、同一组室内外轨迹，把四款最常被人形 / 移动机器人引用的商用闭源 VIO拉到同一条尺子上量——结论是 Apple ARKit 综合最稳最准（相对位姿误差 ≈ 0.02 m/s 漂移），但只能跑 iOS、对 ROS / Linux 不友好；T265 和 ZED 2 虽然 ROS 友好，但分别栽在「单目尺度漂移」和「旋转估计破坏正交性」上，给后续工程选型提供了一个可重复的硬证据。"
---

# An Empirical Evaluation of Four Off-the-Shelf Proprietary Visual-Inertial Odometry Systems

**An Empirical Evaluation of Four Off-the-Shelf Proprietary Visual-Inertial Odometry Systems** 收录于 [Robot Learning Paper Notebooks](https://imchong.github.io/Robot_Learning_Paper_Notebooks/index.html)（分类：09_State_Estimation）。本页为 **索引级实体**，链向深读笔记与原始论文；详细机制待从笔记消化后补充。

## 一句话定义

不是又一篇新算法，而是一篇「实测对比」基准论文：作者用同一只「手提四传感器架」、同一组室内外轨迹，把四款最常被人形 / 移动机器人引用的商用闭源 VIO拉到同一条尺子上量——结论是 Apple ARKit 综合最稳最准（相对位姿误差 ≈ 0.02 m/s 漂移），但只能跑 iOS、对 ROS / Linux 不友好；T265 和 ZED 2 虽然 ROS 友好，但分别栽在「单目尺度漂移」和「旋转估计破坏正交性」上，给后续工程选型提供了一个可重复的硬证据。

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
| 分类 | 09_State_Estimation |
| 深读笔记 | <https://imchong.github.io/Robot_Learning_Paper_Notebooks/papers/09_State_Estimation/An_Empirical_Evaluation_of_Four_Off-the-Shelf_Proprietary_VIO_Systems/An_Empirical_Evaluation_of_Four_Off-the-Shelf_Proprietary_VIO_Systems.html> |
| arXiv | <https://arxiv.org/abs/2207.06780> |

## 实验与评测

- 本页为 **策展索引级** 摘要；量化 benchmark、消融与实机指标以 **深读笔记与论文 PDF** 为准（链接见 [参考来源](#参考来源)）。

## 结论

**这篇的价值不在算法而在选型证据：把四款闭源商用 VIO 放到同一套硬件、同一组轨迹上量，得出的结论是「最准的那个最难用」。**

- 真正起作用的是 **控制变量的实验设计**：同一只手提四传感器架 + 同一组室内外轨迹，才让几个黑盒产品之间的比较可复现，这也是它区别于各家自报指标的地方。
- 精度侧 Apple ARKit 综合最稳最准（相对位姿误差约 0.02 m/s 漂移），但 **只能跑 iOS、对 ROS / Linux 不友好**——工程取舍就卡在这一条上。
- 另两个失败模式很具体：T265 栽在 **单目尺度漂移**，ZED 2 栽在 **旋转估计破坏正交性**；这类问题属于系统性缺陷，不是调参能绕过的。
- 适用边界：结论绑定在被测的这四款商用闭源产品上，是选型硬证据而非关于 VIO 方法的普适结论。
- 本页仅为策展索引级摘要，完整量化结果以 [参考来源](#参考来源) 中的深读笔记与论文 PDF 为准。

## 与其他页面的关系

- 分类父节点：[paper-notebook-category-09-state-estimation](../overview/paper-notebook-category-09-state-estimation.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)

## 参考来源

- [humanoid_pnb_an-empirical-evaluation-of-four-off-the-shelf-pr.md](../../sources/papers/humanoid_pnb_an-empirical-evaluation-of-four-off-the-shelf-pr.md)
- 深读笔记：<https://imchong.github.io/Robot_Learning_Paper_Notebooks/papers/09_State_Estimation/An_Empirical_Evaluation_of_Four_Off-the-Shelf_Proprietary_VIO_Systems/An_Empirical_Evaluation_of_Four_Off-the-Shelf_Proprietary_VIO_Systems.html>
- 论文：<https://arxiv.org/abs/2207.06780>

## 推荐继续阅读

- [机器人论文阅读笔记：An Empirical Evaluation of Four Off-the-Shelf Proprietary Visual-Inertial Odometry Systems](https://imchong.github.io/Robot_Learning_Paper_Notebooks/papers/09_State_Estimation/An_Empirical_Evaluation_of_Four_Off-the-Shelf_Proprietary_VIO_Systems/An_Empirical_Evaluation_of_Four_Off-the-Shelf_Proprietary_VIO_Systems.html)
