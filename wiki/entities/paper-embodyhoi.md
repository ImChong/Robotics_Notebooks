---
type: entity
tags:
  - paper
  - humanoid
  - dexterous-manipulation
  - hand-object-interaction
  - motion-retargeting
  - diffusion-or-flow
status: complete
updated: 2026-10-22
project_id: embodyhoi
arxiv: "2610.27242"
project: https://embodyhoi.github.io/
related:
  - ../concepts/motion-retargeting.md
  - ../overview/hub-cross-embodiment.md
  - ../concepts/world-action-models.md
sources:
  - ../../sources/blogs/wechat_senlanke_weekly_humanoid_quadruped_2026-10-19_23.md
  - ../../sources/papers/embodyhoi_arxiv_2610_27242.md
summary: "EmbodyHOI（2026）：在冻结的 flow-matching 手-物交互生成器采样中嵌入可微 IK，使同一生成器适配 Unitree G1 与 H1-2，减少事后 retargeting 对接触的破坏。"
---

# EmbodyHOI（面向人形灵巧操作的具身感知 HOI 生成）

## 一句话定义

**EmbodyHOI 不先在人手空间生成再做 retargeting，而是在采样时就把目标人形机器人的 IK 梯度引入生成过程，让生成结果直接满足该机器人的可达性与接触。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| HOI | Hand-Object Interaction | 手与物体的交互轨迹，本文生成对象 |
| IK | Inverse Kinematics | 求解关节角使末端达到目标位姿；本文用可微版本 |
| flow matching | Flow Matching | 冻结的轨迹生成器所用的生成框架 |
| G1 / H1-2 | Unitree G1 / H1-2 | 本文评测的两款人形机器人 |

## 为什么重要

- 人手示范到机器人的适配，常在 retargeting 后破坏物体接触；本文给出把机器人约束放进生成循环的另一条路线。
- 同一冻结生成器可跨机器人使用，无需为每个本体重训（据摘要）。

## 方法与机制

- 冻结 flow-matching 轨迹生成器；采样时以特定人形的可微 IK 求解器提供梯度，引导采样。
- 梯度同时调整人形机器人相对物体的位置，以联合对齐接触。
- 作者比较了两种替代方案：事后手腕修正、计算量匹配的 best-of-N 选择。

## 实验与评测

- 数据集：ARCTIC 轨迹；机器人：G1、H1-2。
- 手腕误差 8.9 cm → 0.6 cm。
- 任务成功率：G1 68% → 79%，H1-2 62% → 76%。

## 与其他工作对比

- 与事后 retargeting：本文主张在生成时处理约束。
- 与事后手腕修正 / best-of-N：作者报告这两者 IK 能量更低，但训练出的策略更差（据摘要）。这一点说明「运动学误差更小」不等于「对策略学习更有利」。

## 结论

本文的核心判断是：跨本体适配放在生成阶段比放在后处理阶段更有效。

- 将可微 IK 放进冻结生成器的采样循环，可在不重训生成器的前提下适配新本体。
- 评价手-物交互数据时，不能只看 IK 能量或手腕误差；应同时看下游策略的训练结果。
- 当前仅有摘要级证据（ARCTIC 数据、两款机器人），尚未核实跨数据集的泛化。

## 工程实践

- 代码：抓取页面未列出代码仓库；项目页包含仿真结果。
- 复现前需要的输入：目标人形的可微 IK 模型、冻结的生成器权重（权重发布状态未核实）。

## 局限与风险

- 评测范围：摘要仅提及 ARCTIC 数据与 G1、H1-2 两款机器人。
- 代码与权重的公开状态未在抓取页面中确认。
- 推测：该方法依赖目标机器人 IK 模型的精度，模型偏差会直接进入生成梯度。

## 关联页面

- [Motion Retargeting](../concepts/motion-retargeting.md) — 本文对比并替代的事后适配路线
- [跨本体（Cross-Embodiment）](../overview/hub-cross-embodiment.md) — 同一生成器跨机器人复用的问题背景

## 参考来源

- [EmbodyHOI 论文来源归档](../../sources/papers/embodyhoi_arxiv_2610_27242.md)

## 推荐继续阅读

- [arXiv 2610.27242](https://arxiv.org/abs/2610.27242)
- [项目页](https://embodyhoi.github.io/)
