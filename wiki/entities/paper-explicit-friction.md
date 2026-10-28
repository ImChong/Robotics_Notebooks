---
type: entity
tags: [paper, robotics, research]
status: complete
updated: 2026-10-28
project_id: explicit-friction
arxiv: "2610.33492"
project: https://trthanhnguyen.github.io/explicit-friction-ngs/
summary: "显式摩擦布料仿真：神经布料预测器之后加入可调摩擦的接触算子，估计法向载荷、执行库仑修正并传播布料运动。"
related:
  - ../concepts/differentiable-simulation.md
  - ../concepts/robot-simulation-three-layers.md
sources:
  - ../../sources/papers/explicit-friction_arxiv_2610_33492.md
  - ../../sources/sites/explicit-friction-ngs.md
---

# Explicit Frictional Contact Modeling in Neural Garment Simulation

**一句话定义：** 神经布料预测器之后加入可调摩擦的接触算子，估计法向载荷、执行库仑修正并传播布料运动。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| GNN | Graph Neural Network | 图神经网络 |
| MCO | Modular Contact Operator | 模块化接触算子 |
| WACV | Winter Conference on Applications of Computer Vision | 论文标注会议 |

## 方法

神经布料预测器之后加入可调摩擦的接触算子，估计法向载荷、执行库仑修正并传播布料运动。

## 评测与结果

在 HOOD 和 ContourCraft 上，摩擦系数增大时黏附增加、滑移减少；论文报告相对参考求解器约八分之一的运行时间。

## 结论

这项研究提供了显式摩擦布料仿真的一个具体方案。

- 复现时先对齐论文任务和评测条件。
- 将方法收益与实际应用条件分开验证。
- 对代码及数据的可用性持续跟踪。

## 源码运行时序图

不适用：尚未核实可运行的官方实现。

## 局限与开放状态

结果针对论文所测的布料与人体接触；不能据此认定适用于任意机器人接触。项目页访问失败，代码与数据开放状态未核实。

## 关联页面

- [相关方法](../concepts/differentiable-simulation.md)
- [机器人仿真三层结构](../concepts/robot-simulation-three-layers.md)

## 参考来源

- [论文归档](../../sources/papers/explicit-friction_arxiv_2610_33492.md)
- [项目页核查](../../sources/sites/explicit-friction-ngs.md)

## 推荐继续阅读

- [论文原文](https://arxiv.org/abs/2610.33492)
