---
type: overview
tags:
  - control
  - mpc
  - ilc
  - optimization
  - constraints
  - repetitive
status: complete
updated: 2026-07-18
summary: "MPC 处理约束与多目标滚动优化；ILC 利用重复运动历史误差改进跟踪精度。"
related:
  - ../comparisons/robot-control-eight-paradigms-taxonomy.md
  - ../methods/model-predictive-control.md
  - ../methods/iterative-learning-control.md
sources:
  - ../../sources/blogs/wechat_shenlan_robot_control_eight_paradigms.md
---


# 滚动优化与迭代学习控制（体系⑥）

分别面向带硬约束的动态轨迹与高度重复工业轨迹，突破单步反馈的局部性。

## 一句话定义

分别面向带硬约束的动态轨迹与高度重复工业轨迹，突破单步反馈的局部性。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| MPC | Model Predictive Control | 滚动时域优化 |
| ILC | Iterative Learning Control | 重复轨迹批次学习 |
| OCP | Optimal Control Problem | 有限时域最优控制 |

## 为什么重要

人形步态、AGV 避障需要 **显式约束**；流水线搬运、定点打磨需要 **越重复越准**。

## 核心原理

MPC 每步求解有限时域 OCP 仅执行首控制量；ILC 将上批次全程误差映射为下一批次前馈补偿 $u_{k+1}=u_k + L e_k$。

## 代表性算法

| 算法 | 节点 |
|------|------|
| MPC | [model-predictive-control.md](../methods/model-predictive-control.md) |
| ILC | [iterative-learning-control.md](../methods/iterative-learning-control.md) |

## 工程实践

MPC 用 acados/forces/crocoddyl 等实时求解；ILC 需周期一致与初始条件重复；与 CTC 前馈叠加。

## 局限与风险

MPC 算力与建模成本；ILC 仅适用于 **重复轨迹**，对非重复任务无效。

## 关联页面

- [Model Predictive Control](../methods/model-predictive-control.md)
- [Iterative Learning Control](../methods/iterative-learning-control.md)

## 参考来源

- [wechat_shenlan_robot_control_eight_paradigms.md](../../sources/blogs/wechat_shenlan_robot_control_eight_paradigms.md) — 深蓝具身智能《机器人控制算法八大体系详解：从 PID 到强化学习》（<https://mp.weixin.qq.com/s/Kp12BMBiC7YiIiDPi_P8-g>）

## 推荐继续阅读

- 深蓝具身智能原文：<https://mp.weixin.qq.com/s/Kp12BMBiC7YiIiDPi_P8-g>

