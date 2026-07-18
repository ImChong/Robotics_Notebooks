---
type: method
tags:
  - control
  - ilc
  - repetitive
  - trajectory-tracking
  - manufacturing
status: complete
updated: 2026-07-18
summary: "重复轨迹任务的批次学习优化，工业流水线常用。"
related:
  - ../overview/robot-control-paradigm-receding-horizon-ilc.md
  - ./model-predictive-control.md
  - ./computed-torque-control.md
sources:
  - ../../sources/blogs/wechat_shenlan_robot_control_eight_paradigms.md
---


# Iterative Learning Control（迭代学习控制，ILC）

ILC：重复执行同一轨迹时，将上批次全程误差映射为下一批次前馈修正，迭代提升跟踪精度。

## 一句话定义

> ILC：重复执行同一轨迹时，将上批次全程误差映射为下一批次前馈修正，迭代提升跟踪精度。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| ILC | Iterative Learning Control | 迭代学习控制 |
| Batch | Iteration Batch | 单次重复运行 |
| LTI | Linear Time Invariant | 常见 ILC 核设计假设 |

## 为什么重要

与 CTC 前馈叠加可显著降低 **重复加工** 误差而不改反馈律。

## 核心原理

$u_{k+1}(t) = u_k(t) + L e_k(t)$ 或频域 $U_{k+1}=U_k + \Gamma E_k$；$k$ 为批次索引。

## 工程实践

保证初始条件一致；选 Q-filter 保证单调收敛；与 SMC/CTC 并联。

## 主要技术路线

### 1. 批次误差前馈叠加

文内代表实现路径；详见 [关联概念/形式化](../concepts/optimal-control.md)。

### 2. 与相邻体系融合

常与 [八大机器人控制体系分类](../comparisons/robot-control-eight-paradigms-taxonomy.md) 中相邻类别 **级联或并联**（如 CTC+SMC、MPC+ILC、NN 补偿+PID）。

## 局限与风险

非重复轨迹无效；传感器噪声会累积到前馈。

## 关联页面

- [optimal-control](../concepts/optimal-control.md)
- [Receding Horizon & ILC Paradigm](../overview/robot-control-paradigm-receding-horizon-ilc.md)
- [Model Predictive Control](./model-predictive-control.md)

## 参考来源

- [wechat_shenlan_robot_control_eight_paradigms.md](../../sources/blogs/wechat_shenlan_robot_control_eight_paradigms.md) — 深蓝具身智能《机器人控制算法八大体系详解：从 PID 到强化学习》（<https://mp.weixin.qq.com/s/Kp12BMBiC7YiIiDPi_P8-g>）

## 推荐继续阅读

- 深蓝具身智能原文：<https://mp.weixin.qq.com/s/Kp12BMBiC7YiIiDPi_P8-g>

