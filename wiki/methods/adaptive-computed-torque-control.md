---
type: method
tags:
  - control
  - adaptive-control
  - computed-torque
  - manipulation
status: complete
updated: 2026-07-18
summary: "CTC 与自适应辨识结合，缓解变负载导致的模型失配。"
related:
  - ../overview/robot-control-paradigm-adaptive-control.md
  - ./computed-torque-control.md
  - ./recursive-least-squares-control.md
sources:
  - ../../sources/blogs/wechat_shenlan_robot_control_eight_paradigms.md
---


# Adaptive Computed Torque Control（自适应计算力矩，A-CTC）

A-CTC：在 CTC 框架内在线辨识/更新惯性、质量等动力学参数，动态修正前馈补偿。

## 一句话定义

> A-CTC：在 CTC 框架内在线辨识/更新惯性、质量等动力学参数，动态修正前馈补偿。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| A-CTC | Adaptive Computed Torque Control | 自适应 CTC |
| CTC | Computed Torque Control | 基础框架 |
| RLS | Recursive Least Squares | 常用参数更新 |

## 为什么重要

同一机械臂抓取不同重量工件时的 **工程常用升级路径**。

## 核心原理

用 RLS/梯度法更新 $\hat{M},\hat{C},\hat{g}$；$\tau$ 用当前在线估计参数算 CTC 律；持续激励保证收敛。

## 工程实践

抓取序列中插入辨识激励；监控 $\hat{\theta}$ 有界性；与固定鲁棒层并联。

## 主要技术路线

### 1. 在线参数更新 + CTC 前馈

文内代表实现路径；详见 [关联概念/形式化](../concepts/system-identification.md)。

### 2. 与相邻体系融合

常与 [八大机器人控制体系分类](../comparisons/robot-control-eight-paradigms-taxonomy.md) 中相邻类别 **级联或并联**（如 CTC+SMC、MPC+ILC、NN 补偿+PID）。

## 局限与风险

激励不足时参数不可辨识；与强鲁棒切换可能冲突。

## 关联页面

- [system-identification](../concepts/system-identification.md)
- [Computed Torque Control](./computed-torque-control.md)
- [Recursive Least Squares Control](./recursive-least-squares-control.md)

## 参考来源

- [wechat_shenlan_robot_control_eight_paradigms.md](../../sources/blogs/wechat_shenlan_robot_control_eight_paradigms.md) — 深蓝具身智能《机器人控制算法八大体系详解：从 PID 到强化学习》（<https://mp.weixin.qq.com/s/Kp12BMBiC7YiIiDPi_P8-g>）

## 推荐继续阅读

- 深蓝具身智能原文：<https://mp.weixin.qq.com/s/Kp12BMBiC7YiIiDPi_P8-g>

