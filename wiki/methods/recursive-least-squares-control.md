---
type: method
tags:
  - control
  - adaptive-control
  - system-identification
  - rls
status: complete
updated: 2026-07-18
summary: "自适应控制的基础辨识工具，实时更新参数向量。"
related:
  - ../overview/robot-control-paradigm-adaptive-control.md
  - ../concepts/system-identification.md
  - ./adaptive-computed-torque-control.md
sources:
  - ../../sources/blogs/wechat_shenlan_robot_control_eight_paradigms.md
---


# Recursive Least Squares Control（RLS 递归最小二乘辨识）

RLS：递推最小化预测误差，在线更新动力学参数估计，为自适应/前馈控制提供参数流。

## 一句话定义

> RLS：递推最小化预测误差，在线更新动力学参数估计，为自适应/前馈控制提供参数流。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RLS | Recursive Least Squares | 递归最小二乘 |
| SI | System Identification | 系统辨识 |
| PE | Persistent Excitation | 持续激励条件 |

## 为什么重要

几乎所有在线自适应前馈都依赖某种 **RLS/梯度辨识** 内核。

## 核心原理

$\hat{\theta}_{k+1}=\hat{\theta}_k + K_k (y_k - \phi_k^T \hat{\theta}_k)$；$K_k$ 由协方差递推得。

## 工程实践

选回归向量 $\phi$ 含 $H(q),\dot{q}$ 等基函数；遗忘因子应对慢时变；辨识结果喂 A-CTC。

## 主要技术路线

### 1. 递推参数辨识

文内代表实现路径；详见 [关联概念/形式化](../concepts/system-identification.md)。

### 2. 与相邻体系融合

常与 [八大机器人控制体系分类](../comparisons/robot-control-eight-paradigms-taxonomy.md) 中相邻类别 **级联或并联**（如 CTC+SMC、MPC+ILC、NN 补偿+PID）。

## 局限与风险

噪声与未建模动态导致偏估；需激励设计。

## 关联页面

- [System Identification](../concepts/system-identification.md)
- [Adaptive CTC](./adaptive-computed-torque-control.md)

## 参考来源

- [wechat_shenlan_robot_control_eight_paradigms.md](../../sources/blogs/wechat_shenlan_robot_control_eight_paradigms.md) — 深蓝具身智能《机器人控制算法八大体系详解：从 PID 到强化学习》（<https://mp.weixin.qq.com/s/Kp12BMBiC7YiIiDPi_P8-g>）

## 推荐继续阅读

- 深蓝具身智能原文：<https://mp.weixin.qq.com/s/Kp12BMBiC7YiIiDPi_P8-g>

