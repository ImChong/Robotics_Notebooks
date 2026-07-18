---
type: method
tags:
  - control
  - nonlinear-control
  - feedback-linearization
  - model-based
status: complete
updated: 2026-07-18
summary: "通用非线性控制方法；CTC 可视为机器人动力学上的反馈线性化实例。"
related:
  - ../overview/robot-control-paradigm-model-based-nonlinear-dynamics.md
  - ./computed-torque-control.md
  - ../formalizations/control-lyapunov-function.md
sources:
  - ../../sources/blogs/wechat_shenlan_robot_control_eight_paradigms.md
---


# Feedback Linearization Control（反馈线性化控制）

反馈线性化：通过状态反馈与坐标变换消去系统非线性，化为可控线性形式后复用 LQR/PID。

## 一句话定义

> 反馈线性化：通过状态反馈与坐标变换消去系统非线性，化为可控线性形式后复用 LQR/PID。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| FL | Feedback Linearization | 反馈线性化 |
| DI | Dynamic Inversion | 动态逆/微分同胚 |
| CTC | Computed Torque Control | 机器人领域典型实现 |

## 为什么重要

为「先消非线性再线性控制」提供统一数学框架。

## 核心原理

寻找微分同胚 $z=\phi(x)$ 与反馈 $u=\alpha(x)+\beta(x)v$ 使 $\dot{z}=Az+Bv$；奇异点处不可线性化。

## 工程实践

验证相对阶与可控性；奇异构型附近增益调度或避障。

## 主要技术路线

### 1. 微分同胚消非线性

文内代表实现路径；详见 [关联概念/形式化](../formalizations/control-lyapunov-function.md)。

### 2. 与相邻体系融合

常与 [八大机器人控制体系分类](../comparisons/robot-control-eight-paradigms-taxonomy.md) 中相邻类别 **级联或并联**（如 CTC+SMC、MPC+ILC、NN 补偿+PID）。

## 局限与风险

依赖精确模型；存在内动态不稳定风险。

## 关联页面

- [Computed Torque Control](./computed-torque-control.md)
- [Control Lyapunov Function](../formalizations/control-lyapunov-function.md)

## 参考来源

- [wechat_shenlan_robot_control_eight_paradigms.md](../../sources/blogs/wechat_shenlan_robot_control_eight_paradigms.md) — 深蓝具身智能《机器人控制算法八大体系详解：从 PID 到强化学习》（<https://mp.weixin.qq.com/s/Kp12BMBiC7YiIiDPi_P8-g>）

## 推荐继续阅读

- 深蓝具身智能原文：<https://mp.weixin.qq.com/s/Kp12BMBiC7YiIiDPi_P8-g>

