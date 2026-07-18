---
type: method
tags:
  - control
  - model-based
  - inverse-dynamics
  - manipulation
status: complete
updated: 2026-07-18
summary: "以前馈逆动力学为主、弱反馈为辅的轨迹跟踪控制。"
related:
  - ../overview/robot-control-paradigm-model-based-nonlinear-dynamics.md
  - ./computed-torque-control.md
sources:
  - ../../sources/blogs/wechat_shenlan_robot_control_eight_paradigms.md
---


# Inverse Dynamics Control（逆动力学控制，IDC）

IDC：由期望轨迹 $(q_d,\dot{q}_d,\ddot{q}_d)$ 经动力学逆解直接得前馈力矩，辅以少量反馈修正。

## 一句话定义

> IDC：由期望轨迹 $(q_d,\dot{q}_d,\ddot{q}_d)$ 经动力学逆解直接得前馈力矩，辅以少量反馈修正。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| IDC | Inverse Dynamics Control | 逆动力学控制 |
| RNEA | Recursive Newton-Euler Algorithm | O(n) 逆动力学 |
| CTC | Computed Torque Control | IDC + 闭环反馈强化 |

## 为什么重要

计算开销低于完整 CTC 闭环线性化时，IDC 是 **轻量前馈跟踪** 常用形态。

## 核心原理

$\tau_{ff} = M(q)\ddot{q}_d + C(q,\dot{q})\dot{q} + g(q)$；$\tau = \tau_{ff} + K_p e + K_d \dot{e}$。

## 工程实践

离线轨迹规划 + 在线 IDC 前馈；反馈增益宜小，依赖模型精度；摩擦补偿见 [Friction Compensation](../concepts/friction-compensation.md)。

## 主要技术路线

### 1. 逆动力学前馈为主

文内代表实现路径；详见 [Friction Compensation](../concepts/friction-compensation.md)。

### 2. 与相邻体系融合

常与 [八大机器人控制体系分类](../comparisons/robot-control-eight-paradigms-taxonomy.md) 中相邻类别 **级联或并联**（如 CTC+SMC、MPC+ILC、NN 补偿+PID）。

## 局限与风险

闭环修正能力弱于 CTC；突变扰动下跟踪误差更大。

## 关联页面

- [pinocchio](../entities/pinocchio.md)
- [Computed Torque Control](./computed-torque-control.md)
- [Feedback Linearization](./feedback-linearization-control.md)

## 参考来源

- [wechat_shenlan_robot_control_eight_paradigms.md](../../sources/blogs/wechat_shenlan_robot_control_eight_paradigms.md) — 深蓝具身智能《机器人控制算法八大体系详解：从 PID 到强化学习》（<https://mp.weixin.qq.com/s/Kp12BMBiC7YiIiDPi_P8-g>）

## 推荐继续阅读

- 深蓝具身智能原文：<https://mp.weixin.qq.com/s/Kp12BMBiC7YiIiDPi_P8-g>

