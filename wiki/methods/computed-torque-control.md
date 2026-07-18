---
type: method
tags:
  - control
  - model-based
  - manipulation
  - computed-torque
  - dynamics
status: complete
updated: 2026-07-18
summary: "机械臂经典非线性控制：模型前馈线性化 + PD 反馈消除残差。"
related:
  - ../overview/robot-control-paradigm-model-based-nonlinear-dynamics.md
  - ./inverse-dynamics-control.md
  - ./feedback-linearization-control.md
  - ../entities/pinocchio.md
sources:
  - ../../sources/blogs/wechat_shenlan_robot_control_eight_paradigms.md
---


# Computed Torque Control（计算力矩控制，CTC）

CTC：用动力学模型计算前馈力矩抵消非线性耦合，再叠加反馈使闭环近似线性解耦系统。

## 一句话定义

> CTC：用动力学模型计算前馈力矩抵消非线性耦合，再叠加反馈使闭环近似线性解耦系统。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| CTC | Computed Torque Control | 计算力矩控制 |
| FF | Feedforward | 模型前馈 |
| PD | Proportional–Derivative | 常见反馈层 |

## 为什么重要

工业协作臂标准中层方案；是理解 WBC/逆动力学控制的入门枢纽。

## 核心原理

$\tau = M(q)(\ddot{q}_d + K_p e + K_d \dot{e}) + C(q,\dot{q})\dot{q} + g(q)$；前馈抵消 $C,g$，反馈处理模型误差。

## 工程实践

Pinocchio/RNEA 实时算 $\tau$；Sim 标定惯性参数；与 [Friction Compensation](../concepts/friction-compensation.md) 并联。

## 主要技术路线

### 1. 完整 CTC 前馈 + PD 反馈

文内代表实现路径；详见 [关联概念/形式化](../concepts/friction-compensation.md)。

### 2. 与相邻体系融合

常与 [八大机器人控制体系分类](../comparisons/robot-control-eight-paradigms-taxonomy.md) 中相邻类别 **级联或并联**（如 CTC+SMC、MPC+ILC、NN 补偿+PID）。

## 局限与风险

参数失配导致残余耦合；高速运动需考虑柔性未建模项。

## 关联页面

- [Model-based Nonlinear Dynamics](../overview/robot-control-paradigm-model-based-nonlinear-dynamics.md)
- [Inverse Dynamics Control](./inverse-dynamics-control.md)
- [Pinocchio](../entities/pinocchio.md)

## 参考来源

- [wechat_shenlan_robot_control_eight_paradigms.md](../../sources/blogs/wechat_shenlan_robot_control_eight_paradigms.md) — 深蓝具身智能《机器人控制算法八大体系详解：从 PID 到强化学习》（<https://mp.weixin.qq.com/s/Kp12BMBiC7YiIiDPi_P8-g>）

## 推荐继续阅读

- 深蓝具身智能原文：<https://mp.weixin.qq.com/s/Kp12BMBiC7YiIiDPi_P8-g>

