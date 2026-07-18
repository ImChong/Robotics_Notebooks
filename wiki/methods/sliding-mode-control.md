---
type: method
tags:
  - control
  - robust-control
  - sliding-mode
  - disturbance-rejection
status: complete
updated: 2026-07-18
summary: "工业常用鲁棒控制：滑模面约束 + 切换控制抵抗扰动与模型误差。"
related:
  - ../overview/robot-control-paradigm-robust-control.md
  - ./computed-torque-control.md
  - ./h-infinity-control.md
sources:
  - ../../sources/blogs/wechat_shenlan_robot_control_eight_paradigms.md
---


# Sliding Mode Control（滑模控制，SMC）

SMC：设计滑模面 $s(x)=0$，用不连续/饱和切换律强制状态沿面滑向原点，对匹配扰动不敏感。

## 一句话定义

> SMC：设计滑模面 $s(x)=0$，用不连续/饱和切换律强制状态沿面滑向原点，对匹配扰动不敏感。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| SMC | Sliding Mode Control | 滑模控制 |
| Reaching | Reaching Law | 趋近律 |
| Chattering | Control Chattering | 高频抖振 |

## 为什么重要

接触冲击、负载突变场景下与 CTC 并联可显著提升鲁棒性。

## 核心原理

$s = \dot{e} + \lambda e$；$\tau = \tau_{eq} - K\,\text{sign}(s)$；高阶/终端滑模改良收敛与抖振。

## 工程实践

边界层 $\text{sat}(s/\Phi)$ 减抖振；与观测器估计扰动；MATLAB/Simulink 滑模实例验证。

## 主要技术路线

### 1. 滑模面 + 切换律

文内代表实现路径；详见 [关联概念/形式化](../formalizations/control-lyapunov-function.md)。

### 2. 与相邻体系融合

常与 [八大机器人控制体系分类](../comparisons/robot-control-eight-paradigms-taxonomy.md) 中相邻类别 **级联或并联**（如 CTC+SMC、MPC+ILC、NN 补偿+PID）。

## 局限与风险

原生 sign 切换磨损电机；需调边界层权衡鲁棒与精度。

## 关联页面

- [control-lyapunov-function](../formalizations/control-lyapunov-function.md)
- [Robust Control Paradigm](../overview/robot-control-paradigm-robust-control.md)
- [Computed Torque Control](./computed-torque-control.md)

## 参考来源

- [wechat_shenlan_robot_control_eight_paradigms.md](../../sources/blogs/wechat_shenlan_robot_control_eight_paradigms.md) — 深蓝具身智能《机器人控制算法八大体系详解：从 PID 到强化学习》（<https://mp.weixin.qq.com/s/Kp12BMBiC7YiIiDPi_P8-g>）

## 推荐继续阅读

- 深蓝具身智能原文：<https://mp.weixin.qq.com/s/Kp12BMBiC7YiIiDPi_P8-g>

