---
type: method
tags:
  - control
  - force-control
  - contact-rich
  - manipulation
status: complete
updated: 2026-07-18
summary: "最简力控：力误差直接驱动控制量，用于恒力按压/打磨。"
related:
  - ../overview/robot-control-paradigm-hybrid-position-force.md
  - ../concepts/hybrid-force-position-control.md
  - ./admittance-control.md
sources:
  - ../../sources/blogs/wechat_shenlan_robot_control_eight_paradigms.md
---


# Direct Force Feedback Control（直接力反馈控制）

直接力反馈：以目标接触力为设定值，力传感器闭环直接调节执行器输出，结构最简单。

## 一句话定义

> 直接力反馈：以目标接触力为设定值，力传感器闭环直接调节执行器输出，结构最简单。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Force | Force Control | 力闭环 |
| F/T | Force/Torque Sensor | 力测量 |
| PI | Proportional–Integral | 常用力环调节 |

## 为什么重要

精密按压、恒力打磨等 **单轴力跟踪** 任务足够且易调试。

## 核心原理

$e_f = f_d - f_{meas}$；$u = K_p e_f + K_i \int e_f$；输出为速度/力矩指令。

## 工程实践

Z 轴力控打磨；滤波与饱和防冲击；与位置环分时切换。

## 主要技术路线

### 1. 力传感器直接闭环

文内代表实现路径；详见 [关联概念/形式化](../concepts/force-control-basics.md)。

### 2. 与相邻体系融合

常与 [八大机器人控制体系分类](../comparisons/robot-control-eight-paradigms-taxonomy.md) 中相邻类别 **级联或并联**（如 CTC+SMC、MPC+ILC、NN 补偿+PID）。

## 局限与风险

多轴耦合差；无柔顺建模时刚性碰撞仍可能过大。

## 关联页面

- [force-control-basics](../concepts/force-control-basics.md)
- [Hybrid Force-Position Control](../concepts/hybrid-force-position-control.md)
- [Admittance Control](./admittance-control.md)

## 参考来源

- [wechat_shenlan_robot_control_eight_paradigms.md](../../sources/blogs/wechat_shenlan_robot_control_eight_paradigms.md) — 深蓝具身智能《机器人控制算法八大体系详解：从 PID 到强化学习》（<https://mp.weixin.qq.com/s/Kp12BMBiC7YiIiDPi_P8-g>）

## 推荐继续阅读

- 深蓝具身智能原文：<https://mp.weixin.qq.com/s/Kp12BMBiC7YiIiDPi_P8-g>

