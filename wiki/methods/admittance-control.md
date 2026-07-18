---
type: method
tags:
  - control
  - force-control
  - admittance
  - compliance
  - collaborative-robot
status: complete
updated: 2026-07-18
summary: "力输入、运动输出的柔顺控制，适合协作臂与人机交互。"
related:
  - ../concepts/impedance-control.md
  - ../overview/robot-control-paradigm-hybrid-position-force.md
  - ./direct-force-feedback-control.md
sources:
  - ../../sources/blogs/wechat_shenlan_robot_control_eight_paradigms.md
---


# Admittance Control（导纳控制）

导纳控制：以测得的外力为输入，输出位置/速度修正，实现力→运动的柔顺响应，与阻抗控制对偶。

## 一句话定义

> 导纳控制：以测得的外力为输入，输出位置/速度修正，实现力→运动的柔顺响应，与阻抗控制对偶。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Admittance | Admittance Control | 导纳控制 |
| F/T | Force/Torque Sensor | 外力测量 |
| Impedance | Impedance Control | 位置→力对偶关系 |

## 为什么重要

大负载臂、外环力控时常用 **导纳外环 + 位置内环** 结构。

## 核心原理

$M_d \ddot{x}_c + B_d \dot{x}_c + K_d x_c = f_{ext}$；解出修正轨迹 $x_c$ 送位置控制器。

## 工程实践

低通滤波力信号；调 $M_d,B_d,K_d$ 模拟弹簧阻尼；内环高带宽位置跟踪。

## 主要技术路线

### 1. 力输入外环 + 位置内环

文内代表实现路径；详见 [关联概念/形式化](../concepts/impedance-control.md)。

### 2. 与相邻体系融合

常与 [八大机器人控制体系分类](../comparisons/robot-control-eight-paradigms-taxonomy.md) 中相邻类别 **级联或并联**（如 CTC+SMC、MPC+ILC、NN 补偿+PID）。

## 局限与风险

依赖力传感与内环带宽；纯位置内环不佳时导纳失效。

## 关联页面

- [Impedance Control](../concepts/impedance-control.md)
- [Hybrid Force-Position](../concepts/hybrid-force-position-control.md)

## 参考来源

- [wechat_shenlan_robot_control_eight_paradigms.md](../../sources/blogs/wechat_shenlan_robot_control_eight_paradigms.md) — 深蓝具身智能《机器人控制算法八大体系详解：从 PID 到强化学习》（<https://mp.weixin.qq.com/s/Kp12BMBiC7YiIiDPi_P8-g>）

## 推荐继续阅读

- 深蓝具身智能原文：<https://mp.weixin.qq.com/s/Kp12BMBiC7YiIiDPi_P8-g>

