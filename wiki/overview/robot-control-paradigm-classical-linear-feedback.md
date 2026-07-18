---
type: overview
tags:
  - control
  - classical-control
  - pid
  - lqr
  - linear-control
status: complete
updated: 2026-07-18
summary: "经典线性反馈是机器人伺服底层：PID/LQR/极点配置面向弱耦合线性系统，无需完整动力学模型。"
related:
  - ../comparisons/robot-control-eight-paradigms-taxonomy.md
  - ../methods/pid-control.md
  - ../methods/lqr-ilqr.md
  - ../methods/pole-placement-control.md
sources:
  - ../../sources/blogs/wechat_shenlan_robot_control_eight_paradigms.md
---


# 经典线性反馈控制（体系①）

针对线性、弱扰动系统的底层伺服闭环，是电机/舵机与单关节跟踪的标配。

## 一句话定义

针对线性、弱扰动系统的底层伺服闭环，是电机/舵机与单关节跟踪的标配。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| PID | Proportional–Integral–Derivative | 比例-积分-微分反馈 |
| LQR | Linear Quadratic Regulator | 状态空间最优线性调节 |
| SISO | Single Input Single Output | 单输入单输出系统 |

## 为什么重要

所有多关节、RL、MPC 栈最终都需 **稳定底层执行**；PID 电流/速度环是力矩输出的最后一道闸门。

## 核心原理

闭环反馈采集传感器输出与目标对比修正；LQR 在状态空间同时权衡跟踪误差与控制能耗；极点配置直接指定收敛动态。

## 代表性算法

| 算法 | 节点 |
|------|------|
| PID | [pid-control.md](../methods/pid-control.md) |
| LQR | [lqr-ilqr.md](../methods/lqr-ilqr.md) |
| 极点配置 | [pole-placement-control.md](../methods/pole-placement-control.md) |

## 工程实践

单关节标定从 P→PI→PID 逐级加项；多变量系统用 LQR 设计状态反馈增益；伺服整定用阶跃响应看超调与稳态误差。

## 局限与风险

仅适用于 **弱非线性、弱耦合**；多关节强耦合需升级到非线性动力学控制。

## 关联页面

- [PID Control](../methods/pid-control.md)
- [LQR / iLQR](../methods/lqr-ilqr.md)
- [Pole Placement Control](../methods/pole-placement-control.md)

## 参考来源

- [wechat_shenlan_robot_control_eight_paradigms.md](../../sources/blogs/wechat_shenlan_robot_control_eight_paradigms.md) — 深蓝具身智能《机器人控制算法八大体系详解：从 PID 到强化学习》（<https://mp.weixin.qq.com/s/Kp12BMBiC7YiIiDPi_P8-g>）

## 推荐继续阅读

- 深蓝具身智能原文：<https://mp.weixin.qq.com/s/Kp12BMBiC7YiIiDPi_P8-g>

