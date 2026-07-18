---
type: method
tags:
  - control
  - robust-control
  - h-infinity
  - optimal-control
status: complete
updated: 2026-07-18
summary: "最优鲁棒控制框架，约束最坏扰动下的误差放大倍数。"
related:
  - ../overview/robot-control-paradigm-robust-control.md
  - ./mu-synthesis-control.md
  - ../concepts/optimal-control.md
sources:
  - ../../sources/blogs/wechat_shenlan_robot_control_eight_paradigms.md
---


# H-infinity Control（H∞ 控制）

H∞ 控制：最小化从扰动/不确定性到跟踪误差的 **最坏情况** $H_\infty$ 范数，保证鲁棒性能界。

## 一句话定义

> H∞ 控制：最小化从扰动/不确定性到跟踪误差的 **最坏情况** $H_\infty$ 范数，保证鲁棒性能界。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| H∞ | H-infinity Control | 无穷范数鲁棒控制 |
| Riccati | Algebraic Riccati Equation | 状态反馈求解 |
| LMI | Linear Matrix Inequality | 多变量设计工具 |

## 为什么重要

精密手术、航空级机器人需要 **可证明** 的扰动抑制界。

## 核心原理

广义植物 $P$ 含扰动 $w$ 与性能输出 $z$；求 $K$ 使 $\|T_{wz}\|_\infty < \gamma$；Riccati/LMI 求解。

## 工程实践

线性化工作点设计 $H_\infty$ 控制器；μ 工具分析结构不确定性；与名义控制器切换。

## 主要技术路线

### 1. H∞ 范数约束设计

文内代表实现路径；详见 [关联概念/形式化](../concepts/optimal-control.md)。

### 2. 与相邻体系融合

常与 [八大机器人控制体系分类](../comparisons/robot-control-eight-paradigms-taxonomy.md) 中相邻类别 **级联或并联**（如 CTC+SMC、MPC+ILC、NN 补偿+PID）。

## 局限与风险

设计保守；非线性大范围运动需增益调度或多模型切换。

## 关联页面

- [Mu Synthesis Control](./mu-synthesis-control.md)
- [Optimal Control](../concepts/optimal-control.md)

## 参考来源

- [wechat_shenlan_robot_control_eight_paradigms.md](../../sources/blogs/wechat_shenlan_robot_control_eight_paradigms.md) — 深蓝具身智能《机器人控制算法八大体系详解：从 PID 到强化学习》（<https://mp.weixin.qq.com/s/Kp12BMBiC7YiIiDPi_P8-g>）

## 推荐继续阅读

- 深蓝具身智能原文：<https://mp.weixin.qq.com/s/Kp12BMBiC7YiIiDPi_P8-g>

