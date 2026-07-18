---
type: method
tags:
  - control
  - classical-control
  - linear-control
  - pole-placement
status: complete
updated: 2026-07-18
summary: "通过配置闭环极点位置直接塑造伺服动态响应，用于高精度伺服调校。"
related:
  - ../overview/robot-control-paradigm-classical-linear-feedback.md
  - ../methods/pid-control.md
  - ../methods/lqr-ilqr.md
sources:
  - ../../sources/blogs/wechat_shenlan_robot_control_eight_paradigms.md
---


# Pole Placement Control（极点配置控制）

人为指定闭环极点位置，直接塑造伺服收敛速度与振荡幅度。

## 一句话定义

> 人为指定闭环极点位置，直接塑造伺服收敛速度与振荡幅度。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| CL | Closed Loop | 闭环系统 |
| Pole | System Pole | 决定响应快慢与阻尼 |
| SISO | Single Input Single Output | 常见标定对象 |

## 为什么重要

当需要 **指定超调/调节时间** 而非仅「尽量小误差」时，极点配置比盲目调 PID 更系统。

## 核心原理

对状态空间 $(A,B)$ 若 $(A,B)$ 可控，选期望极点 $\{p_i\}$，求反馈 $K$ 使 $\text{eig}(A-BK)=\{p_i\}$；左半平面极点保证稳定。

## 工程实践

单关节伺服用二阶近似选 $\zeta,\omega_n$ 映射极点；多变量用 Ackermann 公式或 place()；阶跃响应验证。

## 主要技术路线

### 1. 指定极点求状态反馈

文内代表实现路径；详见 [关联概念/形式化](../formalizations/lqr.md)。

### 2. 与相邻体系融合

常与 [八大机器人控制体系分类](../comparisons/robot-control-eight-paradigms-taxonomy.md) 中相邻类别 **级联或并联**（如 CTC+SMC、MPC+ILC、NN 补偿+PID）。

## 局限与风险

依赖准确线性化模型；强非线性区需分段或增益调度。

## 关联页面

- [lqr](../formalizations/lqr.md)
- [Classical Linear Feedback](../overview/robot-control-paradigm-classical-linear-feedback.md)
- [PID Control](./pid-control.md)
- [LQR / iLQR](./lqr-ilqr.md)

## 参考来源

- [wechat_shenlan_robot_control_eight_paradigms.md](../../sources/blogs/wechat_shenlan_robot_control_eight_paradigms.md) — 深蓝具身智能《机器人控制算法八大体系详解：从 PID 到强化学习》（<https://mp.weixin.qq.com/s/Kp12BMBiC7YiIiDPi_P8-g>）

## 推荐继续阅读

- 深蓝具身智能原文：<https://mp.weixin.qq.com/s/Kp12BMBiC7YiIiDPi_P8-g>

