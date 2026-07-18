---
type: method
tags:
  - control
  - adaptive-control
  - mrac
status: complete
updated: 2026-07-18
summary: "经典自适应控制：参考模型对标 + 参数自适应律。"
related:
  - ../overview/robot-control-paradigm-adaptive-control.md
  - ./adaptive-computed-torque-control.md
  - ../formalizations/lyapunov.md
sources:
  - ../../sources/blogs/wechat_shenlan_robot_control_eight_paradigms.md
---


# MRAC（模型参考自适应控制）

MRAC：指定理想参考模型动态，用自适应律在线调节控制器，使实际输出渐近匹配参考模型。

## 一句话定义

> MRAC：指定理想参考模型动态，用自适应律在线调节控制器，使实际输出渐近匹配参考模型。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| MRAC | Model Reference Adaptive Control | 模型参考自适应 |
| Lyapunov | Lyapunov Stability | 自适应律设计依据 |
| MIT | MIT Rule | 经典自适应律（历史） |

## 为什么重要

变负载抓取时无需重调固定增益，让闭环动态 **跟随理想模型**。

## 核心原理

参考模型 $\dot{x}_m = A_m x_m + B_m r$；实际系统 $\dot{x}=Ax+Bu$；自适应律 $\dot{\theta} = -\Gamma e P b$ 使 $e=x-x_m\to 0$。

## 工程实践

选参考模型反映期望超调/带宽；监控参数漂移；与鲁棒项并联防未建模动态。

## 主要技术路线

### 1. 参考模型 + 自适应律

文内代表实现路径；详见 [关联概念/形式化](../formalizations/lyapunov.md)。

### 2. 与相邻体系融合

常与 [八大机器人控制体系分类](../comparisons/robot-control-eight-paradigms-taxonomy.md) 中相邻类别 **级联或并联**（如 CTC+SMC、MPC+ILC、NN 补偿+PID）。

## 局限与风险

仅对满足匹配条件的扰动有效；错误参考模型导致性能劣化。

## 关联页面

- [Adaptive Control Paradigm](../overview/robot-control-paradigm-adaptive-control.md)
- [Lyapunov](../formalizations/lyapunov.md)

## 参考来源

- [wechat_shenlan_robot_control_eight_paradigms.md](../../sources/blogs/wechat_shenlan_robot_control_eight_paradigms.md) — 深蓝具身智能《机器人控制算法八大体系详解：从 PID 到强化学习》（<https://mp.weixin.qq.com/s/Kp12BMBiC7YiIiDPi_P8-g>）

## 推荐继续阅读

- 深蓝具身智能原文：<https://mp.weixin.qq.com/s/Kp12BMBiC7YiIiDPi_P8-g>

