---
type: overview
tags:
  - control
  - adaptive-control
  - mrac
  - system-identification
  - parameter-estimation
status: complete
updated: 2026-07-18
summary: "自适应控制主动辨识并修正模型参数，适配变负载与工况漂移。"
related:
  - ../comparisons/robot-control-eight-paradigms-taxonomy.md
  - ../methods/mrac.md
  - ../methods/adaptive-computed-torque-control.md
  - ../methods/recursive-least-squares-control.md
sources:
  - ../../sources/blogs/wechat_shenlan_robot_control_eight_paradigms.md
---


# 自适应控制（体系④）

在线辨识时变参数并修正控制律，解决负载变化、磨损与摩擦漂移，与鲁棒「被动抵抗」形成互补。

## 一句话定义

在线辨识时变参数并修正控制律，解决负载变化、磨损与摩擦漂移，与鲁棒「被动抵抗」形成互补。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| MRAC | Model Reference Adaptive Control | 跟踪参考模型动态 |
| A-CTC | Adaptive Computed Torque Control | 在线更新动力学参数 |
| RLS | Recursive Least Squares | 递推最小二乘参数辨识 |

## 为什么重要

抓取不同重量工件、长期运行摩擦变化时，固定参数 CTC 精度衰减，自适应可 **无需重标定全模型**。

## 核心原理

MRAC 用自适应律调节控制器使输出逼近参考模型；A-CTC 在线更新 $M,C,g$ 等参数；RLS 递推最小化预测误差更新参数向量。

## 代表性算法

| 算法 | 节点 |
|------|------|
| MRAC | [mrac.md](../methods/mrac.md) |
| A-CTC | [adaptive-computed-torque-control.md](../methods/adaptive-computed-torque-control.md) |
| RLS | [recursive-least-squares-control.md](../methods/recursive-least-squares-control.md) |

## 工程实践

激励轨迹需 **持续激励** 保证可辨识；监控参数收敛与发散；与 RLS 辨识结果喂给前馈通道。

## 局限与风险

错误激励导致参数漂移；与鲁棒层并联时需防止自适应与鲁棒项争用同一误差通道。

## 关联页面

- [MRAC](../methods/mrac.md)
- [Adaptive CTC](../methods/adaptive-computed-torque-control.md)
- [System Identification](../concepts/system-identification.md)

## 参考来源

- [wechat_shenlan_robot_control_eight_paradigms.md](../../sources/blogs/wechat_shenlan_robot_control_eight_paradigms.md) — 深蓝具身智能《机器人控制算法八大体系详解：从 PID 到强化学习》（<https://mp.weixin.qq.com/s/Kp12BMBiC7YiIiDPi_P8-g>）

## 推荐继续阅读

- 深蓝具身智能原文：<https://mp.weixin.qq.com/s/Kp12BMBiC7YiIiDPi_P8-g>

