---
type: overview
tags:
  - control
  - force-control
  - impedance
  - admittance
  - contact-rich
  - manipulation
status: complete
updated: 2026-07-18
summary: "力控体系解决纯位置控制在接触时的冲击问题，涵盖阻抗、导纳、力位混合与直接力反馈。"
related:
  - ../comparisons/robot-control-eight-paradigms-taxonomy.md
  - ../concepts/impedance-control.md
  - ../methods/admittance-control.md
  - ../concepts/hybrid-force-position-control.md
  - ../methods/direct-force-feedback-control.md
  - ../overview/topic-contact-force-control.md
sources:
  - ../../sources/blogs/wechat_shenlan_robot_control_eight_paradigms.md
---


# 位置/力混合控制（体系⑤）

接触作业专用：在任务空间分解位置与力约束，实现打磨、装配、人机协作的柔顺交互。

## 一句话定义

接触作业专用：在任务空间分解位置与力约束，实现打磨、装配、人机协作的柔顺交互。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Wrench | Force/Torque Wrench | 六维力/力矩螺旋 |
| F/T | Force/Torque Sensor | 末端六维力传感 |
| Compliance | Active Compliance | 主动柔顺退让 |

## 为什么重要

插拔、恒力打磨、协作推送等任务必须在 **力维度** 上可控，否则毫米级误差即可产生破坏级接触力。

## 核心原理

阻抗调节末端等效刚度阻尼；导纳以力为输入修正位置；力位混合在约束/自由方向分别闭环位置与力；直接力反馈跟踪目标接触力。

## 代表性算法

| 算法 | 节点 |
|------|------|
| 阻抗 | [impedance-control.md](../concepts/impedance-control.md) |
| 导纳 | [admittance-control.md](../methods/admittance-control.md) |
| 力位混合 | [hybrid-force-position-control.md](../concepts/hybrid-force-position-control.md) |
| 直接力反馈 | [direct-force-feedback-control.md](../methods/direct-force-feedback-control.md) |

## 工程实践

配置六维力传感器与滤波；按任务选阻抗刚度；装配任务在约束方向力控、自由方向位控。

## 局限与风险

力传感噪声与延迟限制带宽；高刚度环境仍需准确环境模型或柔顺策略。

## 关联页面

- [Impedance Control](../concepts/impedance-control.md)
- [Topic: Contact Force Control](../overview/topic-contact-force-control.md)

## 参考来源

- [wechat_shenlan_robot_control_eight_paradigms.md](../../sources/blogs/wechat_shenlan_robot_control_eight_paradigms.md) — 深蓝具身智能《机器人控制算法八大体系详解：从 PID 到强化学习》（<https://mp.weixin.qq.com/s/Kp12BMBiC7YiIiDPi_P8-g>）

## 推荐继续阅读

- 深蓝具身智能原文：<https://mp.weixin.qq.com/s/Kp12BMBiC7YiIiDPi_P8-g>

