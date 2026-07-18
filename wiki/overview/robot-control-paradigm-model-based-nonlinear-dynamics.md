---
type: overview
tags:
  - control
  - model-based
  - dynamics
  - computed-torque
  - manipulation
status: complete
updated: 2026-07-18
summary: "非线性动力学控制用模型前馈将复杂系统等效线性化，CTC 是机械臂经典方案。"
related:
  - ../comparisons/robot-control-eight-paradigms-taxonomy.md
  - ../methods/computed-torque-control.md
  - ../methods/inverse-dynamics-control.md
  - ../methods/feedback-linearization-control.md
sources:
  - ../../sources/blogs/wechat_shenlan_robot_control_eight_paradigms.md
---


# 基于模型的非线性动力学控制（体系②）

依赖精确动力学方程，用前馈抵消惯性/重力/耦合，服务多自由度机械臂与人形中层控制。

## 一句话定义

依赖精确动力学方程，用前馈抵消惯性/重力/耦合，服务多自由度机械臂与人形中层控制。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| CTC | Computed Torque Control | 计算力矩/逆动力学前馈+反馈 |
| IDC | Inverse Dynamics Control | 逆动力学前馈为主 |
| RNEA | Recursive Newton-Euler Algorithm | 高效逆动力学递推 |

## 为什么重要

中高端工业臂与协作臂主流中层算法；是 WBC/TSID 与 MPC 的共同建模基础。

## 核心原理

动力学方程 $\tau = M(q)\ddot{q} + C(q,\dot{q})\dot{q} + g(q)$；CTC 用模型计算前馈力矩并叠加 PD 消除残差；反馈线性化通过微分同胚消非线性。

## 代表性算法

| 算法 | 节点 |
|------|------|
| CTC | [computed-torque-control.md](../methods/computed-torque-control.md) |
| IDC | [inverse-dynamics-control.md](../methods/inverse-dynamics-control.md) |
| 反馈线性化 | [feedback-linearization-control.md](../methods/feedback-linearization-control.md) |

## 工程实践

用 [Pinocchio](../entities/pinocchio.md) 等库实时算逆动力学；标定质量/惯量参数；Sim 中验证模型失配敏感度。

## 局限与风险

**模型失配**（负载变化、摩擦未建模）会显著劣化；需配合鲁棒/自适应或数据补偿。

## 关联页面

- [Computed Torque Control](../methods/computed-torque-control.md)
- [Inverse Dynamics Control](../methods/inverse-dynamics-control.md)
- [Feedback Linearization](../methods/feedback-linearization-control.md)

## 参考来源

- [wechat_shenlan_robot_control_eight_paradigms.md](../../sources/blogs/wechat_shenlan_robot_control_eight_paradigms.md) — 深蓝具身智能《机器人控制算法八大体系详解：从 PID 到强化学习》（<https://mp.weixin.qq.com/s/Kp12BMBiC7YiIiDPi_P8-g>）

## 推荐继续阅读

- 深蓝具身智能原文：<https://mp.weixin.qq.com/s/Kp12BMBiC7YiIiDPi_P8-g>

