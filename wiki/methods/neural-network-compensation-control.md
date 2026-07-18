---
type: method
tags:
  - control
  - machine-learning
  - neural-network
  - residual-learning
status: complete
updated: 2026-07-18
summary: "用 NN 拟合摩擦/柔性等建模残差，与传统控制器并联。"
related:
  - ../overview/robot-control-paradigm-ml-driven-control.md
  - ../concepts/neural-feedback-controller.md
  - ./computed-torque-control.md
sources:
  - ../../sources/blogs/wechat_shenlan_robot_control_eight_paradigms.md
---


# Neural Network Compensation Control（神经网络补偿控制）

NN 补偿：离线训练网络拟合动力学残差，运行时并联输出补偿力矩，修正解析模型未覆盖的非线性。

## 一句话定义

> NN 补偿：离线训练网络拟合动力学残差，运行时并联输出补偿力矩，修正解析模型未覆盖的非线性。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| NN | Neural Network | 神经网络 |
| FFN | Feedforward Network | 全连接补偿 |
| LSTM | Long Short-Term Memory | 时序残差 |

## 为什么重要

复杂摩擦与非标机构上，NN 补偿是 **最低门槛** 的数据驱动增强。

## 核心原理

学习 $\Delta\tau = f_\theta(q,\dot{q},\ddot{q})$；$\tau = \tau_{CTC} + f_\theta$；可用 LSTM/Transformer 捕时序。

## 工程实践

采集多工况轨迹监督训练；注意外推安全；定期再训练。

## 主要技术路线

### 1. 残差网络并联补偿

文内代表实现路径；详见 [关联概念/形式化](../concepts/neural-feedback-controller.md)。

### 2. 与相邻体系融合

常与 [八大机器人控制体系分类](../comparisons/robot-control-eight-paradigms-taxonomy.md) 中相邻类别 **级联或并联**（如 CTC+SMC、MPC+ILC、NN 补偿+PID）。

## 局限与风险

分布外工况失效；需与解析控制器并存保安全。

## 关联页面

- [ML-driven Control Paradigm](../overview/robot-control-paradigm-ml-driven-control.md)
- [Neural Feedback Controller](../concepts/neural-feedback-controller.md)

## 参考来源

- [wechat_shenlan_robot_control_eight_paradigms.md](../../sources/blogs/wechat_shenlan_robot_control_eight_paradigms.md) — 深蓝具身智能《机器人控制算法八大体系详解：从 PID 到强化学习》（<https://mp.weixin.qq.com/s/Kp12BMBiC7YiIiDPi_P8-g>）

## 推荐继续阅读

- 深蓝具身智能原文：<https://mp.weixin.qq.com/s/Kp12BMBiC7YiIiDPi_P8-g>

