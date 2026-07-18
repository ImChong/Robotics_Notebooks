---
type: method
tags:
  - control
  - machine-learning
  - gaussian-process
  - model-based
status: complete
updated: 2026-07-18
summary: "小样本概率建模 + 不确定度量化，适合手术等低数据场景。"
related:
  - ../overview/robot-control-paradigm-ml-driven-control.md
  - ./model-based-rl.md
sources:
  - ../../sources/blogs/wechat_shenlan_robot_control_eight_paradigms.md
---


# Gaussian Process Control（高斯过程控制）

GP 控制：用高斯过程建立概率动力学模型，预测下一状态并给出不确定度，支持安全约束下的决策。

## 一句话定义

> GP 控制：用高斯过程建立概率动力学模型，预测下一状态并给出不确定度，支持安全约束下的决策。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| GP | Gaussian Process | 高斯过程 |
| PILCO | Probabilistic Inference for Learning Control | GP+策略优化代表 |
| UCB | Upper Confidence Bound | 探索-利用平衡 |

## 为什么重要

比确定性 NN 更擅长 **小数据 + 安全界** 的控制设计。

## 核心原理

$f(x)\sim \mathcal{GP}(m,k)$；预测均值 $\mu_*$ 与方差 $\sigma_*^2$；策略优化考虑置信界。

## 工程实践

PILCO 类算法在 Sim 学 GP 动力学再优化；实机用不确定度触发保守模式。

## 主要技术路线

### 1. GP 动力学 + 策略优化

文内代表实现路径；详见 [关联概念/形式化](../formalizations/mdp.md)。

### 2. 与相邻体系融合

常与 [八大机器人控制体系分类](../comparisons/robot-control-eight-paradigms-taxonomy.md) 中相邻类别 **级联或并联**（如 CTC+SMC、MPC+ILC、NN 补偿+PID）。

## 局限与风险

高维状态 GP 计算 $O(n^3)$；需合适核函数与诱导点近似。

## 关联页面

- [mdp](../formalizations/mdp.md)
- [Model-based RL](./model-based-rl.md)
- [ML-driven Control Paradigm](../overview/robot-control-paradigm-ml-driven-control.md)

## 参考来源

- [wechat_shenlan_robot_control_eight_paradigms.md](../../sources/blogs/wechat_shenlan_robot_control_eight_paradigms.md) — 深蓝具身智能《机器人控制算法八大体系详解：从 PID 到强化学习》（<https://mp.weixin.qq.com/s/Kp12BMBiC7YiIiDPi_P8-g>）

## 推荐继续阅读

- 深蓝具身智能原文：<https://mp.weixin.qq.com/s/Kp12BMBiC7YiIiDPi_P8-g>

