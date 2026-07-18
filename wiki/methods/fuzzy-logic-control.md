---
type: method
tags:
  - control
  - fuzzy-logic
  - rule-based
  - nonlinear-control
status: complete
updated: 2026-07-18
summary: "规则驱动的非线性控制，适合难建模非标设备。"
related:
  - ../overview/robot-control-paradigm-ml-driven-control.md
  - ./pid-control.md
sources:
  - ../../sources/blogs/wechat_shenlan_robot_control_eight_paradigms.md
---


# Fuzzy Logic Control（模糊逻辑控制）

模糊逻辑控制：将操作经验编码为 If-Then 模糊规则，经模糊推理与去模糊得到控制量，无需精确动力学方程。

## 一句话定义

> 模糊逻辑控制：将操作经验编码为 If-Then 模糊规则，经模糊推理与去模糊得到控制量，无需精确动力学方程。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| FLC | Fuzzy Logic Control | 模糊逻辑控制 |
| MF | Membership Function | 隶属度函数 |
| Defuzz | Defuzzification | 模糊输出转 crisp |

## 为什么重要

老师傅经验难以公式化时，模糊规则是 **可解释** 的折中。

## 核心原理

模糊化 → 规则推理 → 聚合 → 去模糊（重心法等）；可与 PID 并联调参。

## 工程实践

与专家访谈提炼规则；Sim 调隶属函数；与经典控制器切换。

## 主要技术路线

### 1. If-Then 规则推理

文内代表实现路径；详见 [关联概念/形式化](../concepts/modeling-and-solving-for-control.md)。

### 2. 与相邻体系融合

常与 [八大机器人控制体系分类](../comparisons/robot-control-eight-paradigms-taxonomy.md) 中相邻类别 **级联或并联**（如 CTC+SMC、MPC+ILC、NN 补偿+PID）。

## 局限与风险

规则爆炸；稳定性证明难；精细任务不如模型/学习法。

## 关联页面

- [modeling-and-solving-for-control](../concepts/modeling-and-solving-for-control.md)
- [PID Control](./pid-control.md)
- [ML-driven Control Paradigm](../overview/robot-control-paradigm-ml-driven-control.md)

## 参考来源

- [wechat_shenlan_robot_control_eight_paradigms.md](../../sources/blogs/wechat_shenlan_robot_control_eight_paradigms.md) — 深蓝具身智能《机器人控制算法八大体系详解：从 PID 到强化学习》（<https://mp.weixin.qq.com/s/Kp12BMBiC7YiIiDPi_P8-g>）

## 推荐继续阅读

- 深蓝具身智能原文：<https://mp.weixin.qq.com/s/Kp12BMBiC7YiIiDPi_P8-g>

