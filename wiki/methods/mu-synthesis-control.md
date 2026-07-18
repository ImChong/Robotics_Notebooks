---
type: method
tags:
  - control
  - robust-control
  - mu-synthesis
  - uncertainty
status: complete
updated: 2026-07-18
summary: "多自由度机器人多参数不确定性的进阶鲁棒设计工具。"
related:
  - ../overview/robot-control-paradigm-robust-control.md
  - ./h-infinity-control.md
sources:
  - ../../sources/blogs/wechat_shenlan_robot_control_eight_paradigms.md
---


# Mu Synthesis Control（μ 综合控制）

μ 综合：在 H∞ 框架上显式处理 **结构化参数不确定性**（多关节耦合、多参数漂移），优化稳定裕度。

## 一句话定义

> μ 综合：在 H∞ 框架上显式处理 **结构化参数不确定性**（多关节耦合、多参数漂移），优化稳定裕度。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| μ | Structured Singular Value | 结构化奇异值 |
| DK | D-K Iteration | μ 综合迭代 |
| H∞ | H-infinity Control | 上层鲁棒框架 |

## 为什么重要

比单一 $H_\infty$ 更贴合「每关节参数各不相等」的真实机构。

## 核心原理

将不确定性块 $\Delta$ 嵌入闭环；最小化 $\mu_\Delta(M(j\omega))<1$；D-K 迭代求缩放 $D$ 与控制器 $K$。

## 工程实践

MATLAB `musyn`；与多体标定数据结合界定 $\Delta$ 范围。

## 主要技术路线

### 1. 结构化不确定性 D-K 迭代

文内代表实现路径；详见 [关联概念/形式化](../concepts/optimal-control.md)。

### 2. 与相邻体系融合

常与 [八大机器人控制体系分类](../comparisons/robot-control-eight-paradigms-taxonomy.md) 中相邻类别 **级联或并联**（如 CTC+SMC、MPC+ILC、NN 补偿+PID）。

## 局限与风险

计算与建模成本高；工程上多用于关键子系统而非全身一次性设计。

## 关联页面

- [optimal-control](../concepts/optimal-control.md)
- [H-infinity Control](./h-infinity-control.md)
- [Robust Control Paradigm](../overview/robot-control-paradigm-robust-control.md)

## 参考来源

- [wechat_shenlan_robot_control_eight_paradigms.md](../../sources/blogs/wechat_shenlan_robot_control_eight_paradigms.md) — 深蓝具身智能《机器人控制算法八大体系详解：从 PID 到强化学习》（<https://mp.weixin.qq.com/s/Kp12BMBiC7YiIiDPi_P8-g>）

## 推荐继续阅读

- 深蓝具身智能原文：<https://mp.weixin.qq.com/s/Kp12BMBiC7YiIiDPi_P8-g>

