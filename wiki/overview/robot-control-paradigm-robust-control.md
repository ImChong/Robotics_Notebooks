---
type: overview
tags:
  - control
  - robust-control
  - sliding-mode
  - h-infinity
  - disturbance
status: complete
updated: 2026-07-18
summary: "鲁棒控制被动抵抗扰动与参数偏差，SMC/H∞/μ 综合是工业与航空级常见工具。"
related:
  - ../comparisons/robot-control-eight-paradigms-taxonomy.md
  - ../methods/sliding-mode-control.md
  - ../methods/h-infinity-control.md
  - ../methods/mu-synthesis-control.md
sources:
  - ../../sources/blogs/wechat_shenlan_robot_control_eight_paradigms.md
---


# 鲁棒控制（体系③）

在模型不精确与外部扰动下保证稳定与有界误差，强调「最坏情况」下的性能保证。

## 一句话定义

在模型不精确与外部扰动下保证稳定与有界误差，强调「最坏情况」下的性能保证。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| SMC | Sliding Mode Control | 滑模强制状态贴合滑模面 |
| H∞ | H-infinity Control | 最小化扰动到误差的最大增益 |
| μ | Mu Synthesis | 结构化不确定性综合鲁棒设计 |

## 为什么重要

户外移动机器人、重载臂、接触冲击场景需要 **不依赖精确模型** 的稳定性保证。

## 核心原理

SMC 设计滑模面 $s=0$ 并高频切换使状态沿面收敛；H∞ 将扰动视为输入并约束 $\|T_{wd}\|_\infty$；μ 综合处理多参数不确定性块。

## 代表性算法

| 算法 | 节点 |
|------|------|
| SMC | [sliding-mode-control.md](../methods/sliding-mode-control.md) |
| H∞ | [h-infinity-control.md](../methods/h-infinity-control.md) |
| μ 综合 | [mu-synthesis-control.md](../methods/mu-synthesis-control.md) |

## 工程实践

SMC 需 **边界层/高阶滑模** 抑制抖振；H∞ 用 MATLAB Robust Control Toolbox 或 python-control 设计；与 CTC 并联作残差修正层。

## 局限与风险

SMC 抖振磨损执行器；H∞/μ 设计阶次高、多关节系统标定成本大。

## 关联页面

- [Sliding Mode Control](../methods/sliding-mode-control.md)
- [H-infinity Control](../methods/h-infinity-control.md)
- [Mu Synthesis Control](../methods/mu-synthesis-control.md)

## 参考来源

- [wechat_shenlan_robot_control_eight_paradigms.md](../../sources/blogs/wechat_shenlan_robot_control_eight_paradigms.md) — 深蓝具身智能《机器人控制算法八大体系详解：从 PID 到强化学习》（<https://mp.weixin.qq.com/s/Kp12BMBiC7YiIiDPi_P8-g>）

## 推荐继续阅读

- 深蓝具身智能原文：<https://mp.weixin.qq.com/s/Kp12BMBiC7YiIiDPi_P8-g>

