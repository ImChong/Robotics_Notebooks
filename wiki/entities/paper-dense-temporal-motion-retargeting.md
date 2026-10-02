---
type: entity
tags: [paper, motion-retargeting, humanoid, trajectory-optimization, eth-zurich]
status: complete
updated: 2026-10-02
arxiv: "2609.38617"
related:
  - ./paper-notebook-dynaretarget-dynamically-feasible-retargeting-us.md
  - ./paper-shooting-for-contact.md
sources:
  - ../../sources/blogs/humanoid_motion_intelligence_day1_data_retargeting_2026_10_02.md
  - ../../sources/sites/dense-temporal-retargeting-project.md
summary: "Dense Temporal Motion Retargeting 联合优化机器人动作和参考动作相位，在动态难段调整局部节奏，主要量化证据来自仿真。"
---

# Dense Temporal Motion Retargeting for Legged Robots

## 一句话定义

**Dense Temporal Motion Retargeting** 将“机器人做什么动作”和“动作进行到哪一帧”一起优化，在起跳等困难片段局部调整节奏。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| DTM | Dense Temporal Motion Retargeting | 联合动作与密集时间对齐的重定向 |
| G1 | Unitree G1 | 真机舞蹈片段展示的平台 |
| RL | Reinforcement Learning | 可消费重定向参考的下游控制训练 |

## 为什么重要

固定帧率播放人体参考可能让机器人在动力学关键时刻来不及起跳或落脚。全局放慢则牺牲节奏；局部相位优化能只改困难段。

## 方法

采样式搜索在同一仿真优化中调整机器人关节目标和参考相位进度；相位改变后仍需约束动作连续与总时长。它是**离线参考生成**，真机动作还由跟踪器执行。

## 实验与评测

文章报告约 120 分钟动作适配到多本体后，四种人形的关键点误差中位数平均降低 **13.4%**，时间对齐误差降低 **12.9%**；全语料时长变化控制在 **1%** 以内。G1 真机展示两段舞蹈，大规模数字来自仿真。

## 与其他工作对比

[DynaRetarget](./paper-notebook-dynaretarget-dynamically-feasible-retargeting-us.md) 强调扩展轨迹窗口的动态可行性；本工作特别把**参考相位**纳入密集优化。[Shooting for Contact](./paper-shooting-for-contact.md) 则侧重接触隐式动力学。

## 结论

**难动作参考可以局部调节时间，而不必整体放慢；评测应同时报告姿态误差、时间偏差与总时长变化。**

1. 先定位动态难段，再检查相位变化是否平滑。
2. 相位对齐改善不直接证明真机稳定。
3. 应与固定节奏、全局时间伸缩基线分开比较。

## 工程实践

记录每段参考相位曲线、足端接触与跟踪误差；[项目页](https://jaeryeongnicolekim.com/Dense-Temporal-Motion-Retargeting-For-Legged-Robots/)在本次核查中未能确认可运行代码入口，暂标为**待核实**。

## 局限与风险

大规模统计为仿真；换机器人时需重新考虑其关节范围、惯量和下游跟踪器。

## 源码运行时序图

**不适用**：官方可运行源码未获核实。

## 关联页面

- [DynaRetarget](./paper-notebook-dynaretarget-dynamically-feasible-retargeting-us.md)
- [Shooting for Contact](./paper-shooting-for-contact.md)

## 参考来源

- [Day 1 文章逐篇索引](../../sources/blogs/humanoid_motion_intelligence_day1_data_retargeting_2026_10_02.md)
- [项目页核查](../../sources/sites/dense-temporal-retargeting-project.md)
- [论文](https://arxiv.org/abs/2609.38617)

## 推荐继续阅读

- [项目页](https://jaeryeongnicolekim.com/Dense-Temporal-Motion-Retargeting-For-Legged-Robots/)
