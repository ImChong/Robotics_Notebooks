---
type: entity
project_id: safe-depth
arxiv: "2611.13769"
code: https://github.com/saeed5959/depth_safe_robot
tags: [paper, safe-depth, robotics]
status: draft
updated: 2026-11-14
summary: "Safe Depth 在冻结的 Depth Anything V2 上预测偏差与不确定性，将保守距离 d·exp(μ−kσ) 提供给 Nav2。"
related:
  - ../tasks/zero-shot-object-navigation.md
  - ../concepts/safety-filter.md
original:
  - ../../sources/papers/safe_depth_arxiv_2611_13769.md
sources:
  - ../../sources/papers/safe_depth_arxiv_2611_13769.md
  - ../../sources/repos/safe-depth.md
---

# From Depth to Safe Depth: Uncertainty-Aware Monocular Perception for Mobile Robot Navigation｜不确定性校准单目深度导航

## 一句话定义

Safe Depth 在冻结的 Depth Anything V2 上预测偏差与不确定性，将保守距离 d·exp(μ−kσ) 提供给 Nav2。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|---|---|---|
| Nav2 | Navigation 2 | 本研究涉及的Nav2术语 |
| DA2 | Depth Anything V2 | 本研究涉及的DA2术语 |
| NLL | Negative Log Likelihood | 本研究涉及的NLL术语 |

## 为什么重要

从 RGB 获得单目深度，并用小型 0.58M 参数头估计对数偏差 μ 和方差尺度 σ。

## 核心原理与方法

1. 从 RGB 获得单目深度，并用小型 0.58M 参数头估计对数偏差 μ 和方差尺度 σ。
2. 留出世界做保形标定求 k，形成偏向障碍物一侧的安全深度界。
3. 由 Nav2 消费安全深度并据不确定性判断可通行地面。

## 流程总览

```mermaid
flowchart LR
  A0["RGB 图像"]
  A1["冻结 DA2"]
  A2["偏差与不确定性头"]
  A3["安全深度"]
  A4["Nav2 导航"]
  A0 --> A1
  A1 --> A2
  A2 --> A3
  A3 --> A4
```

> 方法流程根据摘要/官网整理，尚待论文原文逐图核实。

## 实验与评测

论文摘要称 120 次配对仿真中碰撞率 63%→3%、导航成功率 37%→70%，未在真机复测。

上述结果均为作者报告或用户附件转载，**不是本次复测**。

## 结论

**Safe Depth 在冻结的 Depth Anything V2 上预测偏差与不确定性，将保守距离 d·exp(μ−kσ) 提供给 Nav2。**

1. 从 RGB 获得单目深度，并用小型 0.58M 参数头估计对数偏差 μ 和方差尺度 σ。
2. 留出世界做保形标定求 k，形成偏向障碍物一侧的安全深度界。
3. 仿真中采用真值里程计，跨地图/相机/材质的校准失配与真机可靠性仍需验证。

## 与其他工作对比

比较本文与相关基线时应在相同任务、数据、样本量、传感器和部署条件下评估；不能把论文中的单个数字解释为真实部署的通用进步。待全文表格核对后补足逐项量化对照。

## 源码运行时序图

```mermaid
sequenceDiagram
  participant A0 as ROS 2 launch
  participant A1 as mono_nav
  participant A2 as Uncertainty head
  participant A3 as Nav2/Gazebo
  A0->>A1: 启动评测/训练
  A1->>A2: 处理环境与模型输入
  A2->>A3: 执行模型或输出指标
  A3-->>A0: 返回状态
````

**README 入口：** `ros2 launch ./eval_stack.launch.py world:=test_02 method:=mono_safe rviz:=true`。图仅表示已核对到的入口与概念性组件顺序，不代表已运行或逐行审阅。

## 工程实践

- 对齐版本、传感器、动作/观测空间和评测预算后再尝试复现。
- 正式应用应增加跨分布扰动与任务安全边界检查；原文完整复现步骤待核对。
- 本次通读状态：**未完成**；页面仍为 draft。

## 局限与风险

仿真中采用真值里程计，跨地图/相机/材质的校准失配与真机可靠性仍需验证。

未对全部论文正文/附录及运行细节完成审阅，任何进一步量化外推均不成立。

## 关联页面

- [zero-shot-object-navigation](../tasks/zero-shot-object-navigation.md)
- [safety-filter](../concepts/safety-filter.md)

## 参考来源

- [原论文及阅读进度](../../sources/papers/safe_depth_arxiv_2611_13769.md)
- [官方源码入口](../../sources/repos/safe-depth.md)
- [用户附件 13 篇原始资讯整理](../../sources/blogs/embodied-research-digest-2026-11-12.md)

## 推荐继续阅读

- [arXiv 原文](https://arxiv.org/abs/2611.13769)
- [官方源码](https://github.com/saeed5959/depth_safe_robot)
