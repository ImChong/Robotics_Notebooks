---
type: method
tags: [robotics, policy, learning]
status: draft
updated: 2026-11-26
summary: "从机器人任务、数据或交互反馈中学习可执行策略，并在目标环境中评测。"
sources:
  - ../../sources/repos/awesome-agentic-robots-survey-index.md
related:
  - ../concepts/foundation-policy.md
---

# 机器人策略学习流程

## 一句话定义

从机器人任务、数据或交互反馈中学习可执行策略，并在目标环境中评测。

## 英文缩写速查

| 缩写 | 英文全称 | 说明 |
|---|---|---|
| API | Application Programming Interface | 组件交互接口 |

## 关联概念

- [相关概念](../concepts/foundation-policy.md)：了解本方法涉及的背景与边界。

## 为什么重要

明确任务边界、数据来源和验证方式，有助于比较不同系统并避免把背景任务误认为已经完成的实验。

## 主要技术路线

1. **确定输入与动作**：界定机器人本体、传感数据、动作频率和执行接口。
2. **选择学习信号**：可使用示范监督、环境奖励或其他训练信号；各路线的数据、计算与安全要求不同。
3. **验证部署**：报告训练与测试环境、任务成功判据、未见情况及真机边界。

[模仿学习](imitation-learning.md) 和 [强化学习](reinforcement-learning.md) 分别介绍示范监督与奖励驱动的策略学习。

## 局限与后续核查

本页是跨来源的导航与比较框架，具体实现、指标和适用条件应回到相应论文或项目核对；不能据此推断任一条目已实现全部环节。

## 参考来源

- [来源归档](../../sources/repos/awesome-agentic-robots-survey-index.md)
