---
type: method
tags: [humanoid, motion, control]
status: draft
updated: 2026-11-26
summary: "让人形或腿足机器人跟踪参考运动，并在执行中维持可行性和稳定性。"
sources:
  - ../../sources/repos/awesome-agentic-robots-survey-index.md
related:
  - ../concepts/whole-body-tracking-pipeline.md
---

# 人形全身动作跟踪

## 一句话定义

让人形或腿足机器人跟踪参考运动，并在执行中维持可行性和稳定性。

## 英文缩写速查

| 缩写 | 英文全称 | 说明 |
|---|---|---|
| API | Application Programming Interface | 组件交互接口 |

## 关联概念

- [相关概念](../concepts/whole-body-tracking-pipeline.md)：了解本方法涉及的背景与边界。

## 为什么重要

明确任务边界、数据来源和验证方式，有助于比较不同系统并避免把背景任务误认为已经完成的实验。

## 主要技术路线

1. **参考动作**：确定动作来源、本体与时间对齐；人体动作迁移到机器人时需处理结构差异。
2. **跟踪控制**：结合目标姿态、本体状态和接触信息产生动作；实现可采用优化或学习策略。
3. **验证**：比较跟踪误差、接触稳定性、扰动恢复和真机条件；动作逼真不保证真实可执行。

[GMR 重定向](motion-retargeting-gmr.md) 是参考动作跨本体映射的一种实例；[人形运动任务](../tasks/humanoid-locomotion.md) 关注具体行走场景。

## 局限与后续核查

本页是跨来源的导航与比较框架，具体实现、指标和适用条件应回到相应论文或项目核对；不能据此推断任一条目已实现全部环节。

## 参考来源

- [来源归档](../../sources/repos/awesome-agentic-robots-survey-index.md)
