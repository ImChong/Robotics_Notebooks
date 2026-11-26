---
type: method
tags: [sim2real, domain-adaptation, robotics]
status: draft
updated: 2026-11-26
summary: "通过调整图像、特征或策略，使仿真与真实环境之间的分布差异更易处理。"
sources:
  - ../../sources/repos/awesome-sim2real.md
related:
  - ../concepts/sim2real.md
---

# Sim2Real 域适应

## 一句话定义

通过调整图像、特征或策略，使仿真与真实环境之间的分布差异更易处理。

## 英文缩写速查

| 缩写 | 英文全称 | 说明 |
|---|---|---|
| API | Application Programming Interface | 组件交互接口 |

## 关联概念

- [相关概念](../concepts/sim2real.md)：了解本方法涉及的背景与边界。

## 为什么重要

明确任务边界、数据来源和验证方式，有助于比较不同系统并避免把背景任务误认为已经完成的实验。

## 主要技术路线

1. **观测域适应**：对齐图像外观或特征分布，需检查对象语义、几何及标签是否在变换后保留。
2. **动力学与控制迁移**：通过系统辨识、参数变化或真实数据适配处理摩擦、时延与执行器差异；视觉对齐不能保证动力学对齐。
3. **评估迁移效果**：区分源域、无标注目标域和目标域测试数据，记录真实样本预算、可用标签、分布变化和安全边界。

[CycleGAN 与视觉 Sim2Real](cyclegan-sim2real.md) 是视觉外观适应的具体路线；[域随机化](../concepts/domain-randomization.md) 是相关但不同的泛化策略。

## 局限与后续核查

本页是跨来源的导航与比较框架，具体实现、指标和适用条件应回到相应论文或项目核对；不能据此推断任一条目已实现全部环节。

## 参考来源

- [来源归档](../../sources/repos/awesome-sim2real.md)
