---
type: method
tags: [sim2real, simulation, robotics]
status: draft
updated: 2026-11-26
summary: "将仿真中获得的模型或策略迁移到真实环境，并用真实测量检验迁移差异。"
sources:
  - ../../sources/repos/awesome-sim2real.md
related:
  - ../concepts/sim2real.md
---

# Sim2Real 迁移工作流

## 一句话定义

将仿真中获得的模型或策略迁移到真实环境，并用真实测量检验迁移差异。

## 英文缩写速查

| 缩写 | 英文全称 | 说明 |
|---|---|---|
| API | Application Programming Interface | 组件交互接口 |

## 关联概念

- [相关概念](../concepts/sim2real.md)：了解本方法涉及的背景与边界。

## 为什么重要

明确任务边界、数据来源和验证方式，有助于比较不同系统并避免把背景任务误认为已经完成的实验。

## 主要技术路线

1. **明确迁移对象**：区分感知模型、动力学模型和控制策略，记录仿真与真实传感、动作和任务接口。
2. **减少差异**：按问题选用标定、系统辨识、参数随机化或域适应；不同机制处理的差异并不相同。
3. **分阶段验证**：先检查离线数据和受控测试，再以受保护的真机实验测量成功率、失败与安全边界。

[域适应](domain-adaptation-for-sim2real.md) 聚焦源域与目标域差异；[关节辨识实验](sim2real-joint-sysid-experiment-design.md) 是动力学标定实例。

## 局限与后续核查

本页是跨来源的导航与比较框架，具体实现、指标和适用条件应回到相应论文或项目核对；不能据此推断任一条目已实现全部环节。

## 参考来源

- [来源归档](../../sources/repos/awesome-sim2real.md)
