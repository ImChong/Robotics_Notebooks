---
type: method
tags: [agentic, robotics, planning]
status: draft
updated: 2026-11-26
summary: "让任务规划、工具调用、机器人执行和反馈恢复有明确边界的系统组织方法。"
sources:
  - ../../sources/repos/awesome-agentic-robots-survey-index.md
related:
  - ../concepts/llm-robotics-control-interfaces.md
---

# 机器人智能体执行框架

## 一句话定义

让任务规划、工具调用、机器人执行和反馈恢复有明确边界的系统组织方法。

## 英文缩写速查

| 缩写 | 英文全称 | 说明 |
|---|---|---|
| API | Application Programming Interface | 组件交互接口 |

## 关联概念

- [相关概念](../concepts/llm-robotics-control-interfaces.md)：了解本方法涉及的背景与边界。

## 为什么重要

明确任务边界、数据来源和验证方式，有助于比较不同系统并避免把背景任务误认为已经完成的实验。

## 主要技术路线

1. **任务规划与分解**：将用户目标转成可检查的子目标；计划可以由语言模型、规则或两者协同产生。
2. **工具和技能执行**：通过受约束的 API 调用感知、导航、操作或其他技能，记录输入、输出与权限。
3. **观测与恢复**：比较执行状态和预期，必要时重试、重规划、请求帮助或终止；记忆和技能库可帮助跨步骤复用经验。

不同工作可能只研究其中一个环节；语言模型规划不等于低层实时控制。可对照 [智能体操作策略](agentic-manipulation-policies.md)，理解物体操作中的一个具体实例。

## 局限与后续核查

本页是跨来源的导航与比较框架，具体实现、指标和适用条件应回到相应论文或项目核对；不能据此推断任一条目已实现全部环节。

## 参考来源

- [来源归档](../../sources/repos/awesome-agentic-robots-survey-index.md)
