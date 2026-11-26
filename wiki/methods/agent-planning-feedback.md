---
type: method
tags: [agent, planning, feedback]
status: draft
updated: 2026-11-26
summary: "将任务分解、行动选择和环境反馈组织为可检查的智能体执行循环。"
sources:
  - ../../sources/repos/agentic-design-patterns.md
related:
  - ../concepts/agentic-software-factory-harness.md
---

# 智能体规划与反馈执行

## 一句话定义

将任务分解、行动选择和环境反馈组织为可检查的智能体执行循环。

## 英文缩写速查

| 缩写 | 英文全称 | 说明 |
|---|---|---|
| API | Application Programming Interface | 程序交互接口 |

## 关联概念

[相关概念](../concepts/agentic-software-factory-harness.md) 介绍该方法的背景。

## 主要技术路线

1. **规划与状态**：把用户目标转为可执行步骤，明确当前上下文、终止条件与所需权限。
2. **行动与观察**：使用工具或环境接口执行行动，读取结果、失败和状态变化；规划可以是规则、模型或混合实现。
3. **反馈与修正**：根据证据继续、重试、重规划或交给人工；记录轨迹有助于复现与评估。

[工具调用接口](agent-tool-use-interface.md) 聚焦参数与结果契约；本页关注跨步骤执行。

## 局限与后续核查

本页用于跨项目导航；实际功能、性能和许可应以具体项目及环境验证为准。

## 参考来源

- [资料归档](../../sources/repos/agentic-design-patterns.md)
