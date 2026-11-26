---
type: method
tags: [agent, skills, workflow]
status: draft
updated: 2026-11-26
summary: "把可复用任务指引、工具约束与可选脚本组织成可按需加载的技能。"
sources:
  - ../../sources/sites/anthropic_claude_code_best_practices_2026.md
related:
  - ../concepts/agentic-coding-software-fundamentals.md
---

# 智能体技能封装

## 一句话定义

把可复用任务指引、工具约束与可选脚本组织成可按需加载的技能。

## 英文缩写速查

| 缩写 | 英文全称 | 说明 |
|---|---|---|
| API | Application Programming Interface | 软件组件交互接口 |

## 关联概念

[相关概念](../concepts/agentic-coding-software-fundamentals.md) 描述该方法所涉及的背景与边界。

## 主要技术路线

1. **确定边界**：写明适用任务、必要输入、输出及不适用情况，避免把多个互不相关的任务塞入一个技能。
2. **组织资源**：按需组合指引、示例、脚本和参考资料；保留入口与依赖的可追溯关系，不假定每个技能都含可执行代码。
3. **验证和维护**：用代表性任务检查触发、执行结果及失败处理；核对版本、权限与引用路径。

[Agent 工具调用接口](agent-tool-use-interface.md) 关注如何调用工具；技能封装关注何时采用及如何复用一套任务流程。

## 局限与后续核查

本页用于跨项目导航，具体功能、参数、可运行性和授权仍以对应项目或原始资料为准。

## 参考来源

- [相关资料归档](../../sources/sites/anthropic_claude_code_best_practices_2026.md)
