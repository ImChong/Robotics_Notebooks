---
type: method
tags: [agent, skills, workflow]
status: complete
updated: 2026-11-26
summary: "把可复用任务指引、工具约束与可选脚本组织成可按需加载的技能。"
sources:
  - ../../sources/sites/method-agent-skill-packaging-primary-20261126.md
  - ../../sources/sites/anthropic_claude_code_best_practices_2026.md
original:
  - ../../sources/sites/method-agent-skill-packaging-primary-20261126.md
related:
  - ./agent-tool-use-interface.md
  - ./agent-planning-feedback.md
  - ../concepts/agentic-coding-software-fundamentals.md
---

# Agent Skill Packaging · 智能体技能封装

## 一句话定义

把高复用任务的触发描述、主流程和按需资源包装为可安装、可评测的目录。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|---|---|---|
| YAML | YAML Ain’t Markup Language | SKILL.md 的前置信息格式 |
| CLI | Command-Line Interface | 可选脚本的命令行接口 |
| API | Application Programming Interface | 技能步骤可能调用的外部接口 |

## 为什么重要

技能解决重复描述任务操作方式的问题：先暴露简短适用描述，仅在匹配时读取主指引，依赖资料按需打开；既减少上下文占用，也便于独立评测。

## 主要技术路线（核心机制）

把适用与不适用示例同时放进测试集；入口说明写清输入、输出、失败交代和核对点，细则材料放入 references。可选脚本应声明依赖，不能用元数据字段当成权限控制。

```mermaid
flowchart LR
  N0["读取技能元数据"]
  N1["判断任务是否适用"]
  N2["载入主指引"]
  N3["按需读取脚本与参考"]
  N4["在权限内验收结果"]
  N0 --> N1
  N1 --> N2
  N2 --> N3
  N3 --> N4
```

## 工程实践

以结构化机器人日志检查技能为例：一个主指引说明数据记录日期和来源为必需输入，再在需要时引用字段字典；在没有原始记录的负例中必须报告缺失，而不是自动生成样本。

## 局限与风险

规范验证关注名称、目录与前置字段，不验证事实准确性、可移植性或脚本安全。产品可能使用不同的发现及权限模型，跨客户端迁移前要重测。

## 关联页面

- [智能体工具调用接口](agent-tool-use-interface.md)
- [Agent Skill 封装 vs MCP 工具契约 vs CLI/SDK 调用接口](../comparisons/agent-skill-packaging-vs-mcp-cli-tool-interfaces.md)
- [规划与反馈执行](agent-planning-feedback.md)
- [智能体开发工程基础](../concepts/agentic-coding-software-fundamentals.md)

## 参考来源

- [Agent Skills：Specification 核读归档](../../sources/sites/method-agent-skill-packaging-primary-20261126.md)
- [原页面关联资料](../../sources/sites/anthropic_claude_code_best_practices_2026.md)

## 推荐继续阅读

- [Agent Skills：Specification](https://agentskills.io/specification)
