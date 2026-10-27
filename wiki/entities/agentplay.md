---
type: entity
project_id: agentplay
tags:
  - coding-agents
  - agent-observability
  - session-replay
  - developer-tools
status: complete
updated: 2026-10-27
project: https://entelligence.ai/agentplay
code: https://github.com/Entelligence-AI/agentplay
related:
  - ../concepts/agentic-coding-software-fundamentals.md
  - ../concepts/ai-agent-evaluation.md
  - ./open-code-review.md
sources:
  - ../../sources/sites/entelligence-agentplay.md
  - ../../sources/repos/entelligence-agentplay.md
summary: "AgentPlay 将 Claude Code、Codex、Cursor、Copilot 等 coding-agent 本地会话转成可回放的地图与时间线，便于检查执行过程；它是事后观察工具，不能替代测试与代码审查。"
---

# AgentPlay：编码 Agent 会话回放

## 一句话定义

AgentPlay 是 Entelligence 开源的本地优先会话观察器：解析多种 coding agent 写入磁盘的 transcript，生成标准化 session map，并在本机 viewer 中回放工具、文件、错误与交付事件。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| CLI | Command-Line Interface | 安装后通过 agentplay 命令生成和查看会话 |
| MCP | Model Context Protocol | map 可记录会话中涉及的 MCP server 与工具名称 |
| JSONL | JSON Lines | Claude Code、Codex 等会话来源采用的逐行 JSON 记录格式 |
| API | Application Programming Interface | 本机只读 viewer API 连接解析结果与界面 |

## 为什么重要

- **补上过程可见性：** 代码 diff 能显示最终改动，却难以说明 Agent 先读了什么、哪些命令反复失败、上下文何时被压缩，以及子代理返回了什么。AgentPlay 把这些事件放进同一条可定位的时间线。
- **可用于 coding-agent 评估：** 将会话中的探索、错误、重试与最终 PR / commit 一起检查，可辅助分析 agent 的工作习惯和失败模式；它提供观察材料，不自动判定任务质量。
- **跨工具汇总：** 项目为 Claude Code、Codex、Cursor、Copilot 等来源提供解析器，并以 agentplay/v1 作为 viewer 的会话 map 格式。

## 核心原理

会话从本机 transcript 开始，经来源解析器转换为统一事件，再构造成 map 和索引，最后由只绑定本机地址的 viewer 读取。数据库型来源先以只读方式复制和解码；增量构建只处理新增或发生变化的会话。

map 的 agentplay/v1 schema 将会话拆为 session 元数据、总量统计、repositories、files、agents、tasks、pull requests 和按序 events。事件可记录读取 / 编辑、shell、MCP、模型调用、错误、延迟与输出；viewer 将它们映射到目录树和可拖动时间轴。

## 工程实践

- **安装：** 通过 uvx agentplay、pipx run agentplay 或安装 Python 包后运行 agentplay。要求 Python 3.11+。
- **观察视图：** 地图用于浏览仓库、目录、文件、命令、MCP、模型、子代理、错误和输出；时间线可按步骤或时钟时间回放，也可跳转到编辑、错误、延迟、输出和重试。
- **数据格式：** agentplay export <session> 导出单个 JSON / JSON.GZ map；对方可通过 agentplay import 或拖放文件查看。
- **隐私：** 项目声明 transcript 和 map 不上传。map 不保留提示词正文、工具输入输出、文件内容与 shell 命令参数，但保留仓库名、相对路径、工具 / 模型名称、时间、计数和错误信息。公开分享前检查 repo 名及路径；可用 --path-mode dirs_only 减少文件路径暴露。viewer 的可选邮件订阅会发起网络请求，可用 AGENTPLAY_NO_NETWORK=1 禁用。
- **与安全层配合：** 回放能帮助发现遗漏读取、重复失败或失败恢复问题；访问权限、agent 沙箱和高风险操作审批仍需单独配置。可结合 [Open Code Review](./open-code-review.md) 检查变更本身。

## 局限与风险

- **解析器保真度不一：** 不同 agent 的本地格式和暴露字段不同；子代理、token 与成本等细节可能缺失，不能跨工具直接比较所有指标。
- **不评价正确性：** 可视化 transcript 不等于独立验证。需用测试、仿真或人工 review 确认输出；机器人真机任务还需安全监控和独立验收。
- **隐私不是完全匿名：** 虽移除正文与文件内容，仓库名、相对路径、模型、工具与错误元信息仍可能识别项目或工作流。
- **规模边界：** README 说明 map 最多记录 20,000 个事件并标记截断，超过 512 MB 的 transcript 会跳过；Windows 解析器可用但测试较少。
- **本机运行不等于零网络：** 默认无遥测和会话上传，但邮箱订阅是可选网络请求；禁用方式见工程实践。

## 关联页面

- [Agentic Coding 时代的软件工程基础](../concepts/agentic-coding-software-fundamentals.md) — coding agent 的工作过程需要与最终变更一起复盘。
- [AI Agent 评测](../concepts/ai-agent-evaluation.md) — 将回放事件转化为任务成功、成本与失败恢复指标。
- [Open Code Review](./open-code-review.md) — 过程回放与 diff 级审查互补，分别解释过程和检查变更。
- [Agentic Design Patterns](./agentic-design-patterns.md) — coding agent 在通用 agent 系统设计中的位置。

## 参考来源

- [Entelligence AgentPlay 项目页](../../sources/sites/entelligence-agentplay.md)
- [Entelligence-AI/agentplay 源码仓](../../sources/repos/entelligence-agentplay.md)
- [GitHub README](https://github.com/Entelligence-AI/agentplay#readme)
- [AgentPlay JSON Schema](https://github.com/Entelligence-AI/agentplay/blob/main/schema/agentplay-v1.schema.json)

## 推荐继续阅读

- [AgentPlay — GitHub](https://github.com/Entelligence-AI/agentplay) — 安装、支持来源、隐私契约与 map 格式。
- [AgentPlay — Entelligence](https://entelligence.ai/agentplay) — 项目概览、viewer 演示和运行入口。
