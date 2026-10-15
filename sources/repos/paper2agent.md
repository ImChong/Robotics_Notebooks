# jmiao24/Paper2Agent（代码仓库）

> 来源归档（repo）

- **标题：** Paper2Agent: Reimagining Papers As AI Agents
- **代码：** <https://github.com/jmiao24/Paper2Agent>
- **类型：** research workflow / agent skill / MCP server generator
- **License：** MIT
- **论文：** [Nature 2026, DOI 10.1038/s41586-026-11044-y](../papers/paper2agent_nature_2026.md)
- **项目页：** [paper2agent.ai](../sites/paper2agent-ai.md)
- **首次入库：** 2026-10-15

## 一句话摘要

开源 coding-agent skill 将研究论文、相关代码仓库或两者转成 paper skill 与 MCP server；工作流拆分为环境准备、方法提取、工具验证和交付，README 展示 AlphaGenome、Scanpy、TISSUE 的使用示例。

## 公开资源与使用方式

- `skills/paper2agent/`：入口 skill；可按输入路由到 Paper2Skill、Paper2MCP，或分别构建两者。
- `skills/paper2agent/paper2mcp/`：从代码库生成带运行环境说明和交付检查的 MCP server。
- `skills/paper2agent/paper2skill/`：从论文及关联资料生成 paper-specific skill。
- README 提供手动安装方式、Codex / Claude Code / Gemini CLI 的接入指引，以及已托管的 AlphaGenome、Scanpy、TISSUE MCP endpoint。

## 开源边界

- **已开源：** 转换工作流和 skill 源码以 MIT 许可发布。
- 目标 paper agent 不是同一套随仓库附送的模型或通用数据包；复现需要各研究仓库的数据、依赖、算力或密钥。生成 MCP 工具只覆盖可绑定到现有仓库代码且通过检查的能力。

## 对 wiki 的映射

- [`wiki/entities/paper-paper2agent.md`](../../wiki/entities/paper-paper2agent.md)
- [`sources/papers/paper2agent_nature_2026.md`](../papers/paper2agent_nature_2026.md)
- [`sources/sites/paper2agent-ai.md`](../sites/paper2agent-ai.md)
- [Model Context Protocol](../../wiki/concepts/model-context-protocol.md)
