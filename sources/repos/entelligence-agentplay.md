# AgentPlay（Entelligence-AI/agentplay）

> 来源归档（repo；核对日期：2026-10-27）

- **标题：** AgentPlay — coding-agent session replay
- **仓库：** <https://github.com/Entelligence-AI/agentplay>
- **类型：** repo / coding-agents / session-replay / observability
- **组织：** Entelligence AI（[GitHub](https://github.com/Entelligence-AI)）
- **项目页：** <https://entelligence.ai/agentplay>
- **许可：** Apache-2.0（GitHub 仓库元数据与项目页）
- **一句话说明：** 本地优先的编码 Agent 会话回放工具：读取各 Agent 写入本机的 transcript，生成 agentplay/v1 session map，并由本机 viewer 以交互地图和时间线展示工具调用、文件触达、错误、子代理与交付输出。
- **支持的会话来源（README）：** Claude Code、Codex CLI/IDE、Cursor CLI/IDE、GitHub Copilot CLI、GitHub Copilot in VS Code、OpenCode、Kiro、Gemini CLI、Antigravity CLI。不同解析器记录的子代理、token 和成本信息并不完全相同。
- **运行入口：** Python 3.11+；uvx agentplay、pipx run agentplay 或 pip install agentplay。运行时除 Python 标准库外无额外依赖；从源码构建 bundled viewer 还需 Node.js 22+ 与 pnpm。
- **常用命令：** agentplay / agentplay view 构建新增或变化的会话并启动 viewer；build 只生成 map；list 列出本地会话；export 导出单个 map；import 导入共享 map；agents 检查本机发现的 Agent。
- **主要路径：** python/src/agentplay/_core/ 含各来源解析器；builder.py 归一化为 map 并应用脱敏规则；materialize.py 读取 Cursor / OpenCode / Kiro 的本机数据库；server.py 提供本机只读 API；viewer/ 是交互界面。
- **本地数据：** 默认会话地图与索引位于 ~/.agentplay；viewer 服务于 127.0.0.1。可用 AGENTPLAY_HOME 更改目录、AGENTPLAY_NO_NETWORK=1 禁用可选邮箱入口。
- **隐私与分享边界：** README 声明不会上传 session 或 map，也不做遥测或更新检查；map 不包含 prompt / assistant 正文、工具输入输出、文件内容或 shell 参数，但仍含仓库名、repo-relative 文件路径、工具和 MCP / skill 名称、模型、token 数、耗时、错误类别与退出码。导出分享前可用 --path-mode dirs_only 将文件名替换为目录与哈希标识；仓库名与 remote 仍需自行检查。
- **工程限制：** 各 Agent 转录格式不同，解析信息保真度不一；例如 Cursor 和 Kiro 的成本信息受本机记录方式限制。README 标注 Windows 解析器可用但测试较少；单个 map 达 20,000 个事件会截断，超过 512 MB 的 transcript 会跳过。
- **原始来源：** [项目页与本地运行说明](../sites/entelligence-agentplay.md)

## 对 wiki 的映射

- 主实体：[wiki/entities/agentplay.md](../../wiki/entities/agentplay.md)
- 概念关联：[wiki/concepts/agentic-coding-software-fundamentals.md](../../wiki/concepts/agentic-coding-software-fundamentals.md)
