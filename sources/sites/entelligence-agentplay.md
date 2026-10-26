# AgentPlay（Entelligence 项目页）

> 来源归档（site；核对日期：2026-10-27）

- **标题：** AgentPlay — Session replay for coding agents
- **项目页：** <https://entelligence.ai/agentplay>
- **代码：** <https://github.com/Entelligence-AI/agentplay>（公开可读；Apache-2.0）
- **产品定位：** 将本机编码 Agent 的会话记录整理为交互式地图与时间线，查看会话涉及的仓库、文件、命令、MCP 工具、模型、子代理、错误和输出。
- **安装入口：** 提供 uvx agentplay、pipx run agentplay 与 pip install agentplay；要求 Python 3.11+，默认无需账号或 API key，viewer 仅绑定本机 127.0.0.1。
- **支持范围：** 项目页列出 Claude Code、Codex、Cursor、GitHub Copilot、OpenCode 与 Kiro；GitHub README 还列出 Copilot CLI / VS Code、Gemini CLI、Antigravity CLI 的解析器。
- **隐私说明：** 项目页称会话数据在本机处理、无会话上传或遥测；README 补充首次打开 viewer 时可选填写邮箱以接收产品更新，这会发出一条独立网络请求。选择不填写不会发送；AGENTPLAY_NO_NETWORK=1 可禁用该入口与请求。分享导出 map 前应检查其中仍保留的仓库名与相对文件路径。
- **核查结论：** 官方页面链接到 GitHub 仓库；仓库公开且 GitHub 标注 Apache-2.0。开放对象是源码工具，不代表其读取到的会话数据适合公开分享。

## 互链

- [GitHub 源码仓归档](../repos/entelligence-agentplay.md)
- [Wiki 实体](../../wiki/entities/agentplay.md)
