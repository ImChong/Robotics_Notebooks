# Hermes Agent（NousResearch/hermes-agent）

> 来源归档

- **标题：** Hermes Agent
- **类型：** repo
- **作者 / 组织：** [Nous Research](https://nousresearch.com/)
- **代码：** <https://github.com/NousResearch/hermes-agent>
- **文档站：** <https://hermes-agent.nousresearch.com/docs>
- **许可：** MIT（以仓库 `LICENSE` 为准）
- **入库日期：** 2026-05-19
- **一句话说明：** 开源、可长期驻留的自主编码/通用代理运行时：统一 `AIAgent` 对话环、70+ 工具与 20+ 消息网关、跨会话记忆与技能自举闭环、多终端/容器/无服务器沙箱后端，以及面向 RL 的轨迹导出；由训练 Hermes 系列模型的实验室维护。

## 与本仓库知识的关系

| 主题 | 关系 |
|------|------|
| [Hermes Agent](../../wiki/entities/hermes-agent.md) | 实体页：架构分层、学习闭环、与 IDE  tethered copilot / 单 API 聊天壳的边界 |
| [Superpowers（obra）](../../wiki/entities/superpowers-obra.md) | 同属「把代理行为写成可版本化外围」谱系；Superpowers 偏 **交付流程技能**，Hermes 偏 **常驻运行时 + 网关 + 记忆** |
| [Agent Reach](../../wiki/entities/agent-reach.md) | 互补：Agent Reach 聚合 **外网读搜 CLI/MCP**；Hermes 内置 web/browser/MCP 与终端，但定位是完整 agent OS |

## README / 文档要点（归纳，2026-05-19）

- **定位：** *The agent that grows with you* — 非 IDE 绑定的 coding copilot，也非单 API 包装；可部署在 VPS / GPU 集群 / Daytona·Modal 等近零空闲成本环境，经 Telegram 等渠道远程驱动。
- **入口：** CLI（`cli.py`）、Messaging Gateway（`gateway/run.py`，20+ 平台适配器）、ACP（VS Code / Zed / JetBrains）、Batch Runner、API Server、Python 库。
- **核心环：** `AIAgent`（`run_agent.py`）— Prompt Builder、Provider Resolution（18+ 提供商、三种 API mode）、Tool Dispatch（`model_tools.py` + `tools/registry.py`，70+ 工具 / ~28 toolsets）。
- **持久化：** SQLite + FTS5 会话库（`hermes_state.py`）；记忆插件与 **Honcho** 用户建模；技能系统兼容 [agentskills.io](https://agentskills.io/)，支持 Skills Hub。
- **学习闭环（文档强调）：** 代理策展记忆、周期性 nudge、自主创建/改进技能、跨会话 FTS5 召回 + LLM 摘要。
- **执行环境：** 终端 7 后端（local、Docker、SSH、Daytona、Modal、Singularity、Vercel Sandbox 等）；浏览器 5 后端；`execute_code` 程序化工具调用以压缩多步管线。
- **子代理：** `delegate_tool` 隔离子会话、独立终端与 Python RPC。
- **自动化：** 内置 cron（自然语言调度），可向任意已接平台投递结果。
- **研究向：** 批处理、ShareGPT 轨迹导出、Atropos RL 训练管线叙述。
- **LLM 可读文档：** 站点提供 `/llms.txt`（~17KB 索引）与 `/llms-full.txt`（~1.8MB 全文），每次部署重新生成。

## 对 wiki 的映射

- 沉淀 **[`wiki/entities/hermes-agent.md`](../../wiki/entities/hermes-agent.md)**；安装命令与平台列表以克隆时上游为准（此处不复制易变 shell 片段）。
