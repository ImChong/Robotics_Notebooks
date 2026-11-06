# Hermes Agent README 与 Architecture 文档通读

> 来源归档（原文通读 · 支撑 `wiki/entities/hermes-agent.md`）

- **标题：** Hermes Agent README + Developer Guide「Architecture」页
- **作者 / 组织：** [Nous Research](https://nousresearch.com)
- **类型：** repo / docs
- **原文：** <https://github.com/NousResearch/hermes-agent>（`README.md`，main 分支，2026-11 获取全文）；<https://hermes-agent.nousresearch.com/docs/developer-guide/architecture>
- **一句话说明：** 官方自述为 "self-improving AI agent"，内置学习闭环；单一 `AIAgent` 服务 CLI / Gateway / ACP / Batch / API Server，工具自注册，状态落 SQLite+FTS5。
- **核对范围：** 通读 README 全文（特性、安装、命令、Nous Portal、CLI/消息对照表、文档索引、迁移、社区、许可）与 Architecture 页全文（系统总览图、目录结构、三条数据流、子系统概述、设计原则、文件依赖链）。其余子页面（Agent Loop Internals、Gateway Internals、Security 等）仅见链接，未阅读；源码未逐行阅读。
- **通读：** 已完成

## 核心摘录（均出自原文）

- **定位（README）：** "The self-improving AI agent"；闭环学习 = 代理策展记忆 + 周期 nudge + 复杂任务后自主建技能 + 技能使用中自改进 + FTS5 会话检索加 LLM 摘要 + Honcho 辩证用户建模；兼容 agentskills.io。
- **运行位置：** 六个内置终端后端（local、Docker、SSH、Modal、Daytona、Vercel Sandbox），另有 Singularity/Apptainer 等插件后端；Daytona 与 Modal 空闲休眠、按需唤醒。
- **入口与网关：** 一个 gateway 进程接 Telegram / Discord / Slack / WhatsApp / Signal / Email 等；Architecture 页称 25+ 平台适配器（内置 + 捆绑插件）。
- **模型无关：** `hermes model` 切换提供商；Nous Portal 可选，提供 300+ 模型与 Tool Gateway（搜索、图像、TTS、云浏览器）。
- **研究向：** 批量轨迹生成与轨迹压缩，用于训练下一代工具调用模型；Architecture 页称生成 ShareGPT 格式轨迹。
- **迁移：** `hermes import-agent` 可从 OpenClaw / Claude Code / Codex / Cursor 导入设置、记忆、技能。
- **架构（Architecture 页）：** 入口（cli.py、gateway/run.py、acp_adapter/、Batch Runner、API Server、Python Library）→ `AIAgent`（Prompt Builder、Provider Resolution、Tool Dispatch、Compression & Caching）→ 会话存储（SQLite+FTS5）与工具后端（终端 7、浏览器 5、Web 4、MCP 动态）。
- **数据流：** CLI 与 Gateway 均走 `AIAgent.run_conversation()`；Cron 每次创建无历史的新 `AIAgent`，注入附带技能后运行任务并投递。
- **设计原则：** Prompt 稳定（会话中不改 system prompt，保护缓存）、可观测执行、可中断、平台无关核心、松耦合（registry + check_fn 门控）、Profile 隔离（`hermes -p`）。
- **插件：** 三种发现源（用户 / 项目 / pip entry points）；memory provider 与 context engine 为单选。
- **文件依赖链：** `tools/registry.py` 无依赖，各工具 import 时 `registry.register()` 自注册，无需手写导入清单。

## 与 wiki 页的差异备注

- 当前上游文档较页面既有表述更新：平台适配器 25+（页面写 20+）、终端后端 README 称六个内置（Architecture 页部分处写 7 含插件）、工具计数 README 文档表写 40+ 而 Architecture 写 70+。数字随版本变化。

## 对 wiki 的映射

- [`wiki/entities/hermes-agent.md`](../../wiki/entities/hermes-agent.md)
