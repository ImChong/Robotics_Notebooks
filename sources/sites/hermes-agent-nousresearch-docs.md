# Hermes Agent 官方站点与文档（hermes-agent.nousresearch.com）

- **类型：** 网站 / 产品文档（Docusaurus）
- **入口：** <https://hermes-agent.nousresearch.com/>（产品落地页）
- **文档根：** <https://hermes-agent.nousresearch.com/docs>
- **代码仓：** <https://github.com/NousResearch/hermes-agent>（`website/` 为文档站源码）
- **收录日期：** 2026-05-19
- **抓取说明：** 以 **2026-05-19** 对首页、文档首页及 [Architecture](https://hermes-agent.nousresearch.com/docs/developer-guide/architecture) 的公开 HTML 为准；细节以实现与 GitHub 为准。

## 一句话

Nous Research 出品的 **自改进型自主代理** 官方说明：安装向导、用户指南（网关、记忆、技能、MCP、语音、安全）、开发者架构图与 **llms.txt** 机器可读索引。

## 为什么值得保留

- 与 [Hermes Agent 仓库源归档](../repos/nousresearch_hermes_agent.md) 配对，区分 **营销叙事 / 用户路径** 与 **代码树结构**。
- 文档明确 **closed learning loop**、**平台无关的 AIAgent 核心**、**Profile 隔离**（`hermes -p` 独立 `HERMES_HOME`）等设计原则，便于与本站 [LLM Wiki](../../wiki/references/llm-wiki-karpathy.md) 的「持久知识编译」对照。
- 提供 **LLM 维护者入口**（`llms.txt` / `llms-full.txt`），适合作为后续增量 ingest 的索引而非一次性粘贴 1.8MB 正文。

## 公开要点（归纳）

| 区块 | 内容 |
|------|------|
| **安装** | Linux / macOS / WSL2 一键脚本；Windows PowerShell（早期 beta）；Android Termux 同 Linux 流程 |
| **Quick Links** | Installation、Quickstart、Learning Path、Configuration、Messaging Gateway、Tools、Memory、Skills、MCP、Voice、Security、Architecture、FAQ |
| **Key Features** | 6 终端后端；20+ 消息平台；cron；子代理 + `execute_code`；agentskills.io 兼容；MCP；Atropos / 轨迹导出 |
| **架构顶图** | Entry Points → AIAgent → Session Storage + Tool Backends；目录树见 Architecture 页 |
| **设计原则** | Prompt stability、Observable execution、Interruptible、Platform-agnostic core、Loose coupling、Profile isolation |
| **机器可读** | `/docs/llms.txt`、`/docs/llms-full.txt`（部署时生成） |

## 对 wiki 的映射

- 升格页面：[wiki/entities/hermes-agent.md](../../wiki/entities/hermes-agent.md)
- 仓库侧归档：[sources/repos/nousresearch_hermes_agent.md](../repos/nousresearch_hermes_agent.md)
