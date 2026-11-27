# PhyAgentOS/PhyAgentOS-core（官方开源实现归档）

- **仓库名称：** `PhyAgentOS/PhyAgentOS-core`
- **官方地址：** <https://github.com/PhyAgentOS/PhyAgentOS-core>
- **配套仓库：** <https://github.com/PhyAgentOS/PhyAgentOS-DSH-Plugin>（DeepSeek Harness 插件，提供 `@x-era/dsh-forge`、`@x-era/forge-core`、`@x-era/forge-cli`、`@x-era/forge-mcp`）
- **关联论文：** [arXiv:2607.16636](https://arxiv.org/abs/2607.16636)（`PhyAgentOS: A Self-Evolving Operating System for Embodied Agents with Decoupled Cognitive Planning and Physical Execution`）
- **开源协议与分支：** MIT License，默认分支 `main`（归档快照 `3023` stars，最新版本 `v1.2.0`，2026-11-22）
- **开源核查状态：** **已开源**（含完整 Python 包、CLI、测试集、Dockerfile 与仿真/真机快速开始指南）
- **核对边界：** 静态代码核对，未本地运行（基于仓库目录树采样 `files_sample` 与 `README.md` 快照核查）。
- **沉淀到 wiki：** [PhyAgentOS 实体页](../../wiki/entities/paper-agentic-2607-16636.md)

## 已核对目录与核心入口

- **包入口与 CLI（`PhyAgentOS/` & `PhyAgentOS/cli/`）：**
  - `PhyAgentOS/__main__.py`、`PhyAgentOS/cli/commands.py`、`PhyAgentOS/cli/providers.py` — 对应命令行工具 `paos`（支持 `paos onboard`、`paos provider configure/use`、`paos status`、`paos agent`、`paos skill reconcile <skill>`）。
- **智能体主循环、验证与自演化记忆（`PhyAgentOS/agent/`）：**
  - `PhyAgentOS/agent/loop.py`、`context.py`、`budget.py`、`memory.py`、`skills.py`、`subagent.py` — 对话/任务规划循环与上下文构建。
  - `PhyAgentOS/agent/session_verifier.py` — 对应论文中的 `SessionVerifier` 语义验收模块。
  - `PhyAgentOS/agent/interaction/`（`model.py`、`runner.py`、`supervisor.py`）— 智能体侧交互模型、运行器与监督器。
  - `PhyAgentOS/agent/experience/`（`activation.py`、`analyzer.py`、`contracts.py`、`coordinator.py`、`evolution.py`、`redaction.py`、`source.py`、`store.py`）— 经验提取、脱敏、激活、演化协调与持久化存储。
  - `PhyAgentOS/agent/tools/`（`registry.py`、`filesystem.py`、`forge_task.py`、`forge_tool_api.py`、`skill_activation.py`、`scene_graph.py`、`shell.py`、`mcp.py`、`spawn.py` 等）— 受约束的工具注册表与 Forge Task / Tool API 工具桥。
- **物理执行、Forge 交互与技能运行时（`PhyAgentOS/forge/` & `skill_runtime/`）：**
  - `PhyAgentOS/forge/`（`binding.py`、`evidence.py`、`observation.py`、`reconciliation.py`、`task.py`、`tool_client.py`）及 `PhyAgentOS/forge/interaction/`（`binding.py`、`client.py`、`contracts.py`、`evidence.py`、`runner_store.py`、`service.py`）— 负责 `AgentTask` 绑定、观测与证据采集、状态对齐（reconciliation）及 Query/Action/Session Tool API 生命周期服务。
  - `PhyAgentOS/skill_runtime/`（`archive.py`、`catalog.py`、`installer.py`）— 版本化物理技能（Forge/Dora Skills）的目录检索、安装与归档。
  - `PhyAgentOS/embodiment_registry.py`、`PhyAgentOS/session/manager.py`、`PhyAgentOS/heartbeat/service.py` — 本体注册、会话管理与心跳服务。
- **模型供应、通道与配置（`PhyAgentOS/providers/`、`channels/`、`config/`、`bus/`）：**
  - 支持 OpenRouter、OpenAI Responses、LiteLLM、Azure OpenAI 等模型网关管理与多渠道接入（Feishu、Discord、Slack、Telegram 等）。
- **示例与评测文档（`README.md` 索引）：**
  - `docs/quickstart-mujoco.md`（Piper MuJoCo + Dora 仿真）、`example/quickstart/piper-quickstart.md`（真机 Piper 机械臂）、`example/quickstart/libero-quickstart.md`（LIBERO-10 50-episode 评测）、`docs/benchmarks.md`（RoboTwin 2.0 / LIBERO / CALVIN / RoboDojo $\pi_{0.5}$ profile 说明）。
