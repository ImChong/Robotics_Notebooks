# Agent Lightning（microsoft/agent-lightning）

- **标题：** Agent Lightning
- **类型：** repo
- **来源：** 微软（Microsoft / Microsoft Research）
- **链接：** <https://github.com/microsoft/agent-lightning>
- **文档：** <https://microsoft.github.io/agent-lightning/stable/>
- **项目页：** <https://www.microsoft.com/en-us/research/project/agent-lightning/>
- **论文 / 技术报告：**
  - v1.0：<https://arxiv.org/abs/2608.17528>（Towards Harnessed Agentic RL）
  - 初版：<https://arxiv.org/abs/2508.03680>（Train ANY AI Agents with RL）
- **许可：** MIT
- **入库日期：** 2026-09-19
- **一句话说明：** 约 3,500 行 Python 的 **agentic RL 基础设施**：通过 OpenAI 兼容 **API Gateway 代理** 捕获真实 agent harness 的交互轨迹，用 **verl + vLLM** 做策略更新；支持本地 Controller 与 **Kubernetes Job** rollout。
- **沉淀到 wiki：** 是 → [`wiki/entities/agent-lightning.md`](../../wiki/entities/agent-lightning.md)

## 开源状态核查（2026-09-19）

| 项 | 值 |
|----|-----|
| **开放程度** | **已开源** — MIT；完整 Python 包 `agentlightning/`、示例 `examples/`、文档站 `docs/`、训练脚本与 `scripts/setup_verl.sh` |
| Stars / Forks（API） | ~18,340 / ~1,616 |
| 默认分支 | `main` |
| PyPI 包版本（`pyproject.toml`） | **1.0.1** |
| 主要语言 | Python 3.12+ |
| 权重 / 模型 | **不自带** 基座权重；示例可接 Qwen 等，训练依赖 **verl + vLLM** GPU 栈 |
| 项目页 | [Microsoft Research — Agent Lightning](https://www.microsoft.com/en-us/research/project/agent-lightning/) + [官方文档站](https://microsoft.github.io/agent-lightning/stable/) |
| v0.x 历史 | v1.0 完全重构；旧版见 [v0.x 分支](https://github.com/microsoft/agent-lightning/tree/v0.x) |

步骤 2.5：项目页与 GitHub README 均链到本仓与文档站；Quick Start 给出可运行 **Calc-X** 本地训练路径 → **已开源**。

## 仓库概况（README / API）

| 字段 | 值 |
|------|-----|
| 描述 | The absolute trainer to light up AI agents. |
| 创建 | 2025-06-18 |
| 关键词 | agentic-ai, ai-agents, reinforcement-learning |

## v1.0 架构三组件（README）

| 组件 | 职责 |
|------|------|
| **Trainer** | 运行 `verl` 与 vLLM，构建训练样本并更新策略 |
| **API Gateway** | 代理模型请求，捕获训练数据（**agent harness 零改动**） |
| **Rollout Controller** | 本地或 Kubernetes Job 启动 agent rollout |

## 仓库结构要点

| 路径 | 角色 |
|------|------|
| `agentlightning/` | 核心包：`client`、`server`、`controller`、`config`、`verl` 集成 |
| `agentlightning/controller/` | `local_reconciler.py`、`k8s_reconciler.py` — 本地 / K8s rollout |
| `examples/` | Calc-X、GSM8K、ScienceWorld、Search-R1、LLM-in-Sandbox、SWE（`swe_smith`）等 |
| `scripts/setup_verl.sh` | 安装 pin 版 `verl` GPU 依赖 |
| `docs/` | MkDocs 文档源；发布为 `microsoft.github.io/agent-lightning` |

## Quick Start 入口（文档 01-quick-start）

单机单 A100 本地路径（`runner_type=local`）：

1. 完成 Installation + `scripts/setup_verl.sh`
2. 准备 `examples/calc_x/data/`（train/test parquet 等）
3. `examples/calc_x/run_local.sh` — 依次启动 Ray、`verl`/vLLM、`agl-server`（8181）、`agl-controller`、Calc-X 训练

## 公开结果摘要（README）

- **Coding Agent**：6K 样本端到端 Qwen3.5-9B，SWE-bench Verified **41.8% → 56.4%**（+14.6 pp），含数据清洗与 reward hacking 防护脚本
- 另报 Search R1、LLM-in-Sandbox 等域上纯 RL 提升（见 benchmark 图）

## 对 wiki 的映射

- 实体页：[wiki/entities/agent-lightning.md](../../wiki/entities/agent-lightning.md)
- 项目页归档：[sources/sites/agent-lightning-microsoft-research.md](../sites/agent-lightning-microsoft-research.md)
- 技术报告源：[sources/papers/agent_lightning_v1_technical_report.md](../papers/agent_lightning_v1_technical_report.md)
