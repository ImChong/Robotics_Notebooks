# LangChain（langchain-ai/langchain）

> 来源归档（以 GitHub 主仓 README 与 `libs/` 结构为准，2026-09-29 核查）

- **标题：** LangChain
- **类型：** repo
- **组织：** [langchain-ai](https://github.com/langchain-ai)
- **代码：** <https://github.com/langchain-ai/langchain>
- **许可：** MIT（以仓库 `LICENSE` 为准）
- **文档：** <https://docs.langchain.com/oss/python/langchain/overview>（见 [`sources/sites/langchain-docs.md`](../sites/langchain-docs.md)）
- **入库日期：** 2026-09-29
- **一句话说明：** **Agent 工程平台**：用可互操作的组件与海量集成，把 LLM、工具、向量库、检索器串成 **agent 与 LLM 应用**；monorepo 内 `langchain_v1` 为当前主线包，复杂可控工作流推荐 **LangGraph**（独立仓），观测与评测走 **LangSmith** 生态。

## 开源边界（步骤 2.5）

| 已发布 | 备注 |
|--------|------|
| 主仓 `langchain-ai/langchain` | **已开源**；README 与 `libs/` 可克隆运行 |
| PyPI `langchain` 等 | **已发布**；`uv add langchain` / pip 安装路径与版本以 PyPI 为准 |
| LangGraph / LangChain.js | **已开源**；独立仓库（README 交叉链接） |
| LangSmith 托管 | **商业**；框架可 standalone，生产观测非必须但文档强关联 |

## README / monorepo 要点（归纳）

- **定位：** *The agent engineering platform* — 链式组合 **模型、embedding、vector store、tools、retrievers** 等，强调 **模型可替换** 与 **快速原型**。
- **Quickstart：** `init_chat_model("provider:model")` 统一聊天模型入口（README 示例）。
- **生态（文档叙述）：** Deep Agents（高层 agent 模式）、LangGraph（低层可控编排）、Integrations 目录、LangSmith（evals / observability）、LangSmith Deployment（长时 stateful agent 部署）。
- **`libs/` 结构：**
  - `core/` — 核心抽象
  - `langchain/` — langchain-classic
  - `langchain_v1/` — 当前 `langchain` 包
  - `partners/` — 团队直维护的部分 provider 集成（OpenAI、Anthropic、Ollama 等）；多数集成已 **外迁独立仓库**
  - `text-splitters/`、`model-profiles/`、`standard-tests/`
- **与机器人知识库读者：** 常用作 **RAG / tool-calling agent 编排层**，而非运动控制或 sim 物理后端；具身栈中多出现在 **语义规划、文档 grounding、MCP/工具桥** 一侧（对照 [OpenClaw](../../wiki/entities/openclaw.md) 等 **不用 LangChain 的自研 agent 环**）。

## 与本仓库知识的关系

| 主题 | 关系 |
|------|------|
| [RAG](../../wiki/concepts/retrieval-augmented-generation.md) | document loader、retriever、chain 编排的经典工业框架之一 |
| [OpenClaw](../../wiki/entities/openclaw.md) / [ScienceDiscovery](../../wiki/entities/sciencediscovery.md) | 对照：部分 agent 栈 **刻意不依赖** LangChain/LangGraph |
| [Hermes Agent](../../wiki/entities/hermes-agent.md) | 同属 agent 运行时谱系；Hermes 偏网关+记忆 OS，LangChain 偏组件化集成与 LangGraph 编排 |
| [Easy-Vibe](../../wiki/entities/easy-vibe.md) | 教程 Stage 3 常提及 LangChain 生态 |
| awesome-physical-ai #125 | 策展索引 [`pai_awesome_resource_125_langchain.md`](pai_awesome_resource_125_langchain.md) |

## 对 wiki 的映射

- 升格 **[`wiki/entities/langchain.md`](../../wiki/entities/langchain.md)**（由 [`painode-125-langchain.md`](../../wiki/entities/painode-125-langchain.md) 合并为 canonical 实体，保留 painode 页作清单锚点）。
- 轻量更新 **[`wiki/concepts/retrieval-augmented-generation.md`](../../wiki/concepts/retrieval-augmented-generation.md)** 链到深度实体与本文档源。
