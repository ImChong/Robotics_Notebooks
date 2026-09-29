# LangGraph（langchain-ai/langgraph）

> 来源归档（GitHub README + 文档交叉，2026-09-29 核查）

- **标题：** LangGraph
- **类型：** repo
- **组织：** [langchain-ai](https://github.com/langchain-ai)
- **代码：** <https://github.com/langchain-ai/langgraph>
- **产品页：** <https://www.langchain.com/langgraph>
- **文档：** <https://docs.langchain.com/oss/python/langgraph/overview>
- **JS 仓：** <https://github.com/langchain-ai/langgraphjs>
- **许可：** MIT（以仓库 `LICENSE` 为准）
- **入库日期：** 2026-09-29
- **一句话说明：** **有状态 agent 的低层编排框架**：持久执行、人机回路、短期/长期记忆；面向长时 workflow 与 production deployment（常与 LangSmith 联用）。

## 开源边界

| 已发布 | 备注 |
|--------|------|
| `langchain-ai/langgraph` | **已开源**；`pip install -U langgraph` |
| LangGraph.js | **已开源**；独立仓库 |
| LangSmith Deployment | **商业**；文档路径 `docs.langchain.com/langsmith/deployments` |

## README 要点（归纳）

- **定位：** *Low-level orchestration framework for building stateful agents.*
- **能力轴：** Durable execution · Human-in-the-loop · Comprehensive memory · Debugging with LangSmith · Production deployment。
- **与 Deep Agents：** README 指向 Deep Agents 为 **更高层**、基于 LangGraph 的「开箱 agent harness」。
- **与 LangChain：** LangChain 负责组件与快速原型；**复杂可控图** 升格 LangGraph。

## 对 wiki 的映射

- [`wiki/entities/langgraph.md`](../../wiki/entities/langgraph.md)
- 互链 [`langchain.md`](../../wiki/entities/langchain.md)、[`deep-agents.md`](../../wiki/entities/deep-agents.md)、[`langsmith.md`](../../wiki/entities/langsmith.md)
