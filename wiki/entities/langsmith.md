---
type: entity
tags: [entity, llm-agents, observability, evals, langsmith, commercial-saas]
status: complete
updated: 2026-09-29
summary: "LangSmith 是 LangChain 公司的 agent 评测、tracing、调试与部署平台（smith.langchain.com）；OSS 框架可独立运行，生产可观测常选用此商业层。"
related:
  - ./langchain.md
  - ./langgraph.md
  - ./deep-agents.md
  - ./langchain-ai.md
  - ../concepts/ai-auto-research.md
sources:
  - ../../sources/sites/langsmith.md
  - ../../sources/sites/langchain-com-ecosystem.md
---

# LangSmith

**LangSmith**（[langchain.com/langsmith](https://www.langchain.com/langsmith)，控制台 [smith.langchain.com](https://smith.langchain.com/)）是 LangChain 生态的 **商业** 平台：**evals、tracing、调试可视化**，以及文档中的 **agent deployment** 路径。LangChain / LangGraph / Deep Agents **不依赖** LangSmith 即可开发，但官方文档将 **复杂 agent 排障与上线** 强关联到此产品。

## 一句话定义

**给 LangGraph/LangChain agent 用的「实验记录 + 线上 trace + 评测」托管面，不是开源运行时本身。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| LangSmith | LangSmith | 本页商业平台 |
| OSS | Open Source Software | LC/LG/Deep Agents 仍 MIT 开源 |
| Eval | Evaluation | 数据集与指标驱动的 agent 评测 |
| Trace | Distributed Trace | 节点级执行路径与状态变迁 |

## 为什么重要

- **与 LangGraph 成对出现：** LangGraph README 将 **debugging / deployment** 指向 LangSmith；读 LangGraph 工程实践时必须分清 **开源图运行时** vs **付费观测/托管**。
- **机器人/agent 运维：** 长时 tool-calling、RAG 链路需要 **latency、失败步骤、检索 hit** 的可视化；LangSmith 是官方一体化选项，亦可自建 OTel。
- **研究/Auto-Research 侧：** 与 [AI Auto-Research](../concepts/ai-auto-research.md) 中的 **评测与可复现** 需求同轴，但是 **产品化 SaaS** 而非论文基准。

## 核心结构

| 模块（文档归纳） | 作用 |
|------------------|------|
| Tracing / Debugging | 执行路径、state 变迁、runtime metrics |
| Evals | 数据集、评分器、回归对比 |
| Deployment | 长时 stateful agent 托管（商业） |
| 文档 | [docs.langchain.com/langsmith](https://docs.langchain.com/langsmith/home) |

## 工程实践

| 场景 | 做法 |
|------|------|
| **本地开发** | 可选关闭或限流 trace；API key 写入环境变量（以文档为准） |
| **生产** | 评估数据驻留、合规与计费；与自建 observability 对比选型 |
| **不用 LangSmith** | LangChain 仍可用；需自备 log/trace/eval 管线 |
| **源码运行时序图** | **不适用**（托管控制台与 SaaS 后端非 OSS 可运行入口） |

## 局限与风险

- **商业锁定风险：** trace/eval/deployment 深度用 LangSmith 后，迁移成本高于纯 OSS 栈。
- **≠ 训练平台：** 不负责 RL 训练或机器人 sim；只做 **LLM/agent 应用** 观测。

## 关联页面

- [LangChain](./langchain.md)
- [LangGraph](./langgraph.md)
- [Deep Agents](./deep-agents.md)
- [langchain-ai](./langchain-ai.md)

## 参考来源

- [`sources/sites/langsmith.md`](../../sources/sites/langsmith.md)
- [`sources/sites/langchain-com-ecosystem.md`](../../sources/sites/langchain-com-ecosystem.md)

## 推荐继续阅读

- [LangSmith 文档](https://docs.langchain.com/langsmith/home)
- [产品页](https://www.langchain.com/langsmith)
