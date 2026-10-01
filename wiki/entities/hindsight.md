---
type: entity
tags:
  - llm-agents
  - memory
  - rag
  - open-source
  - mcp
status: complete
updated: 2026-10-01
related:
  - ./ecc.md
  - ./hermes-agent.md
  - ./deep-agents.md
  - ./agent-reach.md
  - ../concepts/ai-agent-evaluation.md
  - ../../schema/ingest-workflow.md
sources:
  - ../../sources/repos/hindsight.md
  - ../../sources/sites/hindsight-vectorize.md
summary: "Hindsight（vectorize-io/hindsight）是 MIT 开源的 Agent 记忆系统：强调 learn 而非只 remember，提供 retain/recall/reflect、memory banks 与 LongMemEval 基准叙事；Docker 自托管、LLM wrapper 与 MCP，2026-10 Trendshift 约 +20.6k/月。"
---

# Hindsight

**Hindsight**（[vectorize-io/hindsight](https://github.com/vectorize-io/hindsight)，文档 [hindsight.vectorize.io](https://hindsight.vectorize.io/)）是 **Agent 长期记忆与学习** 的开源栈（MIT）：相对「把聊天记录塞进 RAG」，它强调 **从经验中更新行为** — 通过 **retain / recall / reflect** 与 **observations、mental models、memory banks** 等概念组织记忆。

## 一句话定义

给 Agent 接一层 **会随使用变聪明的记忆服务**，用 **结构化 retain–recall–reflect** 替代纯向量检索对话历史。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RAG | Retrieval-Augmented Generation | 传统对话检索增强；Hindsight 自称超越其局限 |
| MCP | Model Context Protocol | 官方 MCP Server 集成路径 |
| API | Application Programming Interface | 自托管默认 8888；UI 9999 |
| LLM | Large Language Model | wrapper 可两行接入现有 Agent |

## 核心信息

| 字段 | 内容 |
|------|------|
| 论文 | [arXiv:2512.12818](https://arxiv.org/abs/2512.12818) |
| 开源状态 | **已开源** — Docker / PyPI / NPM clients |
| Stars（2026-10-01） | ~44.0k（Trendshift 约 +20.6k/月） |
| 基准 | [benchmarks.hindsight.vectorize.io](https://benchmarks.hindsight.vectorize.io/)（LongMemEval 等） |

## 为什么重要（对本知识库读者）

- **LLM Wiki 维护：** 多次 ingest 的 **教训、lint 踩坑、选型结论** 适合 ** retain 到 memory bank** 而非只留在 chat；与 [Karpathy 模式](../references/llm-wiki-karpathy.md)「写回 wiki」互补 — wiki 是 canonical，Hindsight 是 **会话间 Agent 私有经验**。
- **运行时对照：** [Hermes Agent](hermes-agent.md) 自带记忆/技能；[Deep Agents](deep-agents.md) 偏文件系统；Hindsight 是 **独立 memory 微服务**。
- **编码 Agent：** `npx skills add ... --skill hindsight-docs` 供 Cursor/Claude 边写边查文档。

## 核心操作（概念）

| 操作 | 含义 |
|------|------|
| **retain** | 写入新经验/事实到 memory bank |
| **recall** | 按任务检索相关记忆 |
| **reflect** | 对记忆做归纳/更新 mental model |

## 局限

- 自托管需 **Postgres 卷** 与 LLM API key；生产拓扑见官方 docs。
- 基准分数字段随时间更新，选型以 **live benchmark 站点 + 自家任务** 为准。
- 与 wiki **源文件真相** 冲突时，以 git 中 markdown 为准。

## 关联页面

- [ECC](ecc.md) — harness 内 continuous learning
- [Hermes Agent](hermes-agent.md) — 一体化 Agent OS
- [Deep Agents](deep-agents.md) — LangChain 系 harness 记忆

## 参考来源

- [Hindsight 仓库归档](../../sources/repos/hindsight.md)
- [hindsight.vectorize.io 站点归档](../../sources/sites/hindsight-vectorize.md)
- [vectorize-io/hindsight（GitHub）](https://github.com/vectorize-io/hindsight)

## 推荐继续阅读

- 上游 [Quick Start](https://github.com/vectorize-io/hindsight#quick-start)（Docker）
- [Hindsight Cookbook](https://hindsight.vectorize.io/cookbook)
