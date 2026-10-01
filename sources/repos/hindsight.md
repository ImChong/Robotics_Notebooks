# Hindsight（vectorize-io/hindsight）

> 来源归档

- **标题：** Hindsight — Agent Memory That Learns
- **类型：** repo（Agent 记忆 / 学习系统 + API + 客户端）
- **作者：** Vectorize.io
- **链接：** https://github.com/vectorize-io/hindsight
- **文档 / 云：** https://hindsight.vectorize.io/ — [`sources/sites/hindsight-vectorize.md`](../sites/hindsight-vectorize.md)
- **论文：** https://arxiv.org/abs/2512.12818
- **许可：** MIT
- **入库日期：** 2026-10-01
- **Trendshift（用户触发，2026-10）：** 约 **+20.6k stars/月**；GitHub API 2026-10-01 约 **44.0k** stars
- **一句话说明：** 面向 Agent 的 **会学习的记忆系统**（非单纯聊天历史 RAG）：核心操作 **retain / recall / reflect**，含 memory banks、observations、mental models；Docker 自托管或 embedded Python；提供 LLM wrapper、MCP、编码 Agent 文档 skill 与 LongMemEval 等基准叙事。
- **为什么值得保留：** 与 [Hermes Agent](../../wiki/entities/hermes-agent.md) 运行时记忆、[Deep Agents](../../wiki/entities/deep-agents.md) 文件系统记忆形成对照；对 **长期 ingest / 多会话维护** 的「经验沉淀」选型有参考价值。
- **沉淀到 wiki：** 是 → [`wiki/entities/hindsight.md`](../../wiki/entities/hindsight.md)

## README 要点（归纳）

- **Quick Start：** Docker `ghcr.io/vectorize-io/hindsight:latest`（8888 API / 9999 UI）；PyPI `hindsight-api`、`hindsight-client`；NPM `@vectorize-io/hindsight-client`。
- **编码 Agent：** `npx skills add https://github.com/vectorize-io/hindsight --skill hindsight-docs`
- **基准：** README 称 LongMemEval SOTA；live 看板 <https://benchmarks.hindsight.vectorize.io/>
- **集成：** LLM Wrapper（2 行代码）、MCP Server、多平台 clients。

## 开源状态

- **已开源（MIT）** — 服务端与客户端；Hindsight Cloud 为托管选项。

## 对 wiki 的映射

| 目标 | 链接 |
|------|------|
| 实体页 | [`wiki/entities/hindsight.md`](../../wiki/entities/hindsight.md) |
