# WeKnora：腾讯开源知识库、RAG 与 Agent 平台（官方源码归档）

> 来源归档；本文件仅提炼官方材料，不转存 README 全文。

- **标题：** WeKnora — Open-source LLM Knowledge Framework
- **类型：** repo / 开源知识管理与检索问答平台
- **机构：** 腾讯（Tencent）
- **官方仓库：** https://github.com/Tencent/WeKnora
- **官方文档：** https://weknora.weixin.qq.com/docs/
- **官网：** https://weknora.weixin.qq.com/
- **许可证：** MIT（具体第三方组件以仓库 LICENSE 与 THIRD_PARTY_NOTICES 为准）
- **核查版本：** 上游 main；README 标注 v0.9.0（2026-10-29），2026-11-17 核查
- **核对范围：** 完整阅读官方 [产品介绍](https://github.com/Tencent/WeKnora/blob/main/website-docs/01-getting-started/01-introduction.md) 与 [总体架构](https://github.com/Tencent/WeKnora/blob/main/website-docs/02-architecture/01-overview.md)；阅读 README 的 Overview、Quick Start、主要功能、部署与许可说明及官方 RAG Pipeline 文档的框架与管线组装部分；打开 [cmd/server/main.go](https://github.com/Tencent/WeKnora/blob/main/cmd/server/main.go)、[cmd/server/bootstrap.go](https://github.com/Tencent/WeKnora/blob/main/cmd/server/bootstrap.go)、[docreader/main.py](https://github.com/Tencent/WeKnora/blob/main/docreader/main.py) 核实主服务与解析入口；未逐文件阅读全部 Go 插件及做运行复现。
- **通读：** 已完成
- **开放状态：** 已开源；公开 Go 后端、Vue 前端、Python gRPC 文档解析服务、Docker Compose/Helm、CLI/SDK 源码；模型、向量库、外部服务依配置另行接入。
- **沉淀到 wiki：** [WeKnora 项目实体](../../wiki/entities/weknora.md)

## 一手资料的技术要点

1. **核心定位：** 文档与第三方数据源汇入知识库，采用 RAG 快速问答、Agent 任务执行及带来源引用的 Wiki 组织同一份知识。
2. **关键组件：** `frontend/`（Vue 3 / NGINX）、`cmd/server/` + `internal/`（Go / Gin / Agent / 任务）、`docreader/`（Python / gRPC）。默认 Docker Compose 还使用 PostgreSQL/ParadeDB 与 Redis；Neo4j 等依 profile 启用。
3. **入库：** 文件上传 → Go Handler / Service → PostgreSQL 记录及对象存储 → Asynq 任务队列 → 文档解析 gRPC → 分块 → Embedding / 关键词索引 → 可选摘要、Wiki 和图谱等富化。
4. **问答：** 意图识别 / 问题改写 → 知识检索（向量、BM25 等）→ Rerank → 上下文组装 → 模型流式生成与引用；官方 [RAG 流水线文档](https://github.com/Tencent/WeKnora/blob/main/website-docs/02-architecture/04-rag-pipeline.md) 将其拆为事件驱动插件和 SSE 输出。
5. **部署：** README 提供 `cp .env.example .env`、`docker compose pull` 和 `docker compose up -d`；也有 Helm、Lite 等形态。生产环境官方建议内网部署及加固访问控制。

## 源码入口与执行边界

- Go 服务实际入口：[`cmd/server/main.go`](https://github.com/Tencent/WeKnora/blob/main/cmd/server/main.go) 中的 `main()` 加载环境配置、构造 `container.BuildContainer`、启动 Gin HTTP 服务并处理关闭。
- 文档解析实际入口：[`docreader/main.py`](https://github.com/Tencent/WeKnora/blob/main/docreader/main.py)，其服务对外为 gRPC。
- 官方运行入口：根目录 [README](https://github.com/Tencent/WeKnora/blob/main/README.md) 的 Docker Compose 快速启动。
- 运行时序图参照 [官方总体架构文档中的文档入库调用链](https://github.com/Tencent/WeKnora/blob/main/website-docs/02-architecture/01-overview.md)，**不是** 实测 trace；未核验全部源代码分支。
- 没有查到与机器人控制算法直接集成的官方承诺；将其作为机器人研发资料/日志/手册知识管理候选，不视作 VLA、运动控制或机器人基础模型。

## 来源

- [官方 GitHub 和 README](https://github.com/Tencent/WeKnora)
- [官方产品介绍（完整核对）](https://github.com/Tencent/WeKnora/blob/main/website-docs/01-getting-started/01-introduction.md)
- [官方总体架构（完整核对）](https://github.com/Tencent/WeKnora/blob/main/website-docs/02-architecture/01-overview.md)
- [官方 RAG Pipeline](https://github.com/Tencent/WeKnora/blob/main/website-docs/02-architecture/04-rag-pipeline.md)
- [官方许可证](https://github.com/Tencent/WeKnora/blob/main/LICENSE)

## 对 wiki 的映射

- [WeKnora](../../wiki/entities/weknora.md)：平台结构、入库/问答流程、启动与工程边界
- [RAG 概念](../../wiki/concepts/retrieval-augmented-generation.md)：检索生成的通用机制
- [All-in-RAG](../../wiki/entities/all-in-rag.md)：教学资料与平台的不同定位
