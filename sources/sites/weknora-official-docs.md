# WeKnora 官方产品文档与主页

- **标题：** WeKnora — 官方产品介绍与架构文档
- **类型：** site / 官方产品技术文档
- **项目主页：** https://weknora.weixin.qq.com/
- **技术文档：** https://weknora.weixin.qq.com/docs/
- **官方代码：** https://github.com/Tencent/WeKnora
- **核查版本：** 官方项目文档 main（README 对应 v0.9.0；2026-11-17）
- **核对范围：** 完整阅读官方 [产品介绍](https://github.com/Tencent/WeKnora/blob/main/website-docs/01-getting-started/01-introduction.md) 与 [总体架构](https://github.com/Tencent/WeKnora/blob/main/website-docs/02-architecture/01-overview.md)；另核对 README 部署说明及 RAG Pipeline 的管线结构，未做实际部署测试。
- **通读：** 已完成
- **开源结论：** 已开源。产品介绍链接官方 GitHub；`cmd/server/`、`frontend/`、`docreader/`、`docker-compose.yml`、`helm/` 均可公开阅读。

## 可复用的一手事实

- 三个主要使用路径共享知识库：RAG 问答、Agent 任务、Wiki 知识组织。
- 官方介绍明确支持混合检索（关键词 + 向量）、Rerank、可选 GraphRAG、知识来源引用，以及本地 / 私有化部署。
- 官方总体架构的默认核心是 Go App + Vue 前端 + Python docreader + PostgreSQL + Redis；Neo4j、独立向量库、联网搜索及可观测服务为可选配置。
- 官方提醒：生产系统建议内网或私有网络部署，不应不加防护直接暴露公网。

## 相关材料

- [源码一手归档](../repos/weknora.md)
- [WeKnora 结构化知识节点](../../wiki/entities/weknora.md)
- [README 快速启动](https://github.com/Tencent/WeKnora#run-with-docker-compose)
