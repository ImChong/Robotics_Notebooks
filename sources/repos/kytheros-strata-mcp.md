# kytheros/strata：strata-mcp 跨会话记忆

- **类型：** repo
- **原始链接：** https://github.com/kytheros/strata
- **入库日期：** 2026-11-20
- **核对范围：** README 全文、src/index.ts MCP stdio 入口、src/server.ts 创建服务与处理器注册入口
- **通读：** 已完成

## 原始资料要点（归纳，非全文转载）

- 定位：AI 编码助手共享的跨会话持久记忆，不是本地大模型推理引擎，也不是完整个人 Agent。
- Node.js CLI：npm install -g strata-mcp；strata init 检测并配置兼容的 CLI/MCP 客户端。
- SQLite/FTS5 + BM25 离线关键词检索；可选向量相似度与 RRF，写入经 actionability/specificity/relevance 三类质量门。
- src/index.ts 调用 createServer，绑定 StdioServerTransport；server.ts 接入 SqliteIndexManager、search、entity store、store_memory、search_history 等处理器。
- README 把检索 top20 命中率 94.9% 与依赖回答模型的端到端分数 79.4% 分开披露；仅为项目自报，未经本次复测。
- 项目自标 Beta；本地 SQLite、Cloudflare D1 可用，AWS/GCP 多租户部分是部署模板，须谨慎核对鉴权。

## 对 wiki 的映射

- [kytheros/strata：strata-mcp 跨会话记忆](../../wiki/entities/kytheros-strata-mcp-memory.md)
