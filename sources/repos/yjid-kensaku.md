# Kensaku（yjid/kensaku）官方仓库与技术文档通读

- **类型：** repo / docs
- **标题：** Kensaku — Fast, local desktop file and content search
- **仓库：** https://github.com/yjid/kensaku
- **原文：** [README](https://github.com/yjid/kensaku/blob/9585797888e8b7260daf9d1334ea1c8282dee2f3/README.md)、[Architecture](https://github.com/yjid/kensaku/blob/9585797888e8b7260daf9d1334ea1c8282dee2f3/docs/ARCHITECTURE.md)、[Benchmarks](https://github.com/yjid/kensaku/blob/9585797888e8b7260daf9d1334ea1c8282dee2f3/docs/BENCHMARKS.md)、[Agent API](https://github.com/yjid/kensaku/blob/9585797888e8b7260daf9d1334ea1c8282dee2f3/docs/AGENTS.md)
- **核查版本：** 上游 `master` @ `9585797888e8b7260daf9d1334ea1c8282dee2f3`，2026-11-10；文档自称发布版本 v1.0.16
- **许可：** MIT（仓库 LICENSE 与 README）
- **开源状态：** 已开源；Tauri/React 前端、Rust 索引与检索代码、CLI、MCP/HTTP 服务、测试及基准 harness 可访问。OCR/语义模型资源依配置额外下载。
- **运行入口：** `cli/src/main.rs` → `kensaku_lib::cli::run` → `src-tauri/src/cli.rs`；GUI 入口在 `src-tauri/`，MCP 服务在 `src-tauri/src/agent/mcp.rs`。
- **核对范围：** 阅读官方 README 全文（功能、平台、安装、MCP、测量条件、隐私、已知局限）、`docs/ARCHITECTURE.md`（进程架构、存储、检索四层、查询、索引、OCR、Agent 与产品边界）、`docs/BENCHMARKS.md`（基准工具、机器/语料、跨平台限制）、`docs/AGENTS.md`；并查看 `cli/src/main.rs`、`src-tauri/src/cli.rs`、`src-tauri/src/agent/mod.rs`、`src-tauri/src/agent/mcp.rs`、`src-tauri/src/indexer/mod.rs` 与 `src-tauri/src/search/mod.rs` 的关键入口及调用结构。未在本地编译或运行 benchmark。
- **通读：** 已完成
- **知识页：** [Kensaku：本地全文与语义搜索](../../wiki/entities/kensaku.md)

## 原文要点与工程证据

1. **定位**：跨 Windows、macOS、Linux 的本地文件检索/启动器，覆盖文件名、文档全文、代码、截图 OCR、CJK 和本地语义匹配；GUI 使用 Tauri 2 + React 19，核心 Rust。Agent 可经 CLI、MCP stdio 或 loopback HTTP 搜索同一索引。
2. **数据流**：`jwalk` 遍历和排除规则 → 有界提取池（文本/Office/PDF/OCR）→ 一次读取供给 Tantivy BM25 与语义单元 → SQLite 元数据和内存 arena；`notify` 监听增量变化。四层查询依次为文件名模糊匹配、全文 BM25、静态语义向量、folded/trigram 回退，之后按 RRF 与时间/打开频率信号融合排序。
3. **CLI 实现锚点**：`cli/src/main.rs` 调用 `kensaku_lib::cli::run`；`src-tauri/src/cli.rs` 的 `run_search` 通过 `open_index` 获取 `HeadlessIndex` 并调用 `index.search`；`src-tauri/src/agent/mod.rs` 的 `HeadlessIndex::search_page` 优先转发 daemon 请求，失败后执行 `SearchDispatcher::search_response_fast_sorted_with_snippets`。`mcp.rs` 的 `serve` 提供 JSON-RPC 请求分发和受限工具调用。
4. **测量边界**：官方 `docs/BENCHMARKS.md` 报告 Windows 11 / 100,201 个空文本文件、默认 16 线程的文件名查询平均 17.2–18.8 ms，设置 4 线程为 3.1 ms；Linux 8 vCPU 本地 SSD 的旧版热缓存参考约 1.3 ms。12.2 MB 是这一规模的 `FileEntry` 元数据估计，**不包含**全文索引、向量表或桌面 WebView。
5. **重要限制**：默认 `potion-base-8M` 风格静态 embedding，并非运行大型云端 LLM；视觉相似图像的 CLIP 正式检索路径尚未接通；构建产物尚未代码签名。默认无遥测/云搜索，但更新检查和可选 OCR/模型下载会联网。
6. **安全边界**：Agent 工具默认只读，`reindex` 需显式开启并有 daemon；HTTP 绑定 loopback 且 bearer token，`agent_scoped_roots` 限定路径，排除凭据/敏感文件。索引含文件名及提取文本，仍应视为敏感数据。

## 与本知识库的关系

用于机器人研发资料（PDF、实验日志、代码与 CUDA 错误截图）的本地检索，以及为外部编码 Agent 提供只读检索上下文；**不是**机器人策略、训练框架、VLA 或自动生成答案的 RAG 服务。详情参见 [Kensaku](../../wiki/entities/kensaku.md)。

## 一手资料导航

- [官方源码](https://github.com/yjid/kensaku)
- [Architecture](https://github.com/yjid/kensaku/blob/master/docs/ARCHITECTURE.md)
- [Benchmarks](https://github.com/yjid/kensaku/blob/master/docs/BENCHMARKS.md)
- [Agent API](https://github.com/yjid/kensaku/blob/master/docs/AGENTS.md)
- [安装说明](https://github.com/yjid/kensaku/blob/master/docs/INSTALL.md)
