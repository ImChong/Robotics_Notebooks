# Kensaku（yjid/kensaku）：本地文件检索与 Agent 索引

> 官方仓库与技术文档来源归档

- **标题：** Kensaku（検索）
- **类型：** repo
- **作者：** Yash Jangid（仓库 LICENSE 署名）
- **代码：** <https://github.com/yjid/kensaku>
- **文档：** <https://github.com/yjid/kensaku/tree/master/docs>
- **发行版：** <https://github.com/yjid/kensaku/releases>
- **许可证：** MIT
- **核对版本：** `yjid/kensaku@9585797888e8b7260daf9d1334ea1c8282dee2f3`（2026-11-09 的 `master` HEAD）；README / ARCHITECTURE.md 对 v1.0.16 的描述
- **入库日期：** 2026-11-10
- **核对范围：** 通读官方 README.md、docs/ARCHITECTURE.md、docs/BENCHMARKS.md、docs/AGENTS.md、docs/EXTRACTORS.md；核对 LICENSE、cli/src/main.rs、src-tauri/src/cli.rs 的 `run`/MCP 分发、src-tauri/src/agent/mod.rs 的 `HeadlessIndex`/检索调用、src-tauri/src/agent/mcp.rs 的 `tools/call` 与 `search_page` 分支、src-tauri/src/indexer/mod.rs 的 `start_indexing`、src-tauri/src/search/mod.rs 的检索入口。源码仅核对相关调用链，并未逐行通读全部代码或实际运行基准。
- **通读：** 已完成
- **一句话说明：** 跨 Windows、macOS、Linux 的离线优先文件检索工具，用 Rust/Tauri 建立文件名、全文和静态向量索引，支持 PDF/Office/代码抽取、OCR、CJK 分词、CLI 与 MCP 只读 Agent 接入。
- **为什么值得保留：** 可用于机器人研发资料、ROS 代码仓库、实验日志、失败截图和设计文件的本地跨格式检索；同时是如何把本地检索索引安全暴露给编码代理的工程案例。
- **开源状态：** 官方提供完整桌面/CLI/MCP 代码和发布包；并非论文训练代码或机器人控制算法。
- **沉淀到 wiki：** 是 → [Kensaku 实体节点](../../wiki/entities/kensaku.md)

## 原文要点与技术机制

1. **三种前端，同一索引：** Tauri 2/React 桌面界面、`kensaku-cli` 和 `kensaku-cli mcp` 共享本地索引。Unix 下优先连 `kensaku-daemon`；无 daemon 时 CLI/MCP 使用只读索引回退；Windows GUI 使用进程内索引管理器。
2. **索引机制：** `jwalk` 扫描配置目录，监听增量变更；SQLite WAL 持久化元数据；Tantivy 对文档全文做 BM25；另以量化静态向量表进行语义检索。支持文件名模糊匹配、精确内容与折叠/子串候选，按 RRF 融合重排。
3. **文件内容：** 支持 PDF、DOCX/PPTX/XLSX、EPUB、纯文本、代码等；默认 PP-OCRv6 为图片/扫描文本抽字，Indra 和 Tesseract 为可选引擎。文件默认上限 100 MB、抽取文本 4 MB、语义块 48 个；大文件仍可做文件名检索。
4. **中文与混合语言：** 中文、日文、韩文分词使用 unigram/bigram，兼顾混合脚本边界；并不依赖云端语言模型。
5. **Agent 接入：** `kensaku-cli mcp` 暴露 `search`、`read_snippet`、`index_state`、`health_check`、`regex_search`、`find_symbol`；`reindex` 默认为禁用，需显式配置。HTTP 仅监听 loopback，受 bearer token 保护，并禁用 `reindex`。
6. **安全边界：** 默认跳过隐藏目录、常见构建目录和凭据文件；搜索与片段读取受可配置根目录约束，但本地索引本身储存文件路径和抽取内容，需要按敏感数据管理。

## 性能核对（仅项目方基准，非本次实测）

| 场景 | 官方结果与限制 |
|---|---|
| Windows 11、100,201 个合成文件名、默认 16 线程 | 文件名模糊检索平均 17.2–18.8 ms |
| 同一 Windows 测试、4 线程 | 平均 3.1 ms |
| Linux 8 vCPU、SSD、约 10 万文件、热缓存 | 历史基准平均 1.3 ms（不同硬件，不可直接横比） |
| 100,201 个 FileEntry 的元数据空间 | 12.2 MB；不含 SQLite/Tantivy/向量索引与 GUI |

测量说明见 [官方 BENCHMARKS.md](https://github.com/yjid/kensaku/blob/master/docs/BENCHMARKS.md)，其 `bench` 测试并未覆盖端到端 GUI/MCP 的全部时延；README 中的其他搜索工具对比也不属于严格同条件横评。

## 可复现路径与源码入口

```bash
git clone https://github.com/yjid/kensaku
cd kensaku
bun install
bun run tauri build
kensaku-cli search "docker error"
kensaku-cli mcp
```

依赖 Rust stable、Bun 1.1+ 与平台 WebView 库；`kensaku-cli` 命令需先安装或找到构建后的 CLI 程序，且必须先在桌面端建立索引。更稳妥的首次使用方式是从 Releases 下载对应系统安装包。

- **MCP 入口：** [cli/src/main.rs](https://github.com/yjid/kensaku/blob/master/cli/src/main.rs) → [src-tauri/src/cli.rs](https://github.com/yjid/kensaku/blob/master/src-tauri/src/cli.rs) `mcp::serve` → [src-tauri/src/agent/mcp.rs](https://github.com/yjid/kensaku/blob/master/src-tauri/src/agent/mcp.rs) `tools/call` → [src-tauri/src/agent/mod.rs](https://github.com/yjid/kensaku/blob/master/src-tauri/src/agent/mod.rs) `search_page` → [src-tauri/src/search/mod.rs](https://github.com/yjid/kensaku/blob/master/src-tauri/src/search/mod.rs)。
- **建索引入口：** [src-tauri/src/indexer/mod.rs](https://github.com/yjid/kensaku/blob/master/src-tauri/src/indexer/mod.rs) `start_indexing`；具体抽取、监听与分词模块分别位于 `indexer/extract/`、`indexer/watcher.rs`、`indexer/tokenizer.rs`。

## 已发布与限制

- 已发布：跨平台桌面、文件名/全文/语义检索、OCR、MCP、CLI、按配置受限的增量索引。
- 限制：可视相似图检索在当前发行路径尚未接入；所谓按意义查找基于本地静态向量，不等于调用 LLM；Windows/macOS 未签名，首次安装可能提示警告；OCR 引擎对扫描格式的覆盖不完整。
- 隐私：官方描述为不上传索引或查询，仅有更新检查及可选模型/OCR 下载请求。外部抽取器为显式配置的本机子进程，**不是 OS 沙箱**，只应启用可信程序。

## 对 wiki 的映射

- 主节点：[Kensaku：本地文件搜索与 MCP 检索](../../wiki/entities/kensaku.md)。
- 比较关系：[CodeGraph](../../wiki/entities/codegraph.md) 更侧重源码符号/静态关系图；Kensaku 更侧重本地文件与内容检索。
- 机器人应用：定位 ROS 2 工作区日志、机器人调试截图中的错误字串、PDF/标定报告及配置文件；均属于可行工作流推导，并非官方机器人专项评测结论。
