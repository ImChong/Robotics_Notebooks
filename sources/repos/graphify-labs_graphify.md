# Graphify（Graphify-Labs/graphify）

> 来源归档

- **标题：** graphify
- **类型：** repo
- **作者 / 组织：** Graphify Labs（GitHub Organization）
- **代码：** <https://github.com/Graphify-Labs/graphify>
- **官网：** <https://www.graphify.com>
- **PyPI：** `graphifyy`（包名暂为双 y；CLI 命令仍为 `graphify`）
- **许可：** MIT
- **默认分支：** `v8`（以克隆时为准）
- **入库日期：** 2026-07-16
- **一句话说明：** 面向 20+ 编码代理 harness 的可安装 **Agent Skill + CLI**：把任意文件夹（代码、SQL schema、文档、PDF、图片、音视频等）编译成可查询的 **知识图**（非向量 RAG）；代码走 tree-sitter AST 全本地零 LLM，语义资料可走 IDE 会话或 headless API；输出 `graph.json`、`GRAPH_REPORT.md` 与交互 `graph.html`。
- **为什么值得保留：** 与 Karpathy **LLM Wiki** 模式形成鲜明对照——本仓库是 **人类策展 + 手工 ingest 的持久 wiki**；graphify 是 **从原始资料自动构图 + 图遍历查询**，宣称在混合语料上相对「每次重读原文」有数量级 token 节省；对维护本仓库（`wiki/` + `sources/` + 多语言代码）的代理工作流有直接选型价值。

## README 要点（归纳）

### 定位

- **Agent Skill：** 在 Claude Code、Cursor、Codex、Gemini CLI、OpenCode、Hermes、GitHub Copilot 等环境中触发 `/graphify`（Codex 为 `$graphify`），由技能驱动提取与建图。
- **非向量索引：** 显式 **图结构**（节点 = 概念/符号/文档实体，边 = `calls` / `imports` / `references` / `uses` 等），支持 `graphify query`、`graphify path`、`graphify explain` 等 **遍历式** 查询。
- **边置信度：** 每条边标注 `EXTRACTED`（源中显式）、`INFERRED`（解析推断）或 `AMBIGUOUS`，区分「读到」与「猜到」。

### 提取分层

| 输入类型 | 机制 | 是否离网 |
|----------|------|----------|
| 代码（~36 tree-sitter 语法） | AST + 跨文件 call/import 解析 | 是，零 API |
| 文档 `.md` 等 | 助手模型语义抽取；`[[wikilink]]` / markdown 链接成边 | 否（skill 会话或 headless key） |
| PDF / 图片 / 音视频 | 语义抽取或本地 faster-whisper 转写 | 视模式 |
| SQL / Terraform / MCP 配置等 | 可选 extra 语法包 | 多为本地 |

### 核心产出

```
graphify-out/
├── graph.html          # 浏览器交互图（vis.js）
├── GRAPH_REPORT.md     # god nodes、意外连接、建议问题
├── graph.json          # 持久图，可周后再 query
├── obsidian/           # 可选 Obsidian vault
└── wiki/               # --wiki：按社区生成 agent 可爬 markdown
```

### 技术栈（上游自述）

NetworkX + Leiden 社区检测（graspologic）+ tree-sitter + vis.js；可选 Neo4j / FalkorDB push、MCP stdio/HTTP 服务、`graphify hook install` 在 git commit 后 AST 增量重建。

### 与 Karpathy `/raw` 叙事

README 将 graphify 定位为 Karpathy 式 **`/raw` 资料夹** 的「结构化答案」：混合语料（代码 + 论文 + 图片）上宣称 **71.5×** 单次查询 token 相对重读原文（见上游 `worked/` 与 `BENCHMARKS.md`）。

### 团队工作流

- 建议 **提交 `graphify-out/`** 到 git，全队共享图；`manifest.json` 用相对路径可移植。
- `graphify cursor install` 写 `.cursor/rules/graphify.mdc`（`alwaysApply: true`），引导代理优先 `graphify query` 而非 grep 全库。
- `graphify install --project` 可将 skill 落到仓库内（如 `.cursor/`、`.agents/skills/`）。

### 基准（上游 BENCHMARKS.md 摘要）

- LOCOMO recall@10：**0.497**（对比 mem0 0.048、supermemory 0.149）
- LongMemEval-S QA：**76%**（与 dense RAG 持平）
- 纯代码建图：**0 LLM credits**

## 与本仓库知识的关系

| 主题 | 关系 |
|------|------|
| [LLM Wiki（Karpathy 模式）](../../wiki/references/llm-wiki-karpathy.md) | **对照轴**：wiki = 人工策展的持久编译知识；graphify = 自动构图 + 图查询，可覆盖「探索陌生代码库/资料堆」阶段 |
| [Superpowers（obra）](../../wiki/entities/superpowers-obra.md) | 同属 **可安装 Agent Skill** 谱系；Superpowers 管交付流程，graphify 管 **代码/资料结构理解** |
| [Agent Reach](../../wiki/entities/agent-reach.md) | 互补：Agent Reach 接 **外网读搜**；graphify 接 **本地/仓库内** 多模态语料图 |
| [Caveman](../../wiki/entities/caveman.md) | 互补：Caveman 压 **输出措辞**；graphify 压 **理解语料时的输入 token**（查询图而非重读文件） |
| 本站 `exports/link-graph.json` | 本仓库已有 **wiki 入链统计图**（`make graph`）；graphify 可额外覆盖 **sources + 脚本 + 未升格原文**，二者不互斥 |

## 对 wiki 的映射

- 沉淀 **[`wiki/entities/graphify.md`](../../wiki/entities/graphify.md)**；安装命令与版本矩阵以克隆时上游 README 为准（此处不固化易变 shell）。
