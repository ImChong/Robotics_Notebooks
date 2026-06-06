# Karpathy — LLM Wiki 模式（Gist 原始资料）

- **类型**：blog / idea file（GitHub Gist）
- **作者**：Andrej Karpathy
- **原始链接**：<https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f>
- **收录日期**：2026-06-06
- **关联人物页**：[karpathy.ai](../sites/karpathy-ai.md)

## 一句话

Karpathy 提出的 **LLM 维护持久 wiki** 模式：原始资料只读入库，LLM 增量编译结构化 markdown（实体页、概念页、对比、索引、日志），通过 ingest / query / lint 三类操作让知识 **复利积累**，而非每次 RAG 重新检索。

## 为什么值得保留

- **本仓库 schema 的直接思想来源**：`sources/` → `wiki/` → `schema/` 三层、`log.md` / `index.md` 分工、ingest 一次触达多页、query 答案回写 wiki，均可在 Gist 原文找到对应表述。
- **与 RAG / NotebookLM 的对比清晰**：强调 wiki 是 **persistent, compounding artifact**；cross-reference 与矛盾标记应被 **编译一次并保持更新**。
- **可选工具链提示**：Obsidian、Marp、Dataview、qmd 混合检索等——本项目在 `make export` / BM25 搜索等处的工程化是其 robotics 域实例化。

## 核心结构（Gist 归纳）

### 三层架构

1. **Raw sources** — 人类策展的原始资料，LLM 只读不改。
2. **Wiki** — LLM 生成的 markdown 页（summary、entity、concept、comparison、overview、synthesis）。
3. **Schema** — `AGENTS.md` / `CLAUDE.md` 类文件，约定页面类型、链接规范与 ingest/query/lint 工作流。

### 三类操作

| Op | 作用 |
|----|------|
| **Ingest** | 读新资料 → 写 summary → 更新 index → 触达 10–15 相关页 → 追加 log |
| **Query** | 在 wiki 上搜索综合；**好答案应回写为新页**（comparison、analysis、connection） |
| **Lint** | 矛盾、过时 claim、orphan、缺页概念、缺 cross-reference、可 web 补洞 |

### 两个导航文件

- **index.md** — 内容导向目录（页链 + 一句话摘要 + 分类）。
- **log.md** — 时间线 append-only（ingest / query / lint 记录）。

### 设计动机（原文要点）

- 维护 wiki 的痛点是 **bookkeeping**（交叉引用、摘要更新、矛盾标记），LLM 可低成本批量维护。
- 人类负责 **策展来源、提好问题、判断意义**；LLM 负责 **归纳、链接、归档**。
- 精神关联 Vannevar Bush **Memex**（1945）：个人 curated 知识库 + associative trails。

## 对 wiki 的映射

- 升格页面：[wiki/references/llm-wiki-karpathy.md](../../wiki/references/llm-wiki-karpathy.md)
- 作者实体：[wiki/entities/andrej-karpathy.md](../../wiki/entities/andrej-karpathy.md)
- 项目实现：[schema/ingest-workflow.md](../../schema/ingest-workflow.md)、[AGENTS.md](../../AGENTS.md)

## 参考链接

- Gist：<https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f>
- Karpathy 个人页：<https://karpathy.ai/>
