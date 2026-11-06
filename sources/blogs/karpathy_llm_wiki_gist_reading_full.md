# Karpathy LLM Wiki Gist 全文通读

> 来源归档（原文通读 · 支撑 `wiki/references/llm-wiki-karpathy.md`）

- **标题：** LLM Wiki
- **作者：** Andrej Karpathy
- **类型：** blog / idea file（GitHub Gist）
- **原文：** <https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f>（2026-11 通过 raw 端点获取全文，约 12 KB）
- **一句话说明：** 一份刻意抽象的"想法文件"：让 LLM 增量构建并维护持久 markdown wiki（raw sources / wiki / schema 三层；ingest / query / lint 三操作），知识编译一次、持续更新，而非每次 RAG 重新检索。
- **核对范围：** 通读 Gist 全文：核心想法、示例场景、Architecture、Operations、Indexing and logging、Optional: CLI tools、Tips and tricks、Why this works、Note 各节。未阅读 Gist 评论区，未实际运行文中提到的 qmd、Obsidian 插件；v2 扩展由 rohitg00 的另一份归档覆盖，不在本次范围。
- **通读：** 已完成

## 核心摘录（均出自原文）

- **核心想法：** 常见 RAG / NotebookLM 式体验是查询时检索片段、每次从零重新发现知识，无积累；本模式让 LLM 增量维护"位于用户与原始资料之间"的持久 wiki，新资料进来时更新实体页、修订主题摘要、标注与旧结论的矛盾。wiki 是 persistent, compounding artifact。
- **分工：** 人负责选源、探索、提问；LLM 负责摘要、交叉引用、归档、bookkeeping。作者一侧开 LLM 代理、一侧开 Obsidian："Obsidian 是 IDE，LLM 是程序员，wiki 是代码库"。
- **三层架构：** Raw sources（不可变，LLM 只读）；Wiki（LLM 全权拥有的 markdown 页）；Schema（CLAUDE.md / AGENTS.md，规定结构、约定与工作流，由人与 LLM 共同演化）。
- **Ingest：** 读源 → 与用户讨论要点 → 写摘要页 → 更新 index → 更新相关实体 / 概念页 → 追加 log；一个源可触达 10–15 页；作者偏好逐个摄入并保持参与，也可批量。
- **Query：** 搜相关页、综合带引用的回答，形式可为 markdown、对比表、Marp 幻灯、matplotlib 图；好答案应回写为新页。
- **Lint：** 查矛盾、过时结论、孤儿页、缺页概念、缺交叉引用、可用 web 搜索补的数据缺口。
- **导航文件：** index.md 内容导向目录（中等规模约 100 源、数百页时无需 embedding RAG）；log.md 追加式时间线，统一前缀（如 `## [2026-04-02] ingest | 标题`）可用 grep 解析。
- **可选工具：** qmd（本地 markdown 搜索，BM25/向量混合 + LLM 重排，有 CLI 与 MCP）；也可让 LLM 自写简易搜索脚本。
- **技巧：** Obsidian Web Clipper、本地下载图片、graph view 看 hub / 孤儿、Marp、Dataview、wiki 即 git 仓库。
- **为何有效：** 维护成本（bookkeeping）压垮人类维护的 wiki，LLM 不厌倦且一次可改约 15 个文件；与 Bush 的 Memex（1945）精神相近，LLM 解决了"谁来维护"。
- **Note：** 文档刻意抽象，只传达模式；目录结构、schema、工具均可选、模块化，应交给自己的 LLM 代理共同实例化。
