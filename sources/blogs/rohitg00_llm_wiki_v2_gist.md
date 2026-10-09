# LLM Wiki v2 — extending Karpathy's LLM Wiki pattern

> 来源归档

- **标题：** LLM Wiki v2 — extending Karpathy's LLM Wiki pattern with lessons from building agentmemory
- **类型：** blog / idea file（GitHub Gist）
- **作者：** GitHub 用户 `rohitg00`
- **来源：** https://gist.github.com/rohitg00/2067ab416f7bbe447c1977edaaa681e2
- **上游关系：** Gist 标注 fork 自 [Karpathy 的原始 LLM Wiki](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f)，并称本文补充构建 agentmemory 的实践经验。
- **版本核查：** Gist 页面 2026-10-09 访问；显示 7 revisions，最近活跃于 2026-10-08。内容可能继续变化。
- **许可证：** 页面未声明明确许可证；此处仅归档链接与摘要，不将其标注为开源代码。
- **一句话说明：** 在“sources → wiki → schema”与 ingest/query/lint 基础上，提出知识生命周期、类型化图谱、BM25+向量+图检索、事件自动化、质量控制、协作与隐私治理等扩展方向。
- **为什么值得保留：** 对已有 LLM Wiki 实现给出从轻量手工模式走向多源检索和多代理运行的模块化路线，同时暴露置信度、遗忘、同步与自动修复等需要额外治理的设计问题。

## 核心建议（按原文归纳）

### 1. 给知识加生命周期

- **Confidence scoring：** 为事实记录来源数量、最近核验时间、矛盾情况等证据，再表达可信程度。
- **Supersession：** 新信息更新旧结论时，显式链接替代关系、保留旧版本并标记过时。
- **Forgetting / retention：** 根据知识类型、访问和新证据调整检索优先级；文章提出类似遗忘曲线的思路。
- **Consolidation tiers：** 将新观察逐步整理为 working、episodic、semantic、procedural memory。

这些是作者提出的模式，不是对每条知识自动打分或自动删除的已验证通用规则；实际系统仍需保留来源链与人工可追溯性。

### 2. 从页面链接扩展为类型化知识图谱

文章建议从实体中抽取 people / projects / libraries / concepts 等类型，并用 `uses`、`depends on`、`contradicts`、`supersedes` 等具名关系表达连接语义；查询时沿关系遍历依赖与影响链。图谱用于补充页面导航，不取代可阅读的 wiki。

### 3. 混合搜索与自动化维护

- **Hybrid retrieval：** 组合 BM25 关键词、向量语义和图遍历结果，再用 Reciprocal Rank Fusion（RRF）融合排序。
- **Event-driven hooks：** 在新来源进入、会话开始/结束、查询、知识写入及定时维护时触发抽取、回写或 lint。
- **Quality/self-correction：** 建议对产物做质量检查、修复孤儿与断链，并提出矛盾解决候选。
- **Crystallization：** 将完成的研究、调试或分析过程整理为可溯源的一等 wiki 资料，而不让经验只留在聊天记录。

### 4. 多代理协作、隐私与输出形态

文章讨论多代理同步、私有/共享知识范围、协作进度、入库时的敏感信息过滤、变更审计和可逆批量操作；知识库的呈现也可按任务生成比较表、时间线、依赖图或结构化导出，而非限制在 Markdown。

## 面向实现的阅读方式

Gist 的“implementation spectrum”强调模块化：先有 raw sources、wiki、schema 和基本操作，再按实际规模逐项增加生命周期、图结构、自动化、质量控制与协作。它是架构建议文档，不是包含可复现实验协议的论文，也未给出这些组件在所有语料上的统一基准。

以下量化门槛、置信度浮点分数、遗忘曲线、last-write-wins 同步和自动矛盾修复都应视为待评估选项。若采用，应先定义证据字段、权限、冲突与回滚策略，并用目标数据集测搜索质量、时延和误修率。

## 对 Wiki 的映射

- 主页面：[LLM Wiki（Karpathy 模式与 v2 扩展）](../../wiki/references/llm-wiki-karpathy.md)
- 原始模式来源：[Karpathy LLM Wiki Gist](./karpathy_llm_wiki_gist.md)
- 本仓库实施规范：[schema/ingest-workflow.md](../../schema/ingest-workflow.md) 与 [AGENTS.md](../../AGENTS.md)
