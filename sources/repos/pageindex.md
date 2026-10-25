# PageIndex（VectifyAI）

> 来源归档（ingest）

- **标题：** PageIndex: Vectorless, Reasoning-based RAG
- **类型：** repo / document-retrieval / rag / llm-reasoning
- **作者 / 组织：** VectifyAI
- **代码：** <https://github.com/VectifyAI/PageIndex>（**已开源**，MIT）
- **项目页：** <https://pageindex.ai/>
- **文档：** <https://docs.pageindex.ai/>
- **Python 包：** `pageindex`（`pip install -U pageindex`）
- **许可：** MIT（以仓库 LICENSE 为准）
- **复核日期：** 2026-10-25（GitHub 显示约 39.6k stars）
- **一句话说明：** 以分层文档树和 LLM 推理检索替代向量相似度检索，面向长篇专业文档提供带原文引用的答案；SDK 支持本地 API 调用与 PageIndex Cloud。

## 开源核查

官方 GitHub 仓库公开了 Python 实现、MIT 许可证和 SDK Quickstart；官网链接到该仓库及开发者文档。README 给出本地安装与模型 API key 配置，并提供 PageIndex Cloud 接入方式。代码已开源。

## README 要点（归纳）

- **索引：** 为每份文档构造分层树形结构，保留文档章节与上下文组织。
- **检索：** LLM agent 在树上逐步推理和选择相关节点，而非依赖向量数据库做语义相似度排序。
- **输出：** 检索结果可追溯到明确的原文位置，适合需要审阅证据的长文档问答。
- **适用资料：** README 举例包括财务报告、法律与监管文件、技术手册、医学文献和学术教材。
- **快速开始：** `pip install -U pageindex`，配置模型 API key 后使用 `PageIndexClient` 提交 PDF 并提问；也可连接 PageIndex Cloud。

## 对 wiki 的映射

- [PageIndex 实体页](../../wiki/entities/pageindex.md)：归纳树索引、推理检索、适用场景与工程权衡。
- [DeepTutor 实体页](../../wiki/entities/deeptutor.md)：其 Knowledge Center 已列出 PageIndex 作为可选 RAG 引擎。

## 参考来源

- [官方 GitHub 仓库](https://github.com/VectifyAI/PageIndex)
- [PageIndex 官网](https://pageindex.ai/)
- [PageIndex 文档](https://docs.pageindex.ai/)
