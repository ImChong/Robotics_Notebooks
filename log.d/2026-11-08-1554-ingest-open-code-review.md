## [2026-11-08] ingest | Open Code Review 官方核验与既有节点升级

- **意图：** 收录 Alibaba 官方项目仓库、npm 包 `@alibaba-group/open-code-review` 与 `ocr` CLI，复核现有知识节点，不重复新建实体。
- **开源结论：** `alibaba/open-code-review` 公开核心 Go CLI 与 npm 包装，Apache-2.0；官方支持工作区/分支/commit 评审、`scan`、Delegation Mode、MCP 与 CI 集成。
- **一手通读：** 核对 README、官方快速开始与 CLI 文档，并审阅 `review_cmd.go`、`agent.go`、`selection.go`、`grouping.go` 关键调用链。
- **节点修订：** `wiki/entities/open-code-review.md` 增加 project_id、original 归档、英中双标题，校正分组机制并更新 Mermaid 主流程/源码时序图。
- **范围：** 只改知识源文件，不提交 `catalog.md`、`log.md` 或其他自动生成文件；未实际安装或运行 OCR。
