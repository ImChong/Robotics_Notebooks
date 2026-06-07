# Schema 目录说明

本目录存放知识库的**维护规则与机器可读数据**，不是面向读者的 wiki 正文。维护前建议先读 [ingest-workflow.md](ingest-workflow.md)。

## 文件索引

| 文件 | 用途 |
|------|------|
| [ingest-workflow.md](ingest-workflow.md) | Ingest / Query / Lint / Index 等日常操作步骤与约定 |
| [content-directories.md](content-directories.md) | `wiki/`、`sources/`、`references/` 等根目录如何选用 |
| [linking.md](linking.md) | 内链相对路径写法与关联区块建议 |
| [naming.md](naming.md) | 文件与目录命名约定 |
| [page-types.md](page-types.md) | Wiki 页面类型与 frontmatter 字段 |
| [log-format.md](log-format.md) | `log.md` 条目格式与前缀约定 |
| [canonical-facts.json](canonical-facts.json) | `lint_wiki.py` 矛盾检测用的事实规则数据 |
| [search-regression-cases.json](search-regression-cases.json) | 搜索质量回归测试用例数据 |
| [paper-notebook-index.json](paper-notebook-index.json) | Humanoid Paper Notebooks 137 篇笔记 URL 索引（由脚本同步） |
| [paper-notebook-wiki-overrides.yml](paper-notebook-wiki-overrides.yml) | 论文笔记 → 本库 wiki 路径的手动映射 |
| [paper-notebook-wiki-map.yml](paper-notebook-wiki-map.yml) | 合并后的完整映射（手动 + 自动，供脚本读取） |

## 与其它文档的关系

- 提交与 CI 命令：[../docs/contributing-ci.md](../docs/contributing-ci.md)
- 贡献总入口：[../CONTRIBUTING.md](../CONTRIBUTING.md)
- 自动化维护者说明：[../AGENTS.md](../AGENTS.md)
- 合并前 GitHub Actions 门禁（非 schema 正文）：[../docs/checklists/github-actions-ci-gate.md](../docs/checklists/github-actions-ci-gate.md)
