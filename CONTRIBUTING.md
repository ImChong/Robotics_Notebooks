# 贡献指南

感谢你愿意改进本知识库。请先阅读：

- 知识库维护流程：[`schema/ingest-workflow.md`](schema/ingest-workflow.md)
- **本地与 CI 命令对照**（提交前防踩坑）：[`docs/contributing-ci.md`](docs/contributing-ci.md)
- 协作与提交约定：[`AGENTS.md`](AGENTS.md)（含中文 commit 格式、`make ci-preflight` 要求）

## 快速开始

1. Fork / 分支开发（推荐前缀 `cursor/` 之类由维护者约定）。
2. 安装开发依赖：`pip install -r requirements-dev.txt`，若参与前端脚本检查则 `npm ci`。
3. 修改维护脚本或测试后，建议运行 **`make ci-test`**（与 `tests.yml` 对齐：Ruff、Mypy、pip-audit、ESLint、pytest）。
4. 修改 `wiki/` 或导出/索引链时运行 **`make ci-preflight`**，并提交其生成的导出与统计文件。
5. 日常也可用 `make test` 仅跑 pytest（含覆盖率阈值）。
6. 发起 Pull Request，按模板填写摘要与验证方式。

## 代码风格

- Python：`ruff check` / `ruff format`（配置见 `pyproject.toml`）。
- `docs/main.js`：`npx eslint docs/main.js`（配置见 `eslint.config.js`）。
