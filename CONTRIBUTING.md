# 贡献指南

感谢你愿意改进本知识库。**提交前可先按下面三步选对命令**，不必一次背完整工作流。

## 提交前跑什么（三步）

1. **改了 `wiki/`，或会影响页面目录、导出 JSON、搜索索引、图谱、sitemap、`README` 统计区、`docs/index.html` 等派生物** → 运行 **`make ci-preflight`**，并把命令重新生成后有变化的文件与本次编辑的源文件一并提交。
2. **只改了 `scripts/`、`tests/`、`docs/main.js` 或本地工具配置**（不动上述派生链）→ 运行 **`make ci-test`**（与 [`.github/workflows/tests.yml`](.github/workflows/tests.yml) 对齐：Ruff、Mypy、pip-audit、ESLint、pytest）。
3. **只在本地查词、搜页、阅读 markdown，不写回仓库** → 不必跑 CI 全套；需要时可单独用 `make lint` 或 `python3 scripts/search_wiki.py <关键词>` 做轻量检查。

请先阅读：

- **Schema 与流程索引**：[`schema/README.md`](schema/README.md)
- 知识库维护流程：[`schema/ingest-workflow.md`](schema/ingest-workflow.md)
- **内容进哪个目录**：[`schema/content-directories.md`](schema/content-directories.md)
- **本地与 CI 命令对照**（提交前防踩坑）：[`docs/contributing-ci.md`](docs/contributing-ci.md)
- 协作与提交约定：[`AGENTS.md`](AGENTS.md)（含中文 commit 格式、`make ci-preflight` 要求）

## 快速开始

1. Fork / 分支开发（推荐前缀 `cursor/` 之类由维护者约定）。
2. 安装开发依赖：`pip install -r requirements-dev.txt`（Python 版本见 [`docs/contributing-ci.md`](docs/contributing-ci.md)），若参与前端脚本检查则 `npm ci`。
3. 按上文 **「提交前跑什么（三步）」** 选择 `make ci-preflight` 或 `make ci-test`；日常也可 `make test` 仅跑 pytest（含覆盖率阈值）。
4. 发起 Pull Request，按模板填写摘要与验证方式。

## 代码风格

- Python：`ruff check` / `ruff format`（配置见 `pyproject.toml`）。
- `docs/main.js`：`npx eslint docs/main.js`（配置见 `eslint.config.mjs`）。

## 提交前钩子（可选）

安装：`pip install -r requirements-dev.txt`，然后 **`make install-hooks`**（即 `pre-commit install`）。提交时将自动运行 **Ruff**（与 CI 对齐）。

一次性检查整个仓库：`pre-commit run --all-files`。
