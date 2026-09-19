# 贡献指南

感谢你愿意改进本知识库。**提交前可先按下面三步选对命令**，不必一次背完整工作流。读者入口见 [README.md](README.md)；本页面向维护者。

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

## 项目结构

| 目录 | 用途 |
|------|------|
| `wiki/` | **结构化知识库**。包含 Concepts, Methods, Tasks 等核心页面。 |
| `roadmap/` | **成长路线**。规划了从基础到进阶的系统学习路径。 |
| `tech-map/` | **技术地图**。展示模块间依赖关系与技术栈全景。 |
| `sources/` | **原始资料**。Ingest 之前的原始论文摘录、GitHub 仓库导航。 |
| `references/` | **论文/Repo 索引**。按主题分类的深度阅读资源。 |
| `schema/` | **维护规范**（ingest、命名、内链、页面类型、log 与 lint 用 JSON）。索引见 [schema/README.md](schema/README.md)。 |
| `scripts/` | **维护工具**。用于 lint、搜索、索引生成和统计更新；脚本一览见 [scripts/README.md](scripts/README.md)。 |
| `docs/` | **展示层**。GitHub Pages 托管的 D3.js 交互式图谱与详情页。 |
| `docs/checklists/` | **执行清单归档**。当前技术栈推进、前端优化与历史阶段清单。 |

不知道新资料或新页面该进哪个目录？见 [内容目录怎么选](schema/content-directories.md)。

## 维护看板

- 当前技术栈执行清单：[v31](docs/checklists/tech-stack-next-phase-checklist-v31.md)
- 前端体验优化清单：[frontend-optimization-v1](docs/checklists/frontend-optimization-v1.md)
- 历史执行清单索引：[docs/checklists/README.md](docs/checklists/README.md)

## 日常维护（Ingest / Wiki / Lint / 搜索）

1. **Ingest**：发现好的论文或 Repo，按 [`schema/ingest-workflow.md`](schema/ingest-workflow.md) 先写入 `sources/`。
2. **Wiki 完善**：把 `sources/` 提炼成 `wiki/` 页面，并按 [linking](schema/linking.md) 建立交叉引用（不要把 source 原样转存成 wiki）。
3. **Lint**：提交前用 `make lint`（改了 wiki / 导出链则用 `make ci-preflight`）检查断链与孤儿页。
4. **本地搜索**：`python3 scripts/search_wiki.py <关键词>`。读者走站点搜索即可，不必装 CLI。

## 快速开始

1. Fork / 分支开发（推荐前缀 `cursor/` 之类由维护者约定）。
2. 安装开发依赖：`pip install -r requirements-dev.txt`（Python 版本见 [`docs/contributing-ci.md`](docs/contributing-ci.md)），若参与前端脚本检查则 `npm ci`。
3. 按上文 **「提交前跑什么（三步）」** 选择 `make ci-preflight` 或 `make ci-test`；日常也可 `make test` 仅跑 pytest（含覆盖率阈值）。
4. 发起 Pull Request，按模板填写摘要与验证方式。

## 代码风格

- Python：`ruff check` / `ruff format`（配置见 `pyproject.toml`）。
- `docs/main.js`：`npx eslint docs/main.js`（配置见 `eslint.config.mjs`）。

## 新建知识页

优先使用统一脚手架生成页面骨架，再补充正文与交叉引用：

```bash
python3 scripts/scaffold_wiki_page.py concept "页面标题" --slug page-slug
```

页面主干顺序与各类型要求见 [Page Types](schema/page-types.md)。

## 提交前钩子（可选）

安装：`pip install -r requirements-dev.txt`，然后 **`make install-hooks`**（即 `pre-commit install`）。提交时将自动运行 **Ruff**（与 CI 对齐）。

一次性检查整个仓库：`pre-commit run --all-files`。
