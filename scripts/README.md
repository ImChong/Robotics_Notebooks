# `scripts/` 维护脚本一览

本目录脚本由 `PYTHONPATH=scripts` 调用（见 `Makefile` 与 [贡献指南里的 CI 对照](../docs/contributing-ci.md)）。下表便于快速定位「改什么 / 跑什么」；权威调用方式仍以仓库根目录 **`Makefile`** 为准。

## 入口脚本（按名字母序）

| 脚本 | 作用摘要 | 常见调用 |
|------|----------|----------|
| `append_log.py` | 向 `log.md` 追加维护操作记录 | `make log` |
| `build_search_index.py` | 生成站点搜索索引 JSON | 由 `export_minimal.py` / `make export` 间接使用 |
| `build_vector_index.py` | 构建向量检索索引（可选依赖） | `make vectors` |
| `check_export_quality.py` | 导出 JSON 与 wiki 一致性等检查 | `make export-check`；`make ci-preflight` |
| `ci_preflight.py` | 按固定顺序再生派生物并跑质量门禁 | `make ci-preflight`、`make ci-check` |
| `debug_search.py` | 命令行下调试搜索得分与排名 | `python3 scripts/debug_search.py …` |
| `discover_facts.py` | 扫描 wiki 提取事实候选、相关页建议（辅助维护） | 按需直接运行 |
| `eval_search_quality.py` | 搜索质量回归用例 | `make lint`、`make eval-search`；`make ci-preflight` |
| `export_anki.py` | 导出 Anki 兼容 TSV | `make anki` |
| `export_minimal.py` | 导出 wiki JSON、站点数据、sitemap、搜索索引等 | `make export`；`make ci-preflight` |
| `fetch_to_source.py` | 从 URL 抓取内容进入 `sources/` | `make fetch` |
| `generate_home_stats.py` | 生成首页 Hero 用轻量统计 JSON | `make graph`；`sync_all_stats` / preflight |
| `generate_link_graph.py` | 生成知识图谱边数据等 | `make graph`；`sync_all_stats` / preflight |
| `generate_page_catalog.py` | 生成 `index.md` 页面目录 | `make catalog`；`make ci-preflight` |
| `ingest_coverage.py` | 与覆盖率/ingest 相关的辅助脚本 | `make coverage` |
| `ingest_paper.py` | 生成 `sources/papers/` 论文 ingest 模板 | `make ingest` |
| `lint_wiki.py` | Wiki 健康检查（断链、孤儿页、frontmatter 等） | `make lint`；`make ci-preflight`（`--report`） |
| `migrate_wikilinks.py` | 将 `[[wikilink]]` 迁移为标准 `[text](path)` | 一次性维护时直接运行 |
| `search_wiki.py` | 终端搜索 wiki 内容 | `make search` |
| `sync_all_stats.py` | 图谱、首页统计、README 徽章、`docs/index.html` 等同步 | `make sync-stats`；`make ci-preflight` |
| `update_badge.py` | 更新 `README.md` 顶部徽章区 | `make badge` |
| `wiki_to_marp.py` | 将 wiki 页转为 Marp 幻灯片 Markdown | `make slides` |

## Shell 与其它

| 路径 | 作用摘要 | 常见调用 |
|------|----------|----------|
| `sync_wiki.sh` | 串联 catalog、graph、export 等（历史便捷入口） | `make sync` |

## 主要 Python 模块（通常不直接当 CLI）

| 模块 | 作用摘要 |
|------|----------|
| `search_indexing.py` | 分词、BM25、wiki 文档迭代等搜索索引基础 |
| `search_wiki_core.py` | 终端搜索核心逻辑（被 `search_wiki.py`、`debug_search.py` 使用） |
| `utils/paths.py` | 路径与 ID 规范化工具 |
