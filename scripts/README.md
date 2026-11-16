# `scripts/` 维护脚本一览

本目录脚本由 `PYTHONPATH=scripts` 调用（见 `Makefile` 与 [贡献指南 CI 对照](../docs/contributing-ci.md)）。维护规范总览见 [schema/README.md](../schema/README.md)；权威命令行入口仍以仓库根目录 **`Makefile`** 为准。

## 导出、图谱与站点构建

| 脚本 | 作用摘要 | 常见调用 |
|------|----------|----------|
| `export_minimal.py` | 导出全库 `exports/site-catalog-v1.json`、按页正文、`sitemap.xml` 与搜索索引 | `make export`；`make ci-preflight` |
| `generate_link_graph.py` | 构建知识图谱 `link-graph*.json`、`graph-stats.json`、`hub-rankings.json`、`wiki-activity.json` | `make graph`；`make ci-preflight` |
| `generate_home_stats.py` | 汇总图谱与覆盖率生成 `exports/home-stats.json` | `make graph`；`pages.yml` |
| `graph_exports_sync.py` | 将图谱相关 JSON 从 `exports/` 同步到 `docs/exports/` | `make graph`；`sync_all_stats` |
| `precompute_graph_layout.cjs` | 预计算力导向图谱节点初始坐标供 `generate_link_graph.py` 复用 | 由 `generate_link_graph.py` 调用 |
| `build_search_index.py` | 生成站点全文/轻量 BM25 搜索索引 | 由 `export_minimal.py` 调用 |
| `build_vector_index.py` | 构建可选语义检索向量索引与元信息 | `make vectors` |
| `export_anki.py` | 导出 `wiki/formalizations` 与精选概念为 Anki TSV | `make anki` |
| `generate_page_catalog.py` | 生成本地按需浏览用的 `catalog.md`（已移出 Git，`.gitignore` 忽略） | `make catalog`（本地预览） |
| `generate_star_history.py` | 记录 GitHub Star 日历史供 `docs/star-history.html` 展示 | `star-history.yml` |
| `sync_all_stats.py` | 部署时更新 Shields 徽章、`docs/index.html` Hero 与 `docs/sw.js` 缓存版本号 | `pages.yml`；`make sync-stats` |
| `sync_sw_cache_version.py` | 从 `exports/graph-stats.json` 刷新 `docs/sw.js` `CACHE_NAME` | `make sync-stats` |
| `wiki_to_marp.py` | 将 wiki 页面转换为 Marp 幻灯片 Markdown | `make slides` |

## 质量检查、门禁与诊断

| 脚本 | 作用摘要 | 常见调用 |
|------|----------|----------|
| `lint_wiki.py` | Wiki 全库健康检查（断链、孤儿、frontmatter、canonical-facts、历史基线门禁等） | `make lint`；`make ci-preflight` |
| `check_project_nodes.py` | 检查项目/论文实体的 `project_id` 与 arXiv/DOI 唯一性，防止拆页重复 | `make ci-preflight` |
| `check_export_quality.py` | 校验导出 JSON、搜索索引、图谱同步与节点/页面覆盖一致 | `make export-check`；`make ci-preflight` |
| `ci_preflight.py` | 按 CI 顺序本地再生 gitignore 派生物并执行全部门禁校验 | `make ci-preflight`、`make ci-check` |
| `ci_change_scope.py` | 判断 PR 改动范围，路由轻量 ingest 门禁 vs 全量测试 | GitHub Actions (`tests.yml` / `lint.yml`) |
| `ci_incremental_lint.py` | 仅针对新增/变更 wiki 页做增量快速门禁校验 | GitHub Actions |
| `pr_derived_guard.py` | 校验 PR 未改动 `log.md` 或部署派生物；`sync` 自动合入 `main` 并解冲突 | `make pr-check`；`make sync-main` |
| `eval_search_quality.py` | BM25 / 站点搜索质量回归测试用例集 | `make eval-search`；`make ci-preflight` |
| `debug_search.py` | 命令行调试单条查询的搜索打分与命中排序 | `python3 scripts/debug_search.py …` |
| `audit_depth_health.py` | 审计 `roadmap/depth-*.md` 各 Stage 关联页健康度与缺项 | 按需直接运行 |
| `audit_diagram_coverage.py` | 统计全库 Mermaid 流程图与源码运行时序图覆盖率 | 按需直接运行 |
| `diagnose_topics.py` | 基于 Louvain 结构聚类对照诊断 `schema/topics.json` 主题归属 | `make topic-diagnose` |
| `report_health.py` | 输出变更或全部 wiki 页健康度（0–3）及 `--check-pr-body` PR 正文门禁 | `make health`；ingest PR 规范检查 |
| `check_mermaid.cjs` | 使用与站点一致的 Mermaid 引擎校验全库 Mermaid 代码块语法 | `npm run lint:mermaid` |

## Ingest、元数据同步与日志管理

| 脚本 | 作用摘要 | 常见调用 |
|------|----------|----------|
| `ingest_paper.py` | 生成 `sources/papers/` 论文 ingest 模板 | `make ingest` |
| `ingest_coverage.py` | 统计 `sources/` 到 `wiki/` 的入库覆盖率 | `make coverage` |
| `scaffold_wiki_page.py` | 按页面骨架模板创建新 wiki 页并自检结构 | `python3 scripts/scaffold_wiki_page.py …` |
| `fetch_to_source.py` | 抓取网页原文进入 `sources/` 归档 | `make fetch` |
| `fetch_feishu_doc.py` | 用 Playwright 滚动抓取互联网公开飞书文档纯文本与超链接 | 按需直接运行 |
| `fetch_wechat_article.py` | 用微信内置浏览器 UA 抓取公众号正文纯文本 | 按需直接运行 |
| `bump_institution_tags.py` | 根据页面摘要/首段自动补全 `tags:` 中的机构别名 | 按需直接运行 |
| `bump_wiki_updated_for_sources.py` | 按指定或变更 `sources/papers/` 批量刷新关联 wiki 的 `updated:` 日期 | `make bump-wiki-from-sources` |
| `bootstrap_paper_notebook_knowledge.py` | 批量同步 `Robot_Learning_Paper_Notebooks` 分类树与未映射论文条目 | 按需直接运行 |
| `dedupe_paper_notebook_nodes.py` | 合并重复 `paper-notebook-*` stub 到规范实体页 | 按需直接运行 |
| `discover_facts.py` | 扫描 wiki 候选事实与潜在交叉引用建议 | 按需直接运行 |
| `gen_abbrev_glossary.py` | 为缺失页自动生成 `## 英文缩写速查` 占位表 | 按需直接运行 |
| `reorder_abbrev_glossary.py` | 将 `## 英文缩写速查` 调整到「一句话定义」之后的标准位置 | 按需直接运行 |
| `sync_institution_tags.py` | 从正文机构表、sources 与 GitHub org 同步 frontmatter 机构标签 | 按需直接运行 |
| `sync_paper_notebook_links.py` | 回写深读仓库（`Robot_Learning_Paper_Notebooks`）互链 | 按需直接运行 |
| `sync_paper_notebook_summaries.py` | 将深读仓库摘要同步到 stub 实体页 | 按需直接运行 |
| `sync_rl_sim2sim_demo_links.py` | 同步 `RL Sim2Sim Demo` 交互演示页链接 | 按需直接运行 |
| `append_log.py` | 向 `log.d/` 写入一条独立维护日志碎片 | `make log` |
| `fold_log_fragments.py` | 将 `log.d/` 碎片并入 `log.md` 顶部并触发超期季度归档 (`log-YYYY-QN.md`) | `export.yml` (`main`) |

## 前端验证与内部共用模块

| 脚本 / 模块 | 作用摘要 |
|-------------|----------|
| `screenshot_site_detail.sh` | 本地启静态服务并对 `docs/detail.html` 执行 headless 验证截图 |
| `screenshot_detail_anchor_viewport.cjs` | 检查详情页章节锚点跳转与视口位置回归 |
| `search_indexing.py` | 中英文分词、BM25 计算、YAML frontmatter 解析与摘要提取 |
| `search_wiki_core.py` | CLI 搜索核心逻辑（被 `search_wiki.py`、`debug_search.py` 复用） |
| `search_wiki.py` | 终端搜索 wiki CLI |
| `log_md.py` | `log.md` 顶部插入、`log.d/` 碎片写入/并入与季度归档逻辑 |
| `original_reading.py` | 原文通读结构化核对辅助（对应 `schema/ingest-workflow.md` 步骤 1.6） |
| `wiki_abbrev_section.py` | `## 英文缩写速查` 区块提取与位置判定辅助 |
| `utils/markdown.py` | frontmatter / 代码块剥离 / Markdown 内链 / Chromium 查找公共工具 |
| `utils/community_labels.py` | `schema/topics.json` 图谱主题与检索同义词加载 |
| `utils/paths.py` | 仓库相对路径到站点详情 `id` 与 slug 规范化 |
| `utils/wiki_cache.py` | 全库 `wiki/` 文件名到路径的共享索引缓存 |
