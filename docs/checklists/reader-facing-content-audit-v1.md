# 读者视角内容审计 v1 (Reader-Facing Content Audit)

审计日期：2026-09-17 · 基线提交：`f3d1d87d` · 最近更新：2026-09-18（B 类「ingest 记账」一栏清零）
目标：站点面向 **读者** 而非维护者。本文件全量扫描站点节点，定位仍带维护者口径的内容与页面，并按 checkbox 跟踪进度。

扫描口径：`scripts/export_minimal.py:840-864` 的 `collect_paths()` glob，共 **4135** 个导出节点（`wiki/*`、`roadmap/*`、`references/*`、`tech-map/*`）。

**进度总览**

- [x] **A 类：整页不面向读者的节点**（1071 + 168 + 136 + 15 + 3 页，已全部改写为读者口径）
- [ ] **B 类：正常页面里混入的维护者段落**（升格指令随 A 类清零、ingest 记账 2026-09-18 清零；其余见下）
- [x] **C 类：站点 UI 层的维护者口径**（`tech-map.html` / `module.html` 已删除）
- [ ] **D 类：发布在站点域下但无 UI 入口的维护文档**（待定，倾向保留）

---

## A 类：整页不面向读者的节点 — 已完成

- [x] **A1 · 清单索引论文页（1071 页，`wiki/entities/`）**
  来自 Awesome 清单批量生成（`sun_awesome_wm_catalog.md` 1467 引用、`sun_awesome_ego_catalog.md` 666、`sun_awesome_touch_catalog.md` 171 等）。
  改写内容：页首「本页为知识库 **策展索引级** 详情节点」→「本页是 **清单索引**：给出它在清单中的位置与原文入口」；「为什么重要」里的图谱理由 → 读者动作（横向对照 / 接回学习主线）；小标题去掉 `（索引级）` 后缀；结论段「本条目的站内价值是把 X 提升为可链接的知识节点」→「这一页能给你的是 X 在清单里的坐标与要点」；**824 页共有的**「若该工作成为学习主线，应再升格为深度论文实体」→「要深读这篇，建议直接从原文入手」。站内 `索引级` 一词已统一改为 `清单索引 / 清单摘要`（域内术语「索引级 3D SSG」保留）。

- [x] **A2 · Paper Notebooks 页（304 页）**
  审计首版把这 304 页笼统算作「占位页」，**这是错的**，实际分两种：
  - **168 页真占位**（`status: planned`，tag `paper-notebook-planned`）：确实没有内容。已把「本页作为 **占位子节点**，避免知识图谱缺失该论文实体」「升格条件 / 升格路径」等改成读者口径：「本页 **还没有深读笔记**：只给出分类位置与原文入口」「深读笔记完成后，本页会补上笔记链接与实质要点」；正文里「保持图谱完整 / 图谱连边 / 占住图谱位置」一律换成读者能用的说法（按分类检索、不漏掉这一篇）。
  - **136 页有实质内容**（`status: stub`，tag `paper-notebook-stub`）：正文编译自已完成的深读笔记，并非空页。只改了口径：「本页为 **深读笔记索引实体**」→「本页是 **笔记摘要**」。
  - 另有 1 页 `wiki/comparisons/clf-vs-cbf.md` 内容完整却仍标 `status: stub`，属 frontmatter 陈旧，未改（不影响前端，前端不读 status）。

- [x] **A3 · 空壳模块页 / 参考页（15 页）**
  13 个 `tech-map/modules/*/*.md`（原文仅 28–61 字符，如 `control/mpc.md` 33 字符）已按 `system/ros2.md` 的体例重写为「一段定位 + 站内入口」；`references/benchmarks/humanoid-environments.md`、`locomotion-benchmarks.md`（原文「用于整理…」）重写为可直接用的环境选型表与评测维度表。`references/papers/humanoid-hardware.md`、`repos/humanoid-projects.md`、`papers/survey-papers.md` 的「用于汇总…」导语改为读者口径（内容未扩写）。
  顺带修复：`tech-map/modules/math/linear-algebra.md` 的站内链接少一级 `../`，原为死链。

- [x] **A4 · 覆盖率 / 导读页（3 页，`wiki/queries/*-coverage.md`）**
  标题 `· 本库导读` → `· 阅读导航`；`Query 产物` 问句从「在本库分别对应哪一页？」改为读者会问的「我想找的那个在站内哪一页、同方向还有谁？」；frontmatter summary 去掉「缺口以 cn-os-* 实体补齐」「不重复造节点」等维护者措辞。同步更新 `scripts/generate_china_opensource_coverage_md.py`（生成器，否则重跑会回退）与 `scripts/utils/community_labels.py` 的社区名（图谱上可见）。

## B 类：正常页面里混入的维护者段落 — 部分完成

| 措辞 | 改写前 | 当前 | 状态 |
|------|--------|------|------|
| 「应再升格为深度论文实体」等升格指令 | 872 页 | **0** | [x] 随 A 类清零 |
| 「占位子节点 / 图谱占位」 | 172 页 | **0** | [x] 随 A 类清零 |
| 指向上游 `PROGRESS.md` 的「待深读」状态 | 178 页 | 178 页 | [ ] 表格行 `\| 深读状态 \| 待撰写 \|`，指向上游进度文件，读者可理解但仍偏内部 |
| ingest 记账（「本 ingest 新建 N / 复用 M」「0 重复 arXiv 节点」「不重复造页」「· 新建 / · 复用」列） | 65 页 | **0** | [x] 2026-09-18 改写，见下方说明 |
| 仓库工作流（`schema/ingest-workflow.md`、`make ci-preflight`、「上游更新后需重跑 `scripts/generate_*`」） | 33 页 | 28 页 | [ ] 剩余均在 **以 agent / skill 工具为主题的实体页**，用本仓库流程做对照是读者需要的信息；其余（4 个 `sun-awesome-*` 地图的「需重跑脚本」）已随上条清零 |
| 页首 `> **Query 产物**：…` 标签 | 80 页 | 同左 | [ ] 「Query 产物」是本库 ingest 动作名，读者不需要知道页面由哪次查询触发；改动需同时动 `scripts/lint_wiki.py:1277`（硬校验该字符串）与 `scripts/scaffold_wiki_page.py`，属独立一轮 |
| `（本仓库）` 标注 | 22 个 roadmap 页 / 564 处 | 同左 | [ ] `roadmap/depth-*.md` 链接列表几乎每行一个 |
| `## 参考来源` 直接列 `sources/xxx.md` 仓库路径 | 4080 页 / 25943 条 | 同左 | [ ] 前端降级为 GitHub blob 外链（`docs/main.js:1752-1757`），读者被踢出站到裸 Markdown；应显示为「来源笔记」而非文件路径 |

### B-ingest · ingest 记账口径（2026-09-18 已改写）

起因：读者反馈 [424 项阅读导航](https://imchong.github.io/Robotics_Notebooks/detail.html?id=wiki-queries-china-domestic-opensource-424-coverage) 表格里每行尾部的「· 新建 / · 复用」对读者无意义 —— 那是 ingest 时「这页是不是这轮新造的」的维护者记账。同族措辞全库清理：

- **424 全景两页 + 生成器**（`scripts/generate_china_opensource_coverage_md.py`）：表格去掉 `· 新建 / · 复用` 列尾标注；「规模」表的 `复用既有实体 / 本 ingest 新建实体` 换成 `覆盖机构 / 项目方向`；overview 的 `## 节点策略（本 ingest）` 改为 `## 这份清单能查到什么`。顺带修掉生成器里 HMI 主表的相对路径（`./` → `../queries/`，重跑会写出死链）。
- **26 张技术地图**：`N/N 独立 paper-* 节点：本 ingest 新建 X、复用 Y；0 重复 arXiv 节点` → `N 篇各有一页，可逐篇点开核对…`；`避免 N 个实体成孤岛`（14 处）→ `把 N 篇放在一页里横向对照`。
- **4 张 `sun-awesome-*` 地图 + 生成器**（`scripts/generate_sun254667_awesome_paper_entities.py`）：去掉「Awesome 列表本身不是知识图谱节点」「新建 225 / 复用 24」「上游更新后需重跑 `scripts/generate_*` 再 `make ci-preflight`」，改为读者口径的「原清单每条只有标题 + 链接…」与「本页是 <date> 的快照」。
- **约 30 个实体 / 方法 / 对比页**：`本次 ingest 归档` → `原文归档`；`与同 arXiv 节点不重复造页`、`复用本页不新建实体`、`不要再为 xxx 新建实体` 等直接删去或改为读者能用的说法；`均已升格为 wiki/entities/ 详情页（可搜索、进图谱）` → `各有一页，可直接点开或搜索`。

两个生成器已同步改写 —— 否则下次重跑会把维护者口径写回去。

## C 类：站点 UI 层的维护者口径 — 已完成

- [x] 删除 ~~`docs/tech-map.html`~~、~~`docs/module.html`~~：通篇施工日志语气（kicker「Data-driven tech-map page」、「当前不急着做力导图，先验证 layer / node_kind / summary 是否足够支撑第一版导航页」、footer「第一阶段 data-driven 版本」），且主导航无入口。
- `docs/detail.html` 已完成读者化（「近期相关内容」而非「最近 ingest」），是其余页面的改写基准。
- [ ] 遗留孤儿静态页：`docs/modules/*.html`（3）、`docs/relations/*.html`（4），主导航不链接、内容不随数据更新。**推测**已被 `detail.html` 体系取代，待确认后下线。

## D 类：发布在站点域下但无 UI 入口的维护文档 — 待定

`docs/content-backlog.md`、`docs/frontend-redesign-plan.md`、`docs/contributing-ci.md`、`docs/homepage-copy-v1.md`、`docs/plans/*.md`（8）、`docs/checklists/*.md`（含本文件）均可通过 URL 直达，但 UI 无入口。属维护者看板，可接受；仅需确认不被 sitemap 收录（sitemap 不入库，**未直接验证**）。

## 扫描中发现的其它问题（非本次范围）

- [ ] 前端 **不按 `status` 过滤**（`docs/main.js` 无 status 分支，`scripts/build_search_index.py` 不读 status），168 个「还没读」的占位页在搜索、图谱、「最新知识节点」中与真实内容页同权。改写口径之后这点仍在。
- [ ] 5 条既有死链：`references/README.md → papers/sim2real.md`（2 处）、`references/repos/simulation.md → wiki/entities/easy_quadruped.md`、`references/papers/imitation-learning.md → papers/inverse_reinforcement_learning_primary_refs.md`、`references/papers/locomotion-rl.md → wiki/concepts/locomotion.md`。

## 复现方式

```bash
# 导出节点清单
python3 - <<'PY'
from pathlib import Path
pats = ["wiki/concepts/*.md", "wiki/methods/*.md", "wiki/tasks/*.md", "wiki/comparisons/*.md",
        "wiki/overview/*.md", "wiki/formalizations/*.md", "wiki/queries/*.md", "wiki/entities/*.md",
        "wiki/references/*.md", "wiki/roadmaps/*.md", "roadmap/*.md", "references/papers/*.md",
        "references/repos/*.md", "references/benchmarks/*.md", "tech-map/overview.md",
        "tech-map/dependency-graph.md", "tech-map/modules/*/*.md", "tech-map/research-directions/*.md"]
print("\n".join(str(x) for p in pats for x in sorted(Path('.').glob(p)) if x.name != "README.md"))
PY
# 再对清单 grep 上表各判据即可复现全部计数
```
