# 读者视角内容审计 v1 (Reader-Facing Content Audit)

审计日期：2026-09-17 · 基线提交：`f3d1d87d`
目标：站点面向 **读者** 而非维护者。本文件全量扫描站点节点，定位仍带维护者口径的内容与页面，并给出优先级。

扫描口径：`scripts/export_minimal.py:840-864` 的 `collect_paths()` glob，共 **4135** 个导出节点（`wiki/*`、`roadmap/*`、`references/*`、`tech-map/*`）。

**结论：约 1414 个节点（34.2%）带维护者口径**，集中在三个批量生成的集群。

---

## 一、整页就不面向读者的节点

| 类别 | 数量 | 判据 | 读者实际看到什么 |
|------|------|------|------------------|
| **索引级论文实体**（Awesome 清单批量生成） | **1071**（全在 `wiki/entities/`） | 含 `## 核心信息（索引级）` / `## 与其他工作对比（索引级）`；来源集中在 `sources/papers/sun_awesome_wm_catalog.md`（1467 引用）、`sun_awesome_ego_catalog.md`（666）、`sun_awesome_touch_catalog.md`（171）等 | 页面自己声明「本页 **不做** 与具体基线的逐项数值对比」，即无结论页 |
| **Paper Notebooks 占位页** | **302**（`status: planned` 168 + `status: stub` 134，tag `paper-notebook-planned` / `paper-notebook-stub`） | 例 `wiki/entities/paper-notebook-pimbs-*.md`：正文即「深读笔记尚未撰写」「本页作为 **占位子节点**，避免知识图谱缺失该论文实体」 | 整页零信息量，只有分类名与计划文件夹路径 |
| **tech-map 模块空节点** | **14 个 < 100 字符**（另 6 个 < 800 字符） | `tech-map/modules/control/mpc.md`（33 字符）、`il/behavior-cloning.md`、`robotics/kinematics.md`、`references/benchmarks/*.md` 等 | 点进去只有标题 + 一句话 |
| **本库覆盖率 / 导读页** | 至少 3（`wiki/queries/*-coverage.md`） | `china-domestic-opensource-424-coverage.md` 开篇：「**Query 产物**：…在本库分别对应哪一页？」 | 是仓库自检清单，不是知识内容 |

补充：前端 **不按 `status` 过滤**（`docs/main.js` 无 status 分支，`scripts/build_search_index.py` 不读 status），因此 302 个占位页在搜索、图谱、「最新知识节点」中与真实内容页完全同权。

## 二、正常页面里混入的维护者段落

| 措辞 | 文件数 | 位置样例 |
|------|--------|----------|
| 「应再升格为深度论文实体（补机构、实验表…）」等升格指令 | **872** | 固定出现在 `## 结论` 的最后一条 bullet |
| 「占位子节点 / 图谱占位」 | 172 | 同上 |
| 指向上游 `PROGRESS.md` 的「待深读」状态 | 178 | 表格行 `\| 深读状态 \| 待撰写 \|` |
| 仓库工作流（`schema/ingest-workflow.md`、`make ci-preflight`、「上游更新后需重跑 `python3 scripts/generate_sun254667_...`」） | 33 | `wiki/entities/`（42 处）、`wiki/overview/`（19 处） |
| `（本仓库）` 标注 | **564 处 / 22 个 roadmap 页** | `roadmap/depth-bfm.md` 链接列表几乎每行一个 |
| `## 参考来源` 直接列 `sources/xxx.md` 仓库路径 | **4080 页 / 25943 条** | 前端将其降级为 GitHub blob 外链（`docs/main.js:1752-1757`），读者被踢出站到裸 Markdown |

最后一条量最大但争议也最大：保留一手来源可溯是刻意设计，问题只在 **展示口径**（应显示为「来源笔记」而非仓库文件路径），不是要删。

## 三、站点 UI 层的维护者口径

- ~~`docs/tech-map.html`~~、~~`docs/module.html`~~ —— 通篇施工日志语气（kicker「Data-driven tech-map page」、「当前不急着做力导图，先验证 layer / node_kind / summary 是否已经足够支撑第一版导航页」、footer「第一阶段 data-driven 版本」），且主导航无入口。**已于本次审计的配套 PR 删除**（用不到）。
- 相比之下 `docs/detail.html` 已完成读者化（「近期相关内容」而非「最近 ingest」），可作为其余页面的改写基准。
- 遗留孤儿静态页：`docs/modules/*.html`（3）、`docs/relations/*.html`（4），主导航不链接，内容未随数据更新。**推测**已被 `detail.html` 体系取代，建议后续确认后删除。

## 四、发布在站点域下但无 UI 入口的维护文档

`docs/content-backlog.md`、`docs/frontend-redesign-plan.md`、`docs/contributing-ci.md`、`docs/homepage-copy-v1.md`、`docs/plans/*.md`（8）、`docs/checklists/*.md`（含本文件）均在 GitHub Pages 根目录下可直接访问，但 UI 无入口。属维护者看板，可接受；仅需确认不被 sitemap 收录（sitemap 不入库，**未直接验证**）。

---

## 推进顺序（按「读者受损 / 改动成本」排）

1. **[x] 三、UI 层文案** —— 删除 `tech-map.html` / `module.html`（本次完成）。
2. **[ ] 二、升格指令与 PROGRESS 行** —— 872 页，但同属一套模板，可脚本批改。
3. **[ ] 一、302 个占位页** —— 需先决定策略：隐藏 / 搜索降权 / 合并进分类父页。
4. **[ ] 一、1071 个索引级页** —— 量最大；**建议**保留（图谱完整性），但页首换成读者口径的一句「这是清单索引，正文见原文」。
5. **[ ] 二、`参考来源` 展示口径** —— 把仓库路径渲染成「来源笔记」标签，避免读者跳到裸 Markdown。

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
