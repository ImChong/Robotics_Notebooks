# Wiki Operations

> 本文件定义 Robotics_Notebooks 知识库的日常维护操作规范。
> 基于 Karpathy LLM Wiki 模式：LLM 是维护者，人类是 curator。
> 四种核心 ops：Ingest（吃进）、Query（查询）、Lint（体检）、Index（索引）。

维护者入口：**[schema 目录索引](README.md)**（本目录各规范与数据文件一览）。

---

## Op 1：Ingest — 新资料进入知识库

### 触发条件

满足以下任一条时，执行 ingest：
- 读了一篇值得留存的论文 / 博客 / 教程
- 看了有价值的视频或课程笔记
- 发现了新的工具、仓库、数据集
- 对某个概念有了新的理解或修正

### 步骤 1：判断资料类型

- `paper` — 论文
- `blog` — 博客文章
- `course` — 课程 / 教程
- `video` — 视频
- `repo` — 代码仓库
- `personal` — 个人理解 / 总结

### 步骤 1.5：使用 `ingest_paper.py` 快速生成模板

对于论文类资料，优先使用工具生成 sources/papers/ 模板，避免手动格式出错：

```bash
# 生成模板（填写后手动编辑论文摘录）
make ingest NAME=my_topic TITLE="论文集合标题" DESC="一句话说明"

# 等价命令
python3 scripts/ingest_paper.py my_topic --title "..." --desc "..."
```

模板生成后，编辑文件填写核心论文摘录（至少 3 条），并在每条下方填写 `对 wiki 的映射`。

### 步骤 2：进入 `sources/`

原始资料先收录到 `sources/` 对应位置，保留来源信息。

若不确定应落在 `sources/`、`references/`、`resources/` 还是直接进 `wiki/`，先看 [内容目录怎么选](content-directories.md) 的决策漏斗。

建议记录：
- 标题、链接、类型
- 一句话说明
- 为什么值得保留
- 来源日期（如果知道）

### 步骤 2.5：项目页与源码开放核查（有项目页时必做）

论文、博客或仓库 ingest 时，若资料带有 **项目页**（`*.github.io`、机构 lab 页、`research.nvidia.com/labs/...` 等），**必须先打开项目页**核对源码与数据是否开放，再写 wiki 或 `sources/` 归档。

**核查清单：**

1. **项目页头部 / Footer / Code / Resources 区** — 查找 GitHub、Hugging Face、Zenodo、ModelScope 等链接。
2. **区分开放程度**（写入归档与 wiki，勿含糊带过）：
   - **已开源**：训练/推理/部署代码、权重或数据集至少一项可公开获取 → 在 `sources/sites/<项目>.md` 的元数据行写明 `- **代码：** <url>`（及数据集行，如有）；若仓库值得单独导航，另建 `sources/repos/<name>.md`。
   - **部分开源**：仅 Sim2Sim、checkpoint、数据处理脚本等子集 → 在 wiki「局限与风险」或「工程实践」中写明 **已发布 / 待发布** 边界（参考 [VisualMimic 实体页](../wiki/entities/paper-notebook-visualmimic.md) 的「开源状态」小节）。
   - **宣称将开源 / 未列链接**：项目页或论文写 "code will be released" 但尚无 URL → 写明「截至入库日项目页未列 GitHub」并标注入库日期，便于后续 lint 跟进。
   - **确认未开源**：无代码链接且论文未承诺 → 在 wiki 局限中一句点明，避免读者误以为可复现。
3. **交叉链接**：`sources/sites/` ↔ `sources/repos/` ↔ `sources/papers/` 三处元数据互指（项目页、论文、代码 URL 齐全时）；升格 wiki 后 `## 参考来源` 至少链到对应 `sources/sites/` 或 `sources/repos/`。
4. **勿仅凭论文 PDF 臆断**：Code  availability 以 **项目页实际链接** 为准；PDF 写 open-source 但页上无链时，按「宣称 / 待核实」处理。

> **为什么单独成步**：大量机器人论文的复现入口在项目页而非 arXiv；漏查会导致 wiki 误写「暂无代码」或漏收官方仓库，损害选型与复现判断。

### 步骤 3：判断是否沉淀到 `wiki/`

只有满足以下至少一条，才升格进 `wiki/`：
- 能解释一个重要概念
- 能补全一个方法页
- 能帮助做选型对比
- 能影响学习路线或研究判断

### 步骤 4：判断放进哪个 wiki 子目录

| 资料性质 | 目标位置 |
|---------|---------|
| 形式化基础（数学/理论框架）| `wiki/formalizations/` |
| 核心概念/方法 | `wiki/concepts/` 或 `wiki/methods/` |
| 实体（工具/框架/硬件）| `wiki/entities/` |
| 任务方向 | `wiki/tasks/` |
| 对比分析 | `wiki/comparisons/` |
| 学习路线 | `roadmap/` 或 `wiki/roadmaps/` |

#### `roadmap/` 与 `wiki/roadmaps/` 怎么选

- **`roadmap/`**：默认放这里。面向读者的**主学习路径**、条件分支路线图（例如「若目标是 X 则走哪条」）、与首页/技术地图强绑定的入口；`README` 与站点导航主要指向本目录。
- **`wiki/roadmaps/`**：专题或子路线、与 `wiki/concepts` 等方法页**强互链**、且不必占据仓库根级「主路线」展示位时使用。
- **仍犹豫**：优先 `roadmap/`；若整页几乎只服务某一 `wiki` 子树且读者主要从该主题进入，再考虑 `wiki/roadmaps/`。

### 步骤 5：更新相关页面

若资料有价值：
- 更新一个已有 wiki 页（补充 cross-reference、关联关系）
- 或新建一个 wiki 页

同时补充：
- **`机构标签`** — 论文/实体页须在 frontmatter `tags` 中写入 `schema/institutions.json` 已注册的机构 **alias**（如 `sjtu`、`nvidia`）；正文 **核心信息** 表或 `| 机构 |` 行写明中文全称。多机构联合论文写全；注册表无对应条目时可先略过，或追加注册表后再写 tag。可用 `python3 scripts/sync_institution_tags.py` 从 `| 机构 |` 表与 sources 机构行同步。
  - **新增机构注册时**：`label` 必须为 **`中文（English）`** 全角括号格式（如 `弗莱鑫机器人（Flexion Robotics）`），见 [naming.md § 研究机构命名](naming.md)。`make ci-preflight` **不跑** pytest；改 `institutions.json` 后须额外 `make test`，否则 GitHub Actions **Tests** job 会因 `test_institution_naming` 失败。
- **`英文缩写速查`** — 紧跟页面「一句话定义」之后，固定标题 `## 英文缩写速查`，三列表格（缩写 / 英文全称 / 简要说明）；至少 3 行，覆盖本页核心缩写。格式与编写要求见 [page-types.md](page-types.md)「新增页面最低质量标准」。
- `参考来源` — 必须标注本次 ingest 的原始资料（至少 1 条），格式：
  ```markdown
  ## 参考来源
  - [资料标题](../sources/xxx.md)
  ```
- `关联页面` — 至少 2 个相关 wiki 页
- `推荐继续阅读` — 至少 1 个外部资源
- **源码开放状态（有项目页时）** — 在「工程实践」或「局限与风险」中写明项目页核查结论（见步骤 2.5）；已开源则链到 `sources/repos/` 与官方仓库
- **Mermaid 流程图（推荐，管线类资料建议必做）** — 若资料的主贡献是**多阶段数据流、训练流水线或闭环系统**（例如「采集 → 重定向 → 仿真修正 → 策略训练」），在升格后的 wiki 页中增加一节（如「流程总览」），用 ```mermaid 代码块绘制**一张主干流程图**：节点对应模块边界，边对应数据/监督信号流向；子细节可用文字分节或第二张图，避免单图过度拥挤。渲染侧以 GitHub / 站点 Mermaid 为准，避免使用非标准语法。
- 必要时更新 `index.md`

> **为什么要在 wiki 页面内标注来源**：log.md 记录了操作时间线，但页面本身也应能追溯知识来源，
> 这样读者不依赖 log.md 就能知道这个 wiki 页的知识是从哪里编译来的。

### 步骤 6：更新 catalog.md Page Catalog

每次新增 wiki 页面后，必须重新生成完整目录 `catalog.md`：

```bash
make catalog  # 等价于 python3 scripts/generate_page_catalog.py
```

### 步骤 7：运行导出脚本

如果新增或大幅修改了 wiki 页面，运行导出脚本同步到前端：

```bash
make export  # 等价于 python3 scripts/export_minimal.py
# 同时更新 exports/index-v1.json、exports/site-data-v1.json、docs/sitemap.xml
```

### 步骤 7.5：批量 bump 交叉引用 wiki 的 `updated`（推荐）

更新 `sources/papers/*.md` 并改写其中 `wiki/...` 映射后，**先** bump 链到的 wiki 页日历日，再 **一次性 commit**，最后只跑 **一轮** `make ci-preflight`：

```bash
# 仅 bump 本次改动的 source 所链接的 wiki 页；省略参数则扫描全部 papers
python3 scripts/bump_wiki_updated_for_sources.py sources/papers/your_new_paper.md
# 或：make bump-wiki-from-sources
```

这可避免 lint「sources 比 wiki 新」在多轮 preflight 里反复失败（尤其 agent 在未 commit 时重试）。

### 步骤 8：健康检查

确认知识库健康状态：

```bash
make ci-preflight  # 推荐：同步派生文件 + lint + export 质量门（单次约 2–5 分钟）
# 仅快速体检、不改派生文件时：make lint
```

**提交前完整门禁**（与 GitHub Actions 对齐；`ci-preflight` 本身**不含** pytest / ruff / mypy）：

```bash
make ci-test       # 镜像 .github/workflows/tests.yml（含 pytest）
```

#### ingest 高发 CI 失败点

| 症状 / 触发场景 | 根因 | 本地修复 |
|----------------|------|----------|
| **Tests / pytest** 报 `institution ... label=...` | 新增 `schema/institutions.json` 时 `label` 非 `中文（English）`（纯英文或颠倒格式） | 按 [naming.md § 研究机构命名](naming.md) 改 `label`，再 `make test` |
| **Wiki Lint** 或 **Export Quality** 失败 | 只跑了 `make catalog` / `make graph` 之一，派生 JSON / `index.md` / badge 未同步 | 只跑一轮 `make ci-preflight`，把列出的派生文件一并 commit |
| **CI PR Gate (smoke)** 失败 | 大改后未 `make ci-preflight` 或 `make ci-check` | `make ci-check` 确认工作区与重生派生文件一致 |
| **pytest** `FileNotFoundError`（`link-graph.json` 等） | 全新环境未生成 gitignore 的站点 JSON | 先 `make export graph`，再 `make test` |
| lint「sources 比 wiki 新」反复失败 | 交叉改多个 wiki 后未 bump `updated` | 先 `make bump-wiki-from-sources`（或指定 source），再 **一轮** `make ci-preflight` |
| 首页「最新知识节点」缺本次页 | `log.md` 当日块未写 `wiki/...` 路径 | `make log` 正文显式列出相关 wiki 路径后重跑 `make ci-preflight` |

> **维护者习惯**：wiki / sources / schema 改动后，默认顺序为 `make ci-preflight` →（若动过 `institutions.json`、脚本或 `tests/`）`make test` → commit 全部相关派生文件 → push。

### 步骤 9：记录到 `log.md`

每次 ingest 都写入 `log.md` 顶部（`make log` / `append_log.py` 在首条 `## [` 之前插入，与首页 `latest_wiki_nodes` 解析一致）：

```bash
make log OP=ingest DESC="sources/papers/xxx.md — 简述覆盖的 wiki 页面"
# 等价于 python3 scripts/append_log.py ingest "sources/papers/xxx.md — 简述"
```

格式参考 `schema/log-format.md`。

**首页「最新知识节点」**：静态站 `docs/index.html` 通过 `exports/home-stats.json` 中的 `latest_wiki_nodes`（数组）渲染；`latest_wiki_node` 为列表首项，供兼容旧逻辑。数据由 `make graph`（`scripts/generate_link_graph.py`）写入 `exports/graph-stats.json`，再由 `scripts/generate_home_stats.py` 拷贝。解析规则：在 `log.md` 中**自上而下**取首条 `## [日期] ...` 的**日历日期**为「最新日」，**连续合并**该日期的所有日志块，在这些块正文中按出现顺序收集全部指向现存 wiki 文件、且在图谱中的 `wiki/...` 路径（**去重**；**ingest / structural / query 等任意 op 均可**）。若当日块中没有任何可解析的 wiki 路径，则回退到全库「最近更新」启发式（列表仅一项）。因此：凡是希望读者在首页看到对应更新的 wiki 工作，都应在当日 `log.md` 条目中**显式写出**相关 `wiki/...` 路径；仅写 `sources/` 或脚本路径时，该条不会贡献首页节点。维护完成后运行 `make ci-preflight` 以同步 `exports/` 与 `docs/exports/`。

---

## Op 2：Query — 针对知识库提问

### 触发条件

- 想了解某个概念但不想翻完所有原始资料
- 想找"哪个页面涉及 X 和 Y 的对比"
- 想确认"关于这个主题目前 wiki 里有什么"
- 主动发现了一个值得深挖的连接

### 标准流程

**Step 1：定位相关页面**

先读 `index.md` 和相关分类页，找到最接近的 2-4 个 wiki 页。

**Step 2：综合分析**

精读相关页面，提取：
- 各页的核心结论
- 它们的相同点和分歧点
- 是否有页面已经过时或相互矛盾

**Step 3：形成答案**

好的答案可以是：
- 一个 wiki comparison 页面（两个方法的系统对比）
- 一个新的概念页（把多个来源的洞见综合成一篇）
- 对已有页面的补充（补充 cross-reference 或纠正矛盾）

**Step 4：判断是否写回 wiki**

写回标准：有独立的认知价值——不是简单的复述，而是有新的连接、对比或洞见。

**Step 5：写回到 wiki**

- 独立洞见（跨多个 wiki 页面的综合分析）→ 新建 `wiki/queries/xxx.md`，并在页面顶部注明触发问题
- 对比分析 → 新建 `wiki/comparisons/xxx.md`
- 新概念 → 新建 `wiki/concepts/xxx.md`
- 补充性内容 → 找到最接近的已有页面插入

写回页面格式要求（query 产物特有）：
```markdown
> **Query 产物**：本页由以下问题触发：「<问题一句话>」
> 综合来源：<列出精读的 wiki 页面>
```

**Step 6：记录到 `log.md`**

```markdown
## [YYYY-MM-DD] query | <topic> | <问题一句话>

- Q: <问题摘要>
- A: <结论摘要>
- 写回：<写到哪个页面了，新页面还是已有页面>
- 涉及页面：<精读了哪些 wiki 页>
```

---

## Op 3：Lint — 定期健康检查

### 触发条件

- 每天早上定时任务自动跑（主要）
- 每隔一段时间（建议至少每 1-2 周一次）手动触发
- 新增大量页面后
- 发现某块内容明显过时或矛盾时

### 检查清单

1. **Orphan pages**：是否有页面没有任何 inbound link
2. **矛盾点**：新旧页面之间是否存在事实冲突
3. **缺失的 cross-reference**：某个概念被提到但没有自己的页面
4. **过时内容**：某页面长期未更新但相关来源已有新发展
5. **空壳页面**：只有标题没有实质内容的页面
6. **stale related**：某页面的 `related` 列表已经明显不反映当前知识结构
7. **索引一致性**：index.md 中的页面列表是否和实际 wiki 目录一致
8. **缺失英文缩写速查**：知识页是否缺少 `## 英文缩写速查` 区块（`make lint` 会统计 backlog；**新建或大幅改写页面时必须补齐**）

### Lint 后记录

```markdown
## [YYYY-MM-DD] lint | health-check | <简短描述>

- 发现：<问题 1>
- 发现：<问题 2>
- 修复：<问题 X 已修复>
- 待处理：<问题 Y 暂未处理，原因>
```

---

## Op 4：Catalog — 维护完整页面目录

### 触发条件

- 每次新增 wiki 页面后
- 每次删除或重命名 wiki 页面后
- 定期检查（建议配合 Lint 一起做）

### 维护原则

`index.md` 只保留核心导航与推荐阅读顺序；`catalog.md` 是自动生成的全量页面目录。每次新增页面必须重新生成目录，不允许知识页缺席于 `catalog.md`。

更新内容：
- 运行 `make catalog`，由脚本扫描 `wiki/`、`roadmap/`、`tech-map/` 与 `references/`
- 不要手工编辑 `catalog.md`
- 只有核心入口、推荐顺序或主线变化时才手工更新 `index.md`

---

## 操作符约定

| Op | 含义 |
|----|------|
| `ingest` | 新资料进入知识库 |
| `query` | 向知识库提问并将结果写回 |
| `lint` | 定期健康检查 |
| `catalog` | 重新生成 `catalog.md` 完整页面目录 |
| `structural` | 结构性变更（新增页面类型、路由调整等） |

---

## 与 `docs/change-log.md` 的分工

- **`log.md`** = 每次操作的时间线（ingest / query / lint / structural）
- **`docs/change-log.md`** = 重要结构性变更的里程碑记录（V1→V2、新入口/路由重大调整等）

日常维护优先写 `log.md`；里程碑性质的变化（如版本升级、重大架构调整）才写 `docs/change-log.md`。

---

## 注意事项

- 不要把 source 直接复制成 wiki — 要提炼，不是转存
- 不要把 wiki 页写成纯外链列表 — 要有知识归纳
- 不要为了收集而收集 — 优先服务学习与研究主线
- 不要在 ingest 时一次性做太多事 — 一次一条资料，深度到位再推进
- **有项目页必先查源码是否开放**（步骤 2.5），再写 wiki 开源表述与 `sources/repos/` 归档
- 每次 ingest 都要运行 `make catalog` 更新 `catalog.md`，并追加 `log.md`，不要遗漏
- 子网页优化、纯 wiki 扩写等 **structural** 记录若在当日 `log.md` 正文中写明 `wiki/...`，首页「最新知识节点」会与其他同日条目一并列出；提交前务必 `make ci-preflight` 刷新派生 JSON
