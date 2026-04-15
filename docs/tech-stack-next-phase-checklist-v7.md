# 技术栈项目执行清单 v7

最后更新：2026-04-15
项目仓库：<https://github.com/ImChong/Robotics_Notebooks>
上一版清单：[`docs/tech-stack-next-phase-checklist-v6.md`](tech-stack-next-phase-checklist-v6.md)
方法论参考：[Karpathy LLM Wiki](../wiki/references/llm-wiki-karpathy.md)

---

## V6 完成基线（V7 起点）

| 维度 | V6 末状态 |
|------|----------|
| wiki 页面总数 | ~100（含 teleoperation / reward-design-guide / sim2real-gap-reduction 等新页面） |
| sources/papers/ 文件 | 21 |
| Sources 覆盖率 | 73%（45/62 wiki 页有 ingest 来源） |
| Lint 健康 | ✅ 10 项检测，6 条 CANONICAL_FACTS 规则 |
| Query 产物 | 11 个（wiki/queries/） |
| TF-IDF 搜索 | ✅ search_wiki.py 升级，支持 `--json` 输出 |
| Marp 幻灯片 | ✅ wiki_to_marp.py + `make slides` |
| Web 资料导入 | ✅ fetch_to_source.py + `make fetch` |
| 知识图谱前端 | ✅ docs/graph.html（D3.js 力导向图，物理参数调节 / 主题切换 / 悬停点击卡片） |
| 前端键盘导航 | ❌（V6 P4 全部延续至 V7） |
| GitHub Actions | ❌（V6 P6.2 延续至 V7） |

---

## V7 阶段总目标

> 基于 Karpathy LLM Wiki 原文的三个新洞见做深度对齐：
> 1. **"Obsidian's graph view is the best way to see the shape of your wiki"** — graph.html 已完成独立页面，V7 将其嵌入首页，让图谱成为项目的第一入口而非隐藏功能。
> 2. **"Good answers can be filed back into the wiki as new pages"** — 进一步提升 Query 产物质量，建立 query 回流机制，让每次探索都沉淀为知识。
> 3. **"The wiki stays maintained because the cost of maintenance is near zero"** — 完善 CI/CD 自动化，让 lint / catalog / export 在每次 push 后自动触发，维护成本趋近于零。

---

## P0 · 图谱视图主页集成（前端重设计 Phase 1）

**背景**：docs/graph.html 已完整实现，但入口仍是首页上的一个按钮。Karpathy 原文：*"Obsidian's graph view is the best way to see the shape of your wiki."* V7 把图谱作为首页核心区块嵌入，而非外链跳转。

### 0.1 首页图谱预览区块

- [ ] `docs/index.html` 新增 "知识图谱" section：嵌入一个缩小版力导向图（同 graph.html 但高度固定，禁用物理面板，点击节点直接跳转 detail.html），节点数上限 40（按连接度 Top40 过滤）
- [ ] section 下方显示实时统计：节点数 / 边数 / 覆盖率，从 `docs/exports/link-graph.json` 和 `docs/exports/index-v1.json` 读取

### 0.2 graph.html 交互增强

- [ ] 节点详情侧边栏（替代浮动卡片）：点击节点时右侧滑出侧边栏，显示完整 summary / tags / 关联页面列表，移动端体验更好
- [ ] 节点搜索与定位：搜索命中时自动居中并高亮目标节点（camera fly-to 动画）
- [ ] 孤儿节点高亮模式：过滤器新增 "孤儿" 选项，帮助发现缺少外链的页面

### 完成标准
- 首页可以不跳转直接看到图谱预览
- graph.html 搜索可定位节点
- 完成后 `make export` + 手动 browser 验证

---

## P1 · 前端体验增强（V6 P4 全部延续）

### 1.1 index.html 搜索体验

- [ ] 搜索结果键盘导航：↑↓ 选中高亮，Enter 打开详情页，Esc 清空搜索
- [ ] 搜索框下方标签云：统计高频 tag（Top 20），点击直接过滤；从 `docs/exports/index-v1.json` 的 tags 字段统计

### 1.2 detail.html 增强

- [ ] 关联页面列表改为卡片式（显示 type badge + summary 前 60 字）
- [ ] `og:image` 动态设置：按 page_type 选择预设图标（concept/method/task 各一张），回退到项目 logo

### 1.3 tech-map.html 增强

- [ ] 层级筛选器多选（Ctrl+Click 多层级同时显示）
- [ ] 页面内节点搜索（实时过滤匹配节点）

> 所有前端改动需 `make export && open docs/index.html` browser 验证，无法验证标记 `[-]`。

### 完成标准
- 键盘导航 ↑↓/Enter 可用
- 标签云正常渲染
- detail.html 关联卡片显示

---

## P2 · CI/CD 自动化（V6 P6.2/6.3 延续）

**背景**：Karpathy 原文：*"The wiki is just a git repo of markdown files. You get version history, branching, and collaboration for free."* V7 接上 CI，让每次 push 自动维护知识库健康。

### 2.1 GitHub Actions auto-export

- [ ] `.github/workflows/export.yml`：push 到 main 时自动运行：
  ```bash
  make lint && make catalog && make export && make graph
  ```
  若 lint 有 issues 则失败；成功后将更新的 JSON 文件提交回仓库

### 2.2 动态 Sources 覆盖率 Badge

- [ ] `README.md` 中加入动态覆盖率 badge（通过 shields.io 静态 badge，每次手动 push 时更新数值，目标 ≥75% 时为绿色，50-74% 为黄色）

### 2.3 GitHub Pages 部署稳定性

- [ ] 确认 `docs/exports/link-graph.json` 已被 Pages 正确服务（验证 graph.html 在 GitHub Pages 上可加载图数据）
- [ ] `docs/` 目录结构检查：所有 HTML 引用的相对路径在 Pages 环境中均正确

### 完成标准
- push main 后 Actions 自动运行并成功
- Badge 在 README 中显示绿色

---

## P3 · Sources 层扩充（73% → 82%+）

**背景**：V6 达到 73% 覆盖率，距离 Karpathy 目标 75%+ 仅差 2%，V7 推进到 82%+（约 50/61 页有 ingest 来源）。

### 3.1 新增 sources 文件

| 目标文件 | 覆盖 wiki 页面 | 关键资源 |
|---------|--------------|--------|
| `sources/papers/perception_localization.md` | sensor-fusion.md, state-estimation.md（感知角度） | 视觉惯性里程计、Kalman 滤波类文献 |
| `sources/papers/optimal_control_theory.md` | optimal-control.md, trajectory-optimization.md, mpc.md | Betts 2010 survey; Diehl 2006 NMPC; Kelly 2017 intro to TO |
| `sources/papers/humanoid_hardware.md` | humanoid-robot.md + 新建 wiki/entities/atlas.md, unitree-h1.md | Boston Dynamics Atlas; Unitree H1; Agility Digit 系统设计 |
| `sources/papers/rl_foundation_models.md` | locomotion.md, policy-optimization.md + 新建 wiki/concepts/foundation-policy.md | RT-2, π₀, Octo 等跨任务策略模型 |

- [ ] `sources/papers/perception_localization.md`（覆盖 sensor-fusion / state-estimation）
- [ ] `sources/papers/optimal_control_theory.md`（覆盖 optimal-control / trajectory-optimization / mpc）
- [ ] `sources/papers/humanoid_hardware.md`（覆盖 humanoid-robot + 新建 2 个 entity 页）
- [ ] `sources/papers/rl_foundation_models.md`（覆盖 locomotion / policy-optimization + 新建 foundation-policy 概念页）

### 3.2 Query 产物扩充（11 → 14 个）

基于 Karpathy *"good answers can be filed back into the wiki"*：

- [ ] `wiki/queries/hardware-comparison.md`：主流人形机器人硬件对比（Atlas / H1 / Digit / Spot / TORO）
- [ ] `wiki/queries/rl-hyperparameter-guide.md`：PPO/SAC 超参数调节 checklist（batch size / clip range / entropy coeff / GAE λ）
- [ ] `wiki/queries/when-to-use-wbc-vs-rl.md`：决策树型 query 产物，帮助快速判断控制架构选择

### 完成标准
- sources/papers/ 达到 25 个文件
- Sources 覆盖率 ≥ 82%
- query 产物 14 个，wiki/queries/README.md 同步更新

---

## P4 · Lint 规则完善（CANONICAL_FACTS 6 → 10 条）

**背景**：V6 实现了 6 条 CANONICAL_FACTS（原计划 10 条，4 条因 wiki 内容不足而推迟）。V7 补全剩余 4 条。

### 4.1 补全 CANONICAL_FACTS 规则 7–10

| 规则 ID | 主题 | 正面断言 | 负面断言 |
|--------|------|---------|---------|
| 7 | TSID vs QP 等价性 | TSID 是基于 QP 的 WBC 框架 | TSID 与 QP 是不同的控制方法 |
| 8 | 全身控制必要性 | WBC 在多接触任务中优于独立关节控制 | 独立关节控制可替代 WBC |
| 9 | MuJoCo 接触精度 | MuJoCo 接触模型精度高于早期刚体仿真器 | MuJoCo 接触仿真不适合精密接触任务 |
| 10 | 仿真频率稳定性 | 高仿真频率对接触稳定性至关重要 | 低仿真频率足以支持接触丰富任务 |

- [ ] `scripts/lint_wiki.py`：添加规则 7–10，CANONICAL_FACTS 达到 10 条
- [ ] 验证：`make lint` 仍保持 0 真实 issues

### 4.2 Lint 输出增强

- [ ] `scripts/lint_wiki.py` 新增 `--report` 参数：输出 markdown 格式的健康报告，包含覆盖率统计 / 孤儿页列表 / 矛盾摘要，保存至 `exports/lint-report.md`

### 完成标准
- CANONICAL_FACTS = 10 条
- `make lint --report` 输出格式正确的 lint-report.md

---

## P5 · Search 语义化增强（TF-IDF → 混合检索）

**背景**：Karpathy 原文推荐 *"hybrid BM25/vector search with LLM re-ranking"*（参考 qmd 工具）。V6 实现了 TF-IDF 作为 Step 1。V7 探索向量化方向。

### 5.1 本地向量搜索原型

- [ ] `scripts/embed_wiki.py`：调用本地嵌入模型（sentence-transformers / ollama）或 Anthropic Embeddings API，为每个 wiki 页面生成向量，存储至 `exports/wiki-embeddings.npz`
- [ ] `scripts/search_wiki.py` 新增 `--semantic` 参数：加载向量，计算余弦相似度，与 TF-IDF 分数加权混合后排序

### 5.2 search 结果缓存

- [ ] `exports/search-cache.json`：缓存最近 50 次查询结果（含 query / results / timestamp），`search_wiki.py` 先查缓存再计算，命中时直接返回

> 注：P5 依赖外部模型可用性，如无法运行本地嵌入模型则标记 `[-]`，退而完善 TF-IDF 的 n-gram 支持。

### 完成标准
- `python3 scripts/search_wiki.py "全身控制" --semantic` 结果比纯 TF-IDF 更相关（人工判断）
- 或：TF-IDF 支持 n-gram（bigram），搜索精度提升（验证：`make search Q="model predictive"` 结果含 mpc.md）

---

## P6 · 知识库健康监控（Karpathy 新对齐项）

**背景**：Karpathy 提到 *"Dataview is an Obsidian plugin that runs queries over page frontmatter."* 我们可以用一致的 YAML frontmatter + 脚本查询实现类似效果，并生成图谱健康报告。

### 6.1 YAML Frontmatter 一致性

- [ ] 审计现有 wiki 页面的 frontmatter（type / tags / related / sources 字段），确认格式与 schema/page-types.md 一致
- [ ] `scripts/lint_wiki.py` 新增 frontmatter 格式检查：缺少 `type` / `tags` / `related` 字段时 warning

### 6.2 图谱健康报告

- [ ] `scripts/generate_link_graph.py` 输出增强：新增 `exports/graph-stats.json`，包含：
  - 度中心性 Top 10（最重要的 hub 节点）
  - 孤儿节点列表（入度 = 0 或出度 = 0）
  - 平均路径长度（近似）
  - 按 type 的节点分布
- [ ] `make graph` 同步生成 graph-stats.json

### 6.3 Karpathy Checklist 自评

每次执行清单结束时（下一版清单创建前），对照以下维度自评：

| Karpathy 原则 | 当前状态 | 目标 |
|-------------|---------|------|
| Raw sources（不可变 sources 层） | 21 文件，73% 覆盖 | **25 文件，82% 覆盖** |
| Wiki（LLM 维护的 md 文件集） | ~100 页，互联完整 | **105+ 页** |
| Schema（配置与规范文档） | ✅ schema/ 5 文件 | 同步更新 |
| Ingest "1 source → 10–15 pages" | ✅ coverage checker 辅助（V6 实现） | 持续优化 |
| Query 产物 | 11 个 | **14 个** |
| Lint（矛盾/陈旧/孤儿/覆盖率） | 10 项检测，6 条 CANONICAL_FACTS | **10 条 CANONICAL_FACTS** |
| Search（BM25/vector） | ✅ TF-IDF 排序 + `--json` | **混合检索原型** |
| log.md | ✅ 持续追加 | ✅ 持续追加 |
| Obsidian 图谱视图等效 | ✅ graph.html（独立页面） | **嵌入首页** |
| Dataview 等效（frontmatter 查询） | ❌ | **graph-stats.json + frontmatter lint** |
| CI/CD | ❌ | **✅ GitHub Actions** |

---

## 维护操作标准（V7 更新版）

### Op 1：Ingest（添加新资料）—— 目标 8–12 页面覆盖
```bash
1. make ingest NAME=xxx TITLE="..." DESC="..."
2. 编辑模板，填写论文摘录（至少 3 条核心论文 + wiki 映射）
3. make coverage F=sources/papers/xxx.md       # 审计覆盖范围
4. 在报告建议的 wiki 页面中补充 cross-reference 和 ingest 链接
5. make lint       # 确认 0 issues
6. make catalog    # 更新 index.md
7. make export && make graph   # 同步 JSON + 图谱
8. make log OP=ingest DESC="sources/papers/xxx.md — 简述，覆盖 N 个页面"
```

### Op 2：Query（知识查询）
```bash
1. make search Q=<关键词>                       # TF-IDF 排序搜索
2. python3 scripts/search_wiki.py <关键词> --related  # 加载邻居页面
3. 综合多页面分析，得出结论
4. 如有独立价值 → 保存为 wiki/queries/xxx.md
5. 更新 wiki/queries/README.md 的表格
6. make lint && make catalog && make export && make graph
7. make log OP=query DESC="关键词 → wiki/queries/xxx.md"
```

### Op 3：Lint（健康检查）
```bash
make lint                              # 完整健康检查（10 项 + 覆盖率报告）
make log OP=lint DESC="0 issues，覆盖率 XX%，矛盾 N 个"
```

### Op 4：Index（索引更新）
```bash
make catalog    # 刷新 index.md
make export     # 更新 exports/ JSON + sitemap.xml
make graph      # 更新 link-graph.json + graph-stats.json（V7 新增）
```

---

## 状态约定
- `[ ]` 未开始
- `[~]` 进行中
- `[x]` 已完成
- `[-]` 暂缓
