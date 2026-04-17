# 技术栈项目执行清单 v10

最后更新：2026-04-17（V10 启动，基于 V9 完整交付）
项目仓库：<https://github.com/ImChong/Robotics_Notebooks>
上一版清单：[`docs/tech-stack-next-phase-checklist-v9.md`](tech-stack-next-phase-checklist-v9.md)
方法论参考：[Karpathy LLM Wiki](../wiki/references/llm-wiki-karpathy.md)

---

## V9 完成基线（V10 起点）

| 维度 | V9 末状态 |
|------|----------|
| wiki 节点（图谱） | **75 个**（concepts 23 / query 14 / entity 8 / formalization 8 / method 7 / comparison 6 / task 6） |
| 图谱边数 | **429 条** |
| 总页面（exports） | **110 页**（wiki 65 / tech_map 18 / reference 13 / entity 8 / roadmap 6） |
| Lint 检测项 | **13 项**，0 issues |
| CANONICAL_FACTS | **20 条** |
| Sources 覆盖率 | **100%**（73/73） |
| BM25 搜索 | ✅ 启发式重排 4 规则（title ×5 / summary ×2 / 新鲜度 ×1.2 / query 降权 ×0.7） |
| 向量搜索 | ❌ 尚未实现（V9 P3 延期） |
| GitHub Actions | ✅ weekly-lint.yml + export.yml badge 自动更新 |
| 幻灯片 UX | ✅ graph.html sidebar 📊 按钮 + 搜索卡片 📊 图标 |

---

## V10 阶段总目标

> V10 围绕**三个新主题**推进，同时补完 V9 遗留的向量搜索：
>
> 1. **向量搜索（V9 P3 补完）** — 本地 sentence-transformers，CPU 可用，无需 GPU，真正实现 Karpathy "hybrid BM25/vector" 架构。
>
> 2. **内容深度（方法页扩充）** — DAgger / VLA / contact-rich 等在多个现有页中被引用但缺少独立页的高频概念。同步补全 query 页面覆盖模仿学习和 loco-manipulation 实践问答。
>
> 3. **离线浏览器搜索（docs/ 纯前端 BM25）** — 预计算 BM25 词频索引存入 JSON，`docs/main.js` 在本地 WebWorker 中执行搜索，无需 Python 服务器——让 GitHub Pages 用户也能获得完整搜索体验。

---

## P0 · 向量搜索（V9 P3 补完，最高优先级）

**背景**：V9 已完成 BM25 + 启发式重排，但 Karpathy "hybrid BM25/vector" 的向量半层未实现。V10 补完。

### P0.1 · 本地向量索引构建

- [x] 新建 `scripts/build_vector_index.py`：
  ```python
  # sentence-transformers，CPU 可用（pip install sentence-transformers）
  # 模型：all-MiniLM-L6-v2（22MB，中英文双语可用）
  # 处理：strip frontmatter → 截取前 512 token → encode
  # 输出：exports/vector-index.npz（numpy 矩阵）
  #       exports/vector-index-meta.json（路径/title/type 映射）
  ```
- [x] 在 `Makefile` 中添加 `make vectors` 目标
- [x] 将 `exports/vector-index.npz` 加入 `.gitignore`（体积大，按需本地生成）
- [x] 将 `exports/vector-index-meta.json` **纳入** git（路径映射小，CI 可用）

### P0.2 · 混合搜索模式

- [x] 在 `scripts/search_wiki.py` 中添加 `--semantic` 标志：
  - 若 `exports/vector-index.npz` 存在 → 执行余弦相似度检索
  - 混合得分：`final = 0.6*bm25_norm + 0.4*cosine`
  - 若索引不存在 → fallback 到纯 BM25 + 提示 `make vectors`
- [x] `--json` 输出中添加 `vector_score` 和 `hybrid_score` 字段
- [x] 测试：`python3 scripts/search_wiki.py "腿足稳定性" --semantic` 能找到 `formalizations/lyapunov.md`（即使无关键词匹配）

### 完成标准
`--semantic` 查询 "运动控制稳定性" 返回结果中 `lyapunov.md` 排名前 5，且与纯 BM25 结果至少有 1 条差异（说明向量层有贡献）。

---

## P1 · 内容深度扩充（高频引用但缺独立页的概念）

### P1.1 · 缺失方法页（methods/）

以下概念在现有 wiki 中被高频引用（各引用 ≥ 3 处）但无独立页：

| 文件 | 内容要点 | 主要关联 |
|------|---------|---------|
| `wiki/methods/dagger.md` | DAgger 算法：交互式 IL / 专家干预 / 分布漂移修正 / 与 BC 的对比 | imitation-learning.md, ultra-survey.md |
| `wiki/methods/vla.md` | VLA（Vision-Language-Action）：多模态模型 / π₀ / RT-2 / 端到端策略 | manipulation.md, loco-manipulation.md |
| `wiki/methods/behavior-cloning.md` | BC（行为克隆）：监督学习 IL / compounding error / covariate shift | imitation-learning.md, dagger.md |

- [x] 新建 3 个方法页（满足 schema/page-types.md 规范：frontmatter + 参考来源 + 关联页面）
- [x] 在 `imitation-learning.md` / `manipulation.md` 的 `## 关联页面` 中添加回链
- [x] 为 3 个新页补充 `sources:` frontmatter（引用现有 `sources/papers/` 文件）
- [x] `make lint` 验证 0 issues + 覆盖率维持 100%

### P1.2 · 缺失概念页（concepts/）

| 文件 | 内容要点 | 主要关联 |
|------|---------|---------|
| `wiki/concepts/contact-rich-manipulation.md` | 接触丰富型操作：软/硬接触建模 / 双边约束 / impedance control / 与无接触操作的对比 | manipulation.md, contact-dynamics.md, tsid.md |
| `wiki/concepts/terrain-adaptation.md` | 地形适应：感知 → 规划 → 接触序列 / 腿足跨越障碍 / 高度图 | locomotion.md, footstep-planning.md, sim2real.md |

- [x] 新建 2 个概念页
- [x] 在 `locomotion.md` / `manipulation.md` 关联页中添加回链

### P1.3 · 缺失 Query 页面（queries/）

高价值问题，尚无对应 query 页：

| 文件 | 触发问题 |
|------|---------|
| `wiki/queries/il-for-manipulation.md` | 「做机器人操作用模仿学习还是 RL？怎么收集数据？」 |
| `wiki/queries/vla-deployment-guide.md` | 「如何在真机上部署 VLA 策略？推理延迟怎么控制？」 |

- [x] 新建 2 个 query 页（type: query，包含决策树或对比表格）

### 完成标准
```bash
python3 scripts/search_wiki.py "行为克隆" | head -5  # behavior-cloning.md 应出现
python3 scripts/search_wiki.py "VLA 部署" | head -5  # vla.md 和 query/vla-deployment 应出现
```

---

## P2 · Lint 增强（CANONICAL_FACTS 扩展至 30 条）

**背景**：V9 已有 20 条事实断言，覆盖 PPO / MPC / WBC / IL / Sim2Real 等核心领域。V10 扩展至 30 条，覆盖新增的 DAgger / VLA / 接触 / 地形适应域。

### P2.1 · 新增 10 条 CANONICAL_FACTS

在 `scripts/lint_wiki.py` 的 `CANONICAL_FACTS` 列表中补充以下事实：

| 事实名称 | 正向断言 | 反向断言（不应出现） |
|---------|---------|-------------------|
| DAgger 数据效率 | DAgger 比 BC 数据效率更高（解决分布漂移） | DAgger 与 BC 等价 / DAgger 不处理 covariate shift |
| VLA 推理延迟 | VLA 策略推理通常需要 50ms+ | VLA 实时性与传统控制器相当 |
| 行为克隆 compounding error | BC 存在 compounding error 问题 | BC 累积误差与序列长度无关 |
| 接触力摩擦约束 | 接触力需满足摩擦锥约束（$\|f_{xy}\| \leq \mu f_z$） | 接触力无需摩擦锥约束 |
| 腿足地形感知 | 腿足机器人地形适应通常依赖高度图或点云 | 腿足机器人不需要任何地形感知 |
| MiniLM 向量维度 | all-MiniLM-L6-v2 输出 384 维嵌入 | MiniLM 输出 768 维 |
| Marp 幻灯片格式 | Marp 使用 Markdown + frontmatter 生成幻灯片 | Marp 需要 LaTeX / Beamer |
| sentence-transformers CPU | sentence-transformers 可在 CPU 运行，无需 GPU | sentence-transformers 必须 GPU |
| VLA 训练数据规模 | VLA 通常需要大量多样化演示数据（数千+条） | VLA 可在十条演示上收敛 |
| BM25 参数含义 | BM25 中 k1 控制词频饱和，b 控制文档长度归一化 | BM25 中 b 参数与词频无关 |

- [x] 在 `CANONICAL_FACTS` 列表中添加 10 条（共 30 条）
- [x] `make lint` 验证 0 矛盾报告
- [x] 验证无误判（在新增页面中故意放一条，确认能检测到）

### P2.2 · 新增 lint 规则：`summary:` 字段完整性检查

Karpathy："*frontmatter should be machine-readable*" — 当前 query 页面有 `summary:` 字段，但 concept/method/task 页面缺失统一摘要字段。

- [x] 在 `lint_wiki.py` 中添加规则：检查 `wiki/concepts/`、`wiki/methods/`、`wiki/tasks/` 目录下的页面，若 frontmatter 中既无 `summary:` 也无 `description:` 字段，则输出 `⚠️ 缺少摘要字段`
- [x] 目标：lint 检测项从 13 增至 **14 项**
- [x] 同步修复检测到的页面（批量添加 1-2 行 summary 字段）

### 完成标准
`make lint` 输出 `✅ 所有检查通过！`，检测项 ≥ 14 项。

---

## P3 · 前端离线搜索（GitHub Pages 无服务器 BM25）

**背景**：当前 `search_wiki.py` 需要 Python 环境。GitHub Pages 用户无法通过网站搜索，只能用 graph.html 可视化浏览。V10 将 BM25 索引预计算为 JSON，在浏览器端执行搜索。

### P3.1 · 预计算 BM25 词频索引

- [x] 在 `scripts/export_minimal.py`（或新建 `scripts/build_search_index.py`）中：
  - 遍历所有 wiki 页面，strip frontmatter，分词（中英文空格/标点切分）
  - 计算 TF 词频表 + IDF（基于文档集），存入 `docs/search-index.json`
  - 格式：`{ "meta": {"avgdl": X, "N": Y}, "docs": [{"id":"path","title":"..","tokens":{word:tf,...},"summary":".."}] }`
  - 文件大小控制在 < 500KB（必要时过滤停用词）
- [x] 在 `Makefile` 中集成到 `make export`（确保 Pages 部署时索引自动更新）

### P3.2 · `docs/main.js` 中集成离线搜索

- [x] 在 `docs/main.js` 中加载 `search-index.json`（懒加载，首次搜索时 fetch）
- [x] 实现浏览器端 BM25 评分函数（≤ 50 行 JS）：
  ```javascript
  function bm25Score(tf, idf, dl, avgdl, k1=1.5, b=0.75) {
    return idf * tf * (k1+1) / (tf + k1*(1 - b + b*dl/avgdl));
  }
  ```
- [x] 搜索框输入时触发：分词 → 查 IDF → 累加 BM25 → Top 10 结果渲染
- [x] 与现有 `renderSearchResults()` 复用卡片样式（title / summary / tags）
- [x] 降级策略：若 `search-index.json` 加载失败，显示 "请使用命令行搜索" 提示

### 完成标准
打开 `docs/index.html`（file:// 协议），在搜索框输入 "强化学习"，在 500ms 内返回结果，且 `reinforcement-learning.md` 排名前 3。

---

## P4 · 知识图谱节点聚类（graph.html 社区着色）

**背景**：当前 graph.html 节点颜色按 `type` 字段区分。V10 添加**社区检测**，揭示知识模块边界（RL 社区 / WBC 社区 / 操作社区等）。

- [x] 在 `scripts/generate_link_graph.py` 中输出 `exports/link-graph.json` / `docs/exports/link-graph.json`，新增 `community` / `community_label` 字段（纯 Python Girvan-Newman 近似实现）
  - 可用 `networkx` + `python-louvain` 包：`pip install networkx python-louvain`
  - 社区数上限 8，超出归入 "其他" 社区
- [x] 在 `docs/graph.html` 中：
  - 添加社区切换按钮："按类型" / "按社区"
  - 按社区着色时，使用 d3.schemeTableau10 色板（10 色）
  - 侧边栏显示当前节点所属社区名称（按社区内最高 degree 节点命名，如 "RL 社区"）
- [x] 图例 legend 更新：社区模式下显示社区名 + 颜色块

### 完成标准
`docs/graph.html` 有 "按社区" 按钮，点击后节点重新着色，社区名在 legend 中可见。

---

## P5 · Anki 闪卡导出（知识再利用）

**背景**：Karpathy 强调 "knowledge reuse at zero marginal cost"。Formalization 页面的公式和定义天然适合 Anki 闪卡学习。

### P5.1 · `scripts/export_anki.py`

- [x] 读取 `wiki/formalizations/*.md`，提取：
  - 卡片正面：标题 + 一句话定义（H1 下第一段）
  - 卡片背面：核心公式（第一个 `$$` 块）+ 关联概念（关联页面列表）
- [x] 输出格式：Anki 兼容 TSV（`exports/anki-flashcards.tsv`）
  - 列：`Front\tBack\tTags`
  - Tags：`robotics::formalization::<page_stem>`
- [x] 在 `Makefile` 中添加 `make anki` 目标

### P5.2 · 扩展到 concepts/ 精选页面

- [x] 额外处理 `wiki/concepts/` 中有 `## 一句话定义` 小节的页面（当前导出 19 页）
- [x] 总卡片数目标：≥ 20 张（当前 27 张）

### 完成标准
`make anki` 生成 `exports/anki-flashcards.tsv`，≥ 20 行，可导入 Anki（通过 File > Import 验证格式）。

---

## P6 · README 与指标同步

- [x] 更新 README badge：图谱节点 / 边数与当前导出一致（75 / 429）
- [x] 更新 "常用操作" 命令表：新增 `make vectors`、`make anki`
- [x] README 版本注释从 V9 → V10
- [x] 更新执行清单链接从 v9 → v10

---

## Karpathy 方法论自我评估（V10 目标）

| Karpathy 原则 | V9 末状态 | V10 目标 |
|--------------|---------|---------|
| Three-layer architecture | ✅ | ✅ 维持 |
| 0 orphans，交叉引用完整 | ✅ 75 节点 429 边 | ✅ ~85 节点 ~500 边 |
| "Good answers filed back" | ✅ weekly Actions | ✅ + 2 新 query 页 |
| Lint: 13 检测项 | ✅ 13 项 | ✅ **14 项**（+summary 字段检查） |
| CANONICAL_FACTS | ✅ 20 条 | ✅ **30 条** |
| Sources 覆盖率 100% | ✅ | ✅ 维持 |
| Hybrid BM25/vector | ⚠️ BM25 only | ✅ BM25 + sentence-transformers |
| 浏览器端搜索 | ❌ | ✅ GitHub Pages 离线 BM25 |
| 知识图谱社区检测 | ❌ | ✅ Louvain 社区 + 着色 |
| 知识再利用（Anki） | ❌ | ✅ `make anki` TSV 导出 |

---

## 操作规范（延续 V1→V10）

### Op 1 · 每次改动后必须运行

```bash
make lint      # 0 issues 为通过门槛
make export    # 更新 exports/index-v1.json
make graph     # 更新 link-graph.json + graph-stats.json
make badge     # 同步 README sources 覆盖率 badge
```

### Op 2 · 新建 wiki 页面清单

1. `type:` / `tags:` / `status:` / `related:` / `sources:` 必填
2. `summary:` 字段必填（V10 新要求，lint P2.2 检测）
3. 必须有 `## 参考来源` 区块（≥ 1 条）
4. 必须有 `## 关联页面` 区块（≥ 1 条）
5. 在至少一个现有页的 `## 关联页面` 中添加回链（防孤儿）

### Op 3 · Commit 规范

```
feat(wiki): 新建 <类型>/<页面名> — <一句话说明>
feat(search): 向量索引 / 前端 BM25
fix(lint): <修复内容>
chore(export): 更新前端 JSON / 图谱数据
```

### Op 4 · V10 完成标准（全部满足）

- [x] `make lint` 输出 `✅ 所有检查通过！`，检测项 ≥ 14 项
- [x] CANONICAL_FACTS = 30 条，`make lint` 无误判
- [x] `--semantic` 搜索能找到语义相关页（`lyapunov.md` 出现在 "稳定性" 查询结果前 5）
- [x] `docs/index.html` 浏览器内搜索 "强化学习" 返回结果 < 500ms
- [x] `make anki` 生成 ≥ 20 张闪卡（当前 27 张）
- [x] `docs/graph.html` 有 "按社区" 着色按钮

---

## 状态说明

- `[ ]` 待执行
- `[~]` 进行中
- `[x]` 已完成
- `[−]` 已评估，决定跳过（附理由）
