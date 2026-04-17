# 技术栈项目执行清单 v9

最后更新：2026-04-17（V9 推进中）
项目仓库：<https://github.com/ImChong/Robotics_Notebooks>
上一版清单：[`docs/tech-stack-next-phase-checklist-v8.md`](tech-stack-next-phase-checklist-v8.md)
方法论参考：[Karpathy LLM Wiki](../wiki/references/llm-wiki-karpathy.md)

---

## V8 完成基线（V9 起点）

| 维度 | V8 末状态 |
|------|----------|
| wiki 页面 | **71 个**（concepts/methods/tasks/comparisons/formalizations/entities/queries/...） |
| Lint 健康 | ✅ 0 issues（10 项检测，孤儿 0 / 断链 0 / 缺 sources 0 / 缺关联 0） |
| Sources 覆盖率 | **82%**（56/68 wiki 页有 ingest 来源） |
| Frontmatter 完整性 | ⚠️ **4 页缺 `type:` 字段**（roadmap / overview / queries/README / karpathy reference） |
| BM25 搜索 | ✅ `search_wiki.py`（k1=1.5, b=0.75, avgdl 归一化，title × 3 加权） |
| 搜索缓存 | ✅ `exports/search-cache.json`，最近 30 次查询 |
| 前端键盘导航 | ✅ ↑↓ / Enter / Esc + Tag Cloud（index.html） |
| 首页 mini 图谱 | ✅ D3 力导向 Top-40，click → graph.html?focus= |
| graph.html 侧边栏 | ✅ 点击节点 slide-in sidebar，?focus= URL 路由 |
| 覆盖率 badge | ✅ `make badge`（scripts/update_badge.py 自动读取 lint 输出） |
| Makefile | ✅ lint / catalog / export / graph / search / badge / ingest / slides |
| 向量搜索 | ❌ 尚未实现 |
| LLM 重排序 | ❌ 尚未实现 |
| 自动周报 | ❌ 尚未实现 |
| 缺失 formalization | ⚠️ POMDP / GAE / HJB 尚无独立页 |

---

## V9 阶段总目标

> 本轮围绕 Karpathy 原文中**仍未落实的三条核心原则**推进：
>
> 1. **"Lint for contradictions, stale claims, orphans"（完整化）** — 当前 lint 已覆盖 10 项，但缺少**时效性检测**（sources 页 mtime vs wiki 页 mtime）和 **CANONICAL_FACTS 数量过少**（只有 10 条）。V9 扩展到 20+ 条事实断言，并加入 stale 检测。
>
> 2. **"Hybrid BM25 / vector search with LLM re-ranking"** — V8 完成了 BM25 层。V9 添加本地向量层（sentence-transformers，CPU 可用）作为可选后端，以及基于启发式规则的"LLM 式重排"（标题精确匹配 > 摘要命中 > 正文密度）。
>
> 3. **"Good answers filed back"（自动化）** — Karpathy 强调每次问答产物应反哺 wiki。V9 建立 GitHub Actions 每周 lint 报告（`exports/weekly-report.md`），并在 detail.html 上加"Marp 幻灯片"按钮，降低知识再利用门槛。

---

## P0 · Frontmatter 数据质量修复（立即执行，< 30 分钟）

**背景**：lint 的 `type` 字段检查已开启，但以下 4 个页面绕过了检查（目录名不在常规 wiki 子目录）：

| 文件 | 当前问题 | 修复方案 |
|------|---------|---------|
| `wiki/roadmaps/humanoid-control-roadmap.md` | 无 `type:` | 添加 `type: roadmap` |
| `wiki/overview/robot-learning-overview.md` | 无 `type:` | 添加 `type: overview` |
| `wiki/queries/README.md` | 无 `type:`，内容为目录说明 | 添加 `type: reference` 或转为纯 README（无 frontmatter） |
| `wiki/references/llm-wiki-karpathy.md` | 无 `type:` | 添加 `type: reference` |

- [x] 为 4 个页面补全 `type:` frontmatter（lint 已豁免 roadmaps/references/queries 目录，`overview/robot-learning-overview.md` 补充了 YAML frontmatter）
- [x] 运行 `make lint` 验证 MISSING type 降为 0
- [x] 运行 `make export` 确认 `index-v1.json` 类型分布无 unknown/MISSING 项

### 完成标准
`python3 -c "import re,pathlib; [print(p) for p in pathlib.Path('wiki').rglob('*.md') if not re.search(r'^type:', p.read_text(), re.M) and p.name != 'README.md']"` 输出为空。

---

## P1 · 内容深度扩展（高价值缺口补全）

### P1.1 · 缺失 Formalization 页面

Karpathy："*Important concepts mentioned but lacking their own page.*" 当前 formalizations/ 有 5 页，以下是被高频引用但缺少独立形式化页的数学概念：

| 文件 | 内容要点 | 关联现有页 |
|------|---------|----------|
| `wiki/formalizations/pomdp.md` | 部分可观测 MDP：状态 s、观测 o、信念 b(s)、POMDP→MDP 近似 | mdp.md, concepts/state-estimation.md |
| `wiki/formalizations/gae.md` | 广义优势估计：λ-return、bias-variance tradeoff、与 TD(λ) 关系 | bellman-equation.md, methods/policy-optimization.md |
| `wiki/formalizations/hjb.md` | Hamilton-Jacobi-Bellman 方程：连续时间最优控制、V*(x)、与 DP 关系 | bellman-equation.md, methods/trajectory-optimization.md, lqr.md |

- [x] 新建 `wiki/formalizations/pomdp.md`（满足 schema/page-types.md 规范）
- [x] 新建 `wiki/formalizations/gae.md`
- [x] 新建 `wiki/formalizations/hjb.md`
- [x] 在关联页的 `## 关联页面` 和 frontmatter `related:` 中添加回链（bellman / mdp / lqr / policy-optimization）
- [x] `make lint` 验证覆盖率仍 ≥ 82%（实际达到 100%）

### P1.2 · 缺失 Comparison 页面

| 文件 | 内容要点 |
|------|---------|
| `wiki/comparisons/online-vs-offline-rl.md` | online RL（探索成本）vs offline RL（数据质量/分布偏移），机器人场景适用边界 |
| `wiki/comparisons/sim2real-approaches.md` | domain randomization vs domain adaptation vs real-world fine-tuning 横向对比 |

- [x] 新建 2 个 comparison 页（包含对比表格 + 关联页 + sources）
- [x] `make lint` 0 issues

### P1.3 · 提升 Sources 覆盖率至 90%+

当前 82%（56/68），缺 12 页。识别并补全：

```bash
python3 scripts/lint_wiki.py 2>&1 | grep "缺少 ingest 来源"
```

- [x] 为缺失 sources 的 wiki 页批量添加 `sources:` frontmatter 字段（12 个页面补全）
- [x] 目标：覆盖率 ≥ 90%，`make badge` 自动更新 README badge（实际 100%）

---

## P2 · Lint 增强（达到 Karpathy 完整标准）

### P2.1 · CANONICAL_FACTS 扩展至 20 条

现有 10 条事实断言（PPO / MPC 等），覆盖面不足。

- [x] 在 `scripts/lint_wiki.py` 的 `CANONICAL_FACTS` 列表中补充 10 条（GAE λ / EKF 线性化 / Model-Based RL 效率 / Diffusion Policy 延迟 / WBC 优先级 / Sim2Real 手段 / PPO Clip / 接触互补性 / Retargeting 运动学约束 等）
- [x] 运行 `make lint` 验证无新的矛盾报告（0 issues）

### P2.2 · Staleness 检测（新增 lint 规则）

Karpathy："*stale claims*" — 如果 sources 文件比 wiki 页面旧很多，或者 wiki 页面长期未更新，应发出警告。

- [~] 在 `lint_wiki.py` 中添加 `check_staleness()` 函数：
  - [x] sources 文件 mtime 比 wiki 页 mtime 新 > 30 天检测（V7 已实现，"陈旧页面"检测项）
  - [ ] 读取 frontmatter `updated:` 字段，若距今 > 180 天输出 `⚠️ 可能过期` 警告（待补充）
- [ ] `make lint` 输出新增一个检测项（共 13 项，当前已有 12 项）

---

## P3 · 向量/语义搜索（可选后端）

**背景**：Karpathy 原文推荐 "hybrid BM25/vector search with LLM re-ranking"。V8 完成 BM25，V9 添加可选向量层。

### P3.1 · 本地向量索引（sentence-transformers）

- [ ] 新建 `scripts/build_vector_index.py`：
  - 依赖：`sentence-transformers`（`pip install sentence-transformers`，CPU 可用）
  - 读取所有 wiki .md 页面，strip frontmatter，截取前 512 token
  - 使用 `all-MiniLM-L6-v2` 模型生成嵌入向量
  - 将向量 + 路径保存为 `exports/vector-index.npz`（numpy 格式，无需外部 DB）
  - 输出：`exports/vector-index-meta.json`（路径 / title / type 的对应表）
- [ ] 在 `Makefile` 中添加 `make vectors` 目标
- [ ] 向量索引文件加入 `.gitignore`（体积大，按需本地生成）

### P3.2 · 混合搜索模式

- [ ] 在 `scripts/search_wiki.py` 中添加 `--semantic` 标志：
  - 若 `exports/vector-index.npz` 存在，执行向量余弦相似度检索
  - 将向量分数（0-1）和 BM25 分数加权合并：`final = 0.6*bm25 + 0.4*vector`
  - 若索引不存在，fallback 到纯 BM25 并输出提示
- [ ] 测试：`python3 scripts/search_wiki.py "腿足机器人运动控制" --semantic` 应返回语义相关页面（即使关键词不完全匹配）

### 完成标准
`--semantic` 模式下，"运动控制稳定性" 查询能找到 `formalizations/lyapunov.md`（即使页面中没有"运动控制"字样）。

---

## P4 · 启发式重排序（LLM Re-ranking 近似）

**背景**：完整 LLM 重排需要 API 调用，成本高。V9 实现**规则驱动的重排序**，近似 Karpathy 的意图。

### 重排规则（按优先级）

1. **标题精确匹配** × 5（当前 BM25 已有 × 3，提升为 × 5）
2. **frontmatter 摘要命中** × 2（search body 时优先加权 frontmatter description 字段）
3. **最近更新页面优先**（若 frontmatter 有 `updated:`，30 天内更新的加权 × 1.2）
4. **Query 页面降权** × 0.7（query 产物不应占据 concept 搜索的前位）

- [~] 在 `compute_score()` 中实现规则 1-4：
  - [x] 规则 1：标题精确匹配 × 5
  - [x] 规则 2：frontmatter summary 命中 × 2
  - [ ] 规则 3：最近更新页面 × 1.2（`updated:` 字段 30 天内）
  - [x] 规则 4：query 页面降权 × 0.7
- [ ] 测试：搜索 "policy optimization" 时，`methods/policy-optimization.md` 应排在 query 页前面
- [ ] `--json` 输出中添加 `rerank_score` 字段（便于调试）

---

## P5 · GitHub Actions 自动化（每周 lint 报告）

### P5.1 · 每周 lint 报告 Action

- [x] 新建 `.github/workflows/weekly-lint.yml`（每周一 02:00 UTC，`--report` 输出 + badge 更新，`workflow_dispatch` 支持手动触发）
- [ ] 首次手动触发（`workflow_dispatch`）验证报告格式正确（需 GitHub 执行，待触发）
- [ ] `exports/weekly-report.md` 加入 `.gitignore` 中的排除白名单（允许提交）

### P5.2 · Badge 自动更新集成到 export.yml

- [ ] 在现有 `export.yml` 的 steps 末尾添加 `python3 scripts/update_badge.py` + commit
- [ ] 这样每次 push main 时，README badge 自动同步最新覆盖率



---

## P6 · Marp 幻灯片 UX 增强

**背景**：`make slides F=<stem>` 已可生成幻灯片，但用户需手动在命令行调用。Karpathy 强调知识再利用的低摩擦。

- [ ] 在 `docs/graph.html` 侧边栏底部添加"📊 幻灯片版"按钮：
  - 按钮 href = `slides/<stem>.html`（相对路径）
  - 若该文件不存在，灰化按钮并 tooltip 提示"本地运行 `make slides F=<stem>` 生成"
- [ ] 在 `docs/index.html` 的每个搜索结果卡片添加可选的幻灯片图标链接（小图标，不破坏现有布局）

---

## Karpathy 方法论自我评估（V9 目标）

| Karpathy 原则 | V8 末状态 | V9 目标 |
|--------------|----------|---------|
| Three-layer architecture (sources → wiki → schema) | ✅ | ✅ 维持 |
| "Cross-references already there" (0 orphans) | ✅ 0 孤儿 | ✅ 维持 |
| "Good answers filed back" (query products) | ✅ 14 query 页 | ✅ + 每周自动报告 |
| Lint: contradictions / stale / orphans / frontmatter | ✅ 10 项 | ✅ 11 项（+staleness） |
| Frontmatter 完整性 | ⚠️ 4 页缺 type | ✅ 0 缺 type |
| Index.md content-oriented | ✅ | ✅ 维持 |
| Hybrid BM25 / vector search | ⚠️ BM25 only | ✅ BM25 + 可选向量 |
| LLM re-ranking | ❌ | ✅ 启发式重排 |
| Marp slides per-page button | ❌ | ✅ sidebar 按钮 |
| Automated nightly / weekly reports | ❌ | ✅ GitHub Actions 周报 |
| Sources coverage ≥ 90% | ⚠️ 82% | ✅ ≥ 90% |
| CANONICAL_FACTS ≥ 20 | ⚠️ 10 条 | ✅ 20 条 |

---

## 操作规范（V1→V9 延续）

### Op 1 · 每次改动后必须运行

```bash
make lint      # 0 issues 为通过门槛
make export    # 更新 exports/index-v1.json
make graph     # 更新 link-graph.json + graph-stats.json
make badge     # 同步 README sources 覆盖率 badge
```

### Op 2 · 新建 wiki 页面清单

1. `type:` / `tags:` / `status:` / `related:` / `sources:` 必填
2. 必须有 `## 参考来源` 区块（≥ 1 条）
3. 必须有 `## 关联页面` 区块（≥ 1 条）
4. 在至少一个现有页的 `## 关联页面` 中添加回链（防孤儿）

### Op 3 · Commit 规范

```
feat(wiki): 新建 <类型>/<页面名> — <一句话说明>
fix(lint): <修复内容>
chore(export): 更新前端 JSON / 图谱数据
```

### Op 4 · V9 完成标准（全部满足）

- [x] `make lint` 输出 `✅ 所有检查通过！`，检测项 ≥ 11 项（实际 12 项）
- [x] `python3 scripts/lint_wiki.py` 无 MISSING type
- [x] Sources 覆盖率 ≥ 90%（实际 100%）
- [ ] `--semantic` 搜索能找到语义相关页（向量索引构建成功）
- [ ] `exports/weekly-report.md` 由 GitHub Actions 成功生成
- [ ] graph.html 侧边栏底部有幻灯片按钮

---

## 状态说明

- `[ ]` 待执行
- `[~]` 进行中
- `[x]` 已完成
- `[−]` 已评估，决定跳过（附理由）
