# 技术栈项目执行清单 v8

最后更新：2026-04-16
项目仓库：<https://github.com/ImChong/Robotics_Notebooks>
上一版清单：[`docs/tech-stack-next-phase-checklist-v7.md`](tech-stack-next-phase-checklist-v7.md)
方法论参考：[Karpathy LLM Wiki](../wiki/references/llm-wiki-karpathy.md)

---

## V7 完成基线（V8 起点）

| 维度 | V7 末状态 |
|------|----------|
| wiki 节点（图谱） | 68 nodes，376 edges |
| sources/papers/ 文件 | 24 |
| Sources 覆盖率 | 74%（49/66 wiki 页有 ingest 来源） |
| Lint 健康 | ✅ 0 issues，10 项检测，10 条 CANONICAL_FACTS |
| Query 产物 | 15 个（wiki/queries/） |
| 孤儿节点 | ⚠️ **8 个**（全部是 query 页，入度 = 0） |
| graph-stats.json | ✅ hubs / orphans / type 分布 |
| GitHub Actions | ✅ export.yml（push main 自动 lint+export+graph） |
| graph.html | ✅ 孤儿过滤 + fly-to + 浮动卡片 + 移动端 |
| 前端键盘导航 | ❌（V7 P1 延续） |
| 首页 mini 图谱 | ❌（V7 P0.1 延续） |
| 覆盖率 badge | ❌（V7 P2.2 延续） |
| 向量搜索 | ❌（V7 P5 延续） |

---

## V8 阶段总目标

> 本轮深度对齐 Karpathy 原文三条尚未落实的原则：
>
> 1. **"The cross-references are already there"** — 8 个 query 产物孤儿页意味着交叉引用缺失。V8 的首要任务是修复双向链接，让 query 产物真正成为知识网络节点，而非末端悬挂。
>
> 2. **"Important concepts mentioned but lacking their own page"** — 审计现有页面中频繁提及但缺少独立页的概念，按需新建（sensor-fusion、foundation-policy 等），同步提升覆盖率到 80%+。
>
> 3. **"Lint... look for contradictions, stale claims, orphan pages, missing cross-references, data gaps"** — V7 的 lint 已覆盖孤儿检测和矛盾检测，V8 补全 **frontmatter 一致性** + **log.md 活跃度检查**，让 lint 真正达到 Karpathy 描述的完整健康检查标准。

---

## P0 · 孤儿 Query 页双向链接修复（最高优先级）

**背景**：Karpathy："*The cross-references are already there.*" 当前 8 个 query 页的入度 = 0，意味着核心 wiki 页面没有指向这些 query 产物，知识没有真正连接。这是当前知识库最大的结构缺陷。

### 需要修复的 8 个孤儿页及其推荐回链

| 孤儿 Query 页 | 应添加回链的核心 wiki 页面 |
|--------------|------------------------|
| `control-architecture-comparison.md` | `comparisons/wbc-vs-rl.md`、`concepts/mpc-wbc-integration.md` |
| `humanoid-hardware-selection.md` | `entities/humanoid-robot.md`、`tasks/locomotion.md` |
| `humanoid-rl-cookbook.md` | `tasks/locomotion.md`、`methods/reinforcement-learning.md` |
| `pinocchio-quick-start.md` | `entities/pinocchio.md` |
| `reward-design-guide.md` | `concepts/reward-design.md` |
| `rl-hyperparameter-guide.md` | `methods/policy-optimization.md` |
| `sim2real-gap-reduction.md` | `concepts/sim2real.md` |
| `when-to-use-wbc-vs-rl.md` | `comparisons/wbc-vs-rl.md`、`concepts/whole-body-control.md` |

### 执行方式

在每个"回链目标页"的 `## 关联页面` 区块末尾添加：
```markdown
- [Query：<标题>](../queries/<filename>.md)
```
同时在该目标页的 YAML frontmatter `related:` 列表中补充对应路径。

- [ ] 修复 `comparisons/wbc-vs-rl.md` → 添加 2 条 query 回链
- [ ] 修复 `concepts/mpc-wbc-integration.md` → 添加 1 条 query 回链
- [ ] 修复 `entities/humanoid-robot.md` → 添加 1 条 query 回链
- [ ] 修复 `tasks/locomotion.md` → 添加 2 条 query 回链
- [ ] 修复 `methods/reinforcement-learning.md` → 添加 1 条 query 回链
- [ ] 修复 `entities/pinocchio.md` → 添加 1 条 query 回链
- [ ] 修复 `concepts/reward-design.md` → 添加 1 条 query 回链
- [ ] 修复 `methods/policy-optimization.md` → 添加 1 条 query 回链
- [ ] 修复 `concepts/sim2real.md` → 添加 1 条 query 回链
- [ ] 修复 `concepts/whole-body-control.md` → 添加 1 条 query 回链
- [ ] 运行 `make lint` 验证孤儿节点降为 0

### 完成标准
- `graph-stats.json` 中 `orphan_nodes` 数量 = 0（或仅剩非 query 类型的合理孤儿）
- `make lint` ✅ 0 issues

---

## P1 · 缺失概念页补全（内容深度）

**背景**：Karpathy："*Important concepts mentioned but lacking their own page.*" 当前 sources 映射表中列出但尚未创建的 wiki 页面有多个。

### 1.1 新建缺失概念页（sources 映射要求）

| 文件 | 对应 sources 映射 | 核心内容 |
|------|-----------------|---------|
| `wiki/concepts/sensor-fusion.md` | `perception_localization.md` | IMU + 视觉里程计 + InEKF；感知层状态估计 |
| `wiki/concepts/foundation-policy.md` | `rl_foundation_models.md` | RT-1/RT-2/π₀/Octo；大模型 → 机器人控制 |
| `wiki/concepts/contact-complementarity.md` | 已有同名页面 | 检查是否已完整，补充与 WBC/MPC 的关联 |

- [ ] 新建 `wiki/concepts/sensor-fusion.md`（YAML + 内容 + 关联页面 + 参考来源）
- [ ] 新建 `wiki/concepts/foundation-policy.md`（YAML + 内容 + 关联页面 + 参考来源）
- [ ] 验证 `wiki/formalizations/contact-complementarity.md` 已充分链接

### 1.2 Sources 覆盖率提升（74% → 80%+）

当前差距：49/66 = 74%。目标 80% ≈ 53/66。需要 4 个现有页面补充 sources 引用，或新增 sources 文件。

**策略 A：补全现有 wiki 页面缺少的 sources 链接**（低成本）
- 检查哪些页面在 frontmatter `sources:` 为空但有对应 sources 文件

**策略 B：新增 sources 文件**（高价值）

| 目标文件 | 覆盖 wiki 页面 |
|---------|--------------|
| `sources/papers/contact_control.md` | contact-dynamics.md、contact-estimation.md、contact-complementarity.md |
| `sources/papers/legged_robot_design.md` | balance-recovery.md、footstep-planning.md、gait-generation.md、capture-point-dcm.md |

- [ ] 策略 A：审计并补全 5+ 个现有页面的 `sources:` 字段
- [ ] 策略 B（可选）：新建 `sources/papers/contact_control.md`
- [ ] `make lint` 覆盖率达到 ≥ 80%

### 完成标准
- 新增 2 个概念页，lint ✅
- Sources 覆盖率 ≥ 80%

---

## P2 · Lint 健康检查完善（Karpathy Lint 对齐）

**背景**：Karpathy lint 标准：*"contradictions, stale claims, orphan pages, missing cross-references, data gaps"*。V7 已覆盖矛盾检测和孤儿检测，V8 补全 frontmatter 一致性 + log.md 活跃度。

### 2.1 Frontmatter 一致性检查

- [ ] `scripts/lint_wiki.py` 新增检查：wiki 页面缺少 `type` 字段时 warning（排除 references/、roadmap/、tech-map/ 目录）
- [ ] `scripts/lint_wiki.py` 新增检查：wiki 页面缺少 `related:` 字段（且不是 README）时 warning
- [ ] 运行后审计并修复现有 warning

### 2.2 Log.md 活跃度检查

- [ ] `scripts/lint_wiki.py` 新增检查：`log.md` 最近 30 天内是否有条目（如无，提示"知识库可能已停止维护"）
- [ ] 验证 `grep "^## \[" log.md | tail -5` 输出格式正确

### 2.3 Sources 覆盖率 Badge（V7 P2.2 延续）

- [ ] `README.md` 中加入静态 badge：
  ```markdown
  ![Sources Coverage](https://img.shields.io/badge/sources_coverage-74%25-yellow)
  ```
  每次手动更新数值（≥80% 绿色，60-79% 黄色，<60% 红色）
- [ ] 在 `Makefile` 中加入 `make badge` 目标：读取 lint 输出的覆盖率数值，更新 README 中的 badge URL

### 完成标准
- `make lint` 新增 frontmatter + log.md 检查，0 new false positives
- README 显示覆盖率 badge

---

## P3 · 首页 Mini 图谱嵌入（V7 P0.1 延续）

**背景**：Karpathy："*Obsidian's graph view is the best way to see the shape of your wiki.*" 图谱已是独立页面，V8 将其缩略版嵌入 index.html，让图谱成为知识库的第一视觉入口。

### 3.1 index.html 图谱预览区块

- [ ] `docs/index.html` 新增 "知识图谱" section（在搜索框下方）：
  - 嵌入 300px 高度的迷你力导向图
  - 节点按 Top-40（度数最高）过滤，从 `docs/exports/link-graph.json` 读取
  - 禁用物理面板，禁用悬浮卡片，点击节点直接跳转 `graph.html?focus=<id>`
  - 显示实时统计：`68 节点 · 376 条边 · 覆盖率 74%`
- [ ] graph.html 新增 `?focus=<id>` URL 参数支持：页面加载后自动 fly-to 指定节点并展示详情

### 3.2 graph.html 侧边栏（V7 P0.2 延续）

- [ ] 点击节点时右侧滑出侧边栏（宽度 300px），显示：
  - 节点标题 + type badge
  - tags 列表
  - summary 前 150 字
  - 关联页面列表（可点击定位到目标节点）
  - "打开详情页" 链接
- [ ] 侧边栏替代当前浮动卡片（移动端体验更好，不遮挡图谱）

### 完成标准
- index.html 可见 mini 图谱，不跳转即可感知知识库形态
- graph.html 侧边栏正常渲染，移动端可用

---

## P4 · 前端搜索体验升级（V7 P1 延续）

### 4.1 index.html 搜索增强

- [ ] 搜索结果键盘导航：↑↓ 选中高亮，Enter 打开详情页，Esc 清空
- [ ] 搜索框下方标签云：统计 `docs/exports/index-v1.json` 中 Top-20 高频 tag，点击直接过滤结果

### 4.2 detail.html 关联卡片

- [ ] 关联页面列表从纯文本链接改为卡片式（type badge + summary 前 60 字 + 跳转链接）

### 完成标准
- 键盘导航 ↑↓/Enter 可用
- 标签云正确渲染（来自 index-v1.json）

---

## P5 · 搜索引擎升级（BM25 → 混合检索）

**背景**：Karpathy 推荐 *"hybrid BM25/vector search with LLM re-ranking"*。V7 有 TF-IDF；V8 升级至 BM25 作为 Step 2。

### 5.1 BM25 搜索

- [ ] `scripts/search_wiki.py` 使用 `rank-bm25` 库替换当前 TF-IDF（如无法安装则用手动 BM25 实现）
- [ ] 验证：`make search Q="model predictive control"` 结果含 `model-predictive-control.md` 且排名靠前

### 5.2 搜索结果缓存

- [ ] `exports/search-cache.json`：缓存最近 30 次查询（query / results / timestamp）
- [ ] `search_wiki.py` 命中缓存时跳过计算，直接返回

### 5.3 向量搜索原型（V7 P5，可选）

- [ ] `scripts/embed_wiki.py`：调用 Anthropic Embeddings API（`voyage-3`）或本地 `sentence-transformers`，生成 wiki 页面向量，存储至 `exports/wiki-embeddings.npz`
- [ ] `search_wiki.py --semantic`：加载向量，余弦相似度排序，与 BM25 分数加权混合

> 如无法调用嵌入 API，P5.3 标记 `[-]`，以 BM25 为止。

### 完成标准
- BM25 替换 TF-IDF，搜索质量提升（人工验证 5 个查询）
- 缓存命中率 > 0（运行两次相同查询后）

---

## Karpathy Checklist 自评（V7 → V8）

| Karpathy 原则 | V7 末状态 | V8 目标 |
|-------------|---------|--------|
| Raw sources（不可变 sources 层） | ✅ 24 文件，74% 覆盖 | **26 文件，80%+ 覆盖** |
| Wiki（LLM 维护的 md 文件集） | ✅ 68 节点，376 边，0 断链 | **70+ 节点，0 孤儿 query** |
| Schema（配置与规范文档） | ✅ schema/ 5 文件 | 同步更新 |
| Ingest "1 source → 10–15 pages" | ✅ coverage checker 辅助 | 持续优化 |
| Query 产物（"answers filed back"） | ⚠️ 15 个，8 个孤儿 | **15 个，0 孤儿** |
| Lint（矛盾/孤儿/frontmatter/log） | ✅ 10 项，10 条 CANONICAL_FACTS | **+ frontmatter + log 活跃度** |
| Search（BM25/vector） | TF-IDF | **BM25 + 缓存** |
| log.md | ✅ 284 行，格式正确 | ✅ 持续追加 |
| Obsidian 图谱视图等效 | ✅ graph.html（独立页面） | **嵌入 index.html 首页** |
| Dataview 等效（frontmatter 查询） | ✅ graph-stats.json | **+ frontmatter lint** |
| CI/CD | ✅ GitHub Actions export.yml | ✅ |

---

## 优先级执行顺序

```
P0（孤儿回链）→ P2.1（frontmatter lint）→ P1（缺失概念页）→
P2.2/2.3（badge + log lint）→ P3（mini graph）→ P4（搜索 UI）→
P5.1/5.2（BM25 + cache）→ P5.3（向量，可选）
```

> P0 是本轮最高优先级：修复 8 个孤儿 query 页的回链，成本低（修改 10 个 md 文件），但对知识库结构健康度的改善是立竿见影的。

---

## 维护操作标准（V8 更新版）

### Op 1：Ingest（添加新资料）
```bash
make ingest NAME=xxx TITLE="..." DESC="..."
# 编辑模板，填写论文摘录 + wiki 映射（至少 10 个页面）
make coverage F=sources/papers/xxx.md
# 补充 cross-reference，确保 sources: 字段被相关 wiki 页面引用
make lint && make catalog && make export && make graph
make log OP=ingest DESC="sources/papers/xxx.md — 简述，覆盖 N 个页面"
```

### Op 2：Query（知识查询）
```bash
make search Q=<关键词>
# 综合分析 → 保存为 wiki/queries/xxx.md
# 在回链目标页面的"关联页面"区块添加反向链接
# 更新 wiki/queries/README.md 表格
make lint && make catalog && make export && make graph
make log OP=query DESC="关键词 → wiki/queries/xxx.md"
```

### Op 3：Lint（健康检查）
```bash
make lint                              # 完整检查（孤儿/断链/矛盾/frontmatter/log活跃度）
make lint --report                     # 保存至 exports/lint-report.md
make log OP=lint DESC="0 issues，覆盖率 XX%，孤儿 N 个"
```

### Op 4：Index（索引更新）
```bash
make catalog && make export && make graph
# graph-stats.json 会自动更新 hubs/orphans/type 分布
```

---

## 状态约定
- `[ ]` 未开始
- `[~]` 进行中
- `[x]` 已完成
- `[-]` 暂缓（条件不满足）
