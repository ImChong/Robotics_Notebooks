# 技术栈项目执行清单 v11

最后更新：2026-04-17（V11 规划启动，基于 V10 完整交付）
项目仓库：<https://github.com/ImChong/Robotics_Notebooks>
上一版清单：[`docs/tech-stack-next-phase-checklist-v10.md`](tech-stack-next-phase-checklist-v10.md)
方法论参考：[Karpathy LLM Wiki](../wiki/references/llm-wiki-karpathy.md)

---

## V10 完成基线（V11 起点）

| 维度 | V10 末状态 |
|------|-----------|
| wiki 节点（图谱） | **83 个**（concept 25 / query 16 / method 10 / formalization 9 / entity 8 / comparison 6 / task 6） |
| 图谱边数 | **486 条** |
| 总页面（exports） | **118 页** |
| Lint 检测项 | **14 项**，0 issues |
| CANONICAL_FACTS | **30 条** |
| Sources 覆盖率 | **100%**（81/81） |
| 浏览器搜索 | ✅ 纯前端离线 BM25 |
| 语义搜索 | ✅ `--semantic` 已可用；当前环境 fallback 为 hashed-token-projection |
| 图谱社区 | ✅ 7 个社区，支持按社区着色 |
| Anki 导出 | ✅ 30 张卡（9 formalizations + 21 concepts） |

---

## V11 阶段总目标

> V11 不再追求“补功能点”，而是推进 **检索质量、知识深度、图谱质量、运维稳定性** 四条主线：
>
> 1. **把 V10 的可用版本打磨成稳定版本**：真正落地 sentence-transformers 语义搜索，而不是只停留在 fallback backend。
> 2. **补控制稳定性 / loco-manipulation / 真机部署链路的知识缺口**：从“能搜到”升级到“关键问题能回答透”。
> 3. **提升知识图谱与前端检索质量**：减少单节点社区、补结果解释、提升 GitHub Pages 端使用体验。
> 4. **把维护流程产品化**：加入搜索回归测试、导出质量检查、Anki / graph / search 的 CI 验证。

---

## P0 · 语义搜索从 fallback 升级为真实 sentence-transformers（最高优先级）

**背景**：V10 已打通 `make vectors` 与 `--semantic`，但当前环境仍使用 hashed-token fallback。V11 的重点是把“可用”升级为“真实语义检索”。

### P0.1 · sentence-transformers 真正落地

- [ ] 在 `scripts/build_vector_index.py` 中明确区分两种 backend：`sentence-transformers` / `hashed-token-fallback`
- [ ] 默认优先尝试 `sentence-transformers`：
  - 模型：`all-MiniLM-L6-v2`
  - CPU 可运行
  - 索引 metadata 中写入 `backend` / `model_name` / `embedding_dim`
- [ ] 当 fallback 被启用时，CLI 明确打印黄字告警：`当前不是 sentence-transformers 真向量索引`
- [ ] 在 `exports/vector-index-meta.json` 中保存：
  - `generated_at`
  - `backend`
  - `model_name`
  - `embedding_dim`
  - `doc_count`

### P0.2 · 检索回归样例集

- [ ] 新建 `schema/search-regression-cases.json`
- [ ] 收录至少 12 个查询样例，覆盖：
  - 稳定性 / Lyapunov
  - 行为克隆 / DAgger
  - VLA / 真机部署
  - 地形适应 / 腿足 locomotion
  - 接触丰富操作 / manipulation
  - MPC / WBC / RL 选型
- [ ] 每个样例记录：`query` / `expected_top_k` / `must_include` / `mode(bm25|semantic)`
- [ ] 新建 `scripts/eval_search_quality.py`：
  - 逐条跑查询
  - 输出 recall@5 / hit@5
  - 失败样例打印 diff

### 完成标准
- [ ] `make vectors` 在装有依赖的环境下生成 `backend=sentence-transformers`
- [ ] `python3 scripts/eval_search_quality.py` 通过率 ≥ 80%
- [ ] 至少 3 个 query 在 semantic 模式下优于纯 BM25

---

## P1 · 内容深度扩充：稳定性 / loco-manipulation / 真机部署

### P1.1 · 稳定性与安全控制主线补齐

- [ ] 新建 `wiki/concepts/control-barrier-function.md`
  - 重点：CBF 是什么 / 与 Lyapunov 的关系 / 安全约束用途
- [ ] 新建 `wiki/formalizations/control-lyapunov-function.md`
  - 重点：CLF / CLF-CBF-QP / 与稳定性约束的关系
- [ ] 新建 `wiki/comparisons/clf-vs-cbf.md`
  - 重点：稳定性 vs 安全性 / 何时用哪个
- [ ] 在 `whole-body-control.md` / `locomotion.md` / `lyapunov.md` 中补回链

### P1.2 · loco-manipulation 与 foundation policy 深挖

- [ ] 新建 `wiki/tasks/bimanual-manipulation.md`
- [ ] 新建 `wiki/concepts/whole-body-coordination.md`
- [ ] 新建 `wiki/queries/foundation-policy-for-humanoids.md`
  - 触发问题：「人形机器人 foundation policy 现在到底适合什么，不适合什么？」
- [ ] 更新 `loco-manipulation.md` / `foundation-policy.md` / `manipulation.md` 的关联页面

### P1.3 · 真机部署实践 Query 页

- [ ] 新建 `wiki/queries/sim2real-deployment-checklist.md`
- [ ] 新建 `wiki/queries/robot-policy-debug-playbook.md`
- [ ] 两页都要求包含：
  - 决策树 / 排障流程图
  - 训练端问题 vs 部署端问题区分
  - 建议继续阅读

### 完成标准
- [ ] 新增 ≥ 7 页，且 `make lint` 保持 0 issues
- [ ] `search_wiki.py "CLF"`、`"CBF"`、`"sim2real 部署"` 均能命中新页
- [ ] graph 节点数从 83 提升到 ≥ 90

---

## P2 · 图谱与前端检索质量提升

**背景**：V10 已有社区着色和前端 BM25，但当前社区分布仍偏粗糙，搜索结果也缺少解释层。

### P2.1 · 图谱社区质量提升

- [ ] 在 `scripts/generate_link_graph.py` 中增加“小社区折叠”规则：
  - 节点数 < 3 的社区并入最相近主社区
- [ ] 在 `exports/graph-stats.json` 中输出：
  - `singleton_communities`
  - `largest_community_ratio`
- [ ] 若 `largest_community_ratio > 0.45`，在 stats 中标记 `community_quality_warning`

### P2.2 · 搜索结果解释层

- [ ] 在 `scripts/search_wiki.py --json` 输出中新增 `match_explanation`
  - 例如：`title-hit` / `summary-hit` / `vector-neighbor` / `related-page-boost`
- [ ] `docs/main.js` 搜索结果卡片显示小字说明：
  - “标题命中” / “摘要命中” / “语义相近”
- [ ] 搜索结果卡片增加直达按钮：
  - `打开详情页`
  - `查看图谱邻居`

### P2.3 · 前端搜索体验增强

- [ ] 首次加载 `search-index.json` 时显示 loading 状态
- [ ] 输入为空时显示热门查询建议（固定 6 条）
- [ ] 搜索无结果时显示：
  - 近义词建议
  - 命令行搜索提示
  - 图谱入口链接

### 完成标准
- [ ] `graph-stats.json` 可见社区质量字段
- [ ] GitHub Pages 端搜索结果卡片含解释信息
- [ ] 前端空搜索 / 无结果 / 首次加载三种状态都可用

---

## P3 · Lint / Export / QA 流程产品化

### P3.1 · 搜索与导出质量检查

- [ ] 新建 `scripts/check_export_quality.py`
- [ ] 检查项至少包括：
  - `docs/search-index.json` 是否存在
  - `exports/index-v1.json` / `site-data-v1.json` 文档数是否一致
  - `docs/exports/` 是否与 `exports/` 同步
  - graph node_count 是否与 wiki 页面数大体一致
- [ ] 将该脚本集成进 `make export-check`

### P3.2 · 新增 lint 规则

- [ ] 检查 `wiki/queries/` 页面必须包含：
  - `> **Query 产物**` 触发说明
  - `## 参考来源`
  - `## 关联页面`
- [ ] 检查 `wiki/formalizations/` 页面至少包含一个公式块或形式化定义块
- [ ] 检查 `README.md` 中 graph badge / checklist 链接与当前版本一致

### P3.3 · GitHub Actions 补强

- [ ] 新增 workflow：`search-regression.yml`
- [ ] push / PR 时自动执行：
  - `make lint`
  - `make export`
  - `python3 scripts/eval_search_quality.py`
  - `python3 scripts/check_export_quality.py`
- [ ] CI 失败时输出最小诊断摘要到 job summary

### 完成标准
- [ ] Lint 检测项从 14 提升到 ≥ 17
- [ ] `make export-check` 可稳定通过
- [ ] GitHub Actions 能自动跑搜索回归与导出检查

---

## P4 · 知识再利用 2.0：Anki / 路线 / 学习产物联动

### P4.1 · Anki 导出增强

- [ ] `export_anki.py` 支持 `--deck` 参数
- [ ] deck 至少支持：
  - `formalization`
  - `concepts-core`
  - `control-stability`
- [ ] 为卡片追加来源字段或备注字段（第三/第四列）

### P4.2 · 学习路径联动

- [ ] 新建 `docs/anki-guide.md`
- [ ] 说明如何：
  - 导入 TSV
  - 选择 tag 建子牌组
  - 按 roadmap 学习时配合复习
- [ ] 在 README 的执行清单 / 常用操作区增加 Anki 使用入口

### 完成标准
- [ ] `make anki` 支持按 deck 导出
- [ ] 学习者能从 README 一跳到 Anki 使用说明

---

## P5 · README 与指标同步

- [ ] 更新 README badge：图谱节点 / 边数同步到 V11 当前值
- [ ] 更新 Sources Coverage badge 链接到 `v11`
- [ ] 更新执行清单入口：`v10` → `v11`
- [ ] README 中补一句当前语义搜索状态：是否为 sentence-transformers 真后端

---

## Karpathy 方法论自我评估（V11 目标）

| Karpathy 原则 | V10 末状态 | V11 目标 |
|--------------|-----------|---------|
| Three-layer architecture | ✅ | ✅ 维持 |
| 0 orphans，交叉引用完整 | ✅ 83 节点 486 边 | ✅ ≥ 90 节点，0 orphan |
| Good answers filed back | ✅ query 页持续增加 | ✅ 增加部署 / foundation policy / debug playbook |
| Lint 检测项 | ✅ 14 项 | ✅ ≥ 17 项 |
| CANONICAL_FACTS | ✅ 30 条 | ✅ ≥ 36 条 |
| Sources 覆盖率 100% | ✅ | ✅ 维持 |
| Hybrid BM25/vector | ✅ fallback 可用 | ✅ sentence-transformers 真后端 |
| 浏览器端搜索 | ✅ 离线 BM25 | ✅ 有结果解释 / 空态 / 无结果态 |
| 知识图谱社区检测 | ✅ 基础可用 | ✅ 小社区折叠 + 质量指标 |
| 知识再利用（Anki） | ✅ 基础 TSV 导出 | ✅ 按 deck 导出 + 使用说明 |
| QA / CI | ⚠️ 仅 lint/export 主链 | ✅ 搜索回归 + 导出质量检查 |

---

## 操作规范（延续 V1→V11）

### Op 1 · 每次改动后必须运行

```bash
make lint
make export
make graph
make badge
```

### Op 2 · 完成任务后必须同步打勾

1. 执行清单中的任务完成后，必须立即把对应项从 `[ ]` 改成 `[x]`
2. 不允许“代码已完成，但清单没更新”
3. 若决定跳过，必须写成 `[−]` 并附理由
4. 若任务部分完成，先标 `[~]` 并写清剩余内容

### Op 3 · V11 完成标准（全部满足）

- [ ] `make lint` 输出 `✅ 所有检查通过！`，检测项 ≥ 17 项
- [ ] `make vectors` 在目标环境中生成 `backend=sentence-transformers`
- [ ] `python3 scripts/eval_search_quality.py` 通过率 ≥ 80%
- [ ] 新增 ≥ 7 个高价值页面，graph 节点数 ≥ 90
- [ ] GitHub Pages 搜索结果有解释层，空态 / 无结果态可用
- [ ] `make export-check` 通过
- [ ] GitHub Actions 自动跑搜索回归与导出质量检查

---

## 状态说明

- `[ ]` 待执行
- `[~]` 进行中
- `[x]` 已完成
- `[−]` 已评估，决定跳过（附理由）
