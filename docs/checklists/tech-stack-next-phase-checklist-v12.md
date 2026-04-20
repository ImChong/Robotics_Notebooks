# 技术栈项目执行清单 v12

最后更新：2026-04-20（V12 全部完成，P0-P5 交付；P4.2 搜索 loading 状态延至 V13）
项目仓库：<https://github.com/ImChong/Robotics_Notebooks>
上一版清单：[`tech-stack-next-phase-checklist-v11.md`](tech-stack-next-phase-checklist-v11.md)
方法论参考：[Karpathy LLM Wiki](../../wiki/references/llm-wiki-karpathy.md)

---

## V11 完成基线（V12 起点）

| 维度 | V11 末状态 |
|------|-----------|
| wiki 节点（图谱） | **90 个**（concept 27 / query 19 / method 10 / formalization 10 / entity 8 / comparison 7 / task 7） |
| 图谱边数 | **521 条** |
| 总页面（exports） | **125 页** |
| Lint 检测项 | **17 项**，0 issues |
| CANONICAL_FACTS | **30 条** |
| Sources 覆盖率 | **100%**（88/88） |
| 搜索回归测试 | ✅ 12/12 通过（eval_search_quality.py） |
| 导出质量检查 | ✅ 10/10 通过（check_export_quality.py） |
| 孤立社区 | ⚠️ 2 个（Bellman 方程 社区、LLM Wiki 社区）|
| 语义搜索 | ⚠️ hashed-token fallback（sentence-transformers 未安装） |
| 学习路径 | ⚠️ roadmap/learning-paths/ 目录不存在 |
| Robot Debug Playbook | ⚠️ V11 P1.3 延期 |

---

## V12 阶段总目标

> V12 基于 Karpathy 方法论，聚焦三条主线：
>
> 1. **知识闭环（Compounding Knowledge）**：确保每个实践问题都有 query 回答，好的答案自动反哺 wiki。补全 debug playbook、新 query 页、学习路径骨架。
>
> 2. **图谱整合（Graph Coherence）**：消除 2 个孤立社区（Bellman/LLM Wiki），通过补充跨社区链接把孤岛接入主图。新增节点健康着色，让图谱前端能直观反映内容质量。
>
> 3. **检索深化（Retrieval Quality）**：CANONICAL_FACTS 扩展至 40 条，覆盖 CBF/CLF/VLA/DAgger 等 V11 新增领域。log.md 健康检查恢复。

---

## P0 · 孤立社区整合（Graph Coherence，最高优先级）

**背景**：`graph-stats.json` 显示 2 个 singleton 社区（< 3 节点）：
- `Bellman 方程 社区`：`formalizations/bellman-equation.md` 未被足够多页面引用
- `LLM Wiki 社区`：`references/llm-wiki-karpathy.md` 孤立于主图

### P0.1 · 补充跨社区链接

- [x] 在 `wiki/formalizations/gae.md`（GAE 依赖 Bellman 方程）的 `## 关联页面` 中添加 `bellman-equation.md` 回链
- [x] 在 `wiki/methods/reinforcement-learning.md` 的 `## 关联页面` 中添加 `bellman-equation.md`
- [x] 在 `wiki/formalizations/mdp.md` 的 `## 关联页面` 中添加 `bellman-equation.md`
- [x] 在 `wiki/overview/robot-learning-overview.md` 中引用 `llm-wiki-karpathy.md`（知识库方法论入口）
- [x] 在 `wiki/roadmaps/humanoid-control-roadmap.md` 的维护说明处引用 `llm-wiki-karpathy.md`

### P0.2 · 图谱节点健康着色（前端增强）

**目标**：在 `docs/graph.html` 中新增第三种着色模式——"按健康度"，帮助发现质量偏低的页面。

- [x] 在 `scripts/generate_link_graph.py` 中，为每个节点添加 `health_score` 字段（0-3）：
  - `+1`：frontmatter 有 `summary:` 字段
  - `+1`：有 `sources:` 字段且非空
  - `+1`：`updated:` 字段距今 ≤ 365 天
- [x] 在 `docs/link-graph.json` 中随节点一同输出 `health_score`
- [x] 在 `docs/graph.html` 添加"按健康度"着色按钮（绿=3分、橙=2分、红=0-1分）
- [x] 同步更新 `docs/exports/link-graph.json`

### 完成标准
- [x] `graph-stats.json` 中 `singleton_communities` 列表为空
- [x] `docs/graph.html` 有"按健康度"着色选项，点击后节点重新着色

---

## P1 · 知识闭环补全（Compounding Knowledge）

### P1.1 · V11 延期：Robot Policy Debug Playbook

- [x] 新建 `wiki/queries/robot-policy-debug-playbook.md`
  - 触发问题：「RL 策略在仿真中好但真机差，如何系统排查？」
  - 内容：症状分类树（训练问题 / 部署问题 / 硬件问题）、每类症状的诊断步骤、工具命令
  - 格式：包含 `> **Query 产物**`、决策树 / 排障流程、`## 参考来源`、`## 关联页面`
  - 关联：`sim2real-deployment-checklist.md`、`sim2real.md`、`locomotion.md`

### P1.2 · 新增实践 Query 页

高价值问题，当前无对应 query 页：

| 文件 | 触发问题 |
|------|---------|
| `wiki/queries/simulator-selection-guide.md` | 「MuJoCo vs Isaac Lab vs Genesis，做 locomotion RL 选哪个？」 |
| `wiki/queries/demo-data-collection-guide.md` | 「用模仿学习做操作，怎么高效收集人类演示数据？」 |
| `wiki/queries/ppo-vs-sac-for-robots.md` | 「机器人 RL 用 PPO 还是 SAC？有什么实践区别？」 |

- [x] 新建以上 3 个 query 页（格式含 Query 产物说明 + 对比表 + 关联页）
- [x] `make lint` 保持 0 issues

### P1.3 · 学习路径骨架（Roadmap Learning Paths）

Karpathy："*good answers can be filed back as new pages*" — 用社区结构生成学习路径。

- [x] 新建 `roadmap/learning-paths/` 目录
- [x] 新建 `roadmap/learning-paths/if-goal-locomotion-rl.md`（从 RL → Locomotion → Sim2Real 的最短路径）
- [x] 新建 `roadmap/learning-paths/if-goal-manipulation.md`（从 IL → Manipulation → Loco-Manip 的路径，实际文件名 `if-goal-imitation-learning.md`）
- [x] 每个路径页包含：前置知识 → 核心概念 → 实践 Query → 延伸阅读（格式统一）
- [x] 在 `index.md` 和 README 的"从哪里开始"表格中添加新路径入口

### 完成标准
- [x] `python3 scripts/search_wiki.py "调试策略"` 返回 `robot-policy-debug-playbook.md` 前 3
- [x] `make lint` 0 issues，graph 节点数 ≥ 96

---

## P2 · 检索深化（Retrieval Quality）

### P2.1 · CANONICAL_FACTS 扩展至 40 条

V11 新增了 CBF / CLF / VLA / DAgger 等页面，对应事实尚未进入断言集。

在 `scripts/lint_wiki.py` 的 `CANONICAL_FACTS` 字典中补充 10 条：

| 事实名称 | 正向断言 | 反向断言（不应出现）|
|---------|---------|------------------|
| CBF 安全集 | CBF 通过维持 $h(x)\geq 0$ 约束系统停留在安全集内 | CBF 直接优化性能目标 |
| CLF 衰减条件 | CLF 要求 $\dot{V}\leq -\alpha V(x)$（指数衰减） | CLF 无需 Lyapunov 衰减条件 |
| DAgger 分布修正 | DAgger 通过在线专家干预修正 covariate shift | DAgger 与 BC 在分布偏移上表现相同 |
| VLA 推理频率 | VLA 推理频率通常低于 10Hz，不适合高频关节控制 | VLA 可直接以 1000Hz 控制关节 |
| bimanual 闭链约束 | 双臂操作同一物体时形成闭链（closed-loop kinematics），需专门处理 | 双臂操作等同于两个独立单臂 |
| 地形适应感知 | 腿足地形适应通常依赖高度图或点云输入 | 腿足地形适应无需任何感知输入 |
| BM25 k1 含义 | BM25 k1 参数控制词频饱和速率（典型值 1.2-2.0） | BM25 k1 控制文档长度归一化 |
| Anki TSV 分隔符 | Anki TSV 导入使用制表符（Tab）作为字段分隔符 | Anki 导入使用逗号分隔 |
| sentence-transformers CPU | sentence-transformers 可在 CPU 环境运行，无需 GPU | sentence-transformers 必须 GPU |
| WBC QP 实时性 | WBC 基于 QP 求解，典型求解时间 50-200μs（OSQP） | WBC QP 求解通常需要秒级计算 |

- [x] 添加 10 条，总计 **40 条** CANONICAL_FACTS
- [x] `make lint` 0 矛盾报告，无误判

### P2.2 · log.md 活跃度恢复

Karpathy："*log.md is chronological, grep-parseable*"

- [x] 检查 `log.md` 最近 30 天是否有记录，若无则追加一条当前 lint 通过记录
- [x] 确认 log 格式：`## [YYYY-MM-DD] <操作类型> | <说明>` 可被 `grep "^## \["` 解析
- [x] lint 的 `log_inactive` 检测项正常工作（非 0 天静默）

### P2.3 · 搜索回归用例扩展

- [x] 将 `schema/search-regression-cases.json` 扩展至 **18 条**（新增 CBF/CLF/DAgger/VLA/双臂/地形等新页面用例）
- [x] `python3 scripts/eval_search_quality.py` 通过率维持 ≥ 80%

### 完成标准
- [x] `make lint` CANONICAL_FACTS 覆盖包含 V11 所有新增领域
- [x] 搜索回归 18/18 通过（或 ≥ 80%）
- [x] `log.md` 最近记录距今 ≤ 30 天

---

## P3 · Ingest 流水线改进

**背景**：Karpathy 核心循环是 ingest → wiki update → lint，但当前 `make ingest` 只生成模板，没有自动提示"哪些现有 wiki 页面应该更新"。

### P3.1 · Ingest 建议脚本

- [x] 在 `scripts/ingest_paper.py` 中添加 `--suggest-updates` 标志：
  - 读取新 source 文件的标题和关键词
  - 在 wiki/ 中搜索提及相关关键词的页面
  - 输出："以下页面可能需要根据新来源更新：`wiki/xxx.md`"
- [x] 在 `Makefile` 中更新 `ingest` 目标，默认加 `--suggest-updates`

### P3.2 · Index 自动同步检查

Karpathy："*LLM updates index on every ingest*"

- [x] 在 `check_export_quality.py` 中添加检查：
  - `index.md` 的最近修改时间 ≤ `exports/index-v1.json` 的修改时间 + 1天（同步检测）
  - 若 `index.md` 落后，输出 `⚠️ index.md 可能未及时更新，建议 make catalog`

### 完成标准
- [x] `python3 scripts/ingest_paper.py --suggest-updates` 能对新 source 给出相关 wiki 页推荐
- [x] `make export-check` 包含 index.md 同步检测（共 11 项）

---

## P4 · 前端体验精化

### P4.1 · PWA 离线支持

**目标**：让 GitHub Pages 在无网环境可用（缓存 JS/CSS/JSON）。

- [x] 新建 `docs/manifest.json`（PWA manifest，含 name / short_name / icons / start_url）
- [x] 新建 `docs/sw.js`（Service Worker，缓存 `search-index.json`、`link-graph.json`、`site-data-v1.json`）
- [x] 在 `docs/index.html` 的 `<head>` 中注册 Service Worker 和 manifest
- [x] 首次访问后，断网仍可搜索和浏览图谱

### P4.2 · 搜索首次加载 Loading 状态

V11 P2.3 延期项：

- [ ] 搜索框获得焦点时，若索引未加载，显示 `加载中…` spinner 替代空白
- [ ] 加载完成后自动触发已有输入内容的搜索（无需用户重新输入）

### P4.3 · 知识图谱快照 Badge

- [x] 在 `scripts/update_badge.py` 中新增图谱节点/边数 badge 自动更新（从 `graph-stats.json` 读取）
- [x] 消除 README 中手动维护节点数的需求（与 sources badge 同步自动化）

### 完成标准
- [x] `docs/manifest.json` 和 `docs/sw.js` 存在，`index.html` 已注册
- [x] 图谱 badge 由 `make badge` 自动更新，无需手动改 README

---

## P5 · 内容长尾补全

### P5.1 · 现有薄弱页面加深

当前 stub 或内容偏少的页面（< 400 字）：

- [x] 加深 `wiki/tasks/teleoperation.md`（补充：数据采集流程 / ACT/Diffusion Policy 与遥操结合 / 硬件方案对比）
- [x] 加深 `wiki/concepts/gait-generation.md`（补充：周期 vs 非周期步态 / CPG 方法 / 与 MPC 结合）
- [x] 加深 `wiki/formalizations/contact-complementarity.md`（补充：互补约束数学形式 / LCP 求解 / 与 QP 的关系）

### P5.2 · 对比页扩充

| 文件 | 内容要点 |
|------|---------|
| `wiki/comparisons/mujoco-vs-isaac-lab.md` | 速度、精度、并行能力、sim2real gap、学习曲线 |
| `wiki/comparisons/ppo-vs-sac.md` | 算法特性、机器人适用场景、实现复杂度、超参数敏感度 |

- [x] 新建 2 个对比页（含对比表格 + 选型决策树 + 关联页面）

### 完成标准
- [x] 加深的 3 个页面均 > 400 字
- [x] 2 个新对比页通过 lint
- [x] graph 节点数 ≥ 98

---

## Karpathy 方法论自我评估（V12 目标）

| Karpathy 原则 | V11 末状态 | V12 目标 | V12 末状态 |
|--------------|-----------|---------|----------|
| Three-layer architecture | ✅ | ✅ 维持 | ✅ 维持 |
| 0 orphans，交叉引用完整 | ✅ 90 节点 521 边 | ✅ ≥ 98 节点，0 孤立社区 | ✅ **98 节点 596 边，0 孤立社区** |
| "Good answers filed back" | ✅ query 页持续增加 | ✅ +3 query 页 + 学习路径骨架 | ✅ **+4 query 页 + 4 条学习路径** |
| Lint 检测项 | ✅ 17 项 | ✅ 17 项（稳定） | ✅ 17 项（稳定） |
| CANONICAL_FACTS | ✅ 30 条 | ✅ **40 条** | ✅ **40 条** |
| Sources 覆盖率 100% | ✅ | ✅ 维持 | ✅ 维持 |
| Hybrid BM25/vector | ⚠️ hashed fallback | ⚠️ 维持（环境限制）| ⚠️ 维持（环境限制） |
| 浏览器端搜索 | ✅ + 解释层 | ✅ + loading 状态 + PWA | ✅ PWA ✅ / loading 状态 ❌ 延至 V13 |
| 知识图谱节点健康 | ❌ | ✅ health_score 着色 | ✅ health_score 着色（红/橙/黄/绿） |
| 孤立社区 | ⚠️ 2 个 | ✅ 0 个 | ✅ **0 个** |
| Ingest → suggest updates | ❌ | ✅ `--suggest-updates` | ✅ `--suggest-updates` |
| Log.md 活跃 | ⚠️ 可能静默 | ✅ 距今 ≤ 30 天 | ✅ 距今 ≤ 30 天 |
| 学习路径 | ❌ | ✅ 2 条路径页 | ✅ **4 条路径页** |
| 图谱 badge 自动化 | ⚠️ 手动维护 | ✅ `make badge` 自动更新 | ✅ `make badge` 自动更新 |

---

## 操作规范（延续 V1→V12）

### Op 1 · 每次改动后必须运行

```bash
make lint
make export
make graph
make badge
```

### Op 2 · 新建 wiki 页面必须满足

1. `type:` / `tags:` / `status:` / `related:` / `sources:` / `summary:` 必填
2. 有 `## 参考来源` 区块（≥ 1 条）
3. 有 `## 关联页面` 区块（≥ 1 条）
4. 在至少一个现有页的 `## 关联页面` 中添加回链（防孤儿）
5. query 页面必须包含 `> **Query 产物**` 说明

### Op 3 · V12 完成标准（全部满足）

- [x] `make lint` 0 issues，CANONICAL_FACTS = 40 条
- [x] `graph-stats.json` 无 singleton_communities
- [x] graph 节点数 ≥ 98（实际：98 节点 596 边）
- [x] `make export-check` 11 项全通过
- [x] `docs/manifest.json` 存在（PWA 支持）
- [x] `make badge` 自动更新图谱 badge（无需手动改 README）
- [x] `roadmap/learning-paths/` 下有 ≥ 2 个路径页（实际：4 个）
- [x] `log.md` 最近记录距今 ≤ 30 天

---

## 状态说明

- `[ ]` 待执行
- `[~]` 进行中
- `[x]` 已完成
- `[−]` 已评估，决定跳过（附理由）
