# 技术栈项目执行清单 v13

最后更新：2026-04-20（V13 完成）
项目仓库：<https://github.com/ImChong/Robotics_Notebooks>
上一版清单：[`tech-stack-next-phase-checklist-v12.md`](tech-stack-next-phase-checklist-v12.md)
方法论参考：[Karpathy LLM Wiki](../../wiki/references/llm-wiki-karpathy.md)

---

## V12 完成基线（V13 起点）

| 维度 | V12 末状态 |
|------|-----------|
| wiki 节点（图谱） | **98 个**（concept 27 / query 24 / formalization 10 / method 10 / comparison 9 / entity 8 / task 7 / unknown 2 / overview 1） |
| 图谱边数 | **596 条** |
| 社区数 | **4 个**（Locomotion 41 / WBC 34 / IL 18 / LQR 5） |
| wiki 总页面 | **99 页** |
| CANONICAL_FACTS | **39 条**（V12 目标 40 条，实际差 1） |
| Sources 覆盖率 | **100%**（96/96） |
| 搜索回归测试 | ✅ 18/18 通过 |
| 导出质量检查 | ✅ 11/11 通过 |
| 孤儿节点 | ⚠️ **4 个**（无入链） |
| 缺 `type:` 字段 | ⚠️ **2 页**（roadmap / reference） |
| 薄弱页面（< 200 词） | ⚠️ **7 个** |
| 语义搜索 | ⚠️ hashed-token fallback（sentence-transformers 未安装） |
| PWA 离线支持 | ✅ manifest.json + sw.js |
| 搜索 loading 状态 | ✅ focus 预热索引 |

---

## V13 阶段总目标

> V13 基于 Karpathy 方法论，聚焦三条主线：
>
> 1. **图谱健康（Graph Coherence）**：消除 4 个孤儿节点（无入链页面），修复 2 页缺失 `type:` 字段。图谱无孤儿是 Karpathy Lint 的核心健康指标。
>
> 2. **知识积累（Compounding Knowledge）**：加深 7 个薄弱页面，新建 4 个高价值 Query 页（好问题回写 wiki）。CANONICAL_FACTS 从 39 条扩展至 50 条，覆盖 contact-rich / CLF-CBF / VLA 融合等 V12 新增领域。
>
> 3. **检索深化（Retrieval Quality）**：搜索回归从 18 条扩展至 26 条；新增 Lint 检测项（孤儿节点数量检测）；提升图谱 `health_score` 整体分布。

---

## P0 · 孤儿节点修复（最高优先级）

**背景**：`graph-stats.json` 显示 4 个无入链节点，违反 Karpathy 原则"cross-references are already there"：

| 孤儿页面 | 出链数 | 应从哪里加入链 |
|---------|--------|--------------|
| `wiki/queries/demo-data-collection-guide.md` | 4 | `manipulation.md`、`teleoperation.md` 关联页面 |
| `wiki/queries/humanoid-motion-control-know-how.md` | 12 | `locomotion.md`、`robot-learning-overview.md` 关联页面 |
| `wiki/queries/open-source-motion-control-projects.md` | 12 | `locomotion.md`、`entities/` 相关 entity 页 |
| `wiki/queries/robot-policy-debug-playbook.md` | 5 | `sim2real.md`、`sim2real-deployment-checklist.md` 关联页面 |

### P0.1 · 为孤儿页面补充入链

- [x] 在 `wiki/tasks/manipulation.md` 关联页面中添加 `demo-data-collection-guide.md`
- [x] 在 `wiki/tasks/teleoperation.md` 关联页面中添加 `demo-data-collection-guide.md`
- [x] 在 `wiki/tasks/locomotion.md` 关联页面中添加 `humanoid-motion-control-know-how.md`
- [x] 在 `wiki/overview/robot-learning-overview.md` 关联页面中添加 `humanoid-motion-control-know-how.md`
- [x] 在 `wiki/tasks/locomotion.md` 关联页面中添加 `open-source-motion-control-projects.md`
- [x] 在 `wiki/concepts/sim2real.md` 关联页面中添加 `robot-policy-debug-playbook.md`
- [x] 在 `wiki/queries/sim2real-deployment-checklist.md` 关联页面中添加 `robot-policy-debug-playbook.md`

### P0.2 · 修复缺失 type 字段的页面

- [x] 在 `wiki/roadmaps/humanoid-control-roadmap.md` frontmatter 中添加 `type: roadmap_page`
- [x] 在 `wiki/references/llm-wiki-karpathy.md` frontmatter 中添加 `type: reference`
- [x] 运行 `make lint`，确认 `type_distribution` 中 `unknown` 降为 0

### 完成标准

- [x] `graph-stats.json` 中 `orphan_nodes` 列表为空
- [x] `graph-stats.json` 中 `type_distribution.unknown` 为 0

---

## P1 · CANONICAL_FACTS 修正与扩展（39 → 50 条）

**背景**：V12 目标 40 条，实际仅 39 条，差 1 条未补齐。V12 新增了 contact-rich manipulation、CLF、CBF、VLA 融合等页面，对应事实尚未进入断言集。

在 `scripts/lint_wiki.py` 的 `CANONICAL_FACTS` 字典中补充 11 条：

| 事实名称 | 正向断言 | 反向断言（不应出现）|
|---------|---------|------------------|
| PPO on-policy | PPO 是 on-policy 算法，每次更新后丢弃旧数据 | PPO 可以直接复用历史数据（off-policy）|
| Contact-rich 碰撞处理 | contact-rich manipulation 需要显式建模碰撞力与接触状态 | contact-rich manipulation 可忽略接触力 |
| CLF 控制收敛 | CLF 的使用目标是使系统状态收敛至平衡点 | CLF 用于描述系统安全边界 |
| CBF vs CLF 区别 | CBF 维持安全集不变性，CLF 驱动系统收敛 | CBF 和 CLF 功能完全相同 |
| VLA 动作空间 | VLA 通常输出末端执行器目标位姿而非关节力矩 | VLA 直接输出关节级力矩指令 |
| bimanual 协调需求 | 双臂操作同一刚性物体时需要协调规划以避免内力 | 双臂操作可以完全独立规划 |
| MuJoCo 接触精度 | MuJoCo 使用 soft contact 模型，适合精细接触仿真 | MuJoCo 不支持接触力仿真 |
| Isaac Lab GPU 并行 | Isaac Lab 基于 GPU 并行仿真，支持数千并行环境 | Isaac Lab 不支持大规模并行训练 |
| Domain Randomization 参数分布 | Domain Randomization 对参数施加随机扰动以提升 sim2real 泛化 | Domain Randomization 使用固定参数 |
| Diffusion Policy 多模态 | Diffusion Policy 天然处理多模态动作分布，适合复杂操作任务 | Diffusion Policy 只能生成单峰动作分布 |
| VLA 推理延迟 | VLA 推理延迟通常 100ms 以上，需要异步执行框架 | VLA 可实时同步控制高频关节 |

- [x] 添加 11 条，总计 **50 条** CANONICAL_FACTS
- [x] `make lint` 0 矛盾报告，无误判

### 完成标准

- [x] `scripts/lint_wiki.py` 中 CANONICAL_FACTS 条目数 = 50
- [x] `make lint` 全通过

---

## P2 · 薄弱页面加深（Compounding Knowledge）

**背景**：Karpathy："the wiki keeps getting richer with every source you add"。当前 7 个页面 < 200 词，影响搜索质量和知识完整性。

### P2.1 · stub 页面补全（当前 < 150 词）

| 文件 | 当前词数 | 补充内容要点 |
|------|---------|------------|
| `wiki/tasks/manipulation.md` | ~136 | 操作任务分类 / 关键挑战 / 代表方法（ACT / Diffusion Policy / contact-rich）/ 与 locomotion 的区别 |
| `wiki/concepts/contact-rich-manipulation.md` | ~163 | 接触力建模 / 顺应性控制 / 与 rigid manipulation 的区别 / 典型场景（装配/拧螺丝）|
| `wiki/concepts/terrain-adaptation.md` | ~179 | 地形感知方式（高度图/点云）/ 步态规划策略 / exteroceptive vs proprioceptive / 典型算法 |
| `wiki/tasks/bimanual-manipulation.md` | ~stub | 闭链约束 / 协调控制 / 数据采集难点 / 代表工作（ACT bimanual）|

### P2.2 · 薄弱但有框架的页面加深（150-250 词）

| 文件 | 当前词数 | 补充内容要点 |
|------|---------|------------|
| `wiki/concepts/sensor-fusion.md` | ~195 | IMU + 关节编码器融合 / 卡尔曼滤波 / Legged Robot 状态估计 / 与 SLAM 的关系 |
| `wiki/methods/behavior-cloning.md` | ~249 | Covariate shift 问题 / compounding error / 与 DAgger 对比 / 实践建议 |
| `wiki/methods/vla.md` | ~242 | VLA 架构（Vision-Language-Action）/ RT-1 / RT-2 / 低频动作输出 / 与传统 IL 区别 |

- [x] 加深 P2.1 中 4 个 stub 页面，每页 ≥ 400 字
- [x] 加深 P2.2 中 3 个薄弱页面，每页 ≥ 400 字
- [x] 每个更新页面保持 frontmatter `updated:` 字段为当前日期
- [x] `make lint` 0 issues

### 完成标准

- [x] 上述 7 个页面字数均 ≥ 400 字
- [x] `graph-stats.json` 中各节点 `health_score` 分布：score=3 比例 ≥ 60%

---

## P3 · 新增高价值 Query 页（Good Answers Filed Back）

**背景**：Karpathy："good answers can be filed back into the wiki as new pages"。当前以下高频问题无对应 query 页：

| 文件 | 触发问题 |
|------|---------|
| `wiki/queries/domain-randomization-guide.md` | 「做 sim2real，如何设计 domain randomization 的参数与分布范围？」 |
| `wiki/queries/clf-cbf-in-wbc.md` | 「CLF 和 CBF 如何在 WBC/MPC 中联合使用保证稳定性与安全性？」 |
| `wiki/queries/vla-with-low-level-controller.md` | 「VLA 如何与低级关节控制器（MPC/WBC）融合？有哪些架构？」 |
| `wiki/queries/contact-rich-manipulation-guide.md` | 「做接触丰富的操作任务（装配/拧螺丝），有哪些实践要点？」 |

- [x] 新建以上 4 个 query 页
  - 格式：`> **Query 产物**` 说明 + 对比表或决策树 + `## 参考来源` + `## 关联页面`
  - 每个 query 页在至少 2 个现有 wiki 页的 `## 关联页面` 中添加回链（防孤儿）
- [x] `make lint` 0 issues

### 完成标准

- [x] 4 个 query 页均通过 lint，无孤儿
- [x] `python3 scripts/search_wiki.py "domain randomization"` 返回 `domain-randomization-guide.md` 前 3
- [x] graph 节点数 ≥ 106

---

## P4 · Lint 检测增强（健康检查深化）

**背景**：Karpathy："periodically ask the LLM to health-check the wiki"。当前 lint 17 项检测中缺少对孤儿节点数量的自动预警。

### P4.1 · 孤儿节点数量 Lint 检测项

- [x] 在 `scripts/lint_wiki.py` 中添加检测项 `orphan_count`：
  - 读取 `exports/graph-stats.json` 中 `orphan_nodes` 列表
  - 若 `orphan_nodes` 非空，输出 `⚠️ 发现 N 个孤儿节点（无入链）：[列表]`
  - 纳入 lint 检测项计数（18 项）
- [x] 更新 `check_export_quality.py`：若 `orphan_count` 非零则标记 ⚠️

### P4.2 · 搜索回归用例扩展至 26 条

当前 18 条，V12/V13 新增页面（CLF/CBF/contact-rich/VLA 融合/domain-randomization 等）尚无覆盖：

- [x] 在 `schema/search-regression-cases.json` 中新增 8 条用例（覆盖 P3 新增 query 页 + P2 加深页面）
- [x] `python3 scripts/eval_search_quality.py` 通过率维持 ≥ 80%

### 完成标准

- [x] `make lint` 检测项升至 **18 项**
- [x] 搜索回归 26/26 通过（或 ≥ 80%）
- [x] `make lint` 对孤儿节点自动报警正常工作

---

## P5 · 内容长尾补全

### P5.1 · 学习路径更新

V12 新增了 4 条学习路径，但尚未覆盖"安全控制"和"接触丰富操作"方向：

- [x] 新建 `roadmap/learning-paths/if-goal-safe-control.md`（从 CLF/CBF → WBC → safe RL 的路径）
- [x] 新建 `roadmap/learning-paths/if-goal-contact-manipulation.md`（从 contact model → IL → contact-rich policy 路径）
- [x] 在 `index.md` 和 README 的"从哪里开始"表格中添加新路径入口

### P5.2 · Overview 页面更新

`wiki/overview/robot-learning-overview.md`（118 词）严重偏短，是知识库入口页：

- [x] 加深 `robot-learning-overview.md`（补充：三层架构说明 / 五大主题导航 / 各 community 简介 / 学习建议）
- [x] 字数 ≥ 600 字

### P5.3 · log.md 追加 V13 启动记录

- [x] 追加 `## [2026-04-20] structural | v13-execution | V13 启动，P0-P5 规划`
- [x] 格式符合 `grep "^## \["` 可解析规范

### 完成标准

- [x] `roadmap/learning-paths/` 下有 ≥ 6 个路径页
- [x] `robot-learning-overview.md` 字数 ≥ 600 字
- [x] `log.md` 最近记录距今 ≤ 7 天

---

## Karpathy 方法论自我评估（V13 目标）

| Karpathy 原则 | V12 末状态 | V13 目标 |
|--------------|-----------|---------|
| Three-layer architecture | ✅ | ✅ 维持 |
| 0 orphans，交叉引用完整 | ⚠️ 4 个孤儿节点 | ✅ **0 个孤儿节点** |
| "Good answers filed back" | ✅ 24 query 页 | ✅ **28 query 页** |
| Lint 检测项 | ✅ 17 项 | ✅ **18 项**（新增孤儿计数）|
| CANONICAL_FACTS | ⚠️ 39 条（差 1） | ✅ **50 条** |
| Sources 覆盖率 100% | ✅ | ✅ 维持 |
| Hybrid BM25/vector | ⚠️ hashed fallback | ⚠️ 维持（环境限制）|
| 浏览器端搜索 | ✅ PWA + loading 状态 | ✅ 维持 |
| 知识图谱节点健康 | ✅ health_score 着色 | ✅ score=3 比例 ≥ 60% |
| 孤立社区 | ✅ 0 个 | ✅ 维持 |
| Ingest → suggest updates | ✅ `--suggest-updates` | ✅ 维持 |
| Log.md 活跃 | ✅ | ✅ 距今 ≤ 7 天 |
| 学习路径 | ✅ 4 条 | ✅ **6 条** |
| 图谱 badge 自动化 | ✅ | ✅ 维持 |
| type 字段完整性 | ⚠️ 2 页 unknown | ✅ unknown = 0 |

---

## 操作规范（延续 V1→V13）

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

### Op 3 · V13 完成标准（全部满足）

- [x] `make lint` 0 issues，CANONICAL_FACTS = 50 条，检测项 = 18 项
- [x] `graph-stats.json` 孤儿节点列表为空，`type_distribution.unknown` = 0
- [x] graph 节点数 ≥ 106
- [x] `make export-check` 11 项全通过
- [x] `roadmap/learning-paths/` 下有 ≥ 6 个路径页
- [x] 搜索回归 26/26 通过（或 ≥ 80%）
- [x] `log.md` 最近记录距今 ≤ 7 天
- [x] `robot-learning-overview.md` 字数 ≥ 600 字

---

## 状态说明

- `[ ]` 待执行
- `[~]` 进行中
- `[x]` 已完成
- `[−]` 已评估，决定跳过（附理由）
