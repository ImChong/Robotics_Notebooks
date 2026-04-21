# 技术栈项目执行清单 v14

最后更新：2026-04-20（V14 启动 + P0 搜索回归修复完成 26/26）
项目仓库：<https://github.com/ImChong/Robotics_Notebooks>
上一版清单：[`tech-stack-next-phase-checklist-v13.md`](tech-stack-next-phase-checklist-v13.md)
方法论参考：[Karpathy LLM Wiki](../../wiki/references/llm-wiki-karpathy.md)

---

## V13 完成基线（V14 起点）

| 维度 | V13 末状态 |
|------|-----------|
| wiki 节点（图谱） | **106 个**（concept 30 / query 28 / method 11 / formalization 10 / comparison 9 / entity 8 / task 7 / overview 1 / reference 1 / roadmap_page 1） |
| 图谱边数 | **675 条** |
| 社区数 | **4 个**（Locomotion 40 / WBC 37 / IL 24 / CLF 5） |
| wiki 总页面 | **107 页** |
| CANONICAL_FACTS | **50 条** |
| Sources 覆盖率 | **100%**（104/104） |
| 搜索回归测试 | ⚠️ 26 条，0/26 通过（`search_wiki.py` module-level `import numpy` 导致 BM25 也报错） |
| 导出质量检查 | ✅ 12/12 通过 |
| 孤儿节点 | ✅ **0 个** |
| type 字段缺失 | ✅ **0 页** |
| Lint 检测项 | **18 项** |
| 薄弱页面（< 200 字） | ⚠️ **3 个**（humanoid-robot / hybrid-force-position-control / impedance-control） |
| CLF 安全控制社区 | ⚠️ 仅 **5 节点**（过小，缺少 safe RL / CMDP / trajectory optimization 连接） |
| 学习路径 | ✅ **6 条** |
| 语义搜索 | ⚠️ hashed-token fallback（sentence-transformers 未安装） |
| README 正文 | ⚠️ 硬编码 "98个节点/596条边/110页" 已过时，实际为 106/675/107 |

---

## V14 阶段总目标

> V14 基于 Karpathy 方法论，聚焦三条主线：
>
> 1. **检索可用性（Search Reliability）**：修复 `search_wiki.py` numpy 延迟导入，使 BM25 搜索不依赖 numpy，搜索回归测试从 0/26 恢复至 26/26。
>
> 2. **知识图谱扩展（Graph Growth）**：补全 3 个薄弱页面，新增 formalization / method / query / entity 共 ~10 页，图谱节点从 106 扩至 ≥ 116，同时强化 CLF/安全控制社区（5 节点 → ≥ 10 节点）。
>
> 3. **健康检查深化（Lint Depth）**：Lint 检测项从 18 升至 20 项（新增搜索回归通过率检测 + README 正文版本同步检测）；CANONICAL_FACTS 扩展至 60 条，覆盖 impedance control / action chunking / safety filter / CMDP 等 V13 新增领域。

---

## P0 · 搜索回归可用性修复（最高优先级）

**背景**：`scripts/eval_search_quality.py` 全部 26 条用例报 `No module named 'numpy'`，根因是 `search_wiki.py` 第 15 行为 module-level `import numpy as np`，只有向量搜索路径才需要 numpy，BM25 路径不需要。

### P0.1 · numpy 改为延迟导入

- [x] 在 `scripts/search_wiki.py` 中，将 `import numpy as np` 移出 module 顶部，改为在 `load_vector_resources()` 和 `encode_query_vector()` 函数内部延迟导入
- [x] 修改后运行 `python3 scripts/search_wiki.py PPO`，确认无 numpy 报错
- [x] 运行 `python3 scripts/eval_search_quality.py`，确认 BM25 模式 ≥ 80% 通过

### 完成标准

- [x] `python3 scripts/search_wiki.py PPO` 无任何 import error
- [x] `python3 scripts/eval_search_quality.py` 通过率 ≥ 80%（≥ 21/26）— 实测 **26/26 (100%)**

---

## P1 · 薄弱页面修复（< 200 字）

**背景**：V13 新建的 3 个概念页仍低于 200 字，影响搜索质量和健康评分。

| 文件 | 当前字数 | 补充内容要点 |
|------|---------|------------|
| `wiki/entities/humanoid-robot.md` | ~169 | 人形机器人定义/主流平台（Unitree H1/G1, Apollo, Digit）/与四足机器人的区别/核心挑战 |
| `wiki/concepts/hybrid-force-position-control.md` | ~176 | 力/位混合控制原理/任务空间分解/典型应用（表面打磨/装配插销）/与 impedance control 对比 |
| `wiki/concepts/impedance-control.md` | ~154 | 阻抗控制数学形式（M/B/K 参数）/阻抗 vs 导纳/典型场景/与力控的区别/调参建议 |

- [ ] 加深上述 3 个页面，每页 ≥ 400 字
- [ ] 每页 frontmatter `updated:` 更新为当前日期
- [ ] `make lint` 0 stub_pages 警告

### 完成标准

- [ ] 3 个页面字数均 ≥ 400 字
- [ ] `make lint` 中 stub_pages 列表为空

---

## P2 · 知识图谱扩展（106 → ≥ 116 节点）

**背景**：Karpathy："the wiki keeps getting richer with every source you add"。CLF 社区仅 5 节点（过小），缺少 safe RL / CMDP 等连接页面；entity 层缺少主流四足平台；formalization 层缺少 locomotion 核心数学定义。

### P2.1 · 新增 Formalization 页（+3）

| 文件 | 内容 |
|------|------|
| `wiki/formalizations/cmdp.md` | Constrained MDP（CMDP）形式化：状态/动作/约束集/折扣因子/最优安全策略定义 |
| `wiki/formalizations/zmp-lip.md` | ZMP + LIP（Linear Inverted Pendulum）形式化：ZMP 公式/稳定性条件/CP/LIP 动力学方程 |
| `wiki/formalizations/friction-cone.md` | 摩擦锥形式化：Coulomb 摩擦模型/线性化锥（polyhedral cone）/接触力可行域 |

- [ ] 新建以上 3 个 formalization 页，每页有公式块、`## 参考来源`、`## 关联页面`，并在至少 2 个现有页加回链（防孤儿）
- [ ] `wiki/formalizations/zmp-lip.md` 在 `locomotion.md` / `gait-generation.md` 加回链
- [ ] `wiki/formalizations/cmdp.md` 在 `control-barrier-function.md` / `reinforcement-learning.md` 加回链
- [ ] `wiki/formalizations/friction-cone.md` 在 `contact-dynamics.md` / `contact-rich-manipulation.md` 加回链

### P2.2 · 新增 Method 页（+2）

| 文件 | 内容 |
|------|------|
| `wiki/methods/safe-rl.md` | 安全强化学习：CMDP / Lagrangian 方法 / CPO / safety layer / 与标准 RL 区别 |
| `wiki/methods/trajectory-optimization.md` | 轨迹优化：shooting method / collocation / DDP / iLQR / 与 MPC 的关系 |

- [ ] 新建以上 2 个 method 页，每页 ≥ 400 字，有 frontmatter / 参考来源 / 关联页面
- [ ] `safe-rl.md` 在 `reinforcement-learning.md` / `control-barrier-function.md` / `safety-filter.md` 加回链
- [ ] `trajectory-optimization.md` 在 `model-predictive-control.md` / `whole-body-control.md` 加回链

### P2.3 · 新增 Entity 页（+2）

| 文件 | 内容 |
|------|------|
| `wiki/entities/anymal.md` | ANYmal 四足机器人：ETH 开发/主要硬件规格/代表论文（RMA/ANYmal C）/与 Unitree 的区别 |
| `wiki/entities/boston-dynamics.md` | Boston Dynamics 机器人平台：Spot/Atlas/Stretch/商业化路线/代表技术 |

- [ ] 新建以上 2 个 entity 页，有 frontmatter / 参考来源 / 关联页面，在 `humanoid-robot.md` / `locomotion.md` 加回链

### P2.4 · 新增 Query 页（+3）

| 文件 | 触发问题 |
|------|---------|
| `wiki/queries/reward-shaping-guide.md` | 「训练 locomotion RL 策略时，奖励函数怎么设计？有哪些调教技巧？」 |
| `wiki/queries/locomotion-failure-modes.md` | 「locomotion RL 训练时常见的失败模式有哪些？怎么诊断？」 |
| `wiki/queries/wbc-tuning-guide.md` | 「WBC QP 求解器怎么调参？权重矩阵、约束松弛、热启动有哪些技巧？」 |

- [ ] 新建以上 3 个 query 页，每页 ≥ 400 字，含 `> **Query 产物**` 说明 / 决策树或对比表 / 参考来源 / 关联页面
- [ ] 每个 query 页在至少 2 个现有 wiki 页的关联页面中加回链

### 完成标准

- [ ] 新增 10 页（3 formalization + 2 method + 2 entity + 3 query），`graph-stats.json` 节点数 ≥ 116
- [ ] CLF/安全控制社区节点数 ≥ 10（safe-rl + cmdp + 现有 5 节点 + 新连接）
- [ ] 所有新页面通过 lint（无孤儿、有关联页面和参考来源）

---

## P3 · CANONICAL_FACTS 扩展（50 → 60 条）

**背景**：V13 新增了 impedance control / action chunking / safety filter / hybrid-force-position-control 等页面，对应事实尚未进入断言集。

在 `scripts/lint_wiki.py` 的 `CANONICAL_FACTS` 字典中补充 10 条：

| 事实名称 | 正向断言 | 反向断言（不应出现）|
|---------|---------|------------------|
| Impedance control 参数 | impedance control 通过调节 M/B/K（惯量/阻尼/刚度）矩阵控制末端交互力 | impedance control 直接控制末端位置而不考虑接触力 |
| Action chunking 预测步数 | action chunking 将策略输出扩展为多步动作序列以减少高频推理需求 | action chunking 每步只输出单一动作 |
| Safety filter 叠加方式 | safety filter 作为独立层叠加在原始策略输出之上，不修改策略本身 | safety filter 需要重新训练策略才能保证安全 |
| CMDP 约束形式 | CMDP 在标准 MDP 上增加约束成本函数，要求期望约束成本不超过阈值 | CMDP 把安全约束直接编码到奖励函数中 |
| ZMP 稳定性条件 | ZMP 在支撑多边形内时，足式机器人处于动态平衡状态 | ZMP 可以在支撑多边形外保持稳定 |
| Trajectory optimization 局部性 | 轨迹优化通常求解局部最优轨迹，依赖初始化质量 | 轨迹优化总是找到全局最优轨迹 |
| DDP 计算复杂度 | DDP（Differential Dynamic Programming）计算复杂度与时域长度线性增长 | DDP 计算复杂度随问题维度指数增长 |
| Hybrid force-position 分解 | 混合力/位控制将任务空间分解为力控子空间和位控子空间，各自独立控制 | 混合力/位控制同时在同一自由度上施加力和位置控制 |
| ANYmal 开源生态 | ANYmal 平台配套 ANYpyTools 和 raisimGym，有活跃开源社区 | ANYmal 不支持 sim2real 迁移 |
| LIP 简化条件 | LIP 模型假设质心高度恒定、支撑腿为质量点，简化计算 | LIP 模型精确建模了完整多刚体动力学 |

- [ ] 添加 10 条，总计 **60 条** CANONICAL_FACTS
- [ ] `make lint` 0 矛盾报告，无误判

### 完成标准

- [ ] `scripts/lint_wiki.py` 中 CANONICAL_FACTS 条目数 = 60
- [ ] `make lint` 全通过（0 contradictions）

---

## P4 · Lint 检测项升至 20 项

**背景**：V13 Lint 已有 18 项，但缺少对搜索质量（eval_search 通过率）和 README 正文硬编码值的自动检测。

### P4.1 · 搜索回归通过率检测

- [ ] 在 `scripts/lint_wiki.py` 中新增 `search_regression` 检测项：
  - 调用 `python3 scripts/eval_search_quality.py` 子进程
  - 若通过率 < 80%，输出 `⚠️ 搜索回归通过率 X/26，低于 80%`
  - 纳入 lint 检测项计数（第 19 项）

### P4.2 · README 正文版本同步检测

- [ ] 在 `scripts/lint_wiki.py` 中新增 `readme_content` 检测项：
  - 读取 `README.md` 正文中 `**N 个节点**` 和 `**N 条边**` 的硬编码值
  - 与 `graph-stats.json` 中实际 `node_count`/`edge_count` 对比
  - 若不一致，输出 `⚠️ README 正文节点/边数已过时：正文为 N/M，实际为 X/Y`
  - 纳入 lint 检测项计数（第 20 项）

### P4.3 · 修复 README 正文硬编码值

- [ ] 将 README 中 `**98 个节点**` 更新为实际值（运行 `make graph` 后取 graph-stats.json）
- [ ] 将 README 中 `**596 条边**` 更新为实际值
- [ ] 将 README `make export` 注释中 `110 页` 更新为实际页数

### 完成标准

- [ ] `make lint` 检测项升至 **20 项**
- [ ] README 正文中节点/边数与 `graph-stats.json` 一致
- [ ] `make lint` 对 README 正文版本不一致自动报警正常工作

---

## P5 · 内容长尾与 log.md

### P5.1 · comparison 页面补全

V13 新增了 impedance control / hybrid force-position control，但尚无专门的对比页：

- [ ] 新建 `wiki/comparisons/impedance-vs-force-control.md`（阻抗控制 vs. 力控制：控制目标/稳定性/典型场景/选择依据）
- [ ] 新建 `wiki/comparisons/trajectory-opt-vs-rl.md`（轨迹优化 vs. RL：计算成本/泛化能力/在线/离线/典型选择场景）
- [ ] 在对应概念/方法页添加回链（防孤儿）

### P5.2 · log.md 追加 V14 启动记录

- [ ] 追加 `## [2026-04-20] structural | v14-execution | V14 启动，P0-P5 规划`
- [ ] 格式符合 `grep "^## \["` 可解析规范

### 完成标准

- [ ] `wiki/comparisons/` 下有 ≥ 11 个对比页
- [ ] `log.md` 最近记录距今 ≤ 7 天

---

## Karpathy 方法论自我评估（V14 目标）

| Karpathy 原则 | V13 末状态 | V14 目标 |
|--------------|-----------|---------|
| Three-layer architecture | ✅ | ✅ 维持 |
| 0 orphans，交叉引用完整 | ✅ 0 个孤儿节点 | ✅ 维持（新页面防孤儿）|
| "Good answers filed back" | ✅ 28 query 页 | ✅ **31 query 页** |
| Lint 检测项 | ✅ 18 项 | ✅ **20 项** |
| CANONICAL_FACTS | ✅ 50 条 | ✅ **60 条** |
| Sources 覆盖率 100% | ✅ | ✅ 维持 |
| BM25 搜索回归 | ⚠️ 0/26（numpy 报错） | ✅ **≥ 21/26** |
| Hybrid BM25/vector | ⚠️ hashed fallback | ⚠️ 维持（环境限制）|
| 浏览器端搜索 | ✅ PWA + loading 状态 | ✅ 维持 |
| 知识图谱节点健康 | ✅ health_score 着色 | ✅ 维持 |
| 孤立社区 | ✅ 0 个 | ✅ 维持 |
| Log.md 活跃 | ✅ | ✅ 距今 ≤ 7 天 |
| 学习路径 | ✅ 6 条 | ✅ 维持 |
| 图谱 badge 自动化 | ✅ | ✅ 维持 |
| type 字段完整性 | ✅ unknown = 0 | ✅ 维持 |
| 图谱节点数 | ✅ 106 | ✅ **≥ 116** |
| 安全控制社区大小 | ⚠️ 5 节点 | ✅ **≥ 10 节点** |
| README 正文同步 | ⚠️ 旧值未更新 | ✅ **自动检测 + 修复** |

---

## 操作规范（延续 V1→V14）

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
6. formalization 页面必须包含至少一个公式块（`$$...$$` 或 `$...$`）

### Op 3 · V14 完成标准（全部满足）

- [ ] `make lint` 0 issues，CANONICAL_FACTS = 60 条，检测项 = 20 项
- [ ] `graph-stats.json` 孤儿节点列表为空，节点数 ≥ 116
- [ ] CLF/安全控制社区节点数 ≥ 10
- [ ] `python3 scripts/eval_search_quality.py` 通过率 ≥ 80%
- [ ] README 正文节点/边数与 `graph-stats.json` 一致
- [ ] `log.md` 最近记录距今 ≤ 7 天
- [ ] `wiki/comparisons/` 下有 ≥ 11 个对比页

---

## 状态说明

- `[ ]` 待执行
- `[~]` 进行中
- `[x]` 已完成
- `[−]` 已评估，决定跳过（附理由）
