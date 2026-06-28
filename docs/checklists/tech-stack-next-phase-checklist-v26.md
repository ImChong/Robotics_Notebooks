# 技术栈项目执行清单 v26

最后更新：2026-06-28（P3 图谱页"物理保真度"专题视图：新增 `physics-fidelity` 专题至 16 项，新建汇总枢纽页 `topic-physics-fidelity.md`，专题命中 85 节点，图谱 0 孤儿）
项目仓库：<https://github.com/ImChong/Robotics_Notebooks>
上一版清单：[`tech-stack-next-phase-checklist-v25.md`](archive/tech-stack-next-phase-checklist-v25.md)
方法论参考：[Karpathy LLM Wiki](../../wiki/references/llm-wiki-karpathy.md)

---

## V25 交付基线 (V26 起点)

| 维度 | V25 状态 | V26 目标 |
|------|-----------|---------|
| 知识图谱节点 | 1322 | **≥ 1335** |
| 知识图谱边数 | 8809 | **≥ 8850** |
| 事实库 (CANONICAL_FACTS) | 198 条 | **≥ 208 条** |
| 社区结构 | 16 社区，最大社区占 16.5%（`community_quality_warning: false`） | **保持 ≤ 25%，新增专题不破坏均衡** |
| 技术专题 | 训练数据管线（V25 P1 交付） | **建立"仿真物理保真度链路"专题** |
| 图谱专题视图 | V25 扩至 15 项（新增「训练数据管线」） | **新增「物理保真度」专题至 16 项** |

> 背景：V25 收尾阶段（2026-06-23 前后）密集 ingest 了一批仿真物理底座概念页——`differentiable-simulation`（可微仿真）、`articulated-body-algorithms`（ABA/RNEA 刚体动力学）、`contact-dynamics` / `floating-base-dynamics` / `centroidal-dynamics`（接触与浮动基动力学）、`joint-friction-models` / `friction-compensation`（关节摩擦建模与补偿）、`urdf-robot-description`（机器人描述格式）、`procedural-terrain-generation`（程序化地形）。这些页各自成立，但仍缺一条贯通视角——从**几何/URDF 精度 → 刚体动力学算法 → 接触/摩擦模型 → 执行器模型**逐层「物理保真度」如何分别贡献 sim2real gap、以及各层建模成本与收益的取舍，尚未沉淀为独立 query / concept；事实库也缺「保真度维度间矛盾」（接触保真度 vs 可微性/吞吐、几何精度 vs 仿真速度、刚体假设 vs 软接触等）的矛盾检测规则。V26 优先补齐这条仿真物理保真度知识链，并把分散的动力学/仿真概念页交叉链路规范化。

---

## P0: 自动化与工具链深度强化 (Engineering)

- [x] **动力学/仿真概念页交叉链路巡检 V1**：
    - [x] `scripts/lint_wiki.py` 新增 `_check_physics_concept_crosslink`：对 `tags` 含 `dynamics` / `simulation` / `physics` 的 `concepts/` 与 `formalizations/` 概念页，检查正文是否回链到「仿真物理保真度」专题枢纽（`simulation-physics-fidelity` / `physics-fidelity-sim2real-gap`，缺失给 INFO 级 `physics_concept_crosslink` 提示，不阻塞 CI），枢纽页自身豁免；已写入 lint 报告基线快照（`exports/lint-report.md`，现 15 页待回链，P1 已回链的 5 页正确豁免）；新增 `tests/test_lint_wiki_physics_crosslink.py` 6 个用例覆盖（列表式/内联式 tag、有/无回链、枢纽豁免、INFO 不计失败）。

## P1: 仿真物理保真度链路专题 (Quality)

- [x] **物理保真度知识链 (+2)**：
    - [x] `wiki/queries/simulation-physics-fidelity.md`（端到端 Query：几何/URDF 精度 → 刚体动力学算法（ABA/RNEA）→ 接触/摩擦模型 → 执行器模型四层保真度的取舍决策树，覆盖每层对 sim2real gap 的贡献、建模成本与典型失败模式，配 Mermaid 流程图）。
    - [x] `wiki/concepts/physics-fidelity-sim2real-gap.md`（物理保真度 ↔ sim2real gap 因果概念页：四层保真度维度分层，明示每一维度的简化如何转化为可观测的 sim2real gap，以及与域随机化/系统辨识的互补关系）。
- [x] **仿真物理层专题交叉补强**：
    - [x] 在 `wiki/concepts/contact-dynamics.md`、`joint-friction-models.md`、`articulated-body-algorithms.md`（formalizations/）、`differentiable-simulation.md`、`urdf-robot-description.md` 五页与 P1 新页形成双向回链，明示「几何 → 动力学 → 接触/摩擦 → 执行器」四层在保真度链路中的定位，消除孤儿页。

## P2: 事实库与矛盾检测扩展 (Quantity)

- [x] **事实库扩展**：
    - [x] `schema/canonical-facts.json` 由 198 → **210 条**：新增 12 条物理保真度矛盾检测规则（接触保真度↑ 与可微性/仿真吞吐冲突、几何/URDF 惯量误差被上层逐级放大、硬接触穿透致冲击力偏大、库仑摩擦低估静摩擦致打滑、理想力矩源致执行器力矩 gap、可微仿真梯度受接触不连续制约、硬 LCP 接触不可微、积分步长过大致能量漂移/发散、软接触引入穿透与虚假阻尼、域随机化覆盖残差非替代保真度、保真度+SysID 互补、几何/URDF 层最便宜应优先做）；逐条经脚本校验对现存 wiki 页有 pos 命中且 0 误报（`make lint` 潜在矛盾 0 个）。

## P3: 交互层"物理保真度"增强 (UX/UI)

- [x] **图谱页"物理保真度"专题视图**：
    - [x] `docs/graph.html` / `docs/topic-filters.js` 的专题命中规则在 V25 15 项基础上新增「物理保真度」专题（`physics-fidelity`），复用 path 片段并集机制（`dynamics/contact/friction/articulated-body/differentiable-simulation/urdf/floating-base/centroidal/fidelity`）并按需 `ids` 显式纳入新建 query/concept（`simulation-physics-fidelity` / `physics-fidelity-sim2real-gap` / `articulated-body-algorithms`）；同步在 `#filter-topic-chips` 增加 `data-topic="physics-fidelity"`（⚙️ 物理保真度）chip。新建专题汇总枢纽页 `wiki/overview/topic-physics-fidelity.md` 并从 query/concept 双向回链消除孤儿（`graph-stats.json` 0 orphans）；专题命中 85 节点。Puppeteer 截图归档至 `.cursor-artifacts/screenshots/graph-topic-physics-fidelity.png`（页头实测 `85 / 1512 节点`）。
- [ ] **详情页"同专题相关页"提示**：
    - [ ] 复用 `docs/topic-filters.js` 单一事实源，动力学 / 仿真 / 新建页命中「物理保真度」专题时渲染「⚙️ 物理保真度」轻量徽标 + 跳转 `graph.html?topic=physics-fidelity`（空态降级隐藏）。端到端验证截图归档至 `.cursor-artifacts/screenshots/detail-topic-physics-fidelity.png`。

---

## 验收标准 (Definition of DoD)

- [ ] `make lint`: 0 errors（新引入的 `physics_concept_crosslink_check` 为 INFO 级，不阻塞 CI）。
- [x] 知识图谱节点数 **≥ 1335**，边数 **≥ 8850**（见 `exports/graph-stats.json`：现 1338 节点 / 9145 边）。
- [x] 事实库扩展至 **210 条**（重点补 物理保真度 / 接触摩擦 / 执行器建模 矛盾检测规则）。
- [x] `community_quality_warning` 保持 `false` 且 `largest_community_ratio ≤ 0.25`（现 0.183；新增 `humanoid-soccer` 社区命名 override）。
- [ ] `log.md` 记录 V26 关键改动。

---

## 状态说明

- `[ ]` 待执行
- `[~]` 进行中
- `[x]` 已完成
- `[−]` 已评估，决定跳过（附理由）
