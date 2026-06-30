# 技术栈项目执行清单 v27

最后更新：2026-06-29（v26 全数完成后新建：聚焦「接触力旋量闭环」知识链——把分散的接触感知/估计、力旋量表示、阻抗/混合力位控制、接触丰富操作策略沉淀为贯通 query/concept，并补矛盾检测规则与交互层专题）
项目仓库：<https://github.com/ImChong/Robotics_Notebooks>
上一版清单：[`tech-stack-next-phase-checklist-v26.md`](archive/tech-stack-next-phase-checklist-v26.md)
方法论参考：[Karpathy LLM Wiki](../../wiki/references/llm-wiki-karpathy.md)

---

## V26 交付基线 (V27 起点)

| 维度 | V26 状态 | V27 目标 |
|------|-----------|---------|
| 知识图谱节点 | 1513 | **≥ 1525** |
| 知识图谱边数 | 10195 | **≥ 10260** |
| 事实库 (CANONICAL_FACTS) | 210 条 | **≥ 220 条** |
| 社区结构 | 15 社区，最大社区占 16.7%（`community_quality_warning: false`） | **保持 ≤ 25%，新增专题不破坏均衡** |
| 技术专题 | 仿真物理保真度链路（V26 交付） | **建立"接触力旋量闭环"知识链** |
| 图谱专题视图 | V26 扩至 16 项（新增「物理保真度」） | **新增「接触力控」专题至 17 项** |

> 背景：V26 收尾把仿真物理底座（contact-dynamics / friction / articulated-body / floating-base）沉淀为「物理保真度」链路；与此同时近周密集 ingest 了一批**接触/操作**论文——CHORD（接触力旋量引导灵巧操作）、SceneBot（contact-prompted 全身场景交互跟踪）、HapMorph（可穿戴触觉渲染）。仓库里已各自成页的 `contact-estimation`、`force-control-basics`、`hybrid-force-position-control`、`impedance-control`、`tactile-sensing`、`visuo-tactile-fusion`、`contact-rich-manipulation` 七八页**缺一条贯通视角**——从**接触感知/估计 → 力旋量表示 → 阻抗/导纳/混合力位控制 → 接触丰富操作策略**逐层「闭环」如何分别贡献操作稳定性、各层的带宽/刚度/接触保真度取舍，尚未沉淀为独立 query / concept；事实库也缺「力控范式间矛盾」（力控 vs 位控、阻抗 vs 导纳、刚性高带宽 vs 柔顺安全、视觉 vs 触觉时延等）的矛盾检测规则。V27 优先补齐这条接触力旋量闭环知识链，并把分散的接触/力控/触觉概念页交叉链路规范化。

---

## P0: 自动化与工具链深度强化 (Engineering)

- [ ] **接触/力控/操作概念页交叉链路巡检 V1**：
    - [ ] `scripts/lint_wiki.py` 新增 `_check_contact_control_crosslink`：对 `tags` 含 `contact` / `force-control` / `impedance` / `manipulation` / `tactile` 的 `concepts/` 概念页，检查正文是否回链到「接触力旋量闭环」专题枢纽（`contact-wrench-closed-loop` / `topic-contact-force-control`，缺失给 INFO 级 `contact_control_crosslink` 提示，不阻塞 CI），枢纽页自身豁免；写入 lint 报告基线快照（`exports/lint-report.md`）；新增 `tests/test_lint_wiki_contact_control_crosslink.py` 用例覆盖（列表式/内联式 tag、有/无回链、枢纽豁免、INFO 不计失败）。

## P1: 接触力旋量闭环知识链专题 (Quality)

- [x] **接触力旋量闭环知识链 (+2)**：
    - [x] `wiki/queries/contact-wrench-closed-loop.md`（端到端 Query：接触感知/估计 → 力旋量表示 → 阻抗/导纳/混合力位控制 → 接触丰富操作策略四层闭环的取舍决策树，覆盖每层对操作稳定性/安全性的贡献、带宽/刚度/时延成本与典型失败模式，配 Mermaid 流程图）。
    - [x] `wiki/concepts/contact-force-loop-bandwidth.md`（力控闭环带宽 ↔ 接触稳定性概念页：明示感知时延、控制刚度、接触离散化如何共同决定可达带宽与接触震荡/穿透边界，以及与阻抗/导纳选型的关系）。
- [ ] **接触/力控层专题交叉补强**：
    - [ ] 在 `wiki/concepts/contact-estimation.md`、`force-control-basics.md`、`hybrid-force-position-control.md`、`impedance-control.md`、`visuo-tactile-fusion.md` 五页与 P1 新页形成双向回链，明示「感知 → 力旋量 → 控制 → 操作」四层在闭环链路中的定位，消除孤儿页。

## P2: 事实库与矛盾检测扩展 (Quantity)

- [ ] **事实库扩展**：
    - [ ] `schema/canonical-facts.json` 由 210 → **≥ 220 条**：新增 ≥10 条接触力控矛盾检测规则（力控带宽↑ 与控制刚度/稳定裕度冲突、阻抗 vs 导纳因果对偶在接触刚度未知时失稳、刚性高带宽与柔顺安全取舍、纯视觉时延致接触前过冲、触觉采样率不足致打滑漏检、混合力位控制方向选择错误致约束冲突、力旋量估计依赖准确惯量/雅可比、接触离散化致力旋量偏大、过度柔顺牺牲定位精度、域随机化不替代真机力标定等）；逐条经脚本校验对现存 wiki 页有 pos 命中且 0 误报（`make lint` 潜在矛盾 0 个）。

## P3: 交互层"接触力控"增强 (UX/UI)

- [ ] **图谱页"接触力控"专题视图**：
    - [ ] `docs/topic-filters.js` 单一事实源新增「接触力控」专题（`contact-force-control`，⚙️/🤝 emoji 待定），复用 path 片段并集机制（`impedance` / `admittance` / `wrench` / `compliance` / `hybrid` / `forcecontrol` 等，注意与 `physics-fidelity` / `tactile` / `grasp` 既有专题的最小重叠）并按需 `ids` 显式纳入新建 query/concept；同步在 `docs/graph.html` `#filter-topic-chips` 增加对应 chip。新建专题汇总枢纽页 `wiki/overview/topic-contact-force-control.md` 并从 query/concept 双向回链消除孤儿（`graph-stats.json` 0 orphans）。Puppeteer 截图归档至 `.cursor-artifacts/screenshots/graph-topic-contact-force-control.png`。
- [ ] **详情页"同专题相关页"提示**：
    - [ ] 复用 `docs/topic-filters.js` 单一事实源（`renderMetaTopicBadges` → `topicsForNode` 已数据驱动），接触/力控/新建页命中「接触力控」专题时自动渲染对应轻量徽标 + 跳转 `graph.html?topic=contact-force-control`（空态降级隐藏）。端到端验证截图归档至 `.cursor-artifacts/screenshots/detail-topic-contact-force-control.png`。

---

## 验收标准 (Definition of DoD)

- [ ] `make lint`: 0 errors（新引入的 `contact_control_crosslink_check` 为 INFO 级，不阻塞 CI）。
- [ ] 知识图谱节点数 **≥ 1525**，边数 **≥ 10260**（见 `exports/graph-stats.json`）。
- [ ] 事实库扩展至 **≥ 220 条**（重点补 力控带宽 / 阻抗导纳 / 视触觉时延 矛盾检测规则）。
- [ ] `community_quality_warning` 保持 `false` 且 `largest_community_ratio ≤ 0.25`。
- [ ] `log.md` 记录 V27 关键改动。

---

## 状态说明

- `[ ]` 待执行
- `[~]` 进行中
- `[x]` 已完成
- `[−]` 已评估，决定跳过（附理由）
