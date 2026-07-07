# 技术栈项目执行清单 v28

最后更新：2026-07-05（v27 全数完成后新建：聚焦「具身大模型分类学选型闭环」知识链——把近周密集 ingest 的 VLM / VLN / VLA / VLX / World-Model 五大具身模型家族，从分散的实体/方法/对比页沉淀为一条贯通的「感知理解 → 空间导航 → 动作执行 → 一体化扩展 → 世界模型推演」选型链，补家族间矛盾检测规则与专题视图）
项目仓库：<https://github.com/ImChong/Robotics_Notebooks>
上一版清单：[`tech-stack-next-phase-checklist-v27.md`](archive/tech-stack-next-phase-checklist-v27.md)
方法论参考：[Karpathy LLM Wiki](../../wiki/references/llm-wiki-karpathy.md)

---

## V27 交付基线 (V28 起点)

| 维度 | V27 状态 | V28 目标 |
|------|-----------|---------|
| 知识图谱节点 | 1581 | **≥ 1595** |
| 知识图谱边数 | 10909 | **≥ 10970** |
| 事实库 (CANONICAL_FACTS) | 220 条 | **≥ 230 条** |
| 社区结构 | 16 社区，最大社区占 19.5%（`community_quality_warning: false`） | **保持 ≤ 25%，新增专题不破坏均衡** |
| 技术专题 | 接触力旋量闭环链路（V27 交付） | **建立"具身大模型分类学选型闭环"知识链** |
| 图谱专题视图 | V27 扩至 17 项（新增「接触力控」） | **新增「具身大模型」专题至 18 项** |

> 背景：近周密集 ingest 了一批**具身大模型分类**资料——深蓝「五大具身模型分类」（VLM/VLN/VLA/VLX/World-Model）、human five ViT 入门、MINT 频域意图分词 VLA、SimFoundry Real2Sim 等。仓库里已有 `vlm-vln-vla-vlx-world-model-taxonomy`（对比页）、`topic-vla`、`behavior-foundation-model`、`world-action-models`、`generative-world-models`、`unified-multimodal-tokens` 等页，但**缺一条贯通的选型视角**——从**跨模态感知理解（VLM）→ 空间导航（VLN）→ 动作执行（VLA）→ 一体化多任务扩展（VLX）→ 世界模型时序推演（WM）**逐层如何分工、各家族的 I/O 边界与数据需求/泛化能力/实时性/闭环稳定性取舍，尚未沉淀为独立 query / concept；事实库也缺「家族选型矛盾」（端到端 VLA vs 分层 VLN、显式世界模型预测 vs 无模型反应式、大模型泛化 vs 实时控制带宽、统一 VLX vs 专精分立等）的矛盾检测规则。V28 优先补齐这条具身大模型分类学选型闭环知识链，并把分散的家族概念页交叉链路规范化。

---

## P0: 自动化与工具链深度强化 (Engineering)

- [x] **具身大模型家族概念页交叉链路巡检 V1**：
    - [x] `scripts/lint_wiki.py` 新增 `_check_embodied_fm_crosslink`：对 `tags` 含 `vlm` / `vln` / `vla` / `vlx` / `world-model`（子串匹配派生标签）的 `concepts/` / `comparisons/` 页，检查正文是否回链到「具身大模型分类学选型闭环」专题枢纽（`embodied-fm-taxonomy-loop` / `topic-embodied-foundation-model`，缺失给 INFO 级 `embodied_fm_crosslink` 提示，不阻塞 CI），枢纽页自身豁免；写入 lint 报告基线快照（`exports/lint-report.md`，首批命中 10 页）；新增 `tests/test_lint_wiki_embodied_fm_crosslink.py` 用例覆盖（列表式/内联式 tag、有/无回链、双枢纽、枢纽豁免、INFO 不计失败）。

## P1: 具身大模型分类学选型闭环知识链专题 (Quality)

- [ ] **具身大模型分类学选型闭环知识链 (+2)**：
    - [ ] `wiki/queries/embodied-fm-taxonomy-loop.md`（端到端 Query：VLM 感知理解 → VLN 空间导航 → VLA 动作执行 → VLX 一体化扩展 → WM 世界模型推演 五层选型的取舍决策树，覆盖每层 I/O 边界、数据需求、泛化能力、实时性/控制带宽、闭环稳定性成本与典型失败模式，配 Mermaid 流程图）。
    - [ ] `wiki/concepts/embodied-fm-latency-generalization-tradeoff.md`（具身大模型实时性 ↔ 泛化能力取舍概念页：明示模型规模、多模态跨度、世界模型推演步长如何共同决定推理时延与控制带宽的可达边界，以及与分层/端到端选型的关系）。
- [ ] **具身大模型家族层专题交叉补强**：
    - [ ] 在 `wiki/comparisons/vlm-vln-vla-vlx-world-model-taxonomy.md`、`wiki/methods/vla.md`、`wiki/tasks/vision-language-navigation.md`、`wiki/concepts/world-action-models.md`、`wiki/methods/generative-world-models.md` 五页与 P1 新页形成双向回链，明示「感知 → 导航 → 执行 → 扩展 → 推演」五层在选型闭环中的定位，消除孤儿页。

## P2: 事实库与矛盾检测扩展 (Quantity)

- [ ] **事实库扩展**：
    - [ ] `schema/canonical-facts.json` 由 220 → **230 条**：新增 10 条具身大模型选型矛盾检测规则（端到端 VLA 泛化 vs 分层 VLN 可解释/可调、显式世界模型预测 vs 无模型反应式实时性、大模型参数量↑ 与控制带宽/推理时延冲突、统一 VLX 通用性 vs 专精分立精度、多模态跨度↑ 与数据标注成本、世界模型推演步长↑ 与累积误差、VLM 语义理解 ≠ 可执行动作接口、VLN 离散导航 vs 连续控制粒度、共享 Transformer 底座 ≠ 免调 sim2real、模型规模不替代真机动作数据 等）；逐条经脚本校验对现存 wiki 页有 pos 命中且 0 误报（`make lint` 潜在矛盾 0 个、0 errors）。

## P3: 交互层"具身大模型"增强 (UX/UI)

- [ ] **图谱页"具身大模型"专题视图**：
    - [ ] `docs/topic-filters.js` 单一事实源新增「具身大模型」专题（`embodied-foundation-model`，🧠 emoji），复用 path 片段并集机制（`vlm` / `vln` / `vlx` / `worldmodel` / `worldaction` 等干净片段——刻意剔除过宽的 `vla`（归 `topic-vla`，改用 `ids` 精选纳入），与 `vla` / `vision-backbone` 保持最小重叠）并用 `ids` 显式纳入五层闭环里未被片段命中的家族页（`embodied-fm-taxonomy-loop` / `vlm-vln-vla-vlx-world-model-taxonomy` / `behavior-foundation-model` / `unified-multimodal-tokens` 等）；同步在 `docs/graph.html` `#filter-topic-chips` 增加对应 chip。新建专题汇总枢纽页 `wiki/overview/topic-embodied-foundation-model.md` 并从 query（`embodied-fm-taxonomy-loop`）/concept（`embodied-fm-latency-generalization-tradeoff`）双向回链，`graph-stats.json` 0 orphans。专题视图筛出节点数落稳后 Puppeteer/Playwright 截图归档至 `.cursor-artifacts/screenshots/graph-topic-embodied-foundation-model.png`。
- [ ] **详情页"同专题相关页"提示**：
    - [ ] 复用 `docs/topic-filters.js` 单一事实源（`renderMetaTopicBadges` → `topicsForNode` 已数据驱动），家族/新建页命中「具身大模型」专题时自动渲染对应轻量徽标 + 跳转 `graph.html?topic=embodied-foundation-model`（空态降级隐藏）。端到端验证截图归档至 `.cursor-artifacts/screenshots/detail-topic-embodied-foundation-model.png`。

---

## 验收标准 (Definition of DoD)

- [ ] `make lint`: 0 errors（新引入的 `embodied_fm_crosslink` 为 INFO 级，不阻塞 CI）。
- [ ] 知识图谱节点数 **≥ 1595**，边数 **≥ 10970**（见 `exports/graph-stats.json`）。
- [ ] 事实库扩展至 **230 条**（补齐 端到端 vs 分层 / 世界模型 vs 反应式 / 泛化 vs 实时性 等 10 条具身大模型选型矛盾检测规则）。
- [ ] `community_quality_warning` 保持 `false` 且 `largest_community_ratio ≤ 0.25`。
- [ ] `log.md` 记录 V28 关键改动。

---

## 状态说明

- `[ ]` 待执行
- `[~]` 进行中
- `[x]` 已完成
- `[−]` 已评估，决定跳过（附理由）
