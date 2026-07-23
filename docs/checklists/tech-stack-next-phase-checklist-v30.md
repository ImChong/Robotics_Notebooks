# 技术栈项目执行清单 v30

最后更新：2026-07-20（v29 全数完成后新建：聚焦「执行器驱动链选型闭环」知识链——把近周密集 ingest 的一批**电子硬件 / 驱动 / 执行器建模**资料，从分散的实体页沉淀为一条贯通的「EDA 电路设计 → 电机驱动固件 FOC → 执行器建模与摩擦辨识 → 实时总线闭环集成」选型链，补驱动链层间矛盾检测规则与专题视图）
项目仓库：<https://github.com/ImChong/Robotics_Notebooks>
上一版清单：[`tech-stack-next-phase-checklist-v29.md`](archive/tech-stack-next-phase-checklist-v29.md)
方法论参考：[Karpathy LLM Wiki](../../wiki/references/llm-wiki-karpathy.md)

---

## V29 交付基线 (V30 起点)

| 维度 | V29 状态 | V30 目标 |
|------|-----------|---------|
| 知识图谱节点 | 1687 | **≥ 1698** |
| 知识图谱边数 | 13524 | **≥ 13580** |
| 事实库 (CANONICAL_FACTS) | 240 条 | **≥ 250 条** |
| 社区结构 | 16 社区，最大社区占 17.4%（`community_quality_warning: false`） | **保持 ≤ 25%，新增专题不破坏均衡** |
| 技术专题 | 具身大模型评测基准选型闭环链路（V29 交付） | **建立"执行器驱动链选型闭环"知识链** |
| 图谱专题视图 | V29 扩至 19 项（新增「具身评测基准」） | **新增「执行器驱动链」专题至 20 项** |

> 背景：V28 沉淀了「选哪一类具身大模型」（VLM/VLN/VLA/VLX/World-Model 五层选型链），V29 沉淀了「怎么评测/证明它」（认知→预测保真度→策略成功率→sim↔real gap 四层评测选型链），两条都在软件/算法侧。与之互补的问题是**「策略输出的力矩指令最终由什么样的电子硬件驱动链落地」**。近周密集 ingest 了一批**电子硬件 / 驱动 / 执行器建模**资料——KiCad（开源 EDA）、Altium Designer（商用 EDA）、SimpleFOC（开源 FOC 驱动固件）、NeuralActuator（神经执行器建模）、BAM / BAM-extended（执行器摩擦辨识）、SAGE（sim2real 执行器 gap 估计）等；仓库既有储备还包括 `simplefoc` / `ethercat-protocol` / `ethercat-master-optimization`（实时总线层）、`motor-torque-current-curve` / `motor-torque-speed-curve` / `planetary-roller-screw-humanoid-leg-actuation`（电机与传动层）、`implicit-explicit-actuator-modeling` / `actuator-network`（执行器建模层）与 `depth-torque-motor-design` 纵深路线。这些页各自独立（多为 `entities/` 实体页），但**缺一条贯通的驱动链选型视角**——从**EDA 电路设计（开源 KiCad vs 商用 Altium、驱动板自研 vs 商用一体化关节）→ 电机驱动固件 FOC（电流环带宽/编码器分辨率/标定）→ 执行器建模与摩擦辨识（理想力矩源假设何时破、显式摩擦模型 vs 神经执行器网络）→ 实时总线闭环集成（EtherCAT 周期/抖动与控制带宽的关系）**逐层「每层选什么、数据手册参数与实测曲线差在哪、建模保真度 vs 辨识成本如何取舍、总线周期 ≠ 闭环带宽」，尚未沉淀为独立 query / concept；事实库也缺「驱动链选型矛盾」（理想力矩源假设 vs 摩擦/齿隙实际、数据手册峰值参数 vs 实测持续力矩热约束、FOC 电流环带宽 vs 编码器分辨率制约、总线周期快 ≠ 闭环带宽高、执行器网络拟合好 vs 分布外温升漂移、高减速比力矩大 vs 反驱透明度损失、开源 EDA 够用 vs 高速多层板信号完整性、自研驱动板省钱 vs 可靠性/调试成本、仿真理想执行器 vs 真机 sim2real gap、驱动固件开环标定 vs 负载在环辨识等）的矛盾检测规则。V30 优先补齐这条执行器驱动链选型闭环知识链，并把分散的驱动链实体页交叉链路规范化。

---

## P0: 自动化与工具链深度强化 (Engineering)

- [x] **驱动链页交叉链路巡检 V1**：
    - [x] `scripts/lint_wiki.py` 新增 `_check_actuator_drive_chain_crosslink`：对 `tags` 含 `actuator` / `eda` / `foc`（子串匹配派生标签）的 `entities/` / `comparisons/` / `concepts/` 页，检查正文是否回链到「执行器驱动链选型闭环」专题枢纽（`actuator-drive-chain-selection-loop` / `topic-actuator-drive-chain`，缺失给 INFO 级 `actuator_drive_chain_crosslink` 提示，不阻塞 CI），枢纽页自身豁免；写入 lint 报告基线快照（`exports/lint-report.md`）；新增 `tests/test_lint_wiki_actuator_drive_chain_crosslink.py` 用例覆盖（列表式/内联式 tag、有/无回链、双枢纽、枢纽豁免、INFO 不计失败）。**（2026-07-22 完成：新增 `ACTUATOR_DRIVE_CHAIN_HUBS` / `ACTUATOR_DRIVE_CHAIN_TAG_KEYWORDS` 常量 + `_check_actuator_drive_chain_crosslink`，登记进 `INFO_ONLY_KEYS`/`_empty_results`/`format_report`，`make lint` 0 errors，报告新增 INFO 段（35 页待补回链，不阻塞 CI）；新用例 10 条全过、ruff/mypy 通过）**

## P1: 执行器驱动链选型闭环知识链专题 (Quality)

- [x] **执行器驱动链选型闭环知识链 (+2)**：
    - [x] `wiki/queries/actuator-drive-chain-selection-loop.md`（端到端 Query：EDA 电路设计 → 电机驱动固件 FOC → 执行器建模与摩擦辨识 → 实时总线闭环集成 四层驱动链选型的取舍决策树，覆盖每层选什么工具/方案、开源 vs 商用与自研 vs 一体化关节的取舍、数据手册参数与实测曲线的偏差来源、建模保真度 vs 辨识成本、总线周期与闭环带宽的关系与典型误判，配 Mermaid 决策流程图）。建页后从 `ethercat-master-optimization` query 页与 `implicit-explicit-actuator-modeling` 概念页回链（消孤儿，`graph-stats.json` 0 orphans）。**（2026-07-20 完成：新页含四层决策树 Mermaid、选型矛盾/失败模式速查、缩写表，`make lint` 0 errors，`graph-stats.json` 0 orphans；节点 1687→1716、边 13524→13723）**
    - [x] `wiki/concepts/torque-source-abstraction-gap.md`（「理想力矩源」抽象 ↔ 真实执行器 取舍概念页：明示 RL/MPC 策略把执行器当理想力矩源的抽象在摩擦/齿隙/带宽/热约束下何时破，并把这条 gap 讲成「策略力矩指令能否被真实驱动链忠实执行」的物理根因；配抽象成立条件表、缩小力矩执行 gap 的三条工程路线（摩擦辨识补偿 / 执行器网络 / 力矩传感闭环）与常见误判速查）。与 Query 页双向回链。**（2026-07-21 完成：新页含抽象成立条件表、四层 Mermaid gap 归因图、三条工程路线与常见误判/误区速查，与 Query 页及 implicit-explicit 概念页双向回链；`make lint` 0 errors，`graph-stats.json` 0 orphans）**

- [ ] **驱动链层级专题交叉补强**：
    - [ ] 在 `wiki/entities/kicad.md`（EDA 层，与 `altium-designer.md` 对照）、`wiki/entities/simplefoc.md`（驱动固件层）、`wiki/entities/paper-neuralactuator-neural-actuation-modeling.md` / `wiki/entities/bam-better-actuator-models.md`（执行器建模层）、`wiki/concepts/ethercat-protocol.md`（实时总线层）等页与 P1 新页（`queries/actuator-drive-chain-selection-loop.md`）形成双向回链：各页在 `related` 与「关联页面」补入驱动链选型闭环 Query 页并标注本页所在驱动链层；Query 页 `related` 含全部相关驱动链页，双向闭合，消除孤儿页。

## P2: 事实库与矛盾检测扩展 (Quantity)

- [ ] **事实库扩展**：
    - [ ] `schema/canonical-facts.json` 由 240 → **250 条**：新增 10 条驱动链选型矛盾检测规则（理想力矩源假设 vs 摩擦/齿隙实际、数据手册峰值力矩 vs 持续力矩热约束、FOC 电流环带宽 vs 编码器分辨率制约、总线周期快 ≠ 闭环带宽高、执行器网络拟合好 vs 分布外温升漂移、高减速比力矩大 vs 反驱透明度损失、开源 EDA 够用 vs 高速多层板信号完整性、自研驱动板省钱 vs 可靠性/调试成本、仿真理想执行器 vs 真机 sim2real gap、驱动固件开环标定 vs 负载在环辨识）；逐条经脚本校验对现存 wiki 页有 pos 命中且 0 误报（`make lint` 潜在矛盾 0 个、0 errors）。

## P3: 交互层"执行器驱动链"增强 (UX/UI)

- [ ] **图谱页"执行器驱动链"专题视图**：
    - [ ] `docs/topic-filters.js` 单一事实源新增「执行器驱动链」专题（`actuator-drive-chain`，⚡ emoji），复用 path 片段并集机制（`actuator` / `foc` 等干净片段，与既有专题保持最小重叠）并用 `ids` 显式纳入未被片段命中的驱动链页（`actuator-drive-chain-selection-loop` / `torque-source-abstraction-gap` / `kicad` / `altium-designer` / `simplefoc` / `ethercat-protocol` / `motor-torque-current-curve` / `motor-torque-speed-curve` / `planetary-roller-screw-humanoid-leg-actuation` 等）；同步在 `docs/graph.html` `#filter-topic-chips` 增加对应 chip。专题汇总枢纽页 `wiki/overview/topic-actuator-drive-chain.md` 已建（从相关驱动链/query 页交叉回链），`graph-stats.json` 0 orphans。专题视图落稳后截图归档至 `.cursor-artifacts/screenshots/graph-topic-actuator-drive-chain.png`。
- [ ] **详情页"同专题相关页"提示**：
    - [ ] 复用 `docs/topic-filters.js` 单一事实源（`renderMetaTopicBadges` → `topicsForNode` 已数据驱动），驱动链/新建页命中「执行器驱动链」专题时自动渲染对应轻量徽标 + 跳转 `graph.html?topic=actuator-drive-chain`（空态降级隐藏）。P3① 把 `actuator-drive-chain` 写入单一事实源后，详情页「所属专题」徽标行即自动联动；选一页驱动链实体页端到端验证并归档截图至 `.cursor-artifacts/screenshots/detail-topic-actuator-drive-chain.png`。

---

## 验收标准 (Definition of DoD)

- [ ] `make lint`: 0 errors（新引入的 `actuator_drive_chain_crosslink` 为 INFO 级，不阻塞 CI）。
- [ ] 知识图谱节点数 **≥ 1698**，边数 **≥ 13580**（见 `exports/graph-stats.json`）。
- [ ] 事实库扩展至 **250 条**（补齐 理想力矩源 vs 摩擦实际 / 峰值 vs 持续力矩热约束 / 总线周期 ≠ 闭环带宽 等 10 条驱动链选型矛盾检测规则）。
- [ ] `community_quality_warning` 保持 `false` 且 `largest_community_ratio ≤ 0.25`。
- [ ] `log.md` 记录 V30 关键改动。

---

## 状态说明

- `[ ]` 待执行
- `[~]` 进行中
- `[x]` 已完成
- `[−]` 已评估，决定跳过（附理由）
