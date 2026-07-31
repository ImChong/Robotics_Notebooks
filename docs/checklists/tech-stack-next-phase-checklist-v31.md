# 技术栈项目执行清单 v31

最后更新：2026-07-26（v30 全数完成后新建：聚焦「机器人视觉感知栈选型闭环」知识链——把近周密集 ingest 的一批**目标检测 / 分割 / 2D→3D 语义建图**资料，从分散的实体页沉淀为一条贯通的「传感与标定 → 2D 检测/分割选型 → 2D→3D 提升与语义建图 → 下游策略消费」感知栈选型链，补感知栈层间矛盾检测规则与路线视图）
项目仓库：<https://github.com/ImChong/Robotics_Notebooks>
上一版清单：[`tech-stack-next-phase-checklist-v30.md`](archive/tech-stack-next-phase-checklist-v30.md)
方法论参考：[Karpathy LLM Wiki](../../wiki/references/llm-wiki-karpathy.md)

---

## V30 交付基线 (V31 起点)

| 维度 | V30 状态 | V31 目标 |
|------|-----------|---------|
| 知识图谱节点 | 1913 | **≥ 1924** |
| 知识图谱边数 | 16405 | **≥ 16460** |
| 事实库 (CANONICAL_FACTS) | 250 条 | **≥ 260 条** |
| 社区结构 | 16 社区，最大社区占 21.6%（`community_quality_warning: false`） | **保持 ≤ 25%，新增纵深不破坏均衡** |
| 技术纵深 | 执行器驱动链选型闭环链路（V30 交付） | **建立"机器人视觉感知栈选型闭环"知识链** |
| 图谱路线视图 | V30 扩至 20 项（新增「执行器驱动链」） | **新增「机器人感知栈」纵深至 21 项** |

> 背景：V28 沉淀了「选哪一类具身大模型」（VLM/VLN/VLA/VLX/World-Model 五层选型链），V29 沉淀了「怎么评测/证明它」（认知→预测保真度→策略成功率→sim↔real gap 四层评测选型链），V30 沉淀了「策略输出的力矩指令由什么样的电子硬件驱动链落地」（EDA→FOC→执行器建模→实时总线四层驱动链）。与之互补、位于策略**输入端**的问题是**「策略/操作/导航消费的视觉感知信号从哪来、怎么选感知栈」**。近周密集 ingest 了一批**目标检测 / 分割 / 2D→3D 语义建图**资料——Ultralytics YOLO（单阶段实时检测）、RF-DETR（端到端 DETR）、YOLO v1（单阶段检测奠基论文）、Segment Anything / SAM2（可提示分割）、FindAnything / OV-SAM3D / OVO Semantic Mapping（开放词汇 3D 语义建图）、CMU MSCV Semantic 3D Mapping、GO2 三维语义建图（SAM 2D→3D 流水线）、Booster RoboCup Demo 与足球场线/球门检测等；仓库既有储备还包括 `object-detection` / `object-detection-model-selection`（检测选型 query）、`perception-backbone-selection` / `vision-backbones` / `vision-transformer`（骨干层）、`perception-coordinate-postprocessing`（坐标后处理）、`lovon`（腿式开放词汇导航）等。这些页各自独立（多为 `entities/` 实体页或零散 `methods/` `queries/`），但**缺一条贯通的感知栈选型视角**——从**传感与标定（RGB / RGB-D / LiDAR 输入模态、相机内外参标定、深度精度 vs 成本）→ 2D 检测/分割选型（单阶段 YOLO vs 两阶段、DETR 端到端 vs anchor、闭集检测 vs 开放词汇分割、实时机载 vs 服务器侧）→ 2D→3D 提升与语义建图（深度融合、点云语义、在线 vs 离线建图、稠密 vs 稀疏）→ 下游策略消费（导航/操作/WBC 如何消费感知输出、坐标后处理与感知-控制频率对齐）**逐层「每层选什么、精度 vs 时延/算力如何取舍、闭集准 vs 开放词汇泛、2D 框够用 vs 必须 3D 语义几何、感知频率 ≠ 控制频率」，尚未沉淀为独立 query / concept；事实库也缺「感知栈选型矛盾」的矛盾检测规则。V31 优先补齐这条机器人视觉感知栈选型闭环知识链，并把分散的感知页交叉链路规范化。

---

## P0: 自动化与工具链深度强化 (Engineering)

- [x] **感知栈页交叉链路巡检 V1**：（2026-07-29 完成，基线 72 项 INFO）
    - [x] `scripts/lint_wiki.py` 新增 `_check_perception_stack_crosslink`：对 `tags` 含 `detection` / `segmentation` / `perception` / `semantic-mapping`（连字符 token 前缀匹配派生标签，覆盖 `object-detection` / `instance-segmentation` / `promptable-segmentation` / `semantic-mapping` 等，避免 `reception` / `impedance` 裸子串误判）的 `entities/` / `comparisons/` / `concepts/` / `methods/` 页，检查正文是否回链到「机器人视觉感知栈选型闭环」纵深枢纽（`robot-perception-stack-selection-loop` / `hub-perception-stack`，缺失给 INFO 级 `perception_stack_crosslink` 提示，不阻塞 CI），枢纽页自身豁免；已登记 `INFO_ONLY_KEYS`/`_empty_results`/`format_report` 并写入 lint 报告基线快照（`exports/lint-report.md`，当前 72 项待补回链页）；新增 `tests/test_lint_wiki_perception_stack_crosslink.py`（13 例：列表式/内联式 tag、entities/comparisons/concepts/methods 四类、query 枢纽/topic 枢纽/双枢纽回链、枢纽豁免、无关标签/裸子串不误判、复数与派生 tag、INFO 不计失败）。`make ci-preflight` 通过（0 errors，导出质量 12/12）。参照 V30 `_check_actuator_drive_chain_crosslink` 的常量登记方式（`INFO_ONLY_KEYS`/`_empty_results`/`format_report`）。

## P1: 机器人视觉感知栈选型闭环知识链纵深 (Quality)

- [x] **机器人视觉感知栈选型闭环知识链 (+2)**：（2026-07-27 完成）
    - [x] `wiki/queries/robot-perception-stack-selection-loop.md`（端到端 Query：传感与标定 → 2D 检测/分割选型 → 2D→3D 提升与语义建图 → 下游策略消费 四层感知栈选型的取舍决策树，覆盖每层选什么模型/方案、单阶段 vs 两阶段 vs DETR、闭集 vs 开放词汇、实时机载 vs 服务器侧、2D 框 vs 3D 语义几何、在线 vs 离线建图、感知频率与控制频率对齐的典型误判，配 Mermaid 决策流程图）。建页后从 `object-detection-model-selection` query 页与 `perception-backbone-selection` query 页回链（消孤儿，`graph-stats.json` 0 orphans）。
    - [x] `wiki/concepts/2d-to-3d-semantic-lifting-gap.md`（「2D 检测/分割结果」↔「可供策略消费的 3D 语义几何」取舍概念页：明示把 2D 框/掩码提升到 3D 语义地图时的信息损失与歧义——尺度不确定、遮挡、时序一致性、类别语义 vs 几何占据的分离——并把这条 gap 讲成「感知输出能否被下游导航/操作忠实消费」的物理根因；配 lifting 成立条件表、缩小 gap 的三条工程路线（深度融合 / 多视角一致性 / 语义-几何联合建图）与常见误判速查）。与 Query 页双向回链。

- [x] **感知栈层级纵深交叉补强**：（2026-07-28 完成，+7 边）
    - [x] 在 `wiki/entities/ultralytics.md` / `wiki/entities/rf-detr.md` / `wiki/entities/paper-yolo-unified-realtime-detection.md`（②2D 检测层）、`wiki/entities/paper-segment-anything.md` / `wiki/entities/paper-sam2.md`（②分割层）、`wiki/entities/findanything.md` / `wiki/entities/cmu-mscv-semantic-3d-mapping.md`（③2D→3D 语义建图层）等页与 P1 新页（`queries/robot-perception-stack-selection-loop.md`）形成双向回链：各页在 `related` frontmatter 与「关联页面」正文均补入感知栈选型闭环 Query 页并标注本页所在感知栈层（②/③），Query 页 `related` 已含全部相关感知页，双向闭合。`make ci-preflight` 通过：`graph-stats.json` 0 orphans、边数 17456 → 17463、`community_quality_warning: false`（`largest_community_ratio: 0.145`）。

## P2: 事实库与矛盾检测扩展 (Quantity)

- [x] **事实库扩展**：（2026-07-30 完成，250 → 260 条）
    - [x] `schema/canonical-facts.json` 由 250 → **260 条**：新增 10 条感知栈选型矛盾检测规则（单阶段检测快 vs 两阶段精度高、闭集检测准 vs 开放词汇泛化、实时机载算力受限 vs 服务器侧精度、2D 框够用 vs 必须 3D 语义几何、稠密语义建图信息全 vs 内存/时延、SAM 零样本分割强 vs 类别语义缺失、深度传感精度 vs 成本、在线建图实时 vs 离线建图完整、感知频率高 ≠ 控制闭环带宽高、DETR 端到端简洁 vs 收敛慢/小目标弱）；每条 `terms`/`pos_claims` 均逐条经脚本校验对现存感知栈页（`robot-perception-stack-selection-loop` / `2d-to-3d-semantic-lifting-gap` / `object-detection-model-selection` 等）有 pos 命中，`neg_claims` 采用朴素错误全句、不命中任何页（含未被剥离的「误判速查」表），保证 0 误报。`make lint` 通过：潜在矛盾 0 个、0 errors；`make ci-preflight` 12/12（`graph-stats.json` 0 orphans、`community_quality_warning: false`）。

## P3: 交互层"机器人感知栈"增强 (UX/UI)

- [ ] **图谱页"机器人感知栈"路线视图**：
    - [ ] `docs/depth-filters.js` 单一事实源新增「机器人感知栈」纵深（`perception-stack`，👁 emoji），复用 path 片段并集机制（`detection` / `segment` / `perception` 等干净片段，与既有 `vision-backbone` 纵深保持最小重叠——后者聚焦特征骨干，本纵深聚焦任务级感知流水线）并用 `ids` 显式纳入未被片段命中的感知页（`robot-perception-stack-selection-loop` / `2d-to-3d-semantic-lifting-gap` / `ultralytics` / `rf-detr` / `paper-yolo-unified-realtime-detection` / `paper-segment-anything` / `paper-sam2` / `findanything` / `cmu-mscv-semantic-3d-mapping` / `object-detection-model-selection` 等）；同步在 `docs/graph.html` `#filter-depth-chips` 增加对应 chip。纵深汇总枢纽页 `wiki/overview/hub-perception-stack.md` 需新建（从相关感知/query 页交叉回链），`graph-stats.json` 0 orphans。路线视图落稳后截图归档至 `.cursor-artifacts/screenshots/graph-hub-perception-stack.png`。
- [ ] **详情页"同纵深相关页"提示**：
    - [ ] 复用 `docs/depth-filters.js` 单一事实源（`renderMetaDepthBadges` → `depthsForNode` 已数据驱动），感知/新建页命中「机器人感知栈」纵深时自动渲染对应轻量徽标 + 跳转 `graph.html?depth=perception-stack`（空态降级隐藏）。P3① 把 `perception-stack` 写入单一事实源后，详情页「所属路线」徽标行即自动联动；选一页感知实体页端到端验证并归档截图至 `.cursor-artifacts/screenshots/detail-hub-perception-stack.png`。

---

## 验收标准 (Definition of DoD)

- [ ] `make lint`: 0 errors（新引入的 `perception_stack_crosslink` 为 INFO 级，不阻塞 CI）。
- [ ] 知识图谱节点数 **≥ 1924**，边数 **≥ 16460**（见 `exports/graph-stats.json`）。
- [x] 事实库扩展至 **260 条**（补齐 单阶段 vs 两阶段 / 闭集 vs 开放词汇 / 感知频率 ≠ 控制带宽 等 10 条感知栈选型矛盾检测规则）。（2026-07-30 完成）
- [ ] `community_quality_warning` 保持 `false` 且 `largest_community_ratio ≤ 0.25`。
- [ ] `log.md` 记录 V31 关键改动。

---

## 状态说明

- `[ ]` 待执行
- `[~]` 进行中
- `[x]` 已完成
- `[−]` 已评估，决定跳过（附理由）
