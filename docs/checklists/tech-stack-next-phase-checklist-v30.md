# 技术栈项目执行清单 v30

最后更新：2026-07-19（v29 全数完成后新建：聚焦「免机器人示教数据采集选型闭环」知识链——把近周密集 ingest 的一批**免机器人 / 硬件级低成本示教采集**资料，从分散的实体页沉淀为一条贯通的「免机器人采集范式选型 → embodiment gap 重定向 → 可执行性/接触保真校验 → 策略训练收益 vs 采集成本」选型链，补采集层间矛盾检测规则与专题视图）
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
| 技术专题 | 具身大模型评测基准选型闭环链路（V29 交付） | **建立"免机器人示教数据采集选型闭环"知识链** |
| 图谱专题视图 | V29 扩至 19 项（新增「具身评测基准」） | **新增「免机器人示教」专题至 20 项** |

> 背景：V28 沉淀了「选哪一类具身大模型」（VLM/VLN/VLA/VLX/World-Model 五层选型链），V29 沉淀了「怎么评测/证明它」（认知→预测保真度→策略成功率→sim↔real gap 四层评测选型链），紧接着的问题是**「怎么低成本采集训练它所需的示教数据」**。近周密集 ingest 了一批**免机器人 / 硬件级低成本示教采集**资料——HuMI（无机器人人形全身操作接口）、HandUMI（无机器人双臂示教软件 + LeRobot v3 多臂重定向栈）、HumanoidUMI（PICO VR + UMI 式夹爪无机器人采集稀疏关键点）、DexUMI（human hand as universal manipulation interface）、DexCap（便携 MoCap 采集）、BiFrost-UMI、ActiveUMI 等。仓库里这些页各自独立（多为 `entities/` 实体页，部分已带 `robot-free` 标签），但**缺一条贯通的免机器人采集选型视角**——从**免机器人采集范式选型（手持 UMI 夹爪 / 人手直采 / 便携 MoCap / VR 全身遥操 / 双臂示教）→ 跨 embodiment gap 重定向到目标机器人（腕部位姿/关键点/夹爪开合的动作空间对齐）→ 重定向后可执行性/接触保真校验 → 喂进策略训练的收益 vs 采集成本**逐层「用什么范式采、动作如何对齐、免机器人采集的吞吐/成本/可扩展性优势为何以 embodiment gap / 动作空间错配 / 接触-力反馈缺失 / 运动学可行性为代价」，尚未沉淀为独立 query / concept；事实库也缺「免机器人采集选型矛盾」（免机器人吞吐高 vs 可执行性存疑、人手灵巧 vs 机械手动作空间错配、无力反馈致接触任务失真、腕部位姿采集 ≠ 全身可行轨迹、单一本体采集 vs 跨本体泛化、便携 MoCap 漂移致标注噪声、VR 遥操延迟致动作失真、采集吞吐 ≠ 有效训练样本、免机器人数据需重定向过滤致有效率下降、视频提取动作缺接触/力标签等）的矛盾检测规则。V30 优先补齐这条免机器人示教采集选型闭环知识链，并把分散的免机器人采集实体页交叉链路规范化。

---

## P0: 自动化与工具链深度强化 (Engineering)

- [ ] **免机器人采集页交叉链路巡检 V1**：
    - [ ] `scripts/lint_wiki.py` 新增 `_check_robot_free_collection_crosslink`：对 `tags` 含 `robot-free` / `teleoperation` 且含 `data-collection`（子串匹配派生标签）的 `entities/` / `comparisons/` / `concepts/` 页，检查正文是否回链到「免机器人示教数据采集选型闭环」专题枢纽（`robot-free-demo-collection-selection-loop` / `topic-robot-free-demo-collection`，缺失给 INFO 级 `robot_free_collection_crosslink` 提示，不阻塞 CI），枢纽页自身豁免；写入 lint 报告基线快照（`exports/lint-report.md`）；新增 `tests/test_lint_wiki_robot_free_collection_crosslink.py` 用例覆盖（列表式/内联式 tag、有/无回链、双枢纽、枢纽豁免、INFO 不计失败）。

## P1: 免机器人示教数据采集选型闭环知识链专题 (Quality)

- [ ] **免机器人示教数据采集选型闭环知识链 (+2)**：
    - [ ] `wiki/queries/robot-free-demo-collection-selection-loop.md`（端到端 Query：免机器人采集范式选型 → embodiment gap 重定向 → 可执行性/接触保真校验 → 策略训练收益 vs 采集成本 四层采集选型的取舍决策树，覆盖每层用什么范式/工具采、动作空间如何对齐、免机器人采集的吞吐/成本/可扩展性优势为何以 embodiment gap / 动作错配 / 无力反馈 / 运动学可行性为代价与典型误判，配 Mermaid 决策流程图）。建页后从 `demo-data-collection-guide` / `motion-retargeting-pipeline` 等现存页回链（消孤儿，`graph-stats.json` 0 orphans）。
    - [ ] `wiki/concepts/robot-free-embodiment-gap.md`（免机器人采集吞吐/可扩展性 ↔ 目标机器人可执行性 取舍概念页：明示免机器人采集在吞吐/成本/无本体损耗上的优势为何以牺牲动作空间对齐/接触-力反馈/运动学可行性的代价，并把这条 gap 讲成「采到的人类演示能否被目标机器人可靠复现」的物理根因；配吞吐 vs 可执行性代价表、缩小 embodiment gap 的三条工程路线（重定向过滤/力标签补采/在环校验）与常见误判速查）。与 Query 页双向回链。

- [ ] **免机器人采集范式层专题交叉补强**：
    - [ ] 在 `wiki/entities/handumi.md`（双臂示教层）、`wiki/entities/paper-humanoidumi.md`（VR 全身遥操层）、`wiki/entities/paper-notebook-dexumi-using-human-hand-as-the-universal-manipul.md`（人手直采层）、`wiki/entities/paper-notebook-dexcap-scalable-and-portable-mocap-data-collecti.md`（便携 MoCap 层）等页与 P1 新页（`queries/robot-free-demo-collection-selection-loop.md`）形成双向回链：各页在 `related` 与「关联页面」补入采集选型闭环 Query 页并标注本页所在采集范式层；Query 页 `related` 含全部相关采集页，双向闭合，消除孤儿页。

## P2: 事实库与矛盾检测扩展 (Quantity)

- [ ] **事实库扩展**：
    - [ ] `schema/canonical-facts.json` 由 240 → **250 条**：新增 10 条免机器人采集选型矛盾检测规则（免机器人吞吐高 vs 可执行性存疑、人手灵巧 vs 机械手动作空间错配、无力反馈致接触任务失真、腕部位姿采集 ≠ 全身可行轨迹、单一本体采集 vs 跨本体泛化、便携 MoCap 漂移致标注噪声、VR 遥操延迟致动作失真、采集吞吐 ≠ 有效训练样本、免机器人数据需重定向过滤致有效率下降、视频提取动作缺接触/力标签）；逐条经脚本校验对现存 wiki 页有 pos 命中且 0 误报（`make lint` 潜在矛盾 0 个、0 errors）。

## P3: 交互层"免机器人示教"增强 (UX/UI)

- [ ] **图谱页"免机器人示教"专题视图**：
    - [ ] `docs/topic-filters.js` 单一事实源新增「免机器人示教」专题（`robot-free-demo-collection`，🕹️ emoji），复用 path 片段并集机制（干净片段与既有专题保持最小重叠）并用 `ids` 显式纳入未被片段命中的采集页（`robot-free-demo-collection-selection-loop` / `robot-free-embodiment-gap` / `handumi` / `paper-humanoidumi` / `paper-notebook-dexumi-...` / `paper-notebook-dexcap-...` / `paper-bifrost-umi` 等）；同步在 `docs/graph.html` `#filter-topic-chips` 增加对应 chip。专题汇总枢纽页 `wiki/overview/topic-robot-free-demo-collection.md` 已建（从相关采集/query 页交叉回链），`graph-stats.json` 0 orphans。专题视图落稳后截图归档至 `.cursor-artifacts/screenshots/graph-topic-robot-free-demo-collection.png`。
- [ ] **详情页"同专题相关页"提示**：
    - [ ] 复用 `docs/topic-filters.js` 单一事实源（`renderMetaTopicBadges` → `topicsForNode` 已数据驱动），采集范式/新建页命中「免机器人示教」专题时自动渲染对应轻量徽标 + 跳转 `graph.html?topic=robot-free-demo-collection`（空态降级隐藏）。P3① 把 `robot-free-demo-collection` 写入单一事实源后，详情页「所属专题」徽标行即自动联动；选一页免机器人采集实体页端到端验证并归档截图至 `.cursor-artifacts/screenshots/detail-topic-robot-free-demo-collection.png`。

---

## 验收标准 (Definition of DoD)

- [ ] `make lint`: 0 errors（新引入的 `robot_free_collection_crosslink` 为 INFO 级，不阻塞 CI）。
- [ ] 知识图谱节点数 **≥ 1698**，边数 **≥ 13580**（见 `exports/graph-stats.json`）。
- [ ] 事实库扩展至 **250 条**（补齐 免机器人吞吐 vs 可执行性 / 人手 vs 机械手动作错配 / 无力反馈接触失真 等 10 条免机器人采集选型矛盾检测规则）。
- [ ] `community_quality_warning` 保持 `false` 且 `largest_community_ratio ≤ 0.25`。
- [ ] `log.md` 记录 V30 关键改动。

---

## 状态说明

- `[ ]` 待执行
- `[~]` 进行中
- `[x]` 已完成
- `[−]` 已评估，决定跳过（附理由）
