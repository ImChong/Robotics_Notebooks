# 技术栈项目执行清单 v25

最后更新：2026-06-16（V24 全部条目收口，基于最新数据集 ingest 初始化 V25）
项目仓库：<https://github.com/ImChong/Robotics_Notebooks>
上一版清单：[`tech-stack-next-phase-checklist-v24.md`](archive/tech-stack-next-phase-checklist-v24.md)
方法论参考：[Karpathy LLM Wiki](../../wiki/references/llm-wiki-karpathy.md)

---

## V24 交付基线 (V25 起点)

| 维度 | V24 状态 | V25 目标 |
|------|-----------|---------|
| 知识图谱节点 | 1193 | **≥ 1205** |
| 知识图谱边数 | 7421 | **≥ 7460** |
| 事实库 (CANONICAL_FACTS) | 186 条 | **≥ 196 条** |
| 社区结构 | 14 社区，最大社区占 17.7%（`community_quality_warning: false`） | **保持 ≤ 25%，新增专题不破坏均衡** |
| 技术专题 | 视觉感知骨干与机器人表征（V24 P1 交付） | **建立"人形训练数据管线"专题** |
| 图谱专题视图 | V24 扩至 14 项（新增「视觉感知骨干」） | **新增「训练数据管线」专题至 15 项** |

> 背景：V24 收尾阶段（2026-06-16）密集 ingest 了 AMASS / LaFAN1 / OMOMO / PHUMA / Humanoid Everyday 五套人形参考运动与操作数据集，并新建 `wiki/comparisons/humanoid-reference-motion-datasets.md` 选型对比页与 `wiki/concepts/motion-retargeting.md` 重定向概念页。但「数据层」仍缺一条贯通视角——从**原始动作捕捉 / 视频 → 重定向 → RL/IL 策略训练输入**的端到端选型与质量评估链路尚未沉淀为独立 query / concept，事实库也缺数据质量与物理可行性维度的矛盾检测规则。V25 优先补齐这条训练数据管线知识链，并把分散的数据集实体页元数据规范化。

---

## P0: 自动化与工具链深度强化 (Engineering)

- [x] **数据集页元数据巡检 V1**：
    - [x] `scripts/lint_wiki.py` 新增 `dataset_metadata_check`：对 `tags` 含 `dataset` 的实体页，检查正文是否含规模 / 模态 / 许可证 / 重定向就绪度等标准化速查字段，缺失给出 INFO 级提示（不阻塞 CI），并写入 lint 报告基线快照；新增用例覆盖到 `tests/`。（2026-06-17 完成：新增 `_check_dataset_entity_metadata` 与 result key `dataset_missing_metadata`（INFO 级，不计失败总数）；正文按关键词命中近似四维度，全库巡检命中 17 页基线写入 `exports/lint-report.md`；新增 `tests/test_lint_wiki_dataset_metadata.py` 4 用例。）
- [x] **数据集选型脚手架强化**：
    - [x] 复用 `scripts/scaffold_wiki_page.py`，为 `entity`（数据集）类型补充含「规模 / 模态 / 许可证 / 适配形态 / 重定向就绪度」速查块的骨架模板，降低后续数据集 ingest 的手工成本；自带 `--dry-run` 与 lint 自检。（2026-06-18 完成：新增 `--dataset` 旗标（仅 entity 类型，否则 rc=2），生成「## 数据集速查」五维度速查块并在 frontmatter 写入 `dataset` tag；速查块关键词全覆盖 lint `dataset_metadata_check` 四维度，新建数据集页 0 缺失；新增 3 条用例（速查块/tag/位置、lint 巡检 0 缺失、非 entity 拒绝）至 `tests/test_scaffold_wiki_page.py`，`ruff` 与 `lint_wiki` 通过。）

## P1: 人形训练数据管线专题 (Quality)

- [x] **训练数据管线知识链 (+2)**：
    - [x] `wiki/queries/humanoid-training-data-pipeline.md`（端到端 Query：原始动作捕捉 / 人体视频 → 重定向 → RL/IL 训练输入的选型决策树，覆盖参考运动来源、重定向方案、训练范式三层的取舍与典型失败模式）。（2026-06-19 完成：三层决策树（来源/重定向/范式）+ Mermaid 流程图 + 端到端 pipeline + 5 条误区 + 缩写速查；来源层接 humanoid-reference-motion-datasets 五集表，质量评估接 motion-data-quality 四轴。）
    - [x] `wiki/concepts/motion-data-quality.md`（动作数据质量维度概念页：物理可行性 / 接触一致性 / 形态差距（morphology gap）/ 规模与多样性四个评估轴，与重定向必要性的因果关系）。（2026-06-19 完成：四轴串联门模型 + 四轴↔重定向必要性因果链 + 五集数据集四轴对照表；并补 motion-retargeting.md 与五集对比页对本页/Query 的入链，无孤儿。）
- [x] **数据层专题交叉补强**：
    - [x] 在 `wiki/comparisons/humanoid-reference-motion-datasets.md` 与 `wiki/concepts/motion-retargeting.md` 中明示「数据来源 → 质量评估 → 重定向 → 策略输入」的衔接，并与 P1 新页形成双向回链，消除孤儿页。（2026-06-20 完成：对比页新增「四段衔接」表 + 因果判据段，重定向页新增「上游衔接」表把重定向定位为链路第③段、由 motion-data-quality 四轴的形态差距/接触/物理轴决定触发与补层；两页均显式回链 motion-data-quality 与 humanoid-training-data-pipeline，双向闭环；`make lint` 0 errors。）

## P2: 事实库与矛盾检测扩展 (Quantity)

- [x] **事实库扩展**：
    - [x] `schema/canonical-facts.json` 由 186 → **≥ 196 条**：新增数据层矛盾检测规则（动作捕捉缺接触/力信息导致物理不可行、人体视频规模大但标注/3D 信息弱、形态差距使原始动作不可直接复用、重定向不可省略、物理过滤数据集（PHUMA 类）相对纯 mocap 的可部署性等）；逐条经脚本校验对现存 wiki 页有 pos 命中且 0 误报。（2026-06-21 完成：新增 12 条数据层矛盾检测规则（186 → **198 条**），覆盖纯光学 MoCap 缺力/接触不可直执行、人体视频 3D/接触信息弱、形态差距大重定向不可省略、几何重定向≠物理可执行、PHUMA 物理过滤已重定向免工程、接触一致性为物理可行性前置、规模不能替代物理可行性、真机执行数据天然物理可行但任务窄、四质量轴串联门体检顺序、Humanoid Everyday 非重定向源、已重定向数据集免重定向直接训练、物理不可行参考致 RL 学错力矩；逐条经脚本校验对 `motion-data-quality` / `humanoid-training-data-pipeline` / `motion-retargeting` / `humanoid-reference-motion-datasets` 等现存页有 pos 命中、neg 0 命中，`make lint` 0 errors、潜在矛盾 0 条。）

## P3: 交互层"数据管线"增强 (UX/UI)

- [ ] **图谱页"训练数据管线"专题视图**：
    - [ ] `docs/graph.html` / `docs/topic-filters.js` 的专题命中规则在 V24 14 项基础上新增「训练数据管线」专题（`data-pipeline`），复用 path 片段并集机制（`dataset/datasets/amass/lafan1/omomo/phuma/humanoid-everyday/motion-retargeting/training-data`）并按需 `ids` 显式纳入新建 query/concept；同步在 `#filter-topic-chips` 增加 `data-topic="data-pipeline"`（📦 训练数据）chip。Puppeteer 截图归档至 `.cursor-artifacts/screenshots/graph-topic-data-pipeline.png`。
- [ ] **详情页"同专题相关页"提示**：
    - [ ] 复用 `docs/topic-filters.js` 单一事实源，数据集 / 重定向 / 新建页命中「训练数据管线」专题时渲染「📦 训练数据」轻量徽标 + 跳转 `graph.html?topic=data-pipeline`（空态降级隐藏）。端到端验证截图归档至 `.cursor-artifacts/screenshots/detail-topic-data-pipeline.png`。

---

## 验收标准 (Definition of DoD)

- [ ] `make lint`: 0 errors（新引入的 `dataset_metadata_check` 为 INFO 级，不阻塞 CI）。
- [ ] 知识图谱节点数 **≥ 1205**，边数 **≥ 7460**（见 `exports/graph-stats.json`）。
- [x] 事实库扩展至 **196 条** 以上（重点补 数据质量 / 物理可行性 / 重定向必要性 矛盾检测规则）。（2026-06-21 达成：198 条。）
- [ ] `community_quality_warning` 保持 `false` 且 `largest_community_ratio ≤ 0.25`。
- [ ] `log.md` 记录 V25 关键改动。

---

## 状态说明

- `[ ]` 待执行
- `[~]` 进行中
- `[x]` 已完成
- `[−]` 已评估，决定跳过（附理由）
