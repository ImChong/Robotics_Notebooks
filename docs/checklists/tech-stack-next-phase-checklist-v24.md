# 技术栈项目执行清单 v24

最后更新：2026-06-06（V24 启动，基于 V23 完整交付）
项目仓库：<https://github.com/ImChong/Robotics_Notebooks>
上一版清单：[`tech-stack-next-phase-checklist-v23.md`](tech-stack-next-phase-checklist-v23.md)
方法论参考：[Karpathy LLM Wiki](../../wiki/references/llm-wiki-karpathy.md)

---

## V23 交付基线 (V24 起点)

| 维度 | V23 状态 | V24 目标 |
|------|-----------|---------|
| 知识图谱节点 | 690 | **≥ 705** |
| 知识图谱边数 | 4993 | **≥ 5050** |
| 事实库 (CANONICAL_FACTS) | 172 条 | **≥ 185 条** |
| 社区结构 | 17 社区，最大社区占 10.4%（`community_quality_warning: false`） | **保持 ≤ 25%，新增专题不破坏均衡** |
| 技术专题 | 全身运动跟踪（WBT）/ 跨具身迁移 / 真机安全微调（V23 P1–P3 全部交付） | **建立"视觉感知骨干与机器人表征"专题** |
| 图谱专题视图 | V23 扩至 13 项（新增 WBT / 跨具身 / 真机安全微调） | **新增「视觉感知骨干」专题至 14 项** |

> 背景：V23 期间 ingest 了 ResNet（1512.03385）与 YOLO v1（1506.02640），但视觉感知层 wiki 仍偏薄——仅 `wiki/concepts/vision-backbones.md` 与 `wiki/methods/object-detection.md` 两页核心、无 comparisons / queries 落地，也缺「骨干 → 表征 → 下游策略输入」的衔接视角。V24 优先补齐这条从视觉骨干到机器人策略输入的知识链。

---

## P0: 自动化与工具链深度强化 (Engineering)

- [x] **陈旧声明（stale claim）巡检 V1**：
    - [x] `scripts/lint_wiki.py` 新增 `stale_claim_check`：扫描正文出现「SOTA / 最新 / 当前最强 / state-of-the-art」等绝对化措辞但 frontmatter `updated` 早于库内同主题更晚页面的情形，给出 INFO 级提示（不阻塞 CI），并写入 lint 报告基线快照；新增用例覆盖到 `tests/`。（实现 `_check_stale_claims`，按共享 tag 判定同主题；基线快照 5 条；`tests/test_lint_wiki_stale_claims.py` 6 例覆盖）
- [ ] **缺页概念巡检 V1**：
    - [ ] `scripts/lint_wiki.py` 新增 `missing_concept_page_check`：统计正文中以 `**术语**`/反引号高频出现（≥ N 页引用）但无独立 `wiki/concepts|methods|formalizations` 页的术语，输出"建议新建页"候选清单（INFO 级，不阻塞 CI），作为后续 ingest/query 选题入口。
- [x] **query → wiki 回填脚手架**：
    - [x] 新增 `scripts/scaffold_wiki_page.py`：给定 type（concept/comparison/query/...）与标题，按全库 frontmatter 规范生成骨架（含速查区块锚点、`related`/`sources` 占位、三段式正文骨架），降低把 query 答案沉淀回 wiki 的手工成本；自带 `--dry-run` 与 lint 自检。

## P1: 视觉感知骨干与机器人表征专题 (Quality)

- [x] **视觉表征知识链 (+3)**：
    - [x] `wiki/comparisons/cnn-vs-vit-backbones.md`（CNN（ResNet 系）vs ViT 系视觉骨干对比：归纳偏置、数据量需求、分辨率/吞吐、下游迁移、在机器人感知中的取舍）。
    - [x] `wiki/concepts/visual-representation-for-policy.md`（视觉表征作为策略输入：端到端联合训练 vs 冻结预训练骨干 vs 机器人专用预训练表征（R3M / VC-1 / DINOv2 等）三条路径与取舍）。
    - [x] `wiki/queries/perception-backbone-selection.md`（机器人感知骨干/表征选型 Query：分类骨干 / 检测头 / 通用预训练表征三类，给出选型决策树与典型失败模式）。
- [x] **视觉感知专题交叉补强**：
    - [x] 在 `wiki/concepts/vision-backbones.md` 与 `wiki/methods/object-detection.md` 中明示「骨干特征 → 检测/分割头 → 策略输入」的衔接，并与 P1 新页形成双向回链，消除孤儿页。

## P2: 事实库与矛盾检测扩展 (Quantity)

- [ ] **事实库扩展**：
    - [ ] `schema/canonical-facts.json` 由 172 → **≥ 185 条**：新增视觉骨干（ResNet 残差连接 / ViT 数据量门槛 / YOLO 单阶段 vs 两阶段）与机器人表征（冻结预训练表征 vs 端到端、R3M/VC-1 定位）专题的矛盾检测规则。

## P3: 交互层"专题与感知"增强 (UX/UI)

- [ ] **图谱页"视觉感知骨干"专题视图**：
    - [ ] `docs/graph.html` 的 `TOPIC_FILTERS` 在 V23 13 项基础上新增「视觉感知骨干」专题；命中规则复用 community id + path 片段双路并集机制；同步在 `#filter-topic-chips` 增加对应 `data-topic` chip；新增视图配 Puppeteer 截图归档到 `.cursor-artifacts/screenshots/`。
- [ ] **详情页"同专题相关页"提示（可选）**：
    - [ ] 评估在详情页对命中某专题的页面给出"属于 X 专题"轻量徽标 + 跳转图谱专题视图的链接（空态降级隐藏）。

---

## 验收标准 (Definition of DoD)

- [ ] `make lint`: 0 errors（新引入的 `stale_claim_check` / `missing_concept_page_check` 均为 INFO 级，不阻塞 CI）。
- [ ] 知识图谱节点数 **≥ 705**，边数 **≥ 5050**（见 `exports/graph-stats.json`）。
- [ ] 事实库扩展至 **185 条** 以上（重点补 视觉骨干 / 机器人表征 矛盾检测规则）。
- [ ] `community_quality_warning` 保持 `false` 且 `largest_community_ratio ≤ 0.25`。
- [ ] `log.md` 记录 V24 关键改动。

---

## 状态说明

- `[ ]` 待执行
- `[~]` 进行中
- `[x]` 已完成
- `[−]` 已评估，决定跳过（附理由）
