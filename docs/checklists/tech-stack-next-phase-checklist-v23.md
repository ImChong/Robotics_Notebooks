# 技术栈项目执行清单 v23

最后更新：2026-05-25（V23 启动，基于 V22 完整交付）
项目仓库：<https://github.com/ImChong/Robotics_Notebooks>
上一版清单：[`tech-stack-next-phase-checklist-v22.md`](tech-stack-next-phase-checklist-v22.md)
方法论参考：[Karpathy LLM Wiki](../../wiki/references/llm-wiki-karpathy.md)

---

## V22 交付基线 (V23 起点)

| 维度 | V22 状态 | V23 目标 |
|------|-----------|---------|
| 知识图谱节点 | 429 | **≥ 445** |
| 知识图谱边数 | 3200 | **≥ 3320** |
| 事实库 (CANONICAL_FACTS) | 156 条 | **≥ 170 条** |
| 社区结构 | 17 社区，最大社区占 15.9%（`community_quality_warning: false`） | **保持 ≤ 25%，新增专题不破坏均衡** |
| 技术专题 | 动作重定向与角色化人形（5 页新链 + 双向回链） | **建立"全身运动跟踪（WBT）与跨具身迁移"专题** |

---

## P0: 自动化与工具链深度强化 (Engineering)

- [x] **缩写/别名归一化检索 V2**：
    - [x] `scripts/search_wiki_core.py` 的 `WIKI_ABBREVIATIONS` 在 V22 16 条基础上补 WBT / BFM / DAgger / RSI / RFC / RMA / EMA / LoRA / DoF 等 V22 期间新增的高频缩写，确保新热点直接可检；新增用例覆盖到 `tests/test_search_wiki_core.py`。（2026-05-26：补全 9 条至共 25 条；新增 `test_v22_abbreviations_expand_to_full` / `test_v22_full_phrases_expand_to_abbreviation` 两组子测试，`python -m unittest tests.test_search_wiki_core` 全 26 用例通过；ruff / mypy 同步绿）
- [x] **Entity-Paper 类页元数据 Lint**：
    - [x] `scripts/lint_wiki.py` 新增 `entity_paper_metadata_check`：以 `wiki/entities/paper-*.md` 为目标，校验 frontmatter 至少包含 `arxiv` / `venue` / `code` 三类来源中之一，且正文存在「方法栈 / 评测 / 与其他工作对比」三段式（缺失给出 INFO 级提示，不阻塞 CI）。基线快照写入 lint 报告，作为后续 ingest 工作流自检入口。（2026-05-27：新增 `_check_paper_entity_metadata` 与两条 INFO key `paper_missing_source_meta` / `paper_missing_three_sections`；基线快照：131/131 缺来源键、130/131 缺三段式之一。`tests/test_lint_wiki_paper_metadata.py` 新增 6 用例覆盖来源命中、缺失定位、非论文实体豁免、信息型计数；`python -m pytest tests/ --ignore=test_graph_layout.py` 106 通过，ruff / mypy 同步绿）
- [x] **图谱 latest_wiki_nodes 时间窗口可配置**：
    - [x] `scripts/generate_link_graph.py` 把 `latest_wiki_nodes` 的「最近 N 项」改为可通过 CLI flag / 环境变量配置（默认 10，上限 30）；前端 `docs/main.js` 在详情页底部新增「最近 30 天 ingest 时间线」轻量展示（仅在主入口/首页生效）。（2026-05-28：`latest_wiki_nodes_from_log` 新增 `max_items` / `window_days` 形参；新增 `resolve_latest_nodes_max()` 处理 CLI 标志 `--latest-nodes-max` + 环境变量 `GRAPH_LATEST_NODES_MAX`，默认 10、上限 30、最低 1；窗口默认 30 天，跨日合并保序。前端 `docs/main.js renderLatestWikiNode` 在跨日返回时按日期分组渲染时间线（仅 `homeLatestWikiModule` 挂载点生效，详情/图谱页不受影响）；`docs/style.css` 新增 `.home-latest-wiki-timeline*` 轻量样式。`tests/test_generate_link_graph_latest_nodes.py` 新增 10 用例覆盖 max_items / window 截断 / CLI vs env 优先级 / clamp 边界；`PYTHONPATH=scripts python3 -m unittest discover tests` 87 通过。）

## P1: 全身运动跟踪（WBT）与跨具身迁移专题 (Quality)

- [~] **WBT 知识链 (+3)**：
    - [x] `wiki/concepts/whole-body-tracking-pipeline.md`（WBT 端到端流水线：参考采集 → 重定向 → 训练数据 → 策略学习 → 跨具身迁移 → 真机部署的统一视图，区分 SONIC / SD-AMP / Heracles / Any2Any / BeyondMimic / GMT 等 6 条主流落地路径）。（2026-05-29：新建 `wiki/concepts/whole-body-tracking-pipeline.md`，6 阶段流水线 + 6 路径对比表 + 8 层系统栈映射 + 6 类失败模式 + 评测视角；frontmatter 链入 SONIC / BeyondMimic / SD-AMP / Heracles / Any2Any / RGMT 全部 6 条路径的 method/entity 页与对应 sources；与 `motion-retargeting-pipeline.md` 形成「映射 → 训练 → 迁移」三段流水线衔接的中段。）
    - [ ] `wiki/comparisons/sonic-vs-beyondmimic-vs-sdamp-vs-heracles.md`（四条主流 WBT 方法谱系对比：监督蒸馏 vs AMP 风格化 vs 扩散中间件 vs 物理可行性筛选，重点列出参考来源、训练目标、跨任务一般化、真机指标）。
    - [ ] `wiki/queries/cross-embodiment-transfer-strategy.md`（跨具身策略迁移 Query：单具身训练 + 重定向迁移 vs Any2Any 高效迁移 vs 多具身联合训练，给出选型决策树与典型故障模式）。
- [ ] **跨具身专题交叉补强**：
    - [ ] 在 `wiki/concepts/motion-retargeting.md`（V22 P1 已存在）与 `wiki/concepts/sim2real.md` 中明示「重定向产物 → WBT 训练数据 → 跨具身策略蒸馏」的三段流水线衔接；引用 P1 新页与 V22 motion-retargeting-pipeline / objective 形成「映射 → 训练 → 迁移」三视角闭环。

## P2: 真机安全微调与 Sim2Real 深化 (Quantity)

- [ ] **安全微调知识链 (+3)**：
    - [ ] `wiki/concepts/safe-real-world-rl-fine-tuning.md`（真机安全 RL 微调：从 Sim2Real 残差到真机在线适配的边界与安全约束；覆盖 SLowRL 安全 LoRA、Heracles 扩散兜底、CBF/CLF 安全壳三条主流路径）。
    - [ ] `wiki/formalizations/safe-lora-update-projection.md`（安全 LoRA 投影更新形式化：低秩参数化 + 安全约束投影 $\Pi_{\mathcal{S}}$ 的目标函数；对照 SLowRL 实现）。
    - [ ] `wiki/comparisons/sim2real-vs-real2sim-fine-tuning.md`（Sim2Real 残差适配 vs Real2Sim 真机回放 vs 真机直接 RL 微调三类策略的成本/安全/数据效率三维对比）。
- [ ] **事实库扩展**：
    - [ ] `schema/canonical-facts.json` 由 156 → **≥ 170 条**：新增 WBT 跨具身（SONIC 监督蒸馏 / Any2Any many-to-many 迁移 / BFM 行为基础模型 / SD-AMP 统一走跑起身 / Heracles 扩散兜底中间件）与真机安全微调（SLowRL 安全 LoRA / 真机 RL 安全约束 / Sim2Real 残差 vs Real2Sim）专题的矛盾检测规则。

## P3: 交互层"时间维度"增强 (UX/UI)

- [ ] **详情页"最近相关 ingest"时间线**：
    - [ ] 在 `docs/detail.html` 的「关联页面」附近新增 `#detailRecentIngestTimeline` 容器：基于 `exports/link-graph.json` 的 `latest_wiki_nodes` 与当前节点 1-hop 邻居取交集，展示最近 30 天内入库的相关页面（最多 6 项，按 `recency` 倒序）；空态降级隐藏（不显示标题）。
- [ ] **图谱页"专题视图"扩充**：
    - [ ] `docs/graph.html` 的 `TOPIC_FILTERS` 在 V22 10 项基础上新增「全身运动跟踪（WBT）」「跨具身迁移」「真机安全微调」三个专题；命中规则复用 V22 community id + path 片段双路并集机制；新增视图均配 Puppeteer 截图归档到 `.cursor-artifacts/screenshots/`。

---

## 验收标准 (Definition of DoD)

- [ ] `make lint`: 0 errors（含新引入的 `entity_paper_metadata_check` INFO 级检查不阻塞 CI）。
- [ ] 知识图谱节点数 **≥ 445**，边数 **≥ 3320**（见 `exports/graph-stats.json`）。
- [ ] 事实库扩展至 **170 条** 以上（重点补 WBT 跨具身 / 真机安全微调 矛盾检测规则）。
- [ ] `community_quality_warning` 保持 `false` 且 `largest_community_ratio ≤ 0.25`。
- [ ] `log.md` 记录 V23 关键改动。

---

## 状态说明

- `[ ]` 待执行
- `[~]` 进行中
- `[x]` 已完成
- `[−]` 已评估，决定跳过（附理由）
