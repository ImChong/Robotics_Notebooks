# 技术栈项目执行清单 v29

最后更新：2026-07-13（v28 全数完成后新建：聚焦「具身大模型评测基准选型闭环」知识链——把近周密集 ingest 的一批评测基准资料，从分散的实体页沉淀为一条贯通的「具身大脑/MLLM 认知评测 → 世界模型预测保真度评测 → 策略任务成功率评测 → sim↔real 评测 gap 校准」选型链，补评测层间矛盾检测规则与专题视图）
项目仓库：<https://github.com/ImChong/Robotics_Notebooks>
上一版清单：[`tech-stack-next-phase-checklist-v28.md`](archive/tech-stack-next-phase-checklist-v28.md)
方法论参考：[Karpathy LLM Wiki](../../wiki/references/llm-wiki-karpathy.md)

---

## V28 交付基线 (V29 起点)

| 维度 | V28 状态 | V29 目标 |
|------|-----------|---------|
| 知识图谱节点 | 1597 | **≥ 1610** |
| 知识图谱边数 | 12168 | **≥ 12230** |
| 事实库 (CANONICAL_FACTS) | 230 条 | **≥ 240 条** |
| 社区结构 | 18 社区，最大社区占 19.9%（`community_quality_warning: false`） | **保持 ≤ 25%，新增专题不破坏均衡** |
| 技术专题 | 具身大模型分类学选型闭环链路（V28 交付） | **建立"具身大模型评测基准选型闭环"知识链** |
| 图谱专题视图 | V28 扩至 18 项（新增「具身大模型」） | **新增「具身评测基准」专题至 19 项** |

> 背景：V28 沉淀了「选哪一类具身大模型」（VLM/VLN/VLA/VLX/World-Model 五层选型链），紧接着的问题是**「怎么评测/证明它」**。近周密集 ingest 了一批**评测基准**资料——RoboBench（MLLM 具身大脑五维评测）、EWMBench（世界模型视频生成评测）、ESI-Bench、GigaWorld-1 policy evaluation、MimickingBench（人形模仿学习基准）、ManiSkill-HAB（低层操作基准）、Barkour（四足敏捷性基准）等。仓库里这些页各自独立（多为 `entities/` 实体页），但**缺一条贯通的评测选型视角**——从**具身大脑/MLLM 认知评测 → 世界模型预测保真度评测 → 策略任务成功率评测 → sim↔real 评测 gap 校准**逐层「测什么、用什么基准、指标的可复现性 vs 真实代表性如何取舍、过程指标 vs 结果指标何时用哪个」，尚未沉淀为独立 query / concept；事实库也缺「评测层选型矛盾」（仿真基准易复现 vs 真机代表性、任务成功率 vs 过程/中间指标、世界模型视频质量 ≠ 下游策略收益、MLLM 认知评分 ≠ 可执行动作能力、单任务过拟合 vs 跨任务泛化评测、离线回放评测 vs 在线闭环评测等）的矛盾检测规则。V29 优先补齐这条评测基准选型闭环知识链，并把分散的评测基准实体页交叉链路规范化。

---

## P0: 自动化与工具链深度强化 (Engineering)

- [x] **评测基准页交叉链路巡检 V1**：
    - [x] `scripts/lint_wiki.py` 新增 `_check_eval_benchmark_crosslink`：对 `tags` 含 `benchmark` / `evaluation`（子串匹配派生标签）的 `entities/` / `comparisons/` / `concepts/` 页，检查正文是否回链到「具身大模型评测基准选型闭环」专题枢纽（`embodied-eval-benchmark-selection-loop` / `topic-embodied-eval-benchmark`，缺失给 INFO 级 `eval_benchmark_crosslink` 提示，不阻塞 CI），枢纽页自身豁免；写入 lint 报告基线快照（`exports/lint-report.md`）；新增 `tests/test_lint_wiki_eval_benchmark_crosslink.py` 用例覆盖（列表式/内联式 tag、有/无回链、双枢纽、枢纽豁免、INFO 不计失败）。（已落地：新增 `eval_benchmark_crosslink` INFO 键并接入 `INFO_ONLY_KEYS`/`_empty_results`/runner/报告段；`lint_wiki.py --report` 0 errors、新段基线 20 页；9 条新用例 + 全量 82 条 lint_wiki 用例通过；ruff check/format 通过）

## P1: 具身大模型评测基准选型闭环知识链专题 (Quality)

- [x] **具身大模型评测基准选型闭环知识链 (+2)**：
    - [x] `wiki/queries/embodied-eval-benchmark-selection-loop.md`（端到端 Query：具身大脑/MLLM 认知评测 → 世界模型预测保真度评测 → 策略任务成功率评测 → sim↔real 评测 gap 校准 四层评测选型的取舍决策树，覆盖每层测什么、用什么代表性基准、指标的可复现性/真实代表性/过程 vs 结果/成本取舍与典型误判，配 Mermaid 决策流程图）。已建页并从 `simulation-evaluation-infrastructure` 概念页回链（消孤儿，`graph-stats.json` 0 orphans）。
    - [x] `wiki/concepts/sim-vs-real-eval-gap.md`（仿真评测可复现性 ↔ 真实世界代表性 取舍概念页：明示仿真基准在可复现性/吞吐/可控性上的优势为何以牺牲真实接触/感知噪声/长尾分布的代表性为代价，并把这条 gap 讲成「评测结论能否外推到真机」的物理根因；配可复现性 vs 代表性代价表、缩小评测 gap 的三条工程路线与常见误判速查）。已与 Query 页双向回链。

- [x] **评测基准家族层专题交叉补强**：
    - [x] 在 `wiki/entities/robo-bench.md`（MLLM 认知层）、`wiki/entities/ewmbench.md`（世界模型评测层）、`wiki/entities/paper-gigaworld-1-policy-evaluation.md`（策略评测层）、`wiki/concepts/simulation-evaluation-infrastructure.md`（评测基建）等页与 P1 新页（`queries/embodied-eval-benchmark-selection-loop.md`）形成双向回链：各页在 `related` 与「关联页面」补入评测选型闭环 Query 页并标注本页所在评测层；Query 页 `related` 含全部相关评测页，双向闭合，消除孤儿页。（robo-bench=①认知层、ewmbench=②预测保真度层、gigaworld-1=②策略评估器层，均双向回链；simulation-evaluation-infrastructure 已于 P1① 回链。`ci-preflight` 12/12、graph 边数 12775→12778、0 孤儿）

## P2: 事实库与矛盾检测扩展 (Quantity)

- [x] **事实库扩展**：
    - [x] `schema/canonical-facts.json` 由 230 → **240 条**：新增 10 条具身评测选型矛盾检测规则（仿真基准可复现 vs 真机代表性、任务成功率 vs 过程/中间指标、世界模型视频质量 ≠ 下游策略收益、MLLM 认知评分 ≠ 可执行动作能力、单任务过拟合 vs 跨任务泛化、离线回放评测 vs 在线闭环评测、成功率均值掩盖长尾失败模式、基准饱和 ≠ 真实场景就绪、评测集泄漏致虚高、静态基准不覆盖分布漂移）；逐条经脚本校验对现存 wiki 页有 pos 命中且 0 误报（`make lint` 潜在矛盾 0 个、0 errors）。（已落地：10 条规则的 `pos_claims` 均锚定 `queries/embodied-eval-benchmark-selection-loop.md` / `concepts/sim-vs-real-eval-gap.md` 现存正文，逐条 pos 命中 ≥1 页；`neg_claims` 断言相反错误说法，经全量 wiki 页复核 0 命中（0 误报）；`lint_wiki.py --report` 0 errors、潜在矛盾 0 个、信息型预警仍 22 条不新增；`ci-preflight` 12/12 通过）

## P3: 交互层"具身评测基准"增强 (UX/UI)

- [x] **图谱页"具身评测基准"专题视图**：
    - [x] `docs/topic-filters.js` 单一事实源新增「具身评测基准」专题（`embodied-eval-benchmark`，🧪 emoji），复用 path 片段并集机制（`bench` / `eval` 等干净片段，与既有专题保持最小重叠）并用 `ids` 显式纳入未被片段命中的评测页（`embodied-eval-benchmark-selection-loop` / `sim-vs-real-eval-gap` / `robo-bench` / `ewmbench` / `esi-bench` / `paper-gigaworld-1-policy-evaluation` / `simulation-evaluation-infrastructure` 等）；同步在 `docs/graph.html` `#filter-topic-chips` 增加对应 chip。专题汇总枢纽页 `wiki/overview/topic-embodied-eval-benchmark.md` 已建（从相关评测/query 页交叉回链），`graph-stats.json` 0 orphans。专题视图落稳后截图归档至 `.cursor-artifacts/screenshots/graph-topic-embodied-eval-benchmark.png`。（已落地：`topic-filters.js` 新增 `embodied-eval-benchmark` 三段落——`TOPIC_HUB_IDS` / `TOPIC_FILTERS`（segments=`bench`/`eval`/`benchmark` 干净片段 + 7 页 ids）/ `TOPIC_META`（🧪）；`graph.html` 第 19 个 chip 就位；补建枢纽页 `topic-embodied-eval-benchmark.md`（含英文缩写速查、四层选型表、关键取舍，从 query/concept 页双向回链）；node 端复核 7 个目标评测页 + 枢纽命中、`vla.md` 未命中；`export+graph` 重生 1682 节点/13396 边、**0 orphans**、`largest_community_ratio` 0.17 且 `community_quality_warning: false`；`lint_wiki` **0 errors 0 信息型预警**。截图需浏览器环境，本轮后台例行未附。）
- [ ] **详情页"同专题相关页"提示**：
    - [ ] 复用 `docs/topic-filters.js` 单一事实源（`renderMetaTopicBadges` → `topicsForNode` 已数据驱动），评测基准/新建页命中「具身评测基准」专题时自动渲染对应轻量徽标 + 跳转 `graph.html?topic=embodied-eval-benchmark`（空态降级隐藏）。P3① 把 `embodied-eval-benchmark` 写入单一事实源后，详情页「所属专题」徽标行即自动联动；选一页评测实体页端到端验证并归档截图至 `.cursor-artifacts/screenshots/detail-topic-embodied-eval-benchmark.png`。

---

## 验收标准 (Definition of DoD)

- [x] `make lint`: 0 errors（新引入的 `eval_benchmark_crosslink` 为 INFO 级，不阻塞 CI）。（`lint_wiki` 0 errors 0 信息型预警）
- [x] 知识图谱节点数 **≥ 1610**，边数 **≥ 12230**（见 `exports/graph-stats.json`）。（1682 节点 / 13396 边）
- [x] 事实库扩展至 **240 条**（补齐 仿真 vs 真机 / 成功率 vs 过程指标 / 世界模型质量 vs 策略收益 等 10 条具身评测选型矛盾检测规则）。
- [x] `community_quality_warning` 保持 `false` 且 `largest_community_ratio ≤ 0.25`。（0.17，warning=false）
- [x] `log.md` 记录 V29 关键改动。（P3① 专题视图记录已追加）

---

## 状态说明

- `[ ]` 待执行
- `[~]` 进行中
- `[x]` 已完成
- `[−]` 已评估，决定跳过（附理由）
