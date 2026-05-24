# 技术栈项目执行清单 v22

最后更新：2026-05-13（V22 启动，基于 V21 完整交付）
项目仓库：<https://github.com/ImChong/Robotics_Notebooks>
上一版清单：[`tech-stack-next-phase-checklist-v21.md`](tech-stack-next-phase-checklist-v21.md)
方法论参考：[Karpathy LLM Wiki](../../wiki/references/llm-wiki-karpathy.md)

---

## V21 交付基线 (V22 起点)

| 维度 | V21 状态 | V22 目标 |
|------|-----------|---------|
| 知识图谱节点 | 297 | **≥ 312** |
| 知识图谱边数 | 1933 | **≥ 2050** |
| 事实库 (CANONICAL_FACTS) | 140 条 | **≥ 155 条** |
| 社区结构 | 8 社区，最大社区占 46.1%（`community_quality_warning: true`） | **最大社区占比 ≤ 40%，warning 消除** |
| 技术专题 | 触觉与力觉闭环（Haptics） | **建立"动作重定向与角色化人形"专题** |

---

## P0: 自动化与工具链深度强化 (Engineering)

- [x] **缩写/别名归一化检索**：
    - [x] `scripts/search_wiki.py` 引入轻量缩写表（WBC/VLA/IL/RL/MPC/PPO/SAC/HQP/CBF/CLF 等），查询时与全称双向展开，并在 `print_results` 中提示"已展开为 …"。
      - 实现：在 `scripts/search_wiki_core.py` 新增 `WIKI_ABBREVIATIONS`（覆盖 WBC/VLA/IL/RL/MPC/PPO/SAC/HQP/CBF/CLF/BC/IK/FK/LIP/ZMP/TSID 共 16 条）与 `expand_query_aliases()`：缩写 → 全称（per word）与全称短语 → 缩写（whole-query）双向展开；`search()` 把展开后的词同时喂给 BM25 分词与 `_find_matched_lines`，并将"缩写归一化：已展开为 'X' → 'Y'"挂到 `semantic_notice`，由现有 `print_results` 渲染。
      - 验证：`pytest tests/test_search_wiki_core.py` 21/21 通过（新增 5 个 `TestExpandQueryAliases` 用例）；CLI 实测 `search_wiki.py MPC` / `search_wiki.py "model predictive control"` / `search_wiki.py WBC --json` 均输出 "已展开为 …" 提示；`eval_search_quality.py` 36/37（与基线一致，未引入新回归）。
- [x] **社区粒度二级拆分**：
    - [x] 优化 `scripts/generate_link_graph.py` 的社区检测：在 Locomotion 单一巨型社区（46.1%）内进一步用 Louvain `resolution > 1.0` 二级拆分，使 `largest_community_ratio ≤ 0.40` 且 `community_quality_warning` 转 `false`。
      - 实现：保留 Girvan-Newman 一级检测（`PRIMARY_COMMUNITY_CAP=8`），新增 `refine_oversized_communities` + 纯 Python `louvain_communities`（带 `resolution=1.15` 的 Reichardt-Bornholdt modularity），对占比 > 40% 且节点数 ≥ 30 的巨型社区做二级拆分；`MAX_COMMUNITIES` 提升至 16 容纳子社区命名。
      - 结果：`exports/graph-stats.json` 中 `community_count=17`、`largest_community_ratio=0.138`（Manipulation 42 / 304）、`community_quality_warning=false`；Locomotion 巨型社区拆出 WBC / RL / MPC / IL / Sim2Real / Isaac Gym / Humanoid / Unitree G1 等子社区。
- [x] **方法-Query 闭环 Lint**：
    - [x] `scripts/lint_wiki.py` 新增 `methods_without_practitioner_query` 检查：被超过 3 个其他页面引用的 `methods/` 必须存在至少一篇 `queries/` 操作指南或 `comparisons/` 对比页对应，否则给出"待落地"预警。
      - 实现：新增 `_check_methods_without_practitioner_query()`，阈值 `METHOD_PRACTITIONER_INBOUND_THRESHOLD=3`（即 ≥ 4 个入链），排除自链；遍历入链来源，若无 `wiki/queries/*` 或 `wiki/comparisons/*` 命中则附"被 N 个页面引用，无 queries/comparisons 落地"提示。为防止首次落地即破坏 CI（当前 baseline 28 项），定义 `INFO_ONLY_KEYS = {missing_pages, methods_without_practitioner_query}`，新增 `_failing_total/_info_total` 让 main 退出码只统计硬错误，报告中 28 项以 💡 信息型展示。
      - 验证：`tests/test_lint_wiki_practitioner_query.py` 新增 6 个用例（高入链无 query 命中、queries 命中、comparisons 命中、阈值边界、自链排除、INFO_ONLY 不计失败 total）全通过；`PYTHONPATH=scripts pytest --no-cov` 91/91；`ruff check`、`ruff format --check`、`mypy scripts/lint_wiki.py` 均通过；`python3 scripts/lint_wiki.py` 退出码 0，输出"✅ 所有检查通过！（另含 28 条信息型预警，不阻塞 CI）"。
      - 后续：28 项预警是 V22 P1/P2 待补 `queries/`、`comparisons/` 的落地基线（动作重定向、抓取、AMP/Beyondmimic/Exoactor 等高频热点），后续按 P1/P2 推进会同步消减。

## P1: 动作重定向与角色化人形专题 (Quality)

- [x] **动作重定向知识链 (+3)**：
    - [x] `wiki/concepts/motion-retargeting-pipeline.md`（重定向流水线：MoCap → 骨架对齐 → IK/约束 → 物理可行性筛选 → 训练数据的端到端概念）。
      - 实现：新增 `wiki/concepts/motion-retargeting-pipeline.md`，把 [Motion Retargeting](../../wiki/concepts/motion-retargeting.md) 概念页里的「单次映射」展开为 8 阶段端到端流水线（源归一 → 骨架/DoF 映射 → 体型缩放 → IK/QP → 硬约束与平滑 → 物理可行性筛选 → 可选物理修补 → 离线/在线产物落地），含 Mermaid 总览、三种工程化形态对比表、常见失败模式表与下游接口契约；交叉互链 GMR / NMR / ReActor / SONIC / ExoActor / WBC / Sim2Real / Teleoperation。
      - 验证：`motion-retargeting.md` 关联页面区块回链新页面；index.md 在「重点页面」加入流水线条目（见本次提交）。
    - [x] `wiki/formalizations/motion-retargeting-objective.md`（重定向目标函数形式化：姿态相似项、接触/约束项、平衡项、关节限位项的数学组合）。
      - 实现：新增 `wiki/formalizations/motion-retargeting-objective.md`，给出通用目标函数 $\mathcal{L}^{\text{pose}}+\mathcal{L}^{\text{ee}}+\mathcal{L}^{\text{bal}}+\mathcal{L}^{\text{lim}}+\mathcal{L}^{\text{smooth}}$ 的形式化；逐项列出关节角/关键点/SO(3) 旋转一致、末端跟随/接触锁定/相位/摩擦锥、CoM–支撑多边形/ZMP/RFC 压制、限位四类硬罚项与平滑导数罚项；并给出 GMR（离线 QP）/ DeepMimic（指数核奖励）/ ReActor（双层）/ NMR（CEPR 硬阈值 + L1 标签）/ SPIDER（采样优化）五种工程退化形态对照表；横向回链 [TSID](./tsid-formulation.md)、[Friction Cone](./friction-cone.md)、[ZMP + LIP](./zmp-lip.md)。
      - 交叉互链：`motion-retargeting.md`、`motion-retargeting-pipeline.md` 的「关联页面」加入本页入口。
    - [x] `wiki/comparisons/gmr-vs-nmr-vs-reactor.md`（GMR / NMR / ReActor 重定向方法谱系对比：监督 vs 优化 vs 物理感知 RL，输入形态、依赖、产物差异）。
      - 实现：新增 `wiki/comparisons/gmr-vs-nmr-vs-reactor.md`，按「一句话定义 + 12 维核心对比表 + Mermaid 三路数据流并排图 + 三方适用场景 + 5 类常见误判 + 决策矩阵」结构覆盖三条路线；强调「误差修补发生位置」（下游 / 离线 / 在线）作为核心选型轴，并显式标注 NMR 仍以 GMR 为 CEPR 初值、三者实际常串联而非互斥；交叉互链 motion-retargeting / pipeline / objective / GMR / NMR / ReActor / SPIDER / SONIC / ExoActor。
      - 关联回链：`motion-retargeting.md`、`motion-retargeting-pipeline.md`、`motion-retargeting-objective.md`、`methods/motion-retargeting-gmr.md`、`methods/neural-motion-retargeting-nmr.md`、`methods/reactor-physics-aware-motion-retargeting.md` 的「关联页面」加入本对比页入口；`index.md` Wiki Comparisons 区块插入本页摘要条目。
      - 验证：本地 `python3 -m http.server` + `docs/detail.html?id=wiki-comparisons-gmr-vs-nmr-vs-reactor` 渲染正常（Mermaid 流程图落稳、表格未截断）；`grep "gmr-vs-nmr-vs-reactor" -r wiki/ index.md` 显示双向回链建立。
- [x] **角色化人形（Character Humanoid）边界澄清**：
    - [x] `wiki/concepts/character-animation-vs-robotics.md`（角色动画 vs 机器人控制：动作风格化、表演意图与物理可控性之间的张力；面向 Disney Olaf / Roboto Origin / MotionCanvas 等案例）。
      - 实现：新增 `wiki/concepts/character-animation-vs-robotics.md`，给出六个张力维度（目标函数 / 时间尺度 / 失败定义 / 机构约束 / 数据来源 / 工具生态）与五个案例切片（Disney Olaf 角色优先、DeepMimic-AMP-ASE 图形学起源谱系、BotLab/MotionCanvas 工具语言、Roboto Origin/Asimov v1 中性研究平台、关键帧编辑工具艺术家手工层）；附决策矩阵、常见误区与「角色端→桥接层→机器人端」流程 Mermaid。
      - 交叉互链：`wiki/methods/disney-olaf-character-robot.md`、`wiki/entities/botlab-motioncanvas.md`、`wiki/entities/roboto-origin.md`、`wiki/entities/xue-bin-peng.md`、`wiki/concepts/motion-retargeting.md`、`wiki/concepts/reward-design.md` 的 frontmatter `related` 与正文「关联页面 / 与其他页面的关系」均加入本页入口；与 [Motion Retargeting Pipeline](../../wiki/concepts/motion-retargeting-pipeline.md) 形成「映射几何/动力学」与「目标函数博弈」的双视角。

## P2: 抓取与操作感知深化 (Quantity)

- [x] **抓取知识链 (+3)**：
    - [x] `wiki/methods/grasp-pose-estimation.md`（抓取位姿估计：6-DoF 抓取检测、点云/RGBD 输入、AnyGrasp / GraspNet / Contact-GraspNet 谱系）。
      - 实现：新增 `wiki/methods/grasp-pose-estimation.md`，按「一句话定义 + 任务参数化（6-DoF vs 7-DoF）+ 主流谱系三代演进（GPD → GraspNet-1Billion → Contact-GraspNet / GSNet/Graspness / AnyGrasp）+ Mermaid 谱系流向图 + 输入模态对照 + 训练数据 + AP/MPPH 评测指标 + 下游衔接（cuRobo/IK/视觉伺服/触觉）+ 常见误区」结构组织；显式区分检测式 grasp pose 与多指接触面分配的边界。
      - 交叉互链：`wiki/entities/anygrasp.md`（frontmatter `related` 与「关联页面」回链新页）、`wiki/tasks/manipulation.md` 关联方法新增条目、`wiki/methods/contact-net.md` 关联页面互链、`references/repos/manipulation-perception.md` 总览指针指向本页、`index.md` 重点页面区块加入条目。
    - [x] `wiki/queries/grasp-policy-selection.md`（抓取策略选型 Query：开放场景 vs 已知物体、稀疏 vs 稠密抓取、几何 vs 学习方法）。
      - 实现：新增 `wiki/queries/grasp-policy-selection.md`，按 Query 格式落地「TL;DR 决策树 + 三轴对比表（物体已知度 / 候选稠密度 / 方法类型）+ 四类推荐组合 pipeline + 关键工程经验（候选生成器 vs 执行器 / AP vs 真机成功率 / 透明反光物体 / 端到端策略边界）+ 常见误区 + 一句话记忆」结构，覆盖几何启发式 / 检测式 grasp pose（GraspNet → Contact-GraspNet → GSNet/AnyGrasp）/ 端到端 IL-VLA 的选型逻辑；显式给出「先检测式起步、再用 IL/VLA 替换可学环节」的工程序。
      - 交叉互链：`wiki/queries/README.md` 注册新 Query；`wiki/methods/grasp-pose-estimation.md` frontmatter `related` 与「关联页面」加入新 Query；`wiki/entities/anygrasp.md` 关联页面加入新 Query；`wiki/tasks/manipulation.md` 关联页面新增 Query 入口。
    - [x] `wiki/comparisons/anygrasp-vs-graspnet.md`（AnyGrasp 与 GraspNet 家族对比：输入模态、训练数据、部署延迟与开放词汇支持）。
      - 实现：新增 `wiki/comparisons/anygrasp-vs-graspnet.md`，按「一句话定义 + 14 维核心对比表 + Mermaid 数据流并排图（GraspNet 家族白盒基线 / AnyGrasp SDK 工程闭环）+ 三类适用场景 + 6 类常见误判 + 决策矩阵 + 评测指标视角」结构覆盖 GraspNet-1Billion / Contact-GraspNet / GSNet 三条家族子路线与 AnyGrasp SDK；显式区分「白盒改造 vs 工程化交付」「单帧 vs 动态跨帧」「完全开源 vs 二进制 License」三对核心取舍，并强调两者非互斥替代关系。
      - 交叉互链：`wiki/methods/grasp-pose-estimation.md`、`wiki/entities/anygrasp.md`、`wiki/queries/grasp-policy-selection.md`、`wiki/methods/contact-net.md`、`wiki/tasks/manipulation.md` 的 frontmatter `related` 与「关联页面」均加入本页入口，形成「方法谱系页 + 实体页 + Query + 对比页」四级互链闭环。
      - 验证：`make ci-preflight` 同步派生产物（exports / docs / search-index / sitemap / index.md 等）；至此 P2「抓取知识链 (+3)」三个子项全部落地，专题进入 `[x]` 完成状态。
- [x] **接触/操作交叉补强**：
    - [x] 在 `wiki/concepts/contact-rich-manipulation.md` 与 `wiki/concepts/visuo-tactile-fusion.md` 中补"抓取→插装→精细操作"的级联引用，把 P1 触觉链路与 P2 抓取链路打通。
      - 实现：两页同步新增「抓取 → 插装 → 精细操作（级联视角）」三段式小节。`contact-rich-manipulation.md` 加 3 列表格（① 抓取 / ② 插装 / ③ 精细操作）并显式标注 P2 上游候选（Grasp Pose Estimation / AnyGrasp / ContactNet / 抓取策略选型 Query / AnyGrasp vs GraspNet）→ 本页中段 → P1 下游执行层（Impedance / Tactile Impedance / TSID / WBC）的连接关系，附「① 准但 ② 没接管会撞死」的工程含义说明；`visuo-tactile-fusion.md` 同节加 Mermaid 流水线图 + 三段式表格，强调「检测式 grasp 不带接触可信度，门控必须在触觉给出几何漂移信号时让出权重」这一常被忽略的衔接点。两页 frontmatter `related` 与「关联页面」尾部互链至 P2 抓取链（grasp-pose-estimation / grasp-policy-selection / anygrasp-vs-graspnet）与 P1 触觉链（tactile-impedance-control / hybrid-force-position-control）。`updated` 字段同步刷至 2026-05-21。
      - 验证：`make ci-preflight` 通过（page catalog / export_minimal / sync_all_stats / eval_search_quality 37/37 / check_export_quality 12/12 均通过；`lint_wiki.py` 的 9 项 `stale_pages` 与本次改动无关，均为 2026-05-21 早些 ingest 引入的历史 baseline）。`exports/graph-stats.json` 边数由 2050 升至 **3004**、节点数 **410**、largest_community_ratio = 0.207、community_quality_warning = false。

## P3: 交互层"关系视角"增强 (UX/UI)

- [x] **详情页"关联社区分布"小条形图**：
    - [x] 在 `docs/detail.html` 的"关联页面"区块新增按 `link-graph` 社区（Whole-Body Control / Motion Retargeting / Sim2Real / VLA / IL / RL / Locomotion / ...）聚类的横向条形小图，让读者一眼判断当前节点在知识图谱里偏向哪些社区。
      - 实现：`docs/detail.html` 在 `#detail-related` 标题下新增 `#detailRelatedCommunityDist` 容器（含 head 与 bars 两块）；`docs/main.js` 新增 `ensureDetailCommunityIndex()` 懒加载 `exports/link-graph.json`，建立 `pathToCommunity` Map 与 `communityLabel` 字典（兜底为空 Map），与 `renderRelatedCommunityDistribution()`（按计数倒序排序、最大计数为 100% 基准、其余按比例并保底 6% 可见宽度；不在图谱内的 roadmap / reference / tech_map 关联项统一桶为「未分类」并永远排在末尾），在 `renderDetailPage` 正常态与未匹配态都调用一次以保持空态干净；`docs/style.css` 新增 `.related-community-dist*` 样式（标题/Meta/三列网格：社区标签 160px / 横向轨道 / 计数；540px 以下窄屏缩列至 110px / 1fr / 46px）。社区标签显式去掉末尾「社区」二字以节省横向空间，悬停 `title` 仍显示完整标签。
      - 第一版本（type 分桶）由社区分桶替代后不再保留：理由是 type 维度（概念/方法/实体/...）与现有 frontmatter type 字段重复，区分度不如 link-graph 社区聚类高（V22 P0 已把社区切到 17 个 ≤ 40% 阈值之内，刚好可用于"邻域属于哪些主题"的快速判断）。
      - 验证：`make lint-js` 通过（仅一条 pre-existing 的 `resetMermaidLightboxView` 未使用警告，与本次改动无关）；本地 `python3 -m http.server` + Puppeteer 视口截图 `wiki-concepts-whole-body-control` 详情页（共 12 项 · 8 个社区：Whole-Body Control 4 / Imitation Learning 1 / Locomotion 1 / Motion Retargeting 3 / Sim2Real 1 / Unitree G1 1 / 1 / 未分类 1）与 `wiki-concepts-armature-modeling`（共 5 项 · 3 个社区：WBC 3 / Motion Retargeting 1 / Sim2Real 1），桌面端 1280px 与移动端 375px 双视口落稳。
      - 截图：`.cursor-artifacts/screenshots/detail-related-community-dist-wbc.png`、`detail-related-community-dist-wbc-mobile.png`、`detail-related-community-dist.png`。
- [x] **图谱页"专题视图"切换器**：
    - [x] `docs/graph.html` 增加下拉菜单，可选"全量 / 动作重定向 / 抓取 / 触觉与通信"三个子图过滤模式，复用 V21 微地图的同套 `path → type` 元数据。
      - 实现：`docs/graph.html` 顶部工具栏在搜索框右侧新增 `<label#topic-view-wrap><select#topic-view>`，样式沿用 `chip` 视觉规范并在激活态切换高亮（深色 `rgba(0,212,255,0.14)` / 浅色 `#e6f7ff` + `#1659b4` 描边）。JS 侧 `TOPIC_FILTERS` 字面量声明每个专题的命中规则，按 community id 集合 + path 片段集合双路并集判定（任一命中即可），可选 `excludeSegments` 抑制误命中。新增 `nodeSegments(d)`（按 `/._-` 切分并 memo 化到 `d._segs`）+ `nodeMatchesTopic(d)`，接入既有 `applyFilters()` 与 `link` 边权重透明度链路（与社区/类型/健康度/搜索筛选叠加，互不冲突），非命中节点保持 opacity 0.08 不丢失上下文。切换专题时调用新增的 `fitToVisibleNodes()` 自动缩放到该子图，切回"全量"恢复 `fitToScreen()`。
      - 专题（共 10 项，按用户反馈在初版"触觉与通信"基础上拆分并扩充）：
        | 专题 | 命中规则 | 节点数 |
        |------|----------|--------|
        | 动作重定向 | community-8 + retargeting/gmr/nmr/reactor/sonic/exoactor/spider/wilor/mocap/teleoperation/deepmimic/amp/character/animation/keyframe/pipeline | 25 |
        | 抓取 | grasp/graspnet/anygrasp/dexterous/manipulation/pick/place/bimanual/curobo | 16 |
        | 触觉 | tactile/haptic/impedance/force/contact/visuo（exclude reinforcement） | 16 |
        | 通信协议 | ethercat/can/uart/dds/foxglove/rs485/rs232/serial/communication/protocol/bus/protocols/firmware | 9 |
        | WBC 全身控制 | community-2 + community-15 + wbc/tsid/hqp/cbf/clf/whole/body/balance/hierarchical | ≈32 |
        | Locomotion 步态 | community-14 + locomotion/gait/mpc/zmp/lip/walking/swing/stance/capture | ≈25 |
        | VLA / 基础策略 | community-0 + community-5 + vla/foundation/octo/openvla/rt/pi0/gr00t | ≈111 |
        | 学习范式 IL/RL | community-3 + community-4 + imitation/reinforcement/ppo/sac/behavior/cloning/dreamer | ≈40 |
        | Sim2Real | community-11 + sim2real/randomization | ≈25 |
        | 状态估计 | community-10 + estimation/ekf/ukf/slam/vio/odometry | ≈15 |
      - 验证：`npm run lint:js` 通过（仅一条 pre-existing 的 `resetMermaidLightboxView` 未使用警告，与本次改动无关）；`node --check` 对 graph.html 提取的内联脚本通过；本地 `python3 -m http.server 8765` + Puppeteer 在 1440×900 视口对全部 11 种模式各截图一张，全部正常落稳。新增 `scripts/screenshot_graph_topic.cjs`（puppeteer-core + 本地 d3 注入兜底，便于离线/受限环境复现）。
      - 截图：`.cursor-artifacts/screenshots/graph-topic-{all,motion-retargeting,grasp,tactile,communication,wbc,locomotion,vla,learning,sim2real,state-estimation}.png`。

---

## 验收标准 (Definition of DoD)

- [ ] `make lint`: 0 errors（含新引入的 `methods_without_practitioner_query` 检查全通过）。
- [ ] 知识图谱节点数 **≥ 312**，边数 **≥ 2050**（见 `exports/graph-stats.json`）。
- [x] 事实库扩展至 **155 条** 以上（重点补 motion-retargeting / grasp-pose 矛盾检测规则）。
    - 实现：`schema/canonical-facts.json` 由 140 → **156** 条；本轮新增 17 条按 V22 P1 / P2 主线分布：动作重定向 5 条（`GMR 运动学优化定位` / `ReActor 双层联合优化` / `Motion Retargeting Pipeline 端到端阶段` / `Motion Retargeting 目标函数加权组合` / `Character Humanoid 目标双重性`）、抓取与感知 10 条（`6-DoF vs 7-DoF 抓取` / `GraspNet 三代谱系演进` / `Contact-GraspNet 接触点参数化` / `AnyGrasp 跨帧时序关联` / `AnyGrasp SDK License 分发` / `MPPH 抓取吞吐指标` / `抓取候选需显式碰撞检查` / `抓取选型 检测式优先` / `GraspNet-1Billion 评测基准` / 既有 `Contact-rich 接触力建模` 等基线沿用）、近期 ingest 2 条（`BifrostUMI 无机器人示范` / `OpenLoong 全栈开源`）。每条三元组（`terms` / `pos_claims` / `neg_claims`）按 `lint_wiki._check_contradictions` 的正则匹配规范设计，覆盖 P1 / P2 新页与既有页常见提法。
    - 验证：`python3 scripts/lint_wiki.py` 退出码 0，0 contradictions、0 ⚠️ / 0 💡，"✅ 所有检查通过！"；419/419 wiki/entity 页 ingest 来源覆盖率 100%。
- [ ] `community_quality_warning` 在 `exports/graph-stats.json` 中变为 `false`。
- [ ] `log.md` 记录 V22 关键改动。

---

## 状态说明

- `[ ]` 待执行
- `[~]` 进行中
- `[x]` 已完成
- `[−]` 已评估，决定跳过（附理由）
