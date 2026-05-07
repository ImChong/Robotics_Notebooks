> 核心规范：所有日常动作（ingest / query / lint / structural）必须追加记录到此文件。

> V21 里程碑追踪（Current）：
> - P0：基础架构防腐（拼写纠错、公式规范检查、图谱数据精简） ✅ 完成
> - P1：触觉与力觉闭环（Contact Wrench Cone 页面、触觉基础页面、融合范式对比）
> - P2：通信与延迟规范（硬件通信链路设计、延迟来源分析）
> - P3：V21 质量回归验证

> V20 里程碑追踪：
> - P0：检索系统增强（多词组合查询、Tag 提权、摘要匹配优化） ✅ 完成
> - P1：依赖可视化（Graph 页面渲染社区划分、重力物理引擎集成） ✅ 完成
> - P2：基础页补全（新增 14 个基础概念/算法/任务页面并保证链接闭环） ✅ 完成
> - P3：结构瘦身（合并零散页、清理无效引用） ✅ 完成

> V14 里程碑追踪：
> - P1：扩展 `CANONICAL_FACTS` 到 50 条，覆盖 PPO on-policy、CLF/CBF、VLA、contact-rich manipulation、Isaac Lab 并行训练等事实
> - P2：加深 manipulation / contact-rich / terrain / bimanual / sensor-fusion / behavior-cloning / VLA 等薄弱页面
> - P3：规划新增 4 个高价值 Query 页（domain-randomization / clf-cbf / vla-low-level / contact-rich manipulation）
> - P4：lint 增加孤儿节点计数检测，搜索回归扩展到 26 条
> - P5：补齐安全控制与接触操作学习路径，扩展 overview / index / README 入口，并保持日志 append-only 更新

## [2026-04-20] fix | v14-execution | P0 搜索回归修复：numpy 延迟导入

- V14 P0 完成：`scripts/search_wiki.py` 移除 module 顶部 `import numpy as np`，改为在 `load_vector_resources` / `encode_query_vector` / `search` 内部使用 numpy 的分支做延迟导入
- 修复前：`python3 scripts/search_wiki.py PPO` 直接 `ModuleNotFoundError: No module named 'numpy'`，回归测试 0/26
- 修复后：`python3 scripts/eval_search_quality.py` 通过率 **26/26 (100%)**，BM25 路径不再依赖 numpy
- 无新增依赖、无行为变更；向量搜索分支在 numpy 可用时保持原逻辑

## [2026-04-21] structural | V14 执行清单完整交付 (新增 11 页 + 深度扩充 3 页 + 脚本优化)

## [2026-04-21] structural | V15 执行清单初始化

## [2026-04-21] structural | V15 执行清单完整交付 (聚焦操作社区与软件栈实体)

## [2026-04-21] structural | V16 执行清单初始化 (具身大模型深度化 + 灵巧操作补完)

## [2026-04-21] structural | V16 执行清单完整交付 (具身大模型深度化 + 灵巧操作补完)

## [2026-04-21] structural | V17 执行清单完整交付 & 初始化 V18 (具身数据流与交互闭环)

## [2026-04-21] structural | V20 执行清单完整交付 & 初始化 V21 (具身触觉专题 + 硬件通信链路形式化)

## [2026-04-22] ingest | sources/repos/fusion2urdf.md, sources/repos/marathongo.md — 接入新仓库资料并更新全站索引

## [2026-04-23] ingest | robot_lab (Repo) & CLAW (Blog) — 接入 IsaacLab 扩展框架与 G1 合成数据管线并更新全站索引

## [2026-04-24] ingest | sources/papers/policy_optimization.md — 补充 BRRL/BPO（Bounded Ratio RL）论文、项目页与代码仓库，并更新 Policy Optimization 方法页参考来源

- 新增来源条目：`Bounded Ratio Reinforcement Learning (Ao et al., 2026)`（arXiv / project page / GitHub）
- 更新 `wiki/methods/policy-optimization.md`：新增“BRRL / BPO（2026）”进展说明与参考来源
- 关联强化：将 BRRL 映射到 policy-optimization / reinforcement-learning / ppo-vs-sac / locomotion

## [2026-04-24] structural | V21 P1 推进 | wiki/formalizations/contact-wrench-cone.md

- V21 P1「触觉与力觉闭环」首项：新建 `wiki/formalizations/contact-wrench-cone.md`，把 Friction Cone 从 3D 点接触力推广到 6D 面接触力旋量（CWC / CWS / GWS）
- 涵盖 V-/H- 表示、ZMP/CoP 几何解读、多接触 Minkowski 和 与 Grasp Wrench Space 的统一视角，附最小 Python 骨架与方法局限性
- 回链：`wiki/formalizations/friction-cone.md`、`wiki/concepts/tactile-sensing.md`、`wiki/concepts/contact-dynamics.md` 的关联页面区块新增 CWC 入链
- V21 checklist 对应条目已勾选；follow-up：原同日 log 条目曾在合并 main 时按“冲突以 main 为准”规则被覆盖，此处以 append-only 方式补回

## [2026-04-24] feat | v21-execution | P0 智能拼写纠错（编辑距离）

- V21 P0 第二项推进：`scripts/search_wiki.py` 集成 Levenshtein 编辑距离算法，当查询无结果时自动推荐最接近的 Tag 或 标题
- 新增 `levenshtein_distance` / `collect_known_terms` / `suggest_terms` 三个辅助函数；阈值取 `max(2, ceil(len(query)/2))`，按距离升序返回 Top-5
- `print_results` 增加“您是否想搜索：”分区；`--json` 输出在有 notice 或 suggestions 时切换为 `{notice, suggestions, results}` 字典形式
- `search()` 返回签名保持 `(results, notice)` 不变，`scripts/eval_search_quality.py` 与 `scripts/debug_search.py` 调用方零侵入
- 验证：`python3 scripts/eval_search_quality.py` 通过率 **37/37 (100%)**；`python3 scripts/search_wiki.py "lokomotion" --json` 正确给出 `locomotion`（距离 1）建议
- V21 checklist 对应条目已勾选

## [2026-04-25] lint | 同步 sources/papers/policy_optimization.md 的最新进展（BRRL/BPO 2026）至全站 8 个相关 wiki 页面，消除陈旧预警并更新索引

## [2026-04-25] ingest | sources/repos/roboto_origin.md — 新增 Roboto Origin 及其五个官方模块仓库与文档资料归档，并沉淀 wiki/entities/roboto-origin.md

## [2026-04-25] ingest | sources/repos/atom01_hardware.md, sources/repos/atom01_deploy.md, sources/repos/atom01_train.md, sources/repos/atom01_description.md, sources/repos/atom01_firmware.md — 接入 Roboparty Atom01 五个官方模块仓库原始资料

## [2026-04-25] ingest | wiki/entities/roboto-origin.md — 回链 Atom01 五个 sources 条目并重新导出图谱索引，确保 sources 节点可见

## [2026-04-25] ingest | wiki/entities/atom01-*.md — 新增 Atom01 五个实体页并同步导出图谱（181 nodes / 1012 edges）

## [2026-04-25] feat | v21-execution | P0 自动化背链一致性 Lint（公式变量物理含义检测）

- V21 P0 第三项推进：`scripts/lint_wiki.py` 新增 `formalization_unexplained_vars` 检查；从 `wiki/formalizations/*.md` 的 `$$...$$` 显示公式中抽取单字母拉丁大写变量，逐一验证正文是否给出物理含义解释
- 启发式定义匹配：列表条目冒号、表格行、动词解释（是/为/表示/代表/denote）、其中/where 子句、等式或集合定义（=、\in、\succeq、\succ、\equiv、\triangleq）；并用 `(?![A-Za-z_(])` 排除函数调用形式（如 `R(s,a,s')`）以避免误报
- 修复 6 个被新规则命中的页面，补齐变量物理含义说明：
  - `wiki/formalizations/bellman-equation.md`：Q-learning 更新中的 $R$ 标量
  - `wiki/formalizations/control-lyapunov-function.md`：LQR-CLF 关系中的 $A, B$
  - `wiki/formalizations/ekf.md`：补全 $A, B, C, Q, R, P, K, I$ 整套矩阵物理含义
  - `wiki/formalizations/hjb.md`：LQR 特例段补 $A, B, Q, R, T$
  - `wiki/formalizations/lqr.md`：线性系统模型段补 $A, B, x, u$
  - `wiki/formalizations/tsid-formulation.md`：浮动基座动力学段补执行选择矩阵 $S$
- 验证：`make lint` 0 errors（所有检查通过）；`python3 scripts/eval_search_quality.py` 通过率 **37/37 (100%)**
- V21 checklist 对应条目已勾选

## [2026-04-26] structural | roadmap | 收敛为单一主路线并补齐阶段链接

- 将 `roadmap/route-a-motion-control.md` 从“路线A”调整为唯一主路线，明确 RL/IL/WBC/安全控制/接触操作是目标分支而非并列主路线。
- 删除 `roadmap/route-b-fullstack.md`，避免 roadmap 页面出现 A/B 两套主线。
- 为 L0-L6 阶段补充仓库内概念、方法、形式化、实体、query 页面链接，并同步更新索引与导出。

## [2026-04-26] structural | roadmap | 主路线 ID 去除 A/B 命名

- 将主路线文件从 `roadmap/route-a-motion-control.md` 重命名为 `roadmap/motion-control.md`。
- 将导出 ID 从 `roadmap-route-a-motion-control` 更新为 `roadmap-motion-control`，并同步 README、index、learning paths、wiki 回链与前端入口。
- 在 `docs/main.js` 保留旧 ID 到新 ID 的兼容映射，避免旧 URL 立即失效。

## [2026-04-26] feat | v21-execution | P0 图谱导出数据精简

- V21 P0 第三项收尾：精简 `exports/link-graph.json` 节点字段，去除每个节点上重复的 `community_label`（与 `communities` 数组中的 `label` 完全冗余）。
- `scripts/generate_link_graph.py` 仅在 `communities` 数组保留 `label`，节点端只输出 `community` id；`docs/graph.html` 的 tooltip 与侧栏改为通过 `communityLabelMap.get(d.community)` 查表。
- 体积变化：`exports/link-graph.json` 168 KB → 159 KB（5.7% 缩减），行数 5551 → 5370；`make lint` 与 `check_export_quality.py` 全通过。
- V21 checklist P0 全部完成，进入 P1 触觉与力觉闭环专题阶段。

## [2026-04-27] ingest | sources/repos/lingbot-map.md — 接入 LingBot-Map 流式 3D 重建基础模型方法页

## [2026-04-27] ingest | sources/repos/booster-robocup-demo.md — 接入 Booster Robotics RoboCup 演示框架

## [2026-04-27] ingest | 接入 htwk-gym 与 HumanoidSoccer (PAiD) 足球技能学习框架

## [2026-04-27] ingest | 补全飞书 Wiki 相关的运动控制进阶主题 (BeyondMimic, Any2Track, HAIC, AMP 等)

## [2026-04-27] feat | v21-execution | P1 触觉专题：视触觉融合 (Visuo-Tactile Fusion) 概念页

- V21 P1 触觉学习知识链推进第一项：新增 `wiki/concepts/visuo-tactile-fusion.md`，聚焦“接触瞬间如何在视觉全局与触觉局部之间动态切换权重”。
- 内容覆盖：视觉/触觉互补维度、阶段切换/软门控/注意力级三种融合范式、接触瞬间难点、训练数据采集要点、与现有相关页面的边界澄清（区分 `sensor-fusion`、`multimodal-fusion-tricks`）。
- 同步在 `tactile-sensing`、`contact-rich-manipulation`、`contact-wrench-cone`、`multimodal-fusion-tricks`、`tactile-feedback-in-rl` 中加入回链，避免新页成为孤儿；并刷新 `index.md` Page Catalog。
- 在 `docs/checklists/tech-stack-next-phase-checklist-v21.md` 中将该项打勾。

## [2026-04-28] ingest | 消化 ETH Zurich 关于扩散模型运动生成的论文 (Unitree G1 实时部署)

## [2026-04-28] ingest | 消化 awesome-humanoid-robot-learning 仓库，更新 Loco-Manipulation Wiki 页

## [2026-04-28] ingest | 接入 MimicKit 仓库及其核心算法系列 (DeepMimic, AMP, AWR, ASE, LCP, ADD, SMP)

## [2026-04-28] ingest | 深度更新 SMP (arXiv:2512.03028v3) 技术细节，补充 SDS/ESM/GSI 架构及 Unitree G1 真机验证

## [2026-04-29] ingest | sources/papers/humanoid_touch_dream.md — 消化 HTD / Touch Dreaming 论文并更新触觉增强人形移动操作知识节点

## [2026-04-29] structural | 更新待办列表与主页统计数据 (205 nodes, 1195 edges)

## [2026-04-29] ingest | sources/repos/embodied-ai-guide.md — 接入具身智能全栈百科并补全 RoboTwin/SAPIEN/ALOHA 实体

## [2026-04-29] ingest | sources/repos/xbotics-embodied-guide.md — 接入 Xbotics 工程指南，补全 LeRobot/Genesis 实体与数据飞轮概念

## [2026-04-29] ingest | sources/papers/zest.md — 接入 Boston Dynamics 跨形态技能迁移框架 ZEST

## [2026-04-29] ingest | sources/papers/sumo.md — 接入 RAI Institute 全身移动操作框架 Sumo (MPC-over-RL)

## [2026-04-29] ingest | sources/repos/{robot-explorer,robot-viewer,urdf-studio}.md — 接入 Web 端机器人可视化与设计工具链

## [2026-04-29] ingest | sources/repos/motphys-motrix.md — 接入 Motphys 高性能仿真平台 Motrix (Rust 后端, MJCF 兼容)

## [2026-04-30] ingest | 接入 NVIDIA MotionBricks (GR00T WBC 核心生成式框架)

## [2026-04-30] ingest | 接入 HKUST Switch 框架 (敏捷技能切换与图搜索)

## [2026-05-01] ingest | sources/papers/universal_skeleton.md — Universal Skeleton HOVL 异构骨架开放词汇动作识别，沉淀到 wiki/methods/skeleton-action-recognition.md

## [2026-05-01] ingest | 修复 10 个低健康度 Wiki 节点，并优化 Imitation Learning 与 Contact Dynamics 的搜索关键词以通过质量回归测试。

## [2026-05-01] ingest | sources/books/udl_book.md — 接入 Understanding Deep Learning 教材基础理论，新增 Deep Learning Foundations 和 Generative Foundations wiki 页面

## [2026-05-01] ingest | 扩充 AMP_mjlab 训练命令、指标分析及部署指南

## [2026-05-01] ingest | sources/repos/amp_mjlab.md — 扩充 AMP_mjlab 训练曲线、ONNX 导出与部署检查要点

## [2026-05-01] structural | roadmap/motion-control.md — 在主路线补入 Modern Robotics 章节映射与练习

## [2026-05-01] ingest | sources/repos/wbc_fsm.md — ccrpRepo wbc_fsm：Unitree G1 C++ WBC+FSM 部署框架，新增 wiki/entities/wbc-fsm.md

## [2026-05-01] ingest | sources/repos/gr00t_visual_sim2real.md — NVIDIA GR00T-VisualSim2Real：VIRAL（arXiv:2511.15200）+ DoorMan（arXiv:2512.01061）双 CVPR 2026，PPO Teacher + DAgger Student RGB 蒸馏 Sim2Real，Unitree G1 零样本迁移，新增 wiki/entities/gr00t-visual-sim2real.md，图谱 232 节点

## [2026-05-02] structural | wiki: 把 29 个页面里的 102 处 [[wikilink]] 全部迁移为标准 Markdown [text](path)，含 10 处隐蔽断链（指向 sources/references/roadmap 的非 wiki 路径），脚本 scripts/migrate_wikilinks.py，图谱 233 节点

## [2026-05-03] ingest | sources/papers/multi-gait-learning.md — Multi-Gait Learning for Humanoid Robots Using Reinforcement Learning with Selective Adversarial Motion Priority

## [2026-05-03] ingest | sources/papers/x2n_transformable.md — 新增 X2-N 论文，补充 hybrid-locomotion 和 loco-manipulation

## [2026-05-03] ingest | sources/papers/chasing_autonomy.md — A pipeline to dynamically retarget human motions and use control-guided RL for performant humanoid running. Mapped to motion-retargeting and humanoid-locomotion wiki pages.

## [2026-05-04] ingest | sources/papers/learn_weightlessness.md — 学习失重：人形机器人在非自稳定运动中的模仿机制

## [2026-05-04] ingest | sources/papers/kung_fu_athlete_bot.md — A Kung Fu Athlete Bot That Can Do It All Day: Highly Dynamic, Balance-Challenging Motion Dataset and Autonomous Fall-Resilient Tracking

## [2026-05-06] ingest | sources/papers/lwd.md — AGIBOT《Learning while Deploying: Fleet-Scale Reinforcement Learning for Generalist Robot Policies》, 新增 wiki/methods/lwd.md, 同步刷新 data-flywheel / online-vs-offline-rl / vla / π0-policy

## [2026-05-06] feat | v21-execution | P3 搜索结果按“置信度”分级（精确匹配 vs 潜在关联）

- V21 P3 第二项推进：`docs/main.js` 在搜索结果渲染前对每条命中按 `classifyTier(item, queryTokens)` 分级——命中标签 / 标题 / 路径归为「精确匹配」，仅命中摘要或正文 token 归为「潜在关联」
- 把原 `renderCards` 的卡片 HTML 拼接抽成 `buildResultCardHtml`，避免两个区块各写一份；仅当存在查询 token 时分组，空查询 + 类型筛选场景保持单区块原行为
- `docs/style.css` 新增 `.search-tier-heading` / `.search-tier-exact` / `.search-tier-potential` 样式：跨整行 grid，分隔线 + 小字号大写标签，强调「精确匹配」用 accent 色
- 键盘导航沿用 `getResultCards()`（仅查 `article.card[data-result-url]`），新增的 `<h4>` 区块标题不会进入 ↑↓/Enter 选区
- 验证：`python3 scripts/eval_search_quality.py` 通过率 **37/37 (100%)**；`node --check docs/main.js` 通过；`make lint` 仅余昨日 lwd ingest 留下的 1 条「陈旧页面」预警，与本次改动无关
- V21 checklist 对应条目已勾选

## [2026-05-07] feat | v21-execution | P3 详情页“知识地图”微地图（1-hop 邻居）

- V21 P3 收口：在 `docs/detail.html` 的 detail-hero 顶部新增 `#detailMiniMapWrap` D3 局部图谱容器，并补齐 `<script defer src="https://cdn.jsdelivr.net/npm/d3@7/dist/d3.min.js"></script>`
- `docs/main.js` 新增 `renderDetailMiniMap(detailPage, detailPages)`：基于 `exports/link-graph.json` 用 `detailPage.path` 定位当前节点，遍历 `edges` 收集 1-hop 邻居（最多 12 个，避免拥挤），D3 force simulation 绘制；点击邻居节点经 `path → detail page id` 反查后跳转 `detail.html?id=...`
- 新增 `TYPE_COLOR_DETAIL_MINI` 与现有 `mini-graph.js` 的色板保持一致；当前节点用 `.mini-node-current` 描边强调；`<title>` 节点 hover 给出完整 label
- `docs/style.css` 新增 `.detail-mini-map` / `.detail-mini-map-head` / `.detail-mini-map-svg` / `.mini-node-current` 样式（高度 180px，跟随 `--bg/--bg-alt/--border` 主题变量）
- 验证：`make lint-js`（eslint）通过；`make test` 55 个测试全通过、覆盖率 57.16%（≥ 52% 阈值）；`make lint` 仅余昨日 lwd ingest 留下的 1 条「陈旧页面」预警，与本次改动无关
- V21 checklist 对应条目已勾选；V21 DoD 中「微地图组件上线」「log.md 记录」两项亦同步勾选
