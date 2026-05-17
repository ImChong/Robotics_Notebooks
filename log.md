> 核心规范：所有日常动作（ingest / query / lint / structural）必须追加记录到此文件。

## [2026-05-17] ingest | InterPrior（arXiv:2602.06035）与 sirui-xu.github.io/InterPrior 站点入库

- 原始资料：`sources/papers/interprior_arxiv_2602_06035.md`、`sources/sites/sirui-xu-interprior-github-io.md`；索引 `sources/README.md`
- 沉淀页面：`wiki/entities/paper-interprior.md`
- 交叉更新：`wiki/tasks/loco-manipulation.md`、`wiki/methods/imitation-learning.md`、`wiki/methods/reinforcement-learning.md`、`index.md`（由 `make ci-preflight` 同步）

## [2026-05-17] ingest | sources/repos/leggedrobotics_robotic_world_model.md、leggedrobotics_robotic_world_model_lite.md — ETH RSL 的 RWM / RWM-U（Isaac Lab 扩展与 Lite 离线仓）

- 原始资料：`sources/repos/leggedrobotics_robotic_world_model.md`、`sources/repos/leggedrobotics_robotic_world_model_lite.md`
- 沉淀页面：`wiki/entities/robotic-world-model-eth-rsl.md`
- 交叉更新：`wiki/methods/model-based-rl.md`、`wiki/methods/generative-world-models.md`、`wiki/entities/isaac-gym-isaac-lab.md`、`sources/README.md`、`index.md`

## [2026-05-17] ingest | sources/repos/zilize-awesome-text-to-motion.md — 文本驱动人体运动生成 Awesome 列表入库；新增 wiki/entities/awesome-text-to-motion-zilize.md

- 原始资料：`sources/repos/zilize-awesome-text-to-motion.md`（<https://github.com/Zilize/awesome-text-to-motion>，项目页 <https://zilize.github.io/awesome-text-to-motion/>）
- 沉淀页面：`wiki/entities/awesome-text-to-motion-zilize.md`
- 交叉更新：`wiki/methods/diffusion-motion-generation.md`、`wiki/methods/genmo.md`、`sources/README.md`、`index.md`（由 `make ci-preflight` 同步目录统计）


## [2026-05-17] ingest | DoorMan（arXiv:2512.01061）与 doorman-humanoid 站点、GR00T-VisualSim2Real 归档入库

- 原始资料：`sources/papers/doorman_opening_sim2real_arxiv_2512_01061.md`、`sources/sites/doorman-humanoid-github-io.md`；更新 `sources/repos/gr00t_visual_sim2real.md`
- 沉淀页面：`wiki/entities/paper-doorman-opening-sim2real-door.md`
- 交叉更新：`wiki/entities/gr00t-visual-sim2real.md`、`wiki/entities/paper-viral-humanoid-visual-sim2real.md`、`wiki/tasks/loco-manipulation.md`、`sources/README.md`、`index.md`

## [2026-05-16] structural | wiki/concepts/motion-retargeting-pipeline.md — V22 P1 动作重定向知识链 (1/3)：新增重定向流水线概念页（8 阶段端到端：源归一 → 骨架/DoF 映射 → 体型缩放 → IK/QP → 硬约束与平滑 → 物理可行性筛选 → 可选物理修补 → 离线/在线产物落地）

- 沉淀页面：`wiki/concepts/motion-retargeting-pipeline.md`（含 Mermaid 流程总览、三种工程化形态对比表、常见失败模式表与下游接口契约）
- 交叉更新：`wiki/concepts/motion-retargeting.md`（关联页面回链流水线页）、`docs/checklists/tech-stack-next-phase-checklist-v22.md`（P1 第 1 项打勾、状态置 `[~]`）
- 派生再生成：`make ci-preflight` → 327 Nodes / 2197 Edges / Coverage 325/325；`scripts/lint_wiki.py` 通过（29 条信息型预警不阻塞）

## [2026-05-16] ingest | sources/sites/worldlabs-ai.md — World Labs 官网与 Marble/Spark 归档；新增 wiki/entities/world-labs.md；交叉更新 wiki/methods/generative-world-models.md、wiki/entities/gs-playground.md、index.md

- 原始资料：`sources/sites/worldlabs-ai.md`（<https://www.worldlabs.ai/> 及 About、Marble、Marble Labs、Spark 技术博客等公开链接归档）
- 沉淀页面：`wiki/entities/world-labs.md`
- 交叉更新：`wiki/methods/generative-world-models.md`、`wiki/entities/gs-playground.md`、`index.md`

## [2026-05-16] ingest | sources/papers/pelican_unified_uei_arxiv_2605_15153.md — Pelican-Unified 1.0（UEI，arXiv:2605.15153）技术报告入库；新增 wiki/methods/pelican-unified-1.md；交叉更新 wiki/methods/vla.md、wiki/concepts/world-action-models.md、wiki/methods/being-h07.md、index.md

- 原始资料：`sources/papers/pelican_unified_uei_arxiv_2605_15153.md`（PDF <https://arxiv.org/pdf/2605.15153>；关联 WAM 综述、Awesome-WAM、StarVLA、Being-H0.7 归档链接）
- 沉淀页面：`wiki/methods/pelican-unified-1.md`
- 交叉更新：`wiki/methods/vla.md`、`wiki/concepts/world-action-models.md`、`wiki/methods/being-h07.md`、`index.md`

## [2026-05-16] ingest | sources/papers/ewmbench.md, sources/repos/ewmbench.md, sources/sites/agibot-world.md — EWMBench（arXiv:2505.09694）与 AgibotTech 仓库及 Agibot-World 关联站点入库；新增 wiki/entities/ewmbench.md；交叉更新 wiki/methods/generative-world-models.md、wiki/concepts/video-as-simulation.md

- 原始资料：`sources/papers/ewmbench.md`、`sources/repos/ewmbench.md`、`sources/sites/agibot-world.md`
- 沉淀页面：`wiki/entities/ewmbench.md`
- 交叉更新：`wiki/methods/generative-world-models.md`、`wiki/concepts/video-as-simulation.md`、`index.md`

## [2026-05-16] structural | wiki/methods/genmo.md, docs/main.js, docs/style.css — GENMO 详情页 Mermaid：节点标签改为引号形式并修正边语法，避免 `~`/括号在方括号语法中被误解析；详情页迷你知识地图支持 d3.zoom 平移（禁用滚轮缩放以免抢页面滚动）

- 沉淀页面：`wiki/methods/genmo.md`（`detail.html?id=wiki-methods-genmo` 流程图可渲染）
- 前端：`docs/main.js`（`renderDetailMiniMap`）、`docs/style.css`（grab 光标 / `touch-action`）

## [2026-05-16] ingest | sources/papers/genmo.md, sources/repos/genmo.md — GENMO/GEM（arXiv:2505.01425v1，ICCV 2025 Highlight）论文与 NVlabs/GENMO 仓库入库；扩充 wiki/methods/genmo.md（dual-mode 训练 / multi-text 注入 / NVIDIA 人形栈 / Mermaid 流程图），交叉补 wiki/methods/diffusion-motion-generation.md 参考来源

- 原始资料：`sources/papers/genmo.md`（arXiv abs + HTML v1 摘录）、`sources/repos/genmo.md`（NVlabs/GENMO 仓库、GEM-SMPL HuggingFace 权重、README 时间线）
- 沉淀页面：`wiki/methods/genmo.md`（双模式训练 / multi-text 注入 / NVIDIA 人形栈关联 / Mermaid 流程总览）
- 交叉更新：`wiki/methods/diffusion-motion-generation.md`（参考来源与人体运动域代表实现链接）

## [2026-05-16] ingest | sources/repos/nvlabs-curobo.md — CuRobo / cuRoboV2（curobo.org、GitHub、arXiv:2310.17274、2603.05493）入库；沉淀 wiki/entities/curobo.md

- 原始资料：`sources/repos/nvlabs-curobo.md`（归档 https://curobo.org/、https://github.com/NVlabs/curobo、https://arxiv.org/abs/2310.17274、https://arxiv.org/abs/2603.05493 及 Isaac ROS cuMotion 等关联链接）
- 沉淀页面：`wiki/entities/curobo.md`

## [2026-05-16] ingest | sources/sites/articraft3d-github-io.md, sources/repos/mattzh72-articraft.md — Articraft 项目页与代码仓；新增 wiki/entities/articraft.md；互链 text-to-cad、URDF-Studio

- 原始资料：`sources/sites/articraft3d-github-io.md`、`sources/repos/mattzh72-articraft.md`
- 沉淀页面：`wiki/entities/articraft.md`（交叉更新 `wiki/concepts/text-to-cad.md`、`wiki/entities/urdf-studio.md`、`index.md`）

## [2026-05-16] ingest | sources/repos/jackhan-mujoco-walke3-simulation.md 等六仓 — JackHan-Sdu WalkE3 / HumanoidE3 / FEAP 工具链入库；新增 wiki/entities/jackhan-walke3-e3-ecosystem.md 及六条子实体页（各含 Mermaid 流程图）

- 原始资料：`sources/repos/jackhan-mujoco-walke3-simulation.md`、`sources/repos/jackhan-walke3-dataset.md`、`sources/repos/jackhan-walke3-controller.md`、`sources/repos/jackhan-algorithm-template-for-developer.md`、`sources/repos/jackhan-feap-mujoco-deployment.md`、`sources/repos/jackhan-feapvision-mujoco-deployment.md`
- 沉淀页面：`wiki/entities/jackhan-walke3-e3-ecosystem.md`、`wiki/entities/jackhan-mujoco-walke3-simulation.md`、`wiki/entities/jackhan-walke3-dataset.md`、`wiki/entities/jackhan-walke3-controller.md`、`wiki/entities/jackhan-yobotics-e3-algorithm-template.md`、`wiki/entities/jackhan-feap-mujoco-deployment.md`、`wiki/entities/jackhan-feapvision-mujoco-deployment.md`

## [2026-05-16] structural | wiki/queries/humanoid-rl-cookbook.md, docs/style.css — Humanoid RL Cookbook：「TL;DR」标题改为「快速决策路径」；任务列表内普通子 bullet 增加左缩进避免与 checkbox 同列重叠

## [2026-05-16] structural | wiki/queries/humanoid-rl-cookbook.md, docs/main.js, docs/style.css — Humanoid RL Cookbook：TL;DR 改为 Mermaid 流程图；详情页支持 `- [ ]` / `- [x]` 任务列表复选框渲染

- `wiki/queries/humanoid-rl-cookbook.md`：用 `flowchart TB` 呈现硬件 / 仿真 / 训练路径分支；更新 `updated` 元数据。
- `docs/main.js`：`renderMarkdownContent` 在无序列表解析中识别 GFM task list，输出 `<input type="checkbox" disabled>`；`flushList` 统一列表项结构。
- `docs/style.css`：`.contains-task-list`、`.task-list-item` 与 label 布局，避免与默认列表圆点重叠。

## [2026-05-15] structural | scripts/lint_wiki.py — V22 P0 方法-Query 闭环 Lint：新增 `methods_without_practitioner_query` 检查 + `INFO_ONLY_KEYS` 信息型分类机制

- `scripts/lint_wiki.py`：新增 `_check_methods_without_practitioner_query()`，阈值 `METHOD_PRACTITIONER_INBOUND_THRESHOLD=3`（即 ≥ 4 个 wiki 入链，自链已排除），若入链来源中无任何 `wiki/queries/*` 或 `wiki/comparisons/*` 命中，则标记为"待落地"信息型预警。
- 失败计数机制：抽出 `_failing_total()` / `_info_total()` 辅助函数，新增 `INFO_ONLY_KEYS = {"missing_pages", "methods_without_practitioner_query"}` 让 main 退出码只统计硬错误，避免首次落地即破坏 CI（baseline 28 项以 💡 信息型展示）。
- 测试：新增 `tests/test_lint_wiki_practitioner_query.py` 6 个用例（高入链无 query 命中、queries 命中、comparisons 命中、阈值边界、自链排除、INFO_ONLY 不计失败 total），`PYTHONPATH=scripts pytest --no-cov` 91/91 通过；`ruff check`、`ruff format --check`、`mypy scripts/lint_wiki.py` 均通过；`scripts/lint_wiki.py` 退出码 0，报告含 28 条信息型预警。
- 落地基线：当前 28 条预警覆盖 exoactor / sonic-motion-tracking / amp-reward / beyondmimic / motion-retargeting-gmr / humanoid-transformer-touch-dreaming / deepmimic / auto-labeling-pipelines / pi07-policy / π0-policy 等高频热点，将在 V22 P1（动作重定向）/ P2（抓取）落地 queries 与 comparisons 时同步消减。
- 清单：`docs/checklists/tech-stack-next-phase-checklist-v22.md` P0 "方法-Query 闭环 Lint" 三项全部勾选。

## [2026-05-15] ingest | sources/sites/amass-dataset.md, sources/repos/ubisoft-laforge-animation-dataset.md, sources/sites/mixamo.md — AMASS / LaFAN1 / Mixamo 入库；新增 wiki/entities/amass.md、wiki/entities/lafan1-dataset.md、wiki/entities/mixamo.md；互链 motion-retargeting、wbc-fsm、ProtoMotions

- 原始资料：`sources/sites/amass-dataset.md`、`sources/repos/ubisoft-laforge-animation-dataset.md`、`sources/sites/mixamo.md`
- 沉淀页面：`wiki/entities/amass.md`、`wiki/entities/lafan1-dataset.md`、`wiki/entities/mixamo.md`

## [2026-05-15] ingest | sources/repos/xiaomi-robotics-0.md — 小米 Xiaomi-Robotics-0（官网/GitHub/arXiv:2602.12684）；新增 wiki/entities/xiaomi-robotics-0.md；互链 VLA、Action Chunking

- 原始资料：`sources/repos/xiaomi-robotics-0.md`
- 沉淀页面：`wiki/entities/xiaomi-robotics-0.md`

## [2026-05-15] ingest | sources/papers/viral-humanoid-visual-sim2real.md — VIRAL arXiv:2511.15200：新增 wiki/entities/paper-viral-humanoid-visual-sim2real.md；GR00T-VisualSim2Real 增补 Mermaid 流程图与论文专档互链

- 原始资料：`sources/papers/viral-humanoid-visual-sim2real.md`
- 沉淀页面：`wiki/entities/paper-viral-humanoid-visual-sim2real.md`

## [2026-05-15] ingest | sources/papers/pi07.md — Physical Intelligence π₀.₇（arXiv:2604.15483 + pi.website/blog/pi07），新增 wiki/methods/pi07-policy.md，交叉更新 wiki/methods/π0-policy.md、wiki/methods/vla.md、wiki/concepts/foundation-policy.md

- 原始资料：`sources/papers/pi07.md`
- 沉淀页面：`wiki/methods/pi07-policy.md`

## [2026-05-15] structural | wiki/methods/actuator-network.md, sources/papers/system_identification.md — 补充 ActuatorNet 一手论文 DOI / arXiv、RSS 2018 PDF、Isaac Lab 执行器代码链接；修正错误题名；`system_identification` ingest 映射增加 `actuator-network`

## [2026-05-15] structural | wiki/methods/actuator-network.md — 增加离线辨识训练与仿真步内闭环的 Mermaid 流程图；更新页面 `updated` 元数据

## [2026-05-15] structural | sources/repos/protomotions.md, wiki/entities/protomotions.md — 对照 NVlabs README 与 protomotions.github.io 扩充原始摘录与实体页：能力表、数据—训练—部署与模块化 MDP 双 Mermaid、局限说明；互链 ADD / DeepMimic / xue-bin-peng

## [2026-05-15] ingest | sources/repos/robot-io-rio.md — 收录 RIO 官网、GitHub、Netlify 文档与 arXiv:2605.11564；新建 `wiki/entities/robot-io-rio.md` 并互链 `wiki/methods/vla.md`、`wiki/entities/lerobot.md`、`wiki/tasks/teleoperation.md`

- 原始资料：`sources/repos/robot-io-rio.md`（注明文档站 `/arXiv` 路径 404，以根路径为准）
- 沉淀页面：`wiki/entities/robot-io-rio.md`

## [2026-05-14] structural | scripts/search_wiki_core.py — V22 P0 缩写/别名归一化检索：新增 `WIKI_ABBREVIATIONS`（16 条：WBC/VLA/IL/RL/MPC/PPO/SAC/HQP/CBF/CLF/BC/IK/FK/LIP/ZMP/TSID）与 `expand_query_aliases()`，缩写 ↔ 全称双向展开后同时喂给 BM25 分词与行匹配，并以"缩写归一化：已展开为 …"提示挂到 `semantic_notice`；新增 5 个单测（21/21 通过），`eval_search_quality.py` 36/37 与基线一致

## [2026-05-14] structural | roadmap & docs/main.js — 主路线 ASCII 图换 mermaid + skip-to 改交互按钮 + 4 条 depth 加 mermaid pipeline

- `docs/main.js`：`renderMarkdownContent` 新增原始 HTML block 透传（div/details/summary/section/aside/figure/figcaption），让 markdown 可嵌入交互组件。
- `roadmap/motion-control.md`：L−1 的 4 盒子全景与 L4.0 的方法链 ASCII 图换 mermaid flowchart；资深读者 skip-to 矩阵改为 7 个 grid-style 按钮（auto-fit minmax 260px）。
- `roadmap/depth-*.md`：4 条独立纵深路线页顶部各加专属 mermaid Stage pipeline（不同 stroke 配色区分主题）。

## [2026-05-14] structural | roadmap & docs/roadmap.html — 主路线重构 + 网页正文渲染升级

将 `roadmap.html?id=roadmap-motion-control` 从「阶段树 + 互链 Top10」升级为「mini-map + 完整 markdown 正文 + TOC 侧栏」，复用 detail 页 markdown 渲染管线（`docs/main.js` 新增 `renderRoadmapMarkdownBody`，挂载点 `#roadmapContent` / `#roadmapTocList`）。同时把四条 if-goal 纵深从 `roadmap/motion-control.md` 拆为 `roadmap/depth-{rl-locomotion,imitation-learning,safe-control,contact-manipulation}.md` 四个独立 roadmap 页，主线只留摘要 + 衔接表；旧 `roadmap-if-goal-*` id 由跳锚点改为跳新 roadmap 页。

主路线内容侧从 L0–L6 扩为 L−1 → L7 单主线：
- L−1 序言：感知/规划/控制/执行四盒子全景、为什么人形为主载体、三种读者读法、Modern Robotics 章节映射、25+ 必备术语速查
- 每个 L 加场景隐喻 + 上一层的局限说明，让"为什么这一层存在"显式可见
- L1 / L2 增加里程碑分步说明（L1 三步 / L2 两步），L4 新增 L4.0 桥段（模型粒度 × 控制频率二维表 + 方法链 ASCII 流程图），L4 后新增方法谱系对比表（PID→LQR→LIP→DCM→Centroidal→TO→MPC→TSID/WBC→PPO→BC→DAgger→AMP→Diffusion）
- 每个 L 加 3-4 道自测题，L−1 新增资深读者 skip-to 矩阵
- L7 出口：感知 / 规划 / 操作 / 系统软件栈扫盲 + 2024–2026 前沿地图（Humanoid FM、VLA、World Model、Teacher-Student、AMP、End-to-End、Loco-Manipulation、Tactile）
- Modern Robotics 入口去重：L0–L4 五处 "本阶段入口" 不再重复引入，统一在 L−1 介绍

## [2026-05-14] structural | scripts/generate_link_graph.py — V22 P0 社区粒度二级拆分：保留 Girvan-Newman 一级检测（`PRIMARY_COMMUNITY_CAP=8`），新增纯 Python Louvain（`resolution=1.15`，Reichardt-Bornholdt modularity）对占比 > 40% 的巨型社区二级拆分；`MAX_COMMUNITIES` 提升至 16 容纳子社区。`exports/graph-stats.json`：`largest_community_ratio` 0.651 → 0.138，`community_quality_warning` true → false，Locomotion 拆出 WBC/RL/MPC/IL/Sim2Real/Isaac Gym/Humanoid/Unitree G1 等子社区

## [2026-05-14] structural | wiki — 扩充 `wiki/tasks/locomotion.md`：补充任务边界、闭环 Mermaid、子问题地图、方法选型与工程落地检查

## [2026-05-14] structural | docs/checklists — 新增 `scripts/screenshot_site_detail.sh`（timeout 包裹 headless Chrome、随机 remote-debugging-port），并更新 cloud-agent-pr-workflow 的截图步骤说明

## [2026-05-14] ingest | sources/repos/sonic-humanoid-motion-tracking.md — 对照 NVIDIA GEAR-SONIC 官网扩充 SONIC 原始资料；更新 wiki/methods/sonic-motion-tracking（规模表、VLA/遥操作/规划器接口、双 Mermaid）；互链 foundation-policy、vla、teleoperation 与 tairan-he 项目 URL

## [2026-05-13] structural | references — 扩充 `references/repos/retarget-tools.md`：分组外链、互链 Motion Retargeting / GMR / NMR / ReActor / IL，消除详情页「空正文」观感

## [2026-05-13] structural | roadmap — 将四条 `learning-paths/if-goal-*.md` 全文并入 `roadmap/motion-control.md`「可选纵深」；站点仅保留单一 `roadmap_page`；旧 `roadmap.html?id=roadmap-if-goal-*` 重定向至 `detail.html?id=roadmap-motion-control#...`

## [2026-05-12] structural | wiki — 扩充 `wiki/entities/mimickit.md`：运行时与算法选型 Mermaid 流程图、架构表与局限说明；补强 related 与参考来源

## [2026-05-12] structural | wiki — 扩充 `wiki/methods/motion-retargeting-gmr.md`：命名辨析、双 Mermaid 流程图、开源工程要点与 arXiv；互链 SONIC

## [2026-05-12] ingest | sources/notes/humanoid-parallel-joint-kinematics.md — 新增人形并联关节解算资料索引与 wiki 概念页，互链 Asimov RSU 踝、灵巧手闭链与 Armature

## [2026-05-12] chore | docs/checklists — 明确 PR 验证截图为站点 detail 页而非 GitHub PR 页

## [2026-05-12] structural | docs/checklists — 新增 Cloud Agent PR 推送与验证截图流程说明（cloud-agent-pr-workflow.md），AGENTS 增加指针并忽略 .cursor-artifacts

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

## [2026-05-11] structural | wiki — 新增四足机器人实体页 `wiki/entities/quadruped-robot.md` 并与 humanoid、市面平台纵览、ANYmal、Boston Dynamics、Unitree、locomotion 任务互链

## [2026-05-11] structural | wiki — 新增轮足四足（四轮足）概念页 `wiki/concepts/wheel-legged-quadruped.md`，互链 Hybrid Locomotion、robot_lab、locomotion、Unitree

## [2026-05-11] chore | roadmap — 删除两条空分支学习路径

- 移除 `roadmap/learning-paths/if-goal-generalist.md`、`if-goal-whole-body-control.md`（仅占位列表，无 L 阶段正文）。
- 更新 `roadmap/motion-control.md`、`roadmap/README.md`、`STRUCTURE_v1.md` 引用；`make ci-preflight` 同步索引与导出。

## [2026-05-08] style | docs | 技术路线页 `roadmap.html` emoji 统一为 🚀

- favicon、顶栏站点标题由 🛣️/🧭 改为 🚀，与首页「技术路线指南」CTA 一致；主题切换仍为 ☀️/🌙。
- 同步 `docs/checklists/frontend-optimization-v1.md`。

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

## [2026-05-07] ingest | sources/papers/genesis_gene_ecosystem.md — 收录 GENE-26.5 与 Genesis 仿真论文/链接，新增实体页并补强 genesis-sim 辨析

## [2026-05-07] docs(wiki): wiki/entities/asimov-v1.md — 补充被动趾（主仓弹簧关节 vs mjlab 固连）、RSU 踝机构与公开 MJCF 等效 2-DOF 接口及 asimov_constants 运动学符号说明

## [2026-05-07] docs(wiki): wiki/entities/asimov-v1.md — 补充主仓仿真定位、asimov-mjlab（PPO+imitation）与 Menlo 博文观测/Sim2Real 叙述；新增 sources/repos/asimov-mjlab.md；mjlab 实体页关联 Asimov 训练 fork

## [2026-05-07] ingest | sources/repos/asimov-v1.md — 收录 Asimov v1 全栈开源人形仓库；新增 wiki/entities/asimov-v1.md 并关联开源硬件对比与人形实体页

## [2026-05-07] ingest | sources/repos/1x-technologies.md、sources/repos/figure-ai.md — 沉淀 wiki/entities/1x-technologies.md、figure-ai.md，并更新 humanoid-robot 与硬件选型 query

## [2026-05-07] ingest | sources/repos/notable-commercial-robot-platforms.md — 市面知名人形/四足平台索引与 overview 纵览页

## [2026-05-07] ingest | sources/repos/motioncode.md — 收录 MotionCode 官网资料并新增 wiki/entities/motioncode.md，回链人形机器人 / 动作重定向 / 平台纵览

## [2026-05-07] ingest | sources/repos/sceneverse-pp.md — SceneVerse++ 入库并新增实体页与 3D 空间 VQA / VLN 任务页

## [2026-05-07] docs(wiki): wiki/entities/gene-26-5-genesis-ai.md — 补充 GENE-26.5 官方 YouTube 演示视频链接与参考来源说明

## [2026-05-08] ingest | sources/papers/neural_motion_retargeting_nmr.md — 入库 NMR 论文并新增 wiki/methods/neural-motion-retargeting-nmr.md（含 Mermaid 流程图）；补强 motion-retargeting / GMR 交叉引用；ingest-workflow 增补 Mermaid 步骤说明

## [2026-05-08] chore | merge | 合并 origin/main，解决 PR #134 冲突

- main 已独立实现 V21 P3「搜索结果按置信度分级」（[2026-05-06] feat）与「详情页微地图」（[2026-05-07] feat），与本分支 [2026-05-07] 同名条目重复
- 解决策略：保留 main 的实现（`buildResultCardHtml` / `.search-tier-heading*` / `renderDetailMiniMap` / `.detail-mini-map*`），覆盖本分支 `renderCardItem` / `.search-tier-header*` 变体；checklist v21 P3「详情页微地图」按 main 勾选
- 自动产物（README badge、`exports/*-stats.json`、`exports/lint-report.md`、`docs/exports/*.json`、`exports/index-v1.json`）以 main 为准，再走 `make ci-preflight` 校核
- 最终 PR #134 仅保留：本次 merge 解决冲突的记录

## [2026-05-08] checklist-v21 | schema/canonical-facts.json — 推进 V21「事实库扩展至 ≥ 140 条」DoD

- 围绕 V21 新增的触觉/通信专题与 NMR 入库新页，向 `schema/canonical-facts.json` 追加 10 条矛盾检测规则：Visuo-Tactile Fusion 接触瞬间、Tactile Impedance Control 变阻抗、Contact Wrench Cone 6 维推广、GelSlim 指节级薄型化、控制环路延迟五段分解、UDP 组播四类事件、PTP 时钟同步精度、EtherCAT DC 同步精度、EtherCAT vs EtherNet/IP 选型、NMR CEPR 管线
- 事实库总条目：130 → **140**，达到 V21 DoD「≥ 140 条」目标
- 同步勾选 V21 DoD「知识图谱节点数 ≥ 190」（当前 252，远超阈值）
- `python3 scripts/lint_wiki.py` 矛盾检测：0 个新矛盾（首次回归出现 lcm-basics 误命中后将「UDP 组播」neg_claims 收紧为 `支持.*重传 / 提供.*ACK / 能.*保证.*顺序`，避免误伤合法描述）
- 剩余 DoD「`make lint` 0 errors」单点未达成，其根因是 10 个 sources 比 wiki 新的「陈旧页面」预警，归并到下一日处理

## [2026-05-09] ingest | sources/papers/hipan.md — HiPAN 分层姿态自适应四足导航，新增 wiki/methods/hipan.md 并交叉引用 locomotion、curriculum-learning

## [2026-05-10] ingest | sources/papers/intentional_streaming_rl.md, sources/repos/intentional_rl.md — 流式 RL 意图更新论文与 Intentional_RL 仓库，新增 wiki/methods/intentional-updates-streaming-rl.md 并回链 RL / online-offline / policy-optimization

## [2026-05-10] ingest | sources/blogs/ted_xiao_embodied_three_eras_primary_refs.md — Ted Xiao 访谈相关一手文献索引与 wiki/queries 叙事消化页

## [2026-05-10] structural | wiki — Ted Xiao 一手索引技术线独立节点补全（概念/方法/实体共 14 页）与图谱交叉引用

## [2026-05-11] ingest | sources/repos/rldx-1.md — 收录 RLDX-1 技术报告与仓库；新增 wiki/entities/rldx-1.md，并自 VLA / Manipulation / StarVLA 交叉引用

## [2026-05-11] ingest | sources/repos/dm_control.md, sources/papers/dm_control_suite.md — 接入 Google DeepMind dm_control 与 arXiv:1801.00690，新建 wiki/entities/dm-control 并互链 MuJoCo、仿真器选型与索引

## [2026-05-12] ingest | sources/papers/being_h07.md — Being-H0.7 潜空间世界–动作模型：归档、wiki 方法页，互链 VLA / 生成式世界模型 / 潜空间想象

## [2026-05-12] ingest | sources/repos/april_tag.md — 收录 AprilTag 官方页与 AprilTag 3 C 库并新增实体 wiki 页

## [2026-05-12] ingest | sources/papers/humanoid_parallel_ankle_kinematics_ingest.md — 并联踝文献包（RG+arXiv）并扩充人形并联关节解算页、互链 Sim2Real 与 TO

## [2026-05-12] ingest | sources/sites/botlab_motioncanvas.md — BotLab MotionCanvas 入库与 wiki/entities/botlab-motioncanvas.md

## [2026-05-12] ingest | sources/sites/wuji_robotics.md — 舞肌科技原始资料与实体页、平台纵览互链

## [2026-05-12] ingest | sources/sites/wuji_robotics.md — 增补 Wuji Hand 灵巧手官方文档中心与 NE 时代盘点锚点

## [2026-05-12] ingest | sources/sites/project_instinct.md — Project Instinct 站群入库并升格 wiki/entities/project-instinct.md（含 Mermaid 流程图）

## [2026-05-12] ingest | sources/repos/mjlab_playground.md, sources/repos/freemocap.md — 接入 mjlab_playground 与 FreeMoCap；新增实体页与动捕→mjlab 流程 Mermaid；互链 mjlab / motion-retargeting / unitree_rl_mjlab

## [2026-05-12] ingest | sources/notes/legged_humanoid_rl_pd_gains.md — 腿足/人形 RL 关节 PD 增益资料索引与 wiki 查询页（含 Mermaid）

## [2026-05-12] ingest | sources/papers/rl_pd_action_interface_locomotion.md — RL+PD 人形/双足/四足文献索引与 Kp/Kd query 页增补 Mermaid 与论文共识

## [2026-05-12] structural | wiki — 新增 10 个 RL+PD 论文实体页（paper-*.md）并互链 Kp/Kd query、locomotion、legged_gym 与 sources 索引

## [2026-05-13] ingest | sources/sites/kleiyn-efgcl.md — EFGCL 项目页入库并升格 wiki/methods/efgcl.md

## [2026-05-13] ingest | sources/papers/reactor_rl_physics_aware_motion_retargeting.md — ReActor（arXiv:2605.06593）物理感知双层 RL 重定向入库与 wiki 方法页

## [2026-05-13] ingest | sources/papers/disney_olaf_character_robot.md — Olaf 实机角色（arXiv:2512.16705）入库并新增 wiki/methods/disney-olaf-character-robot.md

## [2026-05-13] ingest | sources/sites/text-to-cad-tools.md — 收录 Zoo 与文字生成 CAD 工具索引并新增 wiki 概念页

## [2026-05-13] structural | wiki — 扩充 text-to-cad 概念页与 sources 索引：成熟度、Adam、Fusion AI、CadQuery/OpenSCAD、网格向工具

## [2026-05-13] checklist-v21 | DoD 收口 & 初始化 V22

- V21 DoD 最后一项「`make lint` 0 errors」收口：`make lint` 输出「✅ 所有检查通过！共发现 0 个问题」，10 个 sources 比 wiki 新的「陈旧页面」预警已全部消化，勾选 `docs/checklists/tech-stack-next-phase-checklist-v21.md` 中对应条目。
- V21 完整交付：图谱 297 节点 / 1933 边，事实库 140 条，触觉与力觉闭环 / 硬件通信链路两条专题全部上线，详情页"知识地图"微地图与搜索结果分级 UI 已在 V20–V21 期间集成。
- 新建 `docs/checklists/tech-stack-next-phase-checklist-v22.md`，专题选定为「动作重定向与角色化人形」，配合「抓取与操作感知」深化；并基于 `exports/graph-stats.json` 中 `community_quality_warning: true`（最大社区占 46.1%）规划 P0 二级社区拆分任务。
- 同步将 README badge、维护看板、`AGENTS.md`、`docs/README.md` 与 `docs/checklists/README.md` 的「当前清单」指针从 V21 切到 V22；V21 进入历史归档区。

## [2026-05-13] ingest | sources/repos/anygrasp-sdk.md — 收录 AnyGrasp SDK / 论文 / GraspNet 生态，新增 wiki/entities/anygrasp、references/repos/manipulation-perception，互链 Manipulation

## [2026-05-14] lint | health-check — 补齐 deep-learning-foundations 与 generative-foundations 元数据，并为 humanoid-robot 增加整体开发流程图

## [2026-05-14] ingest | sources/papers/world_action_models_survey_2605.md、sources/repos/awesome-wam-openmoss.md、sources/sites/awesome-wam-openmoss.md — WAM 综述与 Awesome-WAM 入库，新增 wiki/concepts/world-action-models.md 并交叉 VLA / 生成式世界模型

## [2026-05-14] ingest | sources/sites/tairan-he.md — 收录个人主页并新增 wiki/entities/tairan-he.md，互链 GR00T-VisualSim2Real

## [2026-05-14] ingest | sources/sites/xue-bin-peng.md, sources/sites/zhengyiluo.md — 归档两位学者主页；升格 wiki/entities/xue-bin-peng.md 与 zhengyi-luo.md，并互链 MimicKit、ProtoMotions、SONIC、Tairan He

## [2026-05-14] ingest | sources/repos/kimodo.md、sources/repos/gr00t_wholebodycontrol.md — 入库 Kimodo 与 GR00T WBC 官方仓；新增 wiki/entities/kimodo、gr00t-wholebodycontrol 并互链 diffusion-motion-generation、motionbricks、foundation-policy、SONIC、ProtoMotions、GR00T-VisualSim2Real

## [2026-05-14] ingest | sources/blogs/menlo_noise_is_all_you_need.md — Menlo「处理器在环+总线抖动」博文入库；新增 wiki/concepts/processor-in-the-loop-sim2real；互链 sim2real、asimov-v1

## [2026-05-14] ingest | sources/papers/humannet.md — HumanNet arXiv:2605.06747 入库；新增 wiki/entities/humannet 并互链 VLA / IL / 人形机器人

## [2026-05-15] ingest | sources/personal/humanoid-policy-network-architecture-faq.md — 新建概念页 humanoid-policy-network-architecture 与 tech-map il 节点 policy-network-architecture

## [2026-05-15] structural | schema/search-regression-cases.json — WBC×QP BM25 回归 expected_top_k 3→5（语料扩张后 whole-body-control 概念页常居第 4–5 位）

## [2026-05-15] ingest | sources/repos/pytorch-official.md — 收录 pytorch.org 与 get-started/docs/tutorials；新建 wiki/entities/pytorch.md；互链深度学习基础、Isaac Lab
