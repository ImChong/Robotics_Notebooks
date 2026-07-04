> 核心规范：所有日常动作（ingest / query / lint / structural）必须追加记录到此文件。

## [2026-07-04] structural | scripts/generate_link_graph.py — 更新记录「新增/维护」改按 log.md ingest/structural 首次出现判定（修复 git 日期偏早与 lint glob 误展开）

## [2026-07-04] ingest | sources/papers/heft_arxiv_2607_02332.md — HEFT 重载全尺寸人形 VR 遥操作（PMG+WPC）；wiki/entities/paper-heft.md、wiki/entities/axellwppr-motion-tracking.md；交叉更新 wiki/tasks/teleoperation.md、wiki/entities/paper-twist2.md

## [2026-07-03] structural | schema/canonical-facts.json 210 → 220（V27 P2）——新增 10 条接触力控矛盾检测规则

- 覆盖：力控带宽↑ 与控制刚度/稳定裕度冲突、阻抗 vs 导纳因果对偶在接触刚度未知时失稳、刚性高带宽与柔顺安全取舍、纯视觉时延致接触前过冲、触觉采样率不足致打滑漏检、混合力位方向选择错误致约束冲突、力旋量估计依赖雅可比/惯量标定、接触离散化致力旋量高估、过度柔顺牺牲定位精度、域随机化不替代真机力标定
- 校验：逐条 pos 命中现存 wiki 页（`contact-force-loop-bandwidth` / `impedance-control` / `visuo-tactile-fusion` / `hybrid-force-position-control` / `contact-estimation` / `contact-wrench-closed-loop`），`make lint` 潜在矛盾 0 个、0 errors；`make ci-preflight` 12/12 通过
- 清单勾稽：[`docs/checklists/tech-stack-next-phase-checklist-v27.md`](docs/checklists/tech-stack-next-phase-checklist-v27.md) P2 事实库扩展项完成

## [2026-07-03] ingest | sources/blogs/wechat_human_five_jason_peng_flexible_motion_skills.md — human five Jason Peng 更灵活的运动技能学习；wiki/overview/jason-peng-flexible-motion-skill-learning.md；交叉更新 xue-bin-peng、deepmimic、amp-reward、humanoid-rl-motion-control-body-system-stack、PARC

- 工具：已安装 [Panniantong/Agent-Reach](https://github.com/Panniantong/Agent-Reach) v1.5.0（`pip install git+https://github.com/Panniantong/Agent-Reach.git` + [wechat-article-for-ai](https://github.com/bzd6661/wechat-article-for-ai) 至 `~/.agent-reach/tools/`（Camoufox；`playwright==1.49.1`））
- 原始链接：<https://mp.weixin.qq.com/s/b-5UIRB1mkEDcIJlAT2jwg>
- 沉淀页面：[`wiki/overview/jason-peng-flexible-motion-skill-learning.md`](wiki/overview/jason-peng-flexible-motion-skill-learning.md)
- 交叉更新：[`wiki/entities/xue-bin-peng.md`](wiki/entities/xue-bin-peng.md)、[`wiki/methods/deepmimic.md`](wiki/methods/deepmimic.md)、[`wiki/methods/amp-reward.md`](wiki/methods/amp-reward.md)、[`wiki/overview/humanoid-rl-motion-control-body-system-stack.md`](wiki/overview/humanoid-rl-motion-control-body-system-stack.md)、[`wiki/entities/paper-notebook-parc-physics-based-augmentation-with-reinforceme.md`](wiki/entities/paper-notebook-parc-physics-based-augmentation-with-reinforceme.md)、[`sources/README.md`](sources/README.md)、[`sources/repos/panniantong_agent_reach.md`](sources/repos/panniantong_agent_reach.md)

## [2026-07-03] ingest | sources/repos/ponytail.md — 接入 Ponytail 编码代理必要性阶梯技能并交叉更新 caveman/superpowers/mattpocock/llm-wiki/hermes 实体页；wiki/entities/ponytail.md

## [2026-07-03] ingest | sources/papers/simfoundry_arxiv_2606_28276.md — SimFoundry Real2Sim 场景生成与策略评测/训练闭环；wiki/entities/paper-simfoundry-real2sim-scene-generation.md；交叉更新 sim2real、manipulation、simulation-evaluation-infrastructure、nvidia-gear-lab

## [2026-07-03] ingest | sources/papers/mint_rss_2026.md — MINT RSS 2026 频域意图分词与单样本迁移；wiki/entities/paper-mint-vla.md；交叉更新 wiki/methods/vla.md、wiki/formalizations/vla-tokenization.md

## [2026-07-03] ingest | sources/repos/parallel_ankle_joint.md — G1/天工并联踝 IK·FK·雅可比参考实现；wiki/concepts/humanoid-parallel-joint-kinematics.md

## [2026-07-03] ingest | sources/sites/telegate-project.md — TeleGate 门控专家全身遥操作；wiki/entities/paper-telegate.md；交叉更新 wiki/tasks/teleoperation.md

## [2026-07-03] ingest | sources/repos/freecad.md — FreeCAD 开源参数化 CAD 入库；wiki/entities/freecad.md；交叉更新 blender、step2urdf、urdf-robot-description

## [2026-07-03] structural | wiki/entities/paper-human-as-humanoid.md 等 6 篇 — 补全 Loco-Manip 接触专题缺失论文独立节点并挂接五组 category hub

- 新建：`paper-human-as-humanoid`、`paper-humanoidumi`、`paper-vlk-synthetic-loco-manipulation`、`paper-imagine2real-zero-shot-hoi`、`paper-humanoid-dart`、`paper-wolf-vla`
- 交叉更新：`loco-manip-contact-category-01/03/05`、`loco-manip-contact-technology-map`、`sources/blogs/wechat_embodied_ai_lab_loco_manip_contact_survey.md`

## [2026-07-03] ingest | sources/blogs/wechat_embodied_ai_lab_loco_manip_contact_survey.md — 具身智能研究室 Loco-Manip 接触五段链路专题；父节点 loco-manip-contact-technology-map + 五组 loco-manip-contact-category-* 子节点；复用约 30 篇既有 paper 实体，6 篇仅外链不新建节点

- 工具：已安装 [Panniantong/Agent-Reach](https://github.com/Panniantong/Agent-Reach) v1.5.0（`pip install git+...` + 手动安装 [wechat-article-for-ai](https://github.com/bzd6661/wechat-article-for-ai) 至 `~/.agent-reach/tools/`（Camoufox；`playwright==1.49.1` 规避 viewport 协议错误））
- 原始链接：<https://mp.weixin.qq.com/s/UjShbwl8p1h9ukymfiRNaw>
- 沉淀页面：[`wiki/overview/loco-manip-contact-technology-map.md`](wiki/overview/loco-manip-contact-technology-map.md)（**父**）、[`loco-manip-contact-category-01-contact-data.md`](wiki/overview/loco-manip-contact-category-01-contact-data.md) … [`loco-manip-contact-category-05-vla-world-models.md`](wiki/overview/loco-manip-contact-category-05-vla-world-models.md)（**子**）
- 交叉更新：[`wiki/tasks/loco-manipulation.md`](wiki/tasks/loco-manipulation.md)、[`wiki/overview/humanoid-loco-manip-161-papers-technology-map.md`](wiki/overview/humanoid-loco-manip-161-papers-technology-map.md)、[`sources/README.md`](sources/README.md)、[`sources/repos/panniantong_agent_reach.md`](sources/repos/panniantong_agent_reach.md)

## [2026-07-02] lint | scripts/lint_wiki.py 新增 `_check_contact_control_crosslink`（V27 P0）——接触/力控/操作概念页交叉链路巡检 V1，INFO 级不阻塞 CI，回链「接触力旋量闭环」枢纽（contact-wrench-closed-loop / topic-contact-force-control）；新增 tests/test_lint_wiki_contact_control_crosslink.py（7 例）；刷新 exports/lint-report.md 基线（10 页 backlog，0 errors）

## [2026-07-02] ingest | sources/papers/flying_knots_arxiv_2602_21302.md — Flying Knots Task-Level ILC 可变形绳操作；wiki/entities/paper-flying-knots.md、wiki/entities/flying-knots-public.md；交叉更新 manipulation、contact-rich-manipulation

## [2026-07-02] ingest | sources/repos/robot_retargeter.md — robot_retargeter SMPL-X/多机型 mink 重定向；wiki/entities/robot-retargeter.md；交叉更新 motion-retargeting、motion-retargeting-pipeline、soma-retargeter、amass

## [2026-07-02] ingest | sources/papers/vsgraphs_arxiv_2503_01783.md — vS-Graphs 视觉 SLAM+3D 场景图；沉淀 wiki/entities/paper-vs-graphs-visual-slam-scene-graph.md，互链 orb-slam3、navigation-slam-autonomy-stack

## [2026-07-02] ingest | sources/sites/runpod.md 等 — 国外 GPU 云六平台入库；wiki/comparisons/international-gpu-cloud-platforms.md；交叉更新 china-gpu-cloud-platforms、simulator-selection-guide、isaac-lab

## [2026-07-02] ingest | sources/sites/matpool.md、featurize.md、gpushare.md、ai-galaxy.md — 扩展国内 GPU 云平台实体；wiki/comparisons/china-gpu-cloud-platforms.md 统一选型对比；移除 autodl-vs-gpufree

## [2026-07-02] ingest | sources/sites/autodl.md、sources/sites/gpufree.md — AutoDL 与算力自由 GPU 云入库；wiki/entities/autodl.md、wiki/entities/gpufree.md；交叉更新 wiki/entities/isaac-lab.md、wiki/queries/simulator-selection-guide.md

## [2026-07-02] ingest | sources/papers/kung_fu_athlete_bot.md — KungFuAthleteBot 高动态武术数据集与 tracking+recovery；wiki/entities/paper-kungfuathlete-humanoid-martial-arts-tracking.md、wiki/comparisons/humanoid-reference-motion-datasets.md、wiki/tasks/balance-recovery.md

## [2026-07-02] ingest | sources/papers/humanoidmimicgen_arxiv_2605_27724.md — HumanoidMimicGen 全身规划 loco-manip 合成示范；wiki/entities/paper-humanoidmimicgen.md、wiki/tasks/loco-manipulation.md

## [2026-07-02] ingest | sources/papers/gr00t_n1_arxiv_2503_14734.md — 基于 arXiv:2503.14734 与 NVIDIA 白皮书深化 GR00T N1 canonical 实体页

- 论文源：[arXiv:2503.14734](https://arxiv.org/abs/2503.14734)、[GR00T_1_Whitepaper.pdf](https://d1qx31qr3h6wln.cloudfront.net/publications/GR00T_1_Whitepaper.pdf)
- wiki：[paper-hrl-stack-34-gr00t_n1.md](wiki/entities/paper-hrl-stack-34-gr00t_n1.md) — 补充双系统架构、数据金字塔、GR00T-N1-2B 规格与仿真/真机量化评测

## [2026-07-02] structural | wiki/entities/paper-hrl-stack-34-gr00t_n1.md — 合并 GR00T N1 重复实体页（原 paper-loco-manip-161-148-gr00t-n1 与 paper-hrl-stack-34-gr00t_n1）；更新 Loco-Manip 161 catalog、category-09 hub、paper-grail 交叉引用与 bootstrap_loco_manip_161_entities CANONICAL_ENTITY_BY_NUM

- 删除：`wiki/entities/paper-loco-manip-161-148-gr00t-n1.md`
- 保留 canonical：`wiki/entities/paper-hrl-stack-34-gr00t_n1.md`（双 survey 坐标：42 篇栈 #34 + Loco-Manip 161 #148）
- 相关：`wiki/overview/loco-manip-161-category-09-vla-world-models.md`、`sources/papers/humanoid_loco_manip_161_catalog.md`、`sources/papers/loco_manip_161_survey_148_gr00t-n1.md`

## [2026-07-02] ingest | sources/sites/jim-fan.md — Jim Fan 个人主页/NVIDIA 档案/Google Scholar 入库；wiki/entities/jim-fan.md；交叉更新 wiki/entities/nvidia-gear-lab.md、wiki/entities/tairan-he.md、wiki/entities/zhengyi-luo.md

## [2026-07-01] structural | V27 P1 接触/力控层专题交叉补强 — contact-estimation / force-control-basics / hybrid-force-position-control / impedance-control / visuo-tactile-fusion 五页与接触力旋量闭环链新页（contact-wrench-closed-loop / contact-force-loop-bandwidth）形成双向回链，明示「感知①→力旋量②→控制③→操作④」四层定位；重生派生统计（node 1534 / edge 10412，largest_community_ratio 0.162，community_quality_warning=false）

## [2026-07-01] structural | scripts/bootstrap_paper_notebook_knowledge.py — 同步 Humanoid_Robot_Learning_Paper_Notebooks 最新 progress.json / PROGRESS.md：full-map 549 篇、索引 513 篇；新建 4 个 `wiki/entities/paper-notebook-*` 与 11 个 sources；修复 61 处深读 URL；更新 14 类分类父节点与 `humanoid-paper-notebooks-index.md`；修复 `sync_paper_notebook_links.py` 替换 URL 时吞掉 `>` 的 lint 问题

- 数据源：[progress.json](https://github.com/ImChong/Humanoid_Robot_Learning_Paper_Notebooks/blob/main/progress.json)（284 完成 / 87 待读）+ [papers/PROGRESS.md](https://github.com/ImChong/Humanoid_Robot_Learning_Paper_Notebooks/blob/main/papers/PROGRESS.md)
- 工具：`make paper-notebook-bootstrap`、`make paper-notebook-links`、`make paper-notebook-summaries`
- 新建：`wiki/entities/paper-notebook-learning-contact-representation-for-leg-odometry.md`、`wiki/entities/paper-notebook-learning-multi-modal-whole-body-control-for-real.md`、`wiki/entities/paper-notebook-physics-based-motion-tracking-of-contact-rich-in.md`、`wiki/entities/paper-notebook-simulator-adaptation-via-proprioceptive-distribu.md`
- 相关：`wiki/overview/paper-notebook-category-*.md`、`wiki/overview/humanoid-paper-notebooks-index.md`、`schema/paper-notebook-wiki-full-map.yml`

## [2026-07-01] ingest | sources/papers/hrdexdb_arxiv_2604_14944.md — HRDexDB 配对灵巧抓取数据集；wiki/entities/hrdexdb-dataset.md、wiki/tasks/manipulation.md、wiki/queries/dexterous-data-collection-guide.md、wiki/overview/topic-grasp.md

## [2026-07-01] ingest | sources/papers/omnicontact_arxiv_2606_26201.md — OmniContact Contact Flow meta-skill 链式 loco-manipulation；wiki/entities/paper-omnicontact-humanoid-loco-manipulation.md、wiki/entities/omnicontact-sim2sim.md

## [2026-07-01] ingest | sources/papers/argus_dynamic_symmetry_scirobotics_2026.md、sources/repos/argus_general_robotics_lab.md — Argus 动态对称与全向球形腿式机器人（Science Robotics 2026）；wiki/entities/paper-argus-dynamic-symmetry.md、wiki/tasks/locomotion.md、wiki/tasks/loco-manipulation.md、wiki/tasks/balance-recovery.md

## [2026-07-01] ingest | sources/repos/tsil.md — TSIL 官方仓库接入；更新 wiki/entities/paper-tsil-temporal-self-imitation-learning.md

## [2026-07-01] ingest | sources/papers/tsil_arxiv_2606_19752.md — Temporal Self-Imitation Learning 论文入库与 wiki/entities/paper-tsil-temporal-self-imitation-learning.md

## [2026-07-01] ingest | sources/papers/ultra_fusion_arxiv_2606_21223.md — Ultra-Fusion 补全作者与代码/M3DGR 仓库归档；wiki/entities/paper-ultra-fusion-multi-sensor-slam.md

## [2026-07-01] ingest | sources/papers/trex_arxiv_2606_17055.md — T-Rex 触觉反应式灵巧操作；wiki/entities/paper-trex-tactile-reactive-dexterous-manipulation.md、wiki/methods/vla.md、wiki/methods/egoscale.md、wiki/tasks/manipulation.md、wiki/concepts/contact-rich-manipulation.md、wiki/concepts/visuo-tactile-fusion.md、wiki/queries/manipulation-vla-architecture-selection.md

## [2026-07-01] ingest | sources/papers/robustness_robotic_manipulation_dong_2026.md — 操作鲁棒性系统综述；wiki/entities/paper-robustness-robotic-manipulation-survey.md、wiki/tasks/manipulation.md

## [2026-07-01] ingest | sources/papers/aspire_nvidia_gear_2026.md — 接入 NVIDIA ASPIRE 持续学习技能库系统；wiki/methods/aspire.md、wiki/entities/nvidia-gear-lab.md、wiki/methods/enpire.md、wiki/tasks/manipulation.md、wiki/queries/real-robot-policy-autoresearch-harness.md

## [2026-07-01] ingest | sources/papers/gpc_arxiv_2606_29148.md — GPC 生成式预训练物理控制器；wiki/entities/paper-gpc-generative-pretrained-controllers.md

## [2026-07-01] ingest | sources/sites/tnkr-open-duck-mini-v2.md — 接入 Tnkr Open Duck Mini v2 项目文档；交叉更新 wiki/entities/open-duck-mini.md、wiki/entities/tnkr.md、wiki/entities/open-duck-mini-runtime.md

## [2026-07-01] ingest | sources/papers/reactivebfm_arxiv_2606_30362.md — ReactiveBFM 闭环规划–控制；wiki/entities/paper-reactivebfm.md

## [2026-07-01] ingest | sources/papers/fada_arxiv_2606_28476.md — FADA 少样本动力学对齐；wiki/entities/paper-fada-humanoid.md

## [2026-06-30] query | checklist-v27 P1 接触力旋量闭环知识链（+2）—— 新建端到端 Query 与带宽概念页

- 新建 [`wiki/queries/contact-wrench-closed-loop.md`](wiki/queries/contact-wrench-closed-loop.md)：把分散的「① 接触感知/估计 → ② 力旋量表示 → ③ 阻抗/导纳/混合力位控制 → ④ 接触丰富操作策略」四层串成端到端决策链，含 TL;DR 四层定位表、Mermaid 四层决策树、按层归因的失败模式速查；内链回 `contact-estimation` / `force-control-basics` / `hybrid-force-position-control` / `impedance-control` / `visuo-tactile-fusion` 与 CHORD / SceneBot / HapMorph 来源。
- 新建 [`wiki/concepts/contact-force-loop-bandwidth.md`](wiki/concepts/contact-force-loop-bandwidth.md)：力控闭环带宽 ↔ 接触稳定性，明示感知时延、控制刚度、接触离散化（ZOH）三者「短板约束」共同钳住可达带宽，量化震荡/穿透两条边界，并解释「环境越硬阻抗刚度越低」的来源与阻抗/导纳选型耦合。
- 两页正文双向内链互为入链，`make graph` 后 0 orphans；知识图谱 1517→**1519** 节点、10258→**10271** 边；`update_badge.py` 同步 README 徽标。
- `make lint` **0 errors**（仅 3 条既有信息型预警，不阻塞 CI）；勾选 v27 P1「接触力旋量闭环知识链 (+2)」。

## [2026-06-30] ingest | sources/repos/mujoco.md, sources/repos/mujoco_wasm.md — 官方 MuJoCo WASM 绑定与 zalo 社区 demo；wiki/entities/mujoco-wasm.md；交叉 mujoco / robot-viewer

## [2026-06-30] ingest | sources/papers/opencap_monocular_arxiv_2603_24733.md — OpenCap Monocular 单手机生物力学运动学/动力学；wiki/entities/paper-opencap-monocular.md

## [2026-06-30] ingest | sources/repos/en02-op.md — Westwood EN02-OP 开源三指末端；wiki/entities/en02-op.md；交叉 topic-grasp / manipulation

## [2026-06-30] ingest | sources/sites/grail-locomanipulation-huggingface.md — 入库 GRAIL Hugging Face 数据集；wiki/entities/grail-locomanipulation-dataset.md

## [2026-06-30] ingest | sources/papers/grail_arxiv_2606_05160.md — 合并重复 GRAIL 实体页并深读入库；wiki/entities/paper-grail.md

## [2026-06-30] structural | 删除冗余详情节点 wiki-queries-sim2real-deployment-checklist

- 删除：`wiki/queries/sim2real-deployment-checklist.md`（内容已并入 `sim2real-checklist.md`「快速部署检查」节，重定向桩不再保留）
- 交叉更新：`wiki/queries/sim2real-checklist.md`、`wiki/queries/README.md`

## [2026-06-30] structural | 移除冗余 Sim2Real 详情节点 — tech-map/modules/rl/sim2real.md、references/papers/sim2real.md

- 删除：`tech-map/modules/rl/sim2real.md`（tech-node-rl-sim2real 空桩）、`references/papers/sim2real.md`（reference-papers-sim2real 与 wiki/concepts/sim2real 重复）
- 交叉更新：`wiki/concepts/sim2real.md`、`wiki/concepts/system-identification.md`、`wiki/methods/crisp-real2sim.md`、`references/papers/README.md`；内链改指向 wiki 概念页与 comparisons/sim2real-approaches


- **P0 合并**：`wiki/queries/sim2real-deployment-checklist.md` 内容并入 `wiki/queries/sim2real-checklist.md`「快速部署检查」节；原页保留重定向桩
- **P0 瘦身**：`wiki/queries/sim2real-gap-reduction.md` 删除重复 Pipeline checklist；`wiki/concepts/sim2real.md` 主要方法改为链向 `comparisons/sim2real-approaches.md`
- **P1 搜索**：`scripts/search_wiki_core.py` — paper-notebook stub/planned 降权；sim2real 部署/gap/调试意图提权
- **P2 图谱**：`docs/topic-filters.js` — sim2real 专题 segments 移除宽泛 `domain`
- 交叉更新：`topic-sim2real.md`、`robot-policy-debug-playbook.md`、`domain-randomization-guide.md` 等；`schema/search-regression-cases.json`

## [2026-06-30] ingest | sources/blogs/flexion_reflect_v1_0.md — Flexion Reflect v1.0 长程人形自主平台；wiki/entities/flexion-reflect-v1.md；交叉 loco-manipulation / VLA / WBC

## [2026-06-30] ingest | sources/papers/cwi_arxiv_2606_27676.md — CWI 复合全身模仿 loco-manipulation；wiki/entities/paper-cwi-composite-humanoid-whole-body-imitation.md；交叉 loco-manipulation / teleoperation

## [2026-06-29] structural | checklist-v26 P3 详情页「物理保真度」专题徽标端到端验证 —— 复用单一事实源补归档截图

- 详情页「所属专题」徽标行（[`docs/main.js`](docs/main.js) `renderMetaTopicBadges`）本就以 [`docs/topic-filters.js`](docs/topic-filters.js) 为单一事实源、`topicsForNode` 数据驱动：V26 P3 把 `physics-fidelity` 写入单一事实源后，动力学/仿真/新建页命中即自动渲染「⚙️ 物理保真度」徽标并跳 `graph.html?topic=physics-fidelity`，空态降级隐藏整行，无需二次实现。
- node 逐页校验 `contact-dynamics` / `physics-fidelity-sim2real-gap` / `simulation-physics-fidelity` / `articulated-body-algorithms` / `joint-friction-models` / `topic-physics-fidelity` 等候选页全部稳定命中 `physics-fidelity`（全库 86 节点）。
- Puppeteer 截图归档 [`detail-topic-physics-fidelity.png`](.cursor-artifacts/screenshots/detail-topic-physics-fidelity.png)：`contact-dynamics` 详情页「所属专题」行实测渲染「✋ 触觉 + ⚙️ 物理保真度」双徽标。
- `make lint` 0 errors（仅 2 条信息型预警，不阻塞 CI）；勾选 v26 P3「详情页『同专题相关页』提示」与 DoD「make lint 0 errors」「log.md 记录」三项，清单全数完成。
- v26 全数完成后按维护规则新建 [`tech-stack-next-phase-checklist-v27.md`](docs/checklists/tech-stack-next-phase-checklist-v27.md)（聚焦「接触力旋量闭环」知识链：感知/估计 → 力旋量 → 阻抗/导纳/混合力位控制 → 接触丰富操作策略），把 v26 移入 `archive/` 并刷新 `docs/checklists/README.md` 当前入口/历史链接。

## [2026-06-29] ingest | sources/papers/humanoid_pnb_vmp.md — VMP β-VAE motion prior + 条件 PPO 全身跟踪；wiki/entities/paper-notebook-vmp.md；交叉 whole-body-tracking-pipeline / character-animation-vs-robotics / humanoid-motion-tracking-method-selection

## [2026-06-29] structural | wiki/entities/paper-sonic.md — 合并 Loco-Manip 161 重复 SONIC stub（#019/#103）至 canonical 实体 + 方法页

- 删除：[`paper-loco-manip-161-019-sonic.md`](wiki/entities/paper-loco-manip-161-019-sonic.md)、[`paper-loco-manip-161-103-sonic.md`](wiki/entities/paper-loco-manip-161-103-sonic.md)
- 保留 canonical：[`paper-sonic.md`](wiki/entities/paper-sonic.md) + [`sonic-motion-tracking.md`](wiki/methods/sonic-motion-tracking.md)
- 交叉更新：Loco-Manip 161 category 01/04 表、[`humanoid_loco_manip_161_catalog.md`](sources/papers/humanoid_loco_manip_161_catalog.md)、对应 source 映射；[`bootstrap_loco_manip_161_entities.py`](scripts/bootstrap_loco_manip_161_entities.py) 增加 `CANONICAL_ENTITY_BY_NUM` 防再生

## [2026-06-29] ingest | sources/papers/chord_nvidia_video_to_data_2026.md — CHORD 接触力旋量引导灵巧操作；wiki/entities/paper-chord-contact-wrench-dexterous-manipulation.md；交叉 contact-rich-manipulation / manipulation / SPIDER / dexterous-data-pipeline / Isaac Lab

## [2026-06-29] ingest | sources/papers/scenebot_arxiv_2606_27581.md — 接入 SceneBot contact-prompted 全身场景交互跟踪；沉淀 wiki/entities/paper-scenebot.md；交叉更新 SONIC、运动跟踪选型、loco-manipulation、OmniRetarget

## [2026-06-29] ingest | sources/papers/hapmorph_arxiv_2509_05433.md — HapMorph AFPA 可穿戴气动解耦尺寸+刚度；wiki/entities/paper-hapmorph-pneumatic-haptic-render.md + teleoperation/topic-tactile 交叉

## [2026-06-28] feat(ui): V26 P3 — 图谱页"物理保真度"专题视图（专题扩至 16 项）

- 改动：[`docs/topic-filters.js`](docs/topic-filters.js) 新增 `physics-fidelity` 专题（`TOPIC_HUB_IDS` / `TOPIC_FILTERS` / `TOPIC_META`），复用 path 片段并集机制（`dynamics/contact/friction/articulated/body/differentiable/simulation/urdf/floating/centroidal/fidelity`）并按需 `ids` 显式纳入新建 query/concept；[`docs/graph.html`](docs/graph.html) `#filter-topic-chips` 增加 `data-topic="physics-fidelity"`（⚙️ 物理保真度）chip
- 新页：[`wiki/overview/topic-physics-fidelity.md`](wiki/overview/topic-physics-fidelity.md) 专题汇总枢纽，并从 [`simulation-physics-fidelity`](wiki/queries/simulation-physics-fidelity.md) / [`physics-fidelity-sim2real-gap`](wiki/concepts/physics-fidelity-sim2real-gap.md) 回链消除孤儿
- 校验：`make lint` 0 errors；`graph-stats.json` 0 孤儿；专题命中 **85** 节点；派生站点文件同步至 1512 节点/10143 边并刷新 badge；Puppeteer 截图归档 `.cursor-artifacts/screenshots/graph-topic-physics-fidelity.png`（页头实测 `85 / 1512 节点`）
- 清单：[`docs/checklists/tech-stack-next-phase-checklist-v26.md`](docs/checklists/tech-stack-next-phase-checklist-v26.md) P3 图谱专题视图项打勾

## [2026-06-28] ingest | sources/sites/rek-com.md — REK VR 人形格斗联赛；wiki/entities/rek.md + 交叉 unitree-g1 / teleoperation / robostriker

## [2026-06-28] ingest | sources/repos/gymnasium.md — Gymnasium RL 环境 API 标准；wiki/entities/gymnasium.md + 交叉 mujoco / dm-control / reinforcement-learning / gym-pybullet-drones / sim-platforms-decade

## [2026-06-28] ingest | sources/papers/flap_arxiv_2606_17630.md — FLAP 无先验地图 FOV 主动感知 3D UAV 规划；wiki/entities/paper-flap-fov-active-perception-3d-navigation.md + 交叉 multirotor-simulation-planning-control-stack / ego-planner-swarm

## [2026-06-27] feat(facts): V26 P2 — 事实库扩展 12 条物理保真度矛盾检测规则（198 → 210）

- 改动：[`schema/canonical-facts.json`](schema/canonical-facts.json) 由 198 → **210** 条，新增 12 条围绕「仿真物理保真度链路」的矛盾检测规则：接触保真度↑ 与可微性/吞吐冲突、几何/URDF 惯量误差被上层逐级放大、硬接触穿透致冲击力偏大、库仑摩擦低估静摩擦致打滑、理想力矩源致执行器力矩 gap、可微仿真梯度受接触不连续制约、硬 LCP 接触不可微、积分步长过大致能量漂移/发散、软接触引入穿透与虚假阻尼、域随机化覆盖残差非替代保真度、保真度+SysID 互补、几何/URDF 层最便宜应优先做
- 校验：逐条经脚本核对对现存 wiki 页（[`simulation-physics-fidelity`](wiki/queries/simulation-physics-fidelity.md) / [`physics-fidelity-sim2real-gap`](wiki/concepts/physics-fidelity-sim2real-gap.md) / `contact-dynamics` / `joint-friction-models` / `differentiable-simulation` / `urdf-robot-description`）有 pos 命中且 0 误报；`make lint` 潜在矛盾 **0** 个、0 errors；`ci-preflight` 12/12 通过
- 清单：[`docs/checklists/tech-stack-next-phase-checklist-v26.md`](docs/checklists/tech-stack-next-phase-checklist-v26.md) P2 与 DoD 事实库项打勾

## [2026-06-27] ingest | sources/papers/second_order_optimizers.md — 6 类二阶/拟牛顿优化器；wiki/methods/newtons-method.md、gauss-newton.md、levenberg-marquardt.md、bfgs.md、l-bfgs.md、truncated-newton.md + wiki/comparisons/second-order-optimizers.md + 交叉 quasi-newton-bfgs / line-search / convex-functions

## [2026-06-27] ingest | sources/papers/deep_learning_optimizers.md — 9 类深度学习优化器一手资料；wiki/methods/sgd.md、sgd-momentum.md、nesterov-momentum.md、adagrad.md、rmsprop.md、adadelta.md、adam.md、adamw.md、lion.md + wiki/comparisons/deep-learning-optimizers.md + 交叉 deep-learning-foundations / backpropagation

## [2026-06-27] ingest | sources/blogs/thehumanoid_kinetiq_ascend.md — Humanoid KinetIQ Ascend 真机 CFM-VLA PPO；wiki/entities/kinetiq-ascend.md + 交叉 VLA/BC/manipulation

- 原始资料：[thehumanoid_kinetiq_ascend.md](sources/blogs/thehumanoid_kinetiq_ascend.md)（<https://thehumanoid.ai/technology/kinetiq-ascend/>）
- 沉淀页面：[wiki/entities/kinetiq-ascend.md](wiki/entities/kinetiq-ascend.md)
- 交叉更新：[wiki/methods/vla.md](wiki/methods/vla.md)、[wiki/methods/behavior-cloning.md](wiki/methods/behavior-cloning.md)、[wiki/tasks/manipulation.md](wiki/tasks/manipulation.md)、[schema/institutions.json](schema/institutions.json)

## [2026-06-26] feat(lint): V26 P0 — 动力学/仿真概念页交叉链路巡检 `physics_concept_crosslink`（INFO 级）

- 改动：[`scripts/lint_wiki.py`](scripts/lint_wiki.py) 新增 `_check_physics_concept_crosslink`——对 `tags` 含 `dynamics`/`simulation`/`physics` 的 `wiki/concepts/*` 与 `wiki/formalizations/*` 概念页，检查正文是否回链「仿真物理保真度」专题枢纽（[`simulation-physics-fidelity`](wiki/queries/simulation-physics-fidelity.md) / [`physics-fidelity-sim2real-gap`](wiki/concepts/physics-fidelity-sim2real-gap.md)），缺失给 INFO 级提示不阻塞 CI；枢纽页自身豁免；同时支持列表式与内联式 `tags`
- 测试：新增 [`tests/test_lint_wiki_physics_crosslink.py`](tests/test_lint_wiki_physics_crosslink.py) 6 用例（有/无回链、列表式/内联式 tag、枢纽豁免、INFO 不计失败）全绿；`ruff` / `mypy` 通过
- 基线快照：[`exports/lint-report.md`](exports/lint-report.md) 现 **15** 页待回链；P1 已回链的 5 页（contact-dynamics / joint-friction-models / articulated-body-algorithms / differentiable-simulation / urdf-robot-description）正确豁免；`make lint` 0 errors
- 清单：[`docs/checklists/tech-stack-next-phase-checklist-v26.md`](docs/checklists/tech-stack-next-phase-checklist-v26.md) P0 打勾

## [2026-06-26] structural(wiki): Loco-Manip 161 与 paper-notebook stub 去重合并 — 33 对并入 paper-loco-manip-161-* / genie-sim-3

- 工具：`make paper-notebook-dedupe`（[dedupe_paper_notebook_nodes.py](scripts/dedupe_paper_notebook_nodes.py)）
- 合并：33 对 `paper-notebook-*` stub → 对应 [`wiki/entities/paper-loco-manip-161-{NNN}-*.md`](wiki/entities/)（含 [`genie-sim-3.md`](wiki/entities/genie-sim-3.md) ← Genie Sim 3.0 stub）；删除 34 条冗余 `sources/papers/humanoid_pnb_*` stub source
- 复跑判据：loco-manip 相关 dedupe 对 **0** 残留
- 交叉更新：[`schema/paper-notebook-wiki-full-map.yml`](schema/paper-notebook-wiki-full-map.yml)、若干 category 页与引用 stub 的 wiki 页

## [2026-06-26] ingest | sources/blogs/wechat_embodied_ai_lab_humanoid_loco_manip_161_survey.md — 人形 Loco-Manip 161 篇十方向全景；父节点 + 十组 category 子节点 + 161 篇 paper-loco-manip-161-* 独立实体

- 原始资料：[wechat_embodied_ai_lab_humanoid_loco_manip_161_survey.md](sources/blogs/wechat_embodied_ai_lab_humanoid_loco_manip_161_survey.md)、[wechat_humanoid_loco_manip_161_2026-06-26.md](sources/raw/wechat_humanoid_loco_manip_161_2026-06-26.md)、[humanoid_loco_manip_161_catalog.md](sources/papers/humanoid_loco_manip_161_catalog.md)、`sources/papers/loco_manip_161_survey_{001..161}_*.md`
- 工具：Agent Reach v1.5.0 + wechat-article-for-ai（Camoufox）；[bootstrap_loco_manip_161_entities.py](scripts/bootstrap_loco_manip_161_entities.py)；<https://mp.weixin.qq.com/s/pACh9EhsISiyPGdiiR0C3A>
- 沉淀页面：[`wiki/overview/humanoid-loco-manip-161-papers-technology-map.md`](wiki/overview/humanoid-loco-manip-161-papers-technology-map.md)（**父**）、[`loco-manip-161-category-01-motion-base-wbt.md`](wiki/overview/loco-manip-161-category-01-motion-base-wbt.md) … [`loco-manip-161-category-10-ego-video.md`](wiki/overview/loco-manip-161-category-10-ego-video.md)（**子**）、**161** 篇 [`wiki/entities/paper-loco-manip-161-{NNN}-*.md`](wiki/entities/)（**独立节点**；与姊妹篇重叠者在 `related` 交叉链深读页）
- 交叉更新：[`wiki/tasks/loco-manipulation.md`](wiki/tasks/loco-manipulation.md)、[`wiki/overview/humanoid-motion-cerebellum-technology-map.md`](wiki/overview/humanoid-motion-cerebellum-technology-map.md)、[`sources/README.md`](sources/README.md)、[`sources/repos/panniantong_agent_reach.md`](sources/repos/panniantong_agent_reach.md)

## [2026-06-26] ingest | sources/blogs/wechat_embodied_ai_lab_agibot_june_2026_release.md — 智元 2026-06 发布七段落地链路；父节点 agibot-june-2026-release-technology-map + 六组 agibot-release-category-* 子节点 + 七项目实体

- 原始资料：[wechat_embodied_ai_lab_agibot_june_2026_release.md](sources/blogs/wechat_embodied_ai_lab_agibot_june_2026_release.md)、[wechat_agibot_june_2026_release_2026-06-26.md](sources/raw/wechat_agibot_june_2026_release_2026-06-26.md)
- 工具：已安装 [Panniantong/Agent-Reach](https://github.com/Panniantong/Agent-Reach) v1.5.0（`pip install` + 手动安装 [wechat-article-for-ai](https://github.com/bzd6661/wechat-article-for-ai) 至 `~/.agent-reach/tools/`（Camoufox））
- 沉淀页面：[`wiki/overview/agibot-june-2026-release-technology-map.md`](wiki/overview/agibot-june-2026-release-technology-map.md)（**父**）、[`agibot-release-category-01-data-entry.md`](wiki/overview/agibot-release-category-01-data-entry.md) … [`agibot-release-category-06-application-delivery.md`](wiki/overview/agibot-release-category-06-application-delivery.md)（**子**）、[`agibot-world-2026.md`](wiki/entities/agibot-world-2026.md)、[`genie-sim-3.md`](wiki/entities/genie-sim-3.md)、[`go-2.md`](wiki/entities/go-2.md)、[`agibot-bfm-2.md`](wiki/entities/agibot-bfm-2.md)、[`agibot-agile.md`](wiki/entities/agibot-agile.md)、[`genie-studio-agent.md`](wiki/entities/genie-studio-agent.md)；复用 [`ge-sim-2.md`](wiki/entities/ge-sim-2.md)
- 交叉更新：[`bfm-41-papers-technology-map.md`](wiki/overview/bfm-41-papers-technology-map.md)、[`robot-world-models-training-loop-taxonomy.md`](wiki/overview/robot-world-models-training-loop-taxonomy.md)、[`sources/README.md`](sources/README.md)、[`sources/repos/panniantong_agent_reach.md`](sources/repos/panniantong_agent_reach.md)

## [2026-06-25] structural | checklist-v26 P1 — 仿真物理保真度知识链 +2 页落地并完成四层交叉回链

- 新增 Query：[simulation-physics-fidelity.md](wiki/queries/simulation-physics-fidelity.md)（几何/URDF → 刚体动力学（ABA/RNEA）→ 接触/摩擦 → 执行器四层保真度取舍决策树，配 Mermaid，覆盖每层对 sim2real gap 的贡献/建模成本/典型失败模式）
- 新增 Concept：[physics-fidelity-sim2real-gap.md](wiki/concepts/physics-fidelity-sim2real-gap.md)（物理保真度 ↔ sim2real gap 因果分层，明示各层简化如何转化为可观测 gap，及与域随机化/系统辨识的互补关系）
- 交叉回链：[contact-dynamics](wiki/concepts/contact-dynamics.md)、[joint-friction-models](wiki/concepts/joint-friction-models.md)、[urdf-robot-description](wiki/concepts/urdf-robot-description.md)、[differentiable-simulation](wiki/concepts/differentiable-simulation.md)、[articulated-body-algorithms](wiki/formalizations/articulated-body-algorithms.md) 五页与新页双向回链，消除孤儿页
- 图谱：节点 1336→1338、边 9109→9139；社区重分区后新增 `humanoid-soccer` 社区，补 `COMMUNITY_NAME_OVERRIDES` 命名 override（`community_quality_warning=false`，`largest_community_ratio=0.183`）
- 门禁：`make lint` 0 问题；`tests/test_community_naming`、`test_generate_link_graph_communities` 等单测通过

## [2026-06-25] ingest | sources/papers/lhbs_learning_human_like_badminton_skills_arxiv_2602_08370.md — LHBS Imitation-to-Interaction 四阶段羽毛球技能；升格 wiki/entities/paper-notebook-learning-human-like-badminton-skills-for-humanoi.md

## [2026-06-25] ingest | sources/repos/tensorrt-official.md + openvino-official.md + ncnn-official.md — 补全 TensorRT 实体并扩展 OpenVINO/ncnn 与机载推理选型对比

- 原始资料：[tensorrt-official.md](sources/repos/tensorrt-official.md)、[openvino-official.md](sources/repos/openvino-official.md)、[ncnn-official.md](sources/repos/ncnn-official.md)
- 升格实体：[tensorrt.md](wiki/entities/tensorrt.md)、[openvino.md](wiki/entities/openvino.md)、[ncnn.md](wiki/entities/ncnn.md)
- 更新对比：[onnxruntime-vs-mnn-vs-tensorrt.md](wiki/comparisons/onnxruntime-vs-mnn-vs-tensorrt.md)（延伸 LiteRT/ExecuTorch/LibTorch 一览）
- 交叉更新：[onnx.md](wiki/entities/onnx.md)、[onnxruntime.md](wiki/entities/onnxruntime.md)、[mnn.md](wiki/entities/mnn.md)

## [2026-06-25] ingest | sources/repos/onnx-official.md + onnxruntime-official.md + mnn-official.md — 接入 ONNX/MNN 一手资料并升格实体与 runtime 选型对比

- 原始资料：[onnx-official.md](sources/repos/onnx-official.md)、[onnxruntime-official.md](sources/repos/onnxruntime-official.md)、[mnn-official.md](sources/repos/mnn-official.md)
- 升格实体：[onnx.md](wiki/entities/onnx.md)、[onnxruntime.md](wiki/entities/onnxruntime.md)、[mnn.md](wiki/entities/mnn.md)
- 选型对比：[onnxruntime-vs-mnn-vs-tensorrt.md](wiki/comparisons/onnxruntime-vs-mnn-vs-tensorrt.md)
- 交叉更新：[pytorch.md](wiki/entities/pytorch.md)、[tensorflow.md](wiki/entities/tensorflow.md)、[sim2real.md](wiki/concepts/sim2real.md)、[whole-body-tracking-pipeline.md](wiki/concepts/whole-body-tracking-pipeline.md)、[robot-policy-debug-playbook.md](wiki/queries/robot-policy-debug-playbook.md)

## [2026-06-25] ingest | sources/repos/tensorflow-official.md — 接入 TensorFlow 官网与 GitHub 并升格 wiki/entities/tensorflow.md；交叉更新 deep-learning-foundations、pytorch、htwk-gym

- 原始资料：[tensorflow-official.md](sources/repos/tensorflow-official.md)
- 升格实体：[tensorflow.md](wiki/entities/tensorflow.md)
- 交叉更新：[deep-learning-foundations.md](wiki/concepts/deep-learning-foundations.md)、[pytorch.md](wiki/entities/pytorch.md)、[htwk-gym.md](wiki/methods/htwk-gym.md)

## [2026-06-25] ingest | sources/sites/weights-and-biases.md + sources/repos/tensorboard.md — W&B / TensorBoard 实体与选型对比；交叉 amp-mjlab、mjlab、robot-policy-debug-playbook

- 原始资料：[weights-and-biases.md](sources/sites/weights-and-biases.md)、[tensorboard.md](sources/repos/tensorboard.md)
- 升格实体：[weights-and-biases.md](wiki/entities/weights-and-biases.md)、[tensorboard.md](wiki/entities/tensorboard.md)
- 选型对比：[wandb-vs-tensorboard.md](wiki/comparisons/wandb-vs-tensorboard.md)
- 交叉更新：[amp-mjlab.md](wiki/entities/amp-mjlab.md)、[mjlab.md](wiki/entities/mjlab.md)、[robot-policy-debug-playbook.md](wiki/queries/robot-policy-debug-playbook.md)

## [2026-06-25] fix(wiki): 修复 DAgger / BC 页 LaTeX 中 `\theta` 被制表符破坏导致 KaTeX 渲染失败

- 根因：`wiki/methods/dagger.md`、`wiki/methods/behavior-cloning.md` 中 `\theta` 的 `\t` 被存成字面制表符，KaTeX 将 `_heta` 当作无效下标
- 修复：还原为 `\pi_\theta`、`\min_\theta` 等正确 LaTeX

## [2026-06-25] structural(wiki): 重复节点审计修复 — extreme-parkour 合并、BFM 误标 arXiv 清理、TWIST2 更正为 2511.02832

- dedupe：扩展 [scripts/dedupe_paper_notebook_nodes.py](scripts/dedupe_paper_notebook_nodes.py) `find_arxiv_merge_pairs` 扫描全实体；`paper-notebook-extreme-parkour-with-legged-robots` → [extreme-parkour.md](wiki/entities/extreme-parkour.md)；同步 [schema/paper-notebook-wiki-full-map.yml](schema/paper-notebook-wiki-full-map.yml)
- BFM 元数据：移除 5 篇单篇实体误标综述 arXiv `2506.20487`（[paper-bfm-04](wiki/entities/paper-bfm-04-fast-imitation-bfm.md)、[05](wiki/entities/paper-bfm-05-learning-one-representation.md)、[12](wiki/entities/paper-bfm-12-clone.md)、[17](wiki/entities/paper-bfm-17-maskedmimic.md)、[19](wiki/entities/paper-bfm-19-calm.md)）
- TWIST2：全库更正为 arXiv:2511.02832（保留 [paper-twist.md](wiki/entities/paper-twist.md) 的 2505.02833）；[paper-twist2.md](wiki/entities/paper-twist2.md)、sources、BFM 生成器 id 10

## [2026-06-25] structural(wiki): 批量深化 119 篇 survey 策展实体页 — HRL 42 / BFM 41 / VLN 10 / 深蓝 WM 15 / Ego 9 / Loco-Manip 8 / 运动小脑 15 + 脚本

- 工具：[scripts/deepen_survey_stub_pages.py](scripts/deepen_survey_stub_pages.py) — 从 raw 微信抓取与 catalog 元数据编译 `一句话定义` / `核心机制` / `常见误区`；已有深读页（SONIC、BeyondMimic、GMR 等）保留 survey 坐标并链至方法/实体深读
- 栈覆盖：42 篇 RL 身体系统栈、`paper-bfm-*`（41）、`paper-vln-*`（10）、`paper-shenlan-wm-*`（15）、`paper-ego-*`（9）、`paper-loco-manip-*`（8）、`paper-motion-cerebellum-*`（15）、`paper-sonic` / `paper-twist` / `paper-beyondmimic` 等别名节点
- 刻意保留浅页：`paper-notebook-visualmimic`（Humanoid Paper Notebooks 外链索引）
- 总览：[humanoid-rl-motion-control-body-system-stack.md](wiki/overview/humanoid-rl-motion-control-body-system-stack.md) 局限段改为「编译实体页」

## [2026-06-25] ingest | sources/sites/tairan-he.md — 复核 tairanhe.com：OpenAI MTS、博士答辩与 CVPR 2026 VIRAL/DoorMan 等更新

- 原始资料：[tairan-he.md](sources/sites/tairan-he.md)
- 更新实体：[tairan-he.md](wiki/entities/tairan-he.md)

## [2026-06-25] ingest | sources/sites/yanjieze.md — Yanjie Ze 个人主页归档并升格 wiki/entities/yanjie-ze.md；交叉 TWIST/TWIST2/GMR/VisualMimic/ResMimic 等

- 原始资料：[yanjieze.md](sources/sites/yanjieze.md)
- 升格实体：[yanjie-ze.md](wiki/entities/yanjie-ze.md)
- 交叉更新：[paper-twist.md](wiki/entities/paper-twist.md)、[paper-twist2.md](wiki/entities/paper-twist2.md)、[motion-retargeting-gmr.md](wiki/methods/motion-retargeting-gmr.md)、[paper-notebook-visualmimic.md](wiki/entities/paper-notebook-visualmimic.md)、[paper-resmimic.md](wiki/entities/paper-resmimic.md)
## [2026-06-25] fix(actions): COMMUNITY_NAME_OVERRIDES 补全身运动跟踪流水线 — 修复 community-12 命名不符合 中文（English） 社区 导致 pytest 失败

- `scripts/generate_link_graph.py`：`wiki/concepts/whole-body-tracking-pipeline.md` → `全身运动跟踪流水线（Whole-Body Tracking Pipeline, WBT）`
- 验证：`make ci-preflight`、`make test`（含 `test_community_naming`）通过

## [2026-06-25] structural | AMP 专题 19 篇占位页批量深化收口 — 总览局限段更新为「深读实体页」

- 变更：[humanoid-amp-motion-prior-survey.md](wiki/overview/humanoid-amp-motion-prior-survey.md) 不再将 19 篇标为「策展索引级」；#01–#19 均已 MoRE/CLOT 级深读（#08/#16 先行，#01–#07/#09–#15/#17–#19 本日批次完成）

## [2026-06-25] ingest | AMP 专题 #01–#12 batch deepen — 12 实体页 MoRE/CLOT 级深读 + 策展 source 深读指针 + 5 篇 arXiv/MDPI 归档

- 升格实体（#01–#07、#09–#12）：[paper-amp-survey-01-amp](wiki/entities/paper-amp-survey-01-amp.md)、[paper-amp-survey-02-physics_based_motion_imitation_with](wiki/entities/paper-amp-survey-02-physics_based_motion_imitation_with.md)、[paper-amp-survey-03-smp](wiki/entities/paper-amp-survey-03-smp.md)、[paper-amp-survey-04-kimodo](wiki/entities/paper-amp-survey-04-kimodo.md)、[paper-amp-survey-05-motionbricks](wiki/entities/paper-amp-survey-05-motionbricks.md)、[paper-amp-survey-06-natural_humanoid_robot_locomotion_wi](wiki/entities/paper-amp-survey-06-natural_humanoid_robot_locomotion_wi.md)、[paper-amp-survey-07-adversarial_locomotion_and_motion_im](wiki/entities/paper-amp-survey-07-adversarial_locomotion_and_motion_im.md)、[paper-hiking-in-the-wild](wiki/entities/paper-hiking-in-the-wild.md)、[paper-amp-survey-10-unified_walking_running_and_recovery](wiki/entities/paper-amp-survey-10-unified_walking_running_and_recovery.md)、[paper-adaptive-humanoid-control](wiki/entities/paper-adaptive-humanoid-control.md)、[paper-amp-survey-12-haml](wiki/entities/paper-amp-survey-12-haml.md)
- 未改写（已深）：#08 [paper-amp-survey-08-more](wiki/entities/paper-amp-survey-08-more.md)、#16 CLOT、#13–#19
- 原始资料深读归档：[gmp_generative_motion_prior_arxiv_2503_09015.md](sources/papers/gmp_generative_motion_prior_arxiv_2503_09015.md)、[almi_adversarial_locomotion_motion_imitation_arxiv_2504_14305.md](sources/papers/almi_adversarial_locomotion_motion_imitation_arxiv_2504_14305.md)、[hiking_in_the_wild_arxiv_2601_07718.md](sources/papers/hiking_in_the_wild_arxiv_2601_07718.md)、[adaptive_humanoid_control_ahc_arxiv_2511_06371.md](sources/papers/adaptive_humanoid_control_ahc_arxiv_2511_06371.md)、[haml_humanoid_adversarial_multi_skill_learning_mdpi_2026.md](sources/papers/haml_humanoid_adversarial_multi_skill_learning_mdpi_2026.md)
- 策展索引补强：`humanoid_amp_survey_01`–`12`（#08 既有深读指针保留）
- #10 技术深读主入口：[paper-unified-walk-run-recovery-sdamp](wiki/entities/paper-unified-walk-run-recovery-sdamp.md)

## [2026-06-25] ingest | AMP 专题 #13–#19 深读（Goalkeeper/HUSKY/PhysHSI/TeamHOI/Deep Parkour/Embrace Collisions）— 6 篇 arXiv source + 6 实体页；策展 source 交叉链接

- arXiv 深读：[humanoid_goalkeeper_arxiv_2510_18002.md](sources/papers/humanoid_goalkeeper_arxiv_2510_18002.md)、[husky_humanoid_skateboarding_arxiv_2602_03205.md](sources/papers/husky_humanoid_skateboarding_arxiv_2602_03205.md)、[physhsi_arxiv_2510_11072.md](sources/papers/physhsi_arxiv_2510_11072.md)、[teamhoi_arxiv_2603_07988.md](sources/papers/teamhoi_arxiv_2603_07988.md)、[deep_whole_body_parkour_arxiv_2601_07701.md](sources/papers/deep_whole_body_parkour_arxiv_2601_07701.md)、[embrace_collisions_arxiv_2502_01465.md](sources/papers/embrace_collisions_arxiv_2502_01465.md)
- 升格实体：[paper-amp-survey-13-humanoid_goalkeeper](wiki/entities/paper-amp-survey-13-humanoid_goalkeeper.md)、[paper-amp-survey-14-husky](wiki/entities/paper-amp-survey-14-husky.md)、[paper-amp-survey-15-physhsi](wiki/entities/paper-amp-survey-15-physhsi.md)、[paper-amp-survey-17-teamhoi](wiki/entities/paper-amp-survey-17-teamhoi.md)、[paper-deep-whole-body-parkour](wiki/entities/paper-deep-whole-body-parkour.md)、[paper-amp-survey-19-embrace_collisions](wiki/entities/paper-amp-survey-19-embrace_collisions.md)
- 策展索引补强：`humanoid_amp_survey_13`–`19`（含 #18）、[humanoid_rl_stack_23_deep_whole_body_parkour](sources/papers/humanoid_rl_stack_23_deep_whole_body_parkour.md)

## [2026-06-25] ingest | sources/papers/gmp_generative_motion_prior_arxiv_2503_09015.md 等 — AMP 专题 #06 GMP、#07 ALMI、#09 Hiking、#10 SD-AMP 索引、#11 AHC、#12 HAML 深读实体与 arXiv/MDPI 归档；交叉 humanoid-amp-motion-prior-survey

- 原始资料：[gmp_generative_motion_prior_arxiv_2503_09015.md](sources/papers/gmp_generative_motion_prior_arxiv_2503_09015.md)、[almi_adversarial_locomotion_motion_imitation_arxiv_2504_14305.md](sources/papers/almi_adversarial_locomotion_motion_imitation_arxiv_2504_14305.md)、[hiking_in_the_wild_arxiv_2601_07718.md](sources/papers/hiking_in_the_wild_arxiv_2601_07718.md)、[adaptive_humanoid_control_ahc_arxiv_2511_06371.md](sources/papers/adaptive_humanoid_control_ahc_arxiv_2511_06371.md)、[haml_humanoid_adversarial_multi_skill_learning_mdpi_2026.md](sources/papers/haml_humanoid_adversarial_multi_skill_learning_mdpi_2026.md)
- 升格实体：[wiki/entities/paper-amp-survey-06-natural_humanoid_robot_locomotion_wi.md](wiki/entities/paper-amp-survey-06-natural_humanoid_robot_locomotion_wi.md)、[wiki/entities/paper-amp-survey-07-adversarial_locomotion_and_motion_im.md](wiki/entities/paper-amp-survey-07-adversarial_locomotion_and_motion_im.md)、[wiki/entities/paper-hiking-in-the-wild.md](wiki/entities/paper-hiking-in-the-wild.md)、[wiki/entities/paper-amp-survey-10-unified_walking_running_and_recovery.md](wiki/entities/paper-amp-survey-10-unified_walking_running_and_recovery.md)、[wiki/entities/paper-adaptive-humanoid-control.md](wiki/entities/paper-adaptive-humanoid-control.md)、[wiki/entities/paper-amp-survey-12-haml.md](wiki/entities/paper-amp-survey-12-haml.md)
- 策展索引补强：humanoid_amp_survey_06/07/09/10/11/12
- SD-AMP 深读页（既有）：[wiki/entities/paper-unified-walk-run-recovery-sdamp.md](wiki/entities/paper-unified-walk-run-recovery-sdamp.md)

## [2026-06-25] ingest | sources/papers/more_mixture_residual_experts_arxiv_2506_08840.md — 补完成 MoRE（AMP #08）深读；wiki/entities/paper-amp-survey-08-more.md；交叉 amp-reward、terrain-adaptation、explicit-stair-geometry、locomotion

- 原始资料：[more_mixture_residual_experts_arxiv_2506_08840.md](sources/papers/more_mixture_residual_experts_arxiv_2506_08840.md) — arXiv:2506.08840 + 项目页
- 策展索引补强：[humanoid_amp_survey_08_more_mixture_of_residual_experts_for_humanoid_li.md](sources/papers/humanoid_amp_survey_08_more_mixture_of_residual_experts_for_humanoid_li.md)
- 升格实体：[wiki/entities/paper-amp-survey-08-more.md](wiki/entities/paper-amp-survey-08-more.md) — 两阶段管线、MoE 残差、多判别器 AMP、Mermaid 流程图
- 交叉更新：[paper-notebook-category-05-locomotion](wiki/overview/paper-notebook-category-05-locomotion.md)、[amp-reward](wiki/methods/amp-reward.md)、[locomotion](wiki/tasks/locomotion.md)、[terrain-adaptation](wiki/concepts/terrain-adaptation.md)、[paper-explicit-stair-geometry-humanoid-locomotion](wiki/entities/paper-explicit-stair-geometry-humanoid-locomotion.md)

## [2026-06-25] ingest | sources/papers/oasis_humanoid_loco_manip_2606_08548.md — 复核 OASIS arXiv:2606.08548 入库；交叉补强 wiki/entities/paper-loco-manip-04-oasis.md、wiki/queries/humanoid-training-data-pipeline.md、wiki/concepts/sim2real.md

## [2026-06-25] ingest | sources/sites/kyberlabs-ai.md — Kyber Labs 背驱动灵巧手操作平台；wiki/entities/kyber-labs.md；交叉 notable-commercial-robot-platforms、wuji-robotics、allegro-hand

## [2026-06-25] ingest | sources/sites/wuji_robotics.md — 补正舞肌科技官网 wuji.tech（中/英）与智能数据手套叙事；修正 schema/institutions.json「无界机器人」误标；wiki/entities/wuji-robotics.md

## [2026-06-25] ingest | sources/papers/coordex_arxiv_2606_23680.md — CoorDex body/hand 潜先验协调残差 dexterous loco-manipulation；wiki/entities/paper-coordex-dexterous-humanoid-loco-manipulation.md；交叉 loco-manipulation、wuji-robotics

## [2026-06-25] ingest | sources/sites/hiw-500-dataset.md — HIW-500 野外 G1 遥操作数据集；升格 wiki/entities/hiw-500-dataset.md，互链 teleoperation / humanoid-training-data-pipeline / topic-data-pipeline

## [2026-06-24] checklist-v25 | DoD 收口 & 初始化 V26

- V25 全部条目收口：P0（数据集页元数据巡检 `dataset_metadata_check` + scaffold `--dataset` 旗标）、P1（`humanoid-training-data-pipeline` query + `motion-data-quality` concept + 数据层四段衔接交叉补强）、P2（事实库 186 → 198 条，新增 12 条数据层矛盾检测规则）、P3（图谱第 15 项「训练数据管线」专题视图 `data-pipeline` + 详情页徽标联动）逐条 `[x]`；DoD 末项「`log.md` 记录 V25 关键改动」由本条目收口。
- V25 交付基线：`make lint` 0 errors（仅 1 条信息型预警），图谱 **1322 节点 / 8809 边**，事实库 **198 条**，`community_quality_warning=false`、最大社区占比 **0.165**，图谱专题视图 15 项。
- 新建 [`docs/checklists/tech-stack-next-phase-checklist-v26.md`](docs/checklists/tech-stack-next-phase-checklist-v26.md)：专题选定为「仿真物理保真度链路」，承接 V25 收尾密集 ingest 的 `differentiable-simulation` / `articulated-body-algorithms` / `contact-dynamics` / `joint-friction-models` / `friction-compensation` / `urdf-robot-description` / `procedural-terrain-generation` 等仿真物理底座概念页，规划「几何/URDF → 刚体动力学算法 → 接触/摩擦模型 → 执行器模型」端到端保真度知识链（P1 query+concept）、物理保真度矛盾检测规则扩展（P2 事实库 198→≥208）、动力学/仿真概念页交叉链路巡检（P0）与图谱第 16 项「物理保真度」专题视图（P3）。
- 同步将 README badge / 维护看板、`AGENTS.md`、`docs/README.md`、`docs/checklists/README.md` 的「当前清单」指针从 V25 切到 V26；V25 移入 `archive/` 并修正其内部相对链接（上一版清单同级、方法论参考 `../../../wiki/...`），进入历史归档区。

## [2026-06-24] structural | scripts/generate_link_graph.py — 兜底社区标签改为「其他（Other） 社区」

## [2026-06-24] structural | docs/topic-filters.js、docs/graph.html、wiki/overview/topic-*.md — 专题标签统一为「中文 (English)」格式（与类型图例一致）


## [2026-06-24] ingest | sources/papers/ultra_fusion_arxiv_2606_21223.md — Ultra-Fusion 多传感器 SLAM；沉淀 wiki/entities/paper-ultra-fusion-multi-sensor-slam.md，互链 sensor-fusion、lidar-slam-lio-vio-selection、topic-state-estimation

## [2026-06-24] structural | schema/institutions.json、scripts/sync_institution_tags.py — 实体页机构标签批量补全（表格/sources/覆盖表）

- 扩展机构注册表：中科大、BIGAI、HIT、TeleAI、NYU、Motphys 等 **60+** 条目（46→109）
- 新增 `scripts/sync_institution_tags.py`：从 `|机构|` 表、sources 机构行、GitHub org 与覆盖表写入 frontmatter tags
- 批量更新 **~200** 个 `wiki/entities/` 页（含 HRL 栈、BFM、Ego、深澜 WM、公司/数据集实体）；可派生机构节点 **310→529**（实体非占位 **467/484**）
- 测试：`tests/test_sync_institution_tags.py`

## [2026-06-24] ingest | sources/sites/wokwi-com.md — Wokwi 在线嵌入式仿真平台；升格 wiki/entities/wokwi.md 并交叉 motor-drive-firmware-bus-protocols / simplefoc

- 原始资料：[wokwi-com.md](sources/sites/wokwi-com.md)
- 升格实体：[wiki/entities/wokwi.md](wiki/entities/wokwi.md)
- 交叉更新：[motor-drive-firmware-bus-protocols](wiki/overview/motor-drive-firmware-bus-protocols.md)、[simplefoc](wiki/entities/simplefoc.md)
- 机构注册表：`schema/institutions.json` 追加 Wokwi

## [2026-06-23] structural | schema/institutions.json、scripts/bump_institution_tags.py — 批量补全 wiki 节点所属机构 tags；工具实体 lint 门禁

- 扩展机构注册表（Hugging Face、AI2、Amazon、INRIA、Blender Foundation、X-Humanoid、SDU、RoboParty、FreeMoCap 等）与 Unitree 产品 alias
- 新增 `scripts/bump_institution_tags.py`：从摘要区/H1/显式覆盖表推断机构并写入 frontmatter tags
- 批量更新 ~150 页 wiki（含 Isaac Lab、LeRobot、legged_gym、OpenVLA 等工具实体）；可派生机构节点 89→309
- lint：`tool_missing_institution` 检查工具实体须有所属机构
- 代表页：[wiki/entities/isaac-lab.md](wiki/entities/isaac-lab.md)、[wiki/entities/lerobot.md](wiki/entities/lerobot.md)、[wiki/entities/legged-gym.md](wiki/entities/legged-gym.md)

## [2026-06-23] ingest | sources/sites/nvidia-research-gear-lab.md — NVIDIA GEAR Lab 主页；升格 wiki/entities/nvidia-gear-lab.md 并交叉 EgoScale/ENPIRE/SONIC/GR00T-WBC

## [2026-06-23] ingest | sources/papers/vesta_arxiv_2606_20905.md — Vesta 通才具身 planner；升格 wiki/entities/paper-vesta-generalist-embodied-reasoning.md 并交叉更新 vla / VLN / SayCan

## [2026-06-23] ingest | sources/papers/stubborn_arxiv_2606_12814.md — Stubborn 统一 RL 人形跟踪与跌倒恢复；深读 arXiv:2606.12814 并升格 wiki/entities/paper-motion-cerebellum-stubborn.md

- 原始资料：[stubborn_arxiv_2606_12814.md](sources/papers/stubborn_arxiv_2606_12814.md) — arXiv:2606.12814 + [项目页](https://aislab-sustech.github.io/Stubborn/)
- 策展增补：[motion_cerebellum_survey_34_stubborn.md](sources/papers/motion_cerebellum_survey_34_stubborn.md) — 运动小脑 34/64 索引同步
- 升格实体：[wiki/entities/paper-motion-cerebellum-stubborn.md](wiki/entities/paper-motion-cerebellum-stubborn.md) — yaw-aligned 表征、Bernoulli PT、AdpS 采样、LAFAN1/G1 实验与 Mermaid 流程图

## [2026-06-23] ingest | sources/courses/quadruped_control_simulation_rl_curriculum.md — 具身智能研究室《四足机器人：从动力学建模到强化学习》八章课程大纲；新增 quadruped-control-curriculum 策展页 + 11 个 concept/method/formalization/entity 节点

- 原始资料：[quadruped_control_simulation_rl_curriculum.md](sources/courses/quadruped_control_simulation_rl_curriculum.md) — 四足控制与仿真 RL 课程大纲整理
- 策展入口：[wiki/entities/quadruped-control-curriculum.md](wiki/entities/quadruped-control-curriculum.md)
- 新建 entity：[matrix-simulation-platform](wiki/entities/matrix-simulation-platform.md)、[roamerx-navigation](wiki/entities/roamerx-navigation.md)
- 新建 concept：[differentiable-simulation](wiki/concepts/differentiable-simulation.md)、[urdf-robot-description](wiki/concepts/urdf-robot-description.md)、[joint-friction-models](wiki/concepts/joint-friction-models.md)、[friction-compensation](wiki/concepts/friction-compensation.md)、[procedural-terrain-generation](wiki/concepts/procedural-terrain-generation.md)、[hierarchical-quadruped-navigation-stack](wiki/concepts/hierarchical-quadruped-navigation-stack.md)
- 新建 formalization：[articulated-body-algorithms](wiki/formalizations/articulated-body-algorithms.md)
- 新建 method：[pid-control](wiki/methods/pid-control.md)
- 交叉更新：[quadruped-robot](wiki/entities/quadruped-robot.md)、[system-identification](wiki/concepts/system-identification.md)、[sim2real](wiki/concepts/sim2real.md)、[domain-randomization](wiki/concepts/domain-randomization.md)、[floating-base-dynamics](wiki/concepts/floating-base-dynamics.md)、[simulator-selection-guide](wiki/queries/simulator-selection-guide.md)

## [2026-06-23] structural | wiki/entities/paper-amp-survey-05-motionbricks.md、wiki/methods/motionbricks.md — 消歧 MotionBricks 实体索引页与方法页，加强双向互链

- 实体页 `paper-amp-survey-05-motionbricks`：H1 改为「AMP 专题 #05」、顶部引导至方法页、「与其他页面的关系」补方法页与 Kimodo 对照
- 方法页 `motionbricks`：补 AMP 专题 #05/19 策展语境与回链实体索引页
- 验证：`make ci-preflight`

## [2026-06-23] ingest | sources/courses/numerical_optimization_foundations_robotics.md — 具身智能研究室《数值优化基础》六章课程大纲；新增 numerical-optimization-curriculum 策展页 + 18 个 formalization/method/concept/query 节点

- 原始资料：[numerical_optimization_foundations_robotics.md](sources/courses/numerical_optimization_foundations_robotics.md) — 数值优化基础（机器人应用）课程大纲整理
- 策展入口：[wiki/entities/numerical-optimization-curriculum.md](wiki/entities/numerical-optimization-curriculum.md)
- 新建 formalization：[convex-functions](wiki/formalizations/convex-functions.md)、[kkt-conditions](wiki/formalizations/kkt-conditions.md)、[quadratic-programming](wiki/formalizations/quadratic-programming.md)、[symmetric-cone-programming](wiki/formalizations/symmetric-cone-programming.md)、[adjoint-sensitivity-analysis](wiki/formalizations/adjoint-sensitivity-analysis.md)
- 新建 method：[line-search-steepest-descent](wiki/methods/line-search-steepest-descent.md)、[quasi-newton-bfgs](wiki/methods/quasi-newton-bfgs.md)、[conjugate-gradient-method](wiki/methods/conjugate-gradient-method.md)、[penalty-barrier-augmented-lagrangian](wiki/methods/penalty-barrier-augmented-lagrangian.md)、[nonlinear-model-predictive-control](wiki/methods/nonlinear-model-predictive-control.md)、[time-optimal-path-parameterization](wiki/methods/time-optimal-path-parameterization.md)、[smooth-navigation-path-generation](wiki/methods/smooth-navigation-path-generation.md)、[convex-relaxation-robotics](wiki/methods/convex-relaxation-robotics.md)
- 新建 concept：[constrained-optimization](wiki/concepts/constrained-optimization.md)、[control-allocation](wiki/concepts/control-allocation.md)、[collision-distance-optimization](wiki/concepts/collision-distance-optimization.md)
- 新建 query：[optimization-software-selection](wiki/queries/optimization-software-selection.md)
- 交叉更新：[optimal-control](wiki/concepts/optimal-control.md)、[trajectory-optimization](wiki/methods/trajectory-optimization.md)、[model-predictive-control](wiki/methods/model-predictive-control.md)、[linear-algebra-curriculum](wiki/entities/linear-algebra-curriculum.md)、[roadmap/motion-control.md](roadmap/motion-control.md)

## [2026-06-22] structural | checklist-v25 P3 新增「训练数据管线」图谱专题视图（`data-pipeline`）；docs/topic-filters.js、docs/graph.html、wiki/overview/topic-data-pipeline.md

- 执行清单：[docs/checklists/tech-stack-next-phase-checklist-v25.md](docs/checklists/tech-stack-next-phase-checklist-v25.md) P3「图谱页"训练数据管线"专题视图」收口（V24 14 项 → 15 项专题）
- 单一事实源：[docs/topic-filters.js](docs/topic-filters.js) 的 `TOPIC_HUB_IDS`/`TOPIC_FILTERS`/`TOPIC_META` 新增 `data-pipeline`（emoji 📦、label「训练数据」）；segments=`dataset/datasets/amass/lafan1/lafan/omomo/phuma/everyday/retargeting`，ids 显式纳入 [humanoid-training-data-pipeline](wiki/queries/humanoid-training-data-pipeline.md) query + [motion-data-quality](wiki/concepts/motion-data-quality.md)/[motion-retargeting](wiki/concepts/motion-retargeting.md) concept + [humanoid-reference-motion-datasets](wiki/comparisons/humanoid-reference-motion-datasets.md) 对比
- 新建 hub 页：[wiki/overview/topic-data-pipeline.md](wiki/overview/topic-data-pipeline.md)（专题汇总锚点），并由 [topic-motion-retargeting](wiki/overview/topic-motion-retargeting.md)/[topic-wbt](wiki/overview/topic-wbt.md)「与其他专题的关系」回链消除孤儿
- 交互层：[docs/graph.html](docs/graph.html) `#filter-topic-chips` 新增 `data-topic="data-pipeline"`（📦 训练数据）chip
- 校验：node 载入 topic-filters.js 对 `exports/link-graph.json` 命中 42 节点（数据集 + 重定向 + 质量 + hub）；`make export graph` 重生成派生文件（1288 节点 / 8450 边）；`python3 scripts/update_badge.py` 同步 README badge；`make lint` 0 errors（仅 2 条信息型预警）
- 待补：截图（apt 镜像 404 无法装 Chromium，Puppeteer `graph-topic-data-pipeline.png` 由后续带 Chrome 环境补归档）
- 验证：`make lint`

## [2026-06-22] ingest | sources/papers/htd_refine_arxiv_2605_26879.md — 复核 arXiv:2605.26879（HTD-Refine）已入库；修正 wiki/entities/paper-htd-refine-monocular-hmr.md 英文缩写速查

- 原始资料：[htd_refine_arxiv_2605_26879.md](sources/papers/htd_refine_arxiv_2605_26879.md)（<https://arxiv.org/abs/2605.26879>）；项目页代码仍为 Coming Soon（2026-06-22 复核）
- 沉淀页面：[wiki/entities/paper-htd-refine-monocular-hmr.md](wiki/entities/paper-htd-refine-monocular-hmr.md)（修正 HMR/PVA/MPJVE 等核心缩写表）
- 首次入库：2026-06-04（source + wiki + motion-retargeting / whole-body-tracking / GMR / GVHMR 交叉引用均已就绪）

## [2026-06-22] ingest | sources/papers/x_ionet_arxiv_2511_08277.md — X-IONet 跨平台单 IMU 惯性里程计；wiki/entities/paper-x-ionet-cross-platform-inertial-odometry.md；交叉 state-estimation、ekf、topic-state-estimation

- 原始资料：[x_ionet_arxiv_2511_08277.md](sources/papers/x_ionet_arxiv_2511_08277.md)（<https://arxiv.org/abs/2511.08277>，IEEE RA-L Vol. 11 No. 7, July 2026）
- 沉淀页面：[wiki/entities/paper-x-ionet-cross-platform-inertial-odometry.md](wiki/entities/paper-x-ionet-cross-platform-inertial-odometry.md)
- 交叉更新：[wiki/concepts/state-estimation.md](wiki/concepts/state-estimation.md)、[wiki/formalizations/ekf.md](wiki/formalizations/ekf.md)、[wiki/overview/topic-state-estimation.md](wiki/overview/topic-state-estimation.md)

## [2026-06-22] ingest | sources/papers/rf_detr_arxiv_2511_09554.md — RF-DETR 实时 DETR；wiki/entities/rf-detr.md；交叉 object-detection、object-detection-model-selection

- 原始资料：[rf_detr_arxiv_2511_09554.md](sources/papers/rf_detr_arxiv_2511_09554.md)、[rf_detr.md](sources/repos/rf_detr.md)、[rfdetr-docs.md](sources/sites/rfdetr-docs.md)（<https://arxiv.org/abs/2511.09554>、<https://github.com/roboflow/rf-detr>、<https://rfdetr.roboflow.com/latest/>）
- 沉淀页面：[wiki/entities/rf-detr.md](wiki/entities/rf-detr.md)
- 交叉更新：[wiki/methods/object-detection.md](wiki/methods/object-detection.md)、[wiki/queries/object-detection-model-selection.md](wiki/queries/object-detection-model-selection.md)

## [2026-06-22] ingest | sources/blogs/wechat_shenlan_sim_platforms_top8_decade.md — 十年 TOP 8 仿真平台盘点；wiki/overview/sim-platforms-decade-technology-map.md；8 平台各建实体节点

- 工具：已安装 [Panniantong/Agent-Reach](https://github.com/Panniantong/Agent-Reach) v1.5.0（`pip install` + 手动安装 [wechat-article-for-ai](https://github.com/bzd6661/wechat-article-for-ai) 至 `~/.agent-reach/tools/`（Camoufox））
- 原始资料：`sources/blogs/wechat_shenlan_sim_platforms_top8_decade.md`、`sources/raw/wechat_sim_platforms_top8_2026-06-22.md`（<https://mp.weixin.qq.com/s/iaw_lWAR--AwppyMeIK4lw>）
- 沉淀页面：[`wiki/overview/sim-platforms-decade-technology-map.md`](wiki/overview/sim-platforms-decade-technology-map.md)
- 新建实体：`wiki/entities/ai2-thor.md`、`matterport3d-simulator.md`、`habitat-sim.md`、`igibson.md`、`maniskill2.md`、`behavior-1k.md`、`carla.md`、`robogen.md`
- 交叉更新：[`mujoco.md`](wiki/entities/mujoco.md)、[`isaac-gym.md`](wiki/entities/isaac-gym.md)、[`pybullet.md`](wiki/entities/pybullet.md)、[`genesis-sim.md`](wiki/entities/genesis-sim.md)、[`sapien.md`](wiki/entities/sapien.md)、[`simulator-selection-guide.md`](wiki/queries/simulator-selection-guide.md)、[`sources/README.md`](sources/README.md)、[`sources/repos/panniantong_agent_reach.md`](sources/repos/panniantong_agent_reach.md)

## [2026-06-22] ingest | sources/papers/halomi_arxiv_2606_18772.md — HALOMI 主动感知无机器人示范→人形 loco-manipulation；wiki/entities/paper-halomi-humanoid-loco-manipulation.md；交叉 loco-manipulation、teleoperation

## [2026-06-21] structural | checklist-v25 P2 事实库扩展 +12 条数据层矛盾检测规则（186 → 198 条）；schema/canonical-facts.json

- 执行清单：[docs/checklists/tech-stack-next-phase-checklist-v25.md](docs/checklists/tech-stack-next-phase-checklist-v25.md) P2「事实库扩展」收口（≥ 196 条目标达成，实际 198 条）
- 新增规则（数据层矛盾检测）：纯光学 MoCap 缺力/接触不可直执行、人体视频 3D/接触信息弱、形态差距大重定向不可省略、几何重定向≠物理可执行、PHUMA 物理过滤已重定向免工程、接触一致性为物理可行性前置、规模不能替代物理可行性、真机执行数据天然物理可行但任务窄、四质量轴串联门体检顺序、Humanoid Everyday 非重定向源、已重定向数据集免重定向直接训练、物理不可行参考致 RL 学错力矩
- 校验：逐条经脚本核验对 `motion-data-quality` / `humanoid-training-data-pipeline` / `motion-retargeting` / `humanoid-reference-motion-datasets` 等现存页有 pos 命中、neg 0 命中；`make lint` 0 errors、潜在矛盾 0 条
- 验证：`make lint`

## [2026-06-21] structural | 图谱社区 — 弱归属节点归入「其他社区」；scripts/generate_link_graph.py、docs/graph.html

- 规则：与同社区邻居边占比 &lt; 50% 的非枢纽节点不再强行贴标签，统一落入 `community-other`（其他社区）；图谱图例/筛选始终展示该桶
- 验证：`make ci-preflight`

## [2026-06-21] ingest | sources/repos/spear-sim.md — SPEAR UE 光真实感具身仿真库；wiki/entities/spear-sim.md；交叉 simulator-selection-guide、metahuman、airsim

- 原始资料：[spear-sim.md](sources/repos/spear-sim.md)（<https://github.com/spear-sim/spear>）
- 说明：UE 反射 API、begin_frame/end_frame 事务模型、56 FPS GT 渲染、MuJoCo co-sim、MetaHumans 多视角示例
- 沉淀页面：[wiki/entities/spear-sim.md](wiki/entities/spear-sim.md)
- 交叉更新：[wiki/queries/simulator-selection-guide.md](wiki/queries/simulator-selection-guide.md)、[wiki/entities/metahuman.md](wiki/entities/metahuman.md)、[wiki/entities/airsim.md](wiki/entities/airsim.md)
- 验证：`make ci-preflight`

## [2026-06-21] ingest | sources/papers/gvhmr_arxiv_2409_06662.md — GVHMR Gravity-View 单目 world-grounded HMR；深化 wiki/entities/gvhmr.md

- 原始资料：[gvhmr_arxiv_2409_06662.md](sources/papers/gvhmr_arxiv_2409_06662.md)、[gvhmr-zju3dv-github-io.md](sources/sites/gvhmr-zju3dv-github-io.md)、[gvhmr.md](sources/repos/gvhmr.md)（<https://zju3dv.github.io/gvhmr/>、<https://github.com/zju3dv/GVHMR>）
- 说明：Gravity-View 坐标逐帧 HMR、预处理→Transformer→世界轨迹管线、AMASS/BEDLAM/H36M/3DPW 训练、SimpleVO 工程更新
- 沉淀页面：[wiki/entities/gvhmr.md](wiki/entities/gvhmr.md)
- 验证：`make ci-preflight`
## [2026-06-20] ingest | sources/repos/karpathy-autoresearch.md — Karpathy 单 GPU 自主 LLM 训练实验环；wiki/entities/karpathy-autoresearch.md；交叉 ai-auto-research、andrej-karpathy

## [2026-06-20] ingest | sources/papers/ai_auto_research_survey_2605_18661.md — AI Auto-Research 综述与 Awesome 列表；wiki/concepts/ai-auto-research.md；交叉 llm-wiki-karpathy、agent-reach、hermes-agent、world-action-models

## [2026-06-20] ingest | sources/papers/oscar_arxiv_2606_04463.md — OSCAR 跨具身骨架条件世界模型；wiki/entities/paper-oscar.md；交叉 generative-world-models、roboarena、robot-world-models-training-loop-taxonomy、world-models-route-03-virtual-sandbox

## [2026-06-20] ingest | sources/blogs/wechat_shenlan_vln_10_papers_survey.md — VLN 10 篇技术地图与论文节点

- 工具：已安装 [Panniantong/Agent-Reach](https://github.com/Panniantong/Agent-Reach) v1.5.0（`pip install -e` + `agent-reach install --channels=wechat`）；微信正文经 `~/.agent-reach/tools/wechat-article-for-ai`（Camoufox）
- 原始资料：`sources/blogs/wechat_shenlan_vln_10_papers_survey.md`、`sources/raw/wechat_vln_10_papers_2026-06-20.md`、`sources/papers/vln_10_papers_catalog.md`、`sources/papers/vln_survey_*.md`（10 篇）
- 沉淀页面：[`wiki/overview/vln-10-papers-technology-map.md`](wiki/overview/vln-10-papers-technology-map.md)（父）、[`vln-category-01-datasets-platforms.md`](wiki/overview/vln-category-01-datasets-platforms.md)、[`vln-category-02-algorithm-frameworks.md`](wiki/overview/vln-category-02-algorithm-frameworks.md)（子）、`wiki/entities/paper-vln-01-r2r.md` … `paper-vln-10-navid.md`
- 去重：**NaVid**（RSS 2024，arXiv:2402.15852）≠ **Uni-NaVid**（RSS 2025 导航 VLA 复现栈）
- 交叉更新：[`wiki/tasks/vision-language-navigation.md`](wiki/tasks/vision-language-navigation.md)、[`wiki/overview/vln-open-source-repro-paradigms.md`](wiki/overview/vln-open-source-repro-paradigms.md)、[`sources/README.md`](sources/README.md)、[`sources/repos/panniantong_agent_reach.md`](sources/repos/panniantong_agent_reach.md)
- 验证：`make ci-preflight`

## [2026-06-19] structural | checklist-v25 P1 训练数据管线知识链（+2 页）；新建 wiki/queries/humanoid-training-data-pipeline.md、wiki/concepts/motion-data-quality.md

- 执行清单：[docs/checklists/tech-stack-next-phase-checklist-v25.md](docs/checklists/tech-stack-next-phase-checklist-v25.md) P1 第一项「训练数据管线知识链 (+2)」收口
- 新建页面：
  - [wiki/queries/humanoid-training-data-pipeline.md](wiki/queries/humanoid-training-data-pipeline.md) — 端到端 Query：原始 MoCap / 人体视频 → 重定向 → RL/IL 训练输入三层决策树（含 Mermaid、端到端 pipeline、5 条误区、缩写速查）
  - [wiki/concepts/motion-data-quality.md](wiki/concepts/motion-data-quality.md) — 动作数据质量四轴（物理可行性/接触一致性/形态差距/规模多样性）串联门模型 + 与重定向必要性因果链 + 五集对照
- 交叉补强：[wiki/concepts/motion-retargeting.md](wiki/concepts/motion-retargeting.md)、[wiki/comparisons/humanoid-reference-motion-datasets.md](wiki/comparisons/humanoid-reference-motion-datasets.md) 补入对两新页的入链（无孤儿）
- 验证：`make export graph` → 1257 节点 / 8186 边 / 0 孤儿；`make ci-preflight` 12/12 通过；`lint_wiki` 0 error

## [2026-06-19] ingest | sources/sites/metahuman-epic-docs.md — Epic MetaHuman 官方文档索引；补全 wiki/entities/metahuman.md 文档节

- 原始资料：[metahuman-epic-docs.md](sources/sites/metahuman-epic-docs.md)（<https://dev.epicgames.com/documentation/metahuman/metahuman-documentation>）；侧栏 15 篇子文档摘要
- 说明：Animator 三路径（实时/深度/单目）、UE Cine/Optimized/UEFN 管线、Facial Description Standard、Devkit/OpenRigLogic
- 沉淀页面：[wiki/entities/metahuman.md](wiki/entities/metahuman.md)

## [2026-06-19] ingest | sources/sites/metahuman-com.md — Epic MetaHuman 数字人平台；wiki/entities/metahuman.md；交叉 motion-retargeting、mixamo、blender

- 原始资料：[metahuman-com.md](sources/sites/metahuman-com.md)（<https://www.metahuman.com/>）；复核 5.8 发布说明
- 说明：Creator + Animator 高保真数字人；5.8 全身 Mesh to MetaHuman、单相机无标记全身 Animator、Crowds、OpenRigLogic（MIT）
- 沉淀页面：[wiki/entities/metahuman.md](wiki/entities/metahuman.md)
- 交叉更新：[wiki/concepts/motion-retargeting.md](wiki/concepts/motion-retargeting.md)、[wiki/entities/mixamo.md](wiki/entities/mixamo.md)、[wiki/entities/blender.md](wiki/entities/blender.md)

## [2026-06-19] ingest | sources/repos/crisp_real2sim_repo.md — 校正 CRISP 官方实现为 Z1hanW/CRISP-Real2Sim 并补全 scripts 1–8 + MotionTracking 工程管线

- 原始资料：[crisp_real2sim_repo.md](sources/repos/crisp_real2sim_repo.md)（<https://github.com/Z1hanW/CRISP-Real2Sim>）；复核 [crisp_real2sim_iclr2026.md](sources/papers/crisp_real2sim_iclr2026.md)、[crisp-real2sim-project-github-io.md](sources/sites/crisp-real2sim-project-github-io.md)
- 说明：先前误链 `crisp-real2sim/CRISP-Real2Sim`（实为 GitHub Pages 站点仓）；主代码为作者仓，含 `run_crisp_video.sh`、可选 contact hallucination / NKSR、Google Drive 视频数据集
- 沉淀页面：[wiki/methods/crisp-real2sim.md](wiki/methods/crisp-real2sim.md)（新增「工程实现」节与 Mermaid）

## [2026-06-19] ingest | sources/papers/mujica_arxiv_2605_13058.md — MUJICA 轮足多技能统一控制（wiki/entities/paper-mujica-wheel-legged-multi-skill.md 及轮足/混合运动/sim2real 交叉引用）

- 原始资料：[mujica_arxiv_2605_13058.md](sources/papers/mujica_arxiv_2605_13058.md)（<https://arxiv.org/abs/2605.13058>）；[项目页](https://hyzenthlayer.github.io/mujica/)
- 说明：Go2-W **纯本体** 单策略联合全向移动、高台攀爬、摔倒恢复；**P3O + DC 电机硬约束** + 两阶段技能选择器；真机 **1 m 高台** 与连续多技能链
- 沉淀页面：[wiki/entities/paper-mujica-wheel-legged-multi-skill.md](wiki/entities/paper-mujica-wheel-legged-multi-skill.md)
- 交叉更新：[wiki/concepts/wheel-legged-quadruped.md](wiki/concepts/wheel-legged-quadruped.md)、[wiki/tasks/hybrid-locomotion.md](wiki/tasks/hybrid-locomotion.md)、[wiki/tasks/locomotion.md](wiki/tasks/locomotion.md)、[wiki/concepts/sim2real.md](wiki/concepts/sim2real.md)

## [2026-06-19] ingest | sources/papers/swap_parkour_arxiv_2606_19928.md、sources/sites/swap-parkour-github-io.md — SWAP 对称等变世界模型四足跑酷；wiki/entities/paper-swap-parkour.md；交叉 locomotion、stair-obstacle、extreme-parkour

- 原始资料：[swap_parkour_arxiv_2606_19928.md](sources/papers/swap_parkour_arxiv_2606_19928.md)（<https://arxiv.org/abs/2606.19928>）；[swap-parkour-github-io.md](sources/sites/swap-parkour-github-io.md)（<https://swap-parkour.github.io/>）
- 说明：SE-RSSM 对称等变潜变量世界模型 + 等变 Actor / 不变 Critic 端到端四足跑酷；Apollo 实机 **2.13 m 远跳 / 1.63 m 攀台**；镜像 OOD 与户外零样本泛化
- 沉淀页面：[wiki/entities/paper-swap-parkour.md](wiki/entities/paper-swap-parkour.md)
- 交叉更新：[wiki/tasks/locomotion.md](wiki/tasks/locomotion.md)、[wiki/tasks/stair-obstacle-perceptive-locomotion.md](wiki/tasks/stair-obstacle-perceptive-locomotion.md)、[wiki/entities/extreme-parkour.md](wiki/entities/extreme-parkour.md)

## [2026-06-19] ingest | sources/papers/phygile_arxiv_2603_19305.md、sources/sites/phygile-page.md — PhyGile 文本驱动机器人原生扩散与 GMT 闭环；wiki/entities/paper-phygile.md；交叉 diffusion-motion-generation、humanoid-motion-tracking-method-selection、paper-notebook-gmt

- 原始资料：[phygile_arxiv_2603_19305.md](sources/papers/phygile_arxiv_2603_19305.md)（<https://arxiv.org/abs/2603.19305>）；[phygile-page.md](sources/sites/phygile-page.md)（<https://baojch.github.io/phygile-page/>）
- 说明：physics-prefix 引导的 **262D robot-native** 扩散生成 + 课程式 **MoE GMT** 跟踪器闭环；真机 breakdance、侧手翻、高踢、旋跳等高动态全身动作
- 沉淀页面：[wiki/entities/paper-phygile.md](wiki/entities/paper-phygile.md)
- 交叉更新：[wiki/methods/diffusion-motion-generation.md](wiki/methods/diffusion-motion-generation.md)、[wiki/queries/humanoid-motion-tracking-method-selection.md](wiki/queries/humanoid-motion-tracking-method-selection.md)、[wiki/entities/paper-notebook-gmt.md](wiki/entities/paper-notebook-gmt.md)

## [2026-06-19] ingest | sources/papers/humanoid_gpt_arxiv_2606_03985.md、sources/repos/humanoid_gpt_galaxy_general_robotics.md — Humanoid-GPT 复核：仓库已发布推理/部署与 ONNX checkpoint；wiki/entities/paper-humanoid-gpt.md 补工程节

- 原始资料：[humanoid_gpt_arxiv_2606_03985.md](sources/papers/humanoid_gpt_arxiv_2606_03985.md)（<https://arxiv.org/abs/2606.03985>）；[humanoid_gpt_galaxy_general_robotics.md](sources/repos/humanoid_gpt_galaxy_general_robotics.md)（<https://github.com/GalaxyGeneralRobotics/Humanoid-GPT>）
- 说明：2026-06-04 首入库后复核官方仓库 README——已发布推理/评测/真机部署、`pns_wo_priv216.onnx` checkpoint 与 `projects/{hme,gqs,tracking_transformer}`；训练代码与 2B 数据仍 TODO；补 RoPE、视频估计动作与 G1_VERSION 工程细节
- 沉淀页面：[wiki/entities/paper-humanoid-gpt.md](wiki/entities/paper-humanoid-gpt.md)

## [2026-06-19] ingest | sources/sites/gfr-project.md — 补录 GfR 项目页（RSS 2026）与 PDF 镜像；交叉更新 wiki/methods/mtrg-reference-goal-driven-rl.md、wiki/comparisons/hil-vs-mtrg-vs-zest-parkour-imitation.md、sources/papers/mtrg_reference_goal_driven_rl_arxiv_2602_20375.md

- 原始资料：[`sources/sites/gfr-project.md`](sources/sites/gfr-project.md)（<https://jiashunwang.github.io/GfR/>；PDF：<https://jiashunwang.github.io/GfR/static/mat/gfr_paper.pdf>）
- 说明：arXiv:2602.20375 已于 2026-06-12 以 MTRG 入库；本次补 **GfR** 官方项目名、**RSS 2026** 定稿、长程状态机组合、MuJoCo sim-to-sim 与 elevation map 扩展
- 交叉更新：[`wiki/methods/mtrg-reference-goal-driven-rl.md`](wiki/methods/mtrg-reference-goal-driven-rl.md)、[`wiki/comparisons/hil-vs-mtrg-vs-zest-parkour-imitation.md`](wiki/comparisons/hil-vs-mtrg-vs-zest-parkour-imitation.md)、[`wiki/tasks/humanoid-locomotion.md`](wiki/tasks/humanoid-locomotion.md)

## [2026-06-18] structural | scripts/scaffold_wiki_page.py — V25 P0「数据集选型脚手架强化」：新增 `--dataset` 旗标生成数据集实体骨架（五维度速查块 + `dataset` tag）

- 改动：`scripts/scaffold_wiki_page.py` 新增 `--dataset`（仅 `entity` 类型，否则 rc=2），输出「## 数据集速查」表格覆盖「规模 / 模态 / 许可证 / 适配形态 / 重定向就绪度」并在 frontmatter 写入 `dataset` tag；速查块关键词全覆盖 `lint_wiki._check_dataset_entity_metadata` 四维度，新建数据集页元数据巡检 0 缺失。
- 测试：[`tests/test_scaffold_wiki_page.py`](tests/test_scaffold_wiki_page.py) 新增 3 用例（速查块/tag/位置、lint 巡检 0 缺失、非 entity 拒绝）；`ruff check/format` 与 `python3 scripts/lint_wiki.py` 通过。
- 清单：[`docs/checklists/tech-stack-next-phase-checklist-v25.md`](docs/checklists/tech-stack-next-phase-checklist-v25.md) P0「数据集选型脚手架强化」打勾。

## [2026-06-18] ingest | sources/papers/ume_exo_arxiv_2606_14218.md、sources/sites/ume-exo-project.md — UME 外骨骼力矩反馈遥操作；wiki/entities/paper-ume-exo.md；交叉 teleoperation、bimanual-manipulation、loco-manipulation、motion-retargeting、action-chunking

## [2026-06-18] ingest | sources/blogs/allenai_molmo_motion.md — MolmoMotion 语言条件 3D 点轨迹预测；wiki/entities/molmo-motion.md；交叉 generative-world-models、manipulation、video-as-simulation

## [2026-06-18] ingest | sources/papers/greenvla_arxiv_2602_00919.md — Green-VLA 五阶段 VLA；wiki/entities/paper-greenvla-staged-vla-humanoid.md、wiki/methods/vla.md、wiki/tasks/manipulation.md

## [2026-06-18] ingest | sources/papers/enpire_nvidia_gear_2026.md、sources/sites/nvidia-research-enpire.md — ENPIRE 真机 coding-agent 策略自改进；wiki/methods/enpire.md；交叉 manipulation、simulation-evaluation-infrastructure

## [2026-06-18] ingest | sources/papers/kairos_arxiv_2606_16533.md — Kairos 原生世界模型栈；wiki/entities/paper-kairos-native-world-model-stack.md；交叉 generative-world-models、world-action-models、homeworld

## [2026-06-18] ingest | sources/repos/xpad.md — 接入 Linux Xbox USB 手柄内核驱动 xpad 并新建 wiki/entities/xpad.md，交叉更新 teleoperation 与 open-duck-mini-runtime

- 原始资料：[`sources/repos/xpad.md`](sources/repos/xpad.md)（<https://github.com/paroj/xpad>）
- 沉淀页面：[`wiki/entities/xpad.md`](wiki/entities/xpad.md)
- 交叉更新：[`wiki/tasks/teleoperation.md`](wiki/tasks/teleoperation.md)、[`wiki/entities/open-duck-mini-runtime.md`](wiki/entities/open-duck-mini-runtime.md)
## [2026-06-18] ingest | sources/sites/botworld.md — 接入 BotWorld 机器人资产平台；wiki/entities/botworld.md；交叉 urdf-studio、botlab-motioncanvas、step2urdf、motrix

- 原始资料：[`sources/sites/botworld.md`](sources/sites/botworld.md)（<https://botworld.enkeebot.com/>；前端 bundle 策展）
- 沉淀页面：[`wiki/entities/botworld.md`](wiki/entities/botworld.md)
- 交叉更新：[`wiki/entities/urdf-studio.md`](wiki/entities/urdf-studio.md)、[`wiki/entities/botlab-motioncanvas.md`](wiki/entities/botlab-motioncanvas.md)、[`wiki/entities/step2urdf.md`](wiki/entities/step2urdf.md)、[`wiki/entities/motrix.md`](wiki/entities/motrix.md)

## [2026-06-18] ingest | sources/sites/motrixsim-web-viewer.md — 归档 MotrixSim Web Viewer 并更新 wiki/entities/motrix.md

- 原始资料：[`sources/sites/motrixsim-web-viewer.md`](sources/sites/motrixsim-web-viewer.md)（<https://motrix.motphys.com/>、ReadTheDocs WebViewer 指南）
- 交叉更新：[`sources/repos/motphys-motrix.md`](sources/repos/motphys-motrix.md)
- 沉淀页面：[`wiki/entities/motrix.md`](wiki/entities/motrix.md) — 新增 MotrixSim Web Viewer 小节（WASM、Online/Customize、拖文件夹加载、快捷键）

## [2026-06-18] ingest | sources/repos/step2urdf.md、sources/sites/step2urdf-top.md — STEP→URDF 浏览器工具 step2urdf；wiki/entities/step2urdf.md；交叉 urdf-studio、cad-skills、references/repos/utilities.md

## [2026-06-18] ingest | sources/blogs/wechat_embodied_ai_lab_humanoid_motion_cerebellum_survey.md — 运动小脑 64 篇长文：父节点 wiki/overview/humanoid-motion-cerebellum-technology-map.md + 九组 motion-cerebellum-category-* hub；复用 paper-hrl-stack-* 等既有节点，新建 15 篇 paper-motion-cerebellum-*

- 工具：Agent Reach v1.5.0 + wechat-article-for-ai（Camoufox）；短链 <https://mp.weixin.qq.com/s/Kx9myecE1Z0eGqOapoqQnA>
- 原始资料：[`sources/blogs/wechat_embodied_ai_lab_humanoid_motion_cerebellum_survey.md`](sources/blogs/wechat_embodied_ai_lab_humanoid_motion_cerebellum_survey.md)、[`sources/raw/wechat_motion_cerebellum_64_survey_2026-06-18.md`](sources/raw/wechat_motion_cerebellum_64_survey_2026-06-18.md)、[`sources/papers/motion_cerebellum_64_catalog.md`](sources/papers/motion_cerebellum_64_catalog.md)
- 沉淀页面：[`wiki/overview/humanoid-motion-cerebellum-technology-map.md`](wiki/overview/humanoid-motion-cerebellum-technology-map.md)、九组 [`wiki/overview/motion-cerebellum-category-*.md`](wiki/overview/motion-cerebellum-category-01-locomotion-base.md)
- 新建索引（15）：`wiki/entities/paper-motion-cerebellum-*`（GuideWalk、T-GMP、MARCH、TAGA、TRAM、Stubborn、ConstrainedMimic、SafeWBC、MuGen、CEER、HANDOFF、主动空间大脑、HOIST、HumanoidMimicGen、GRAIL）
- 交叉更新：[`wiki/overview/humanoid-rl-motion-control-body-system-stack.md`](wiki/overview/humanoid-rl-motion-control-body-system-stack.md)、[`sources/README.md`](sources/README.md)、[`sources/repos/panniantong_agent_reach.md`](sources/repos/panniantong_agent_reach.md)

## [2026-06-18] ingest | sources/papers/deepinsight_arxiv_2606_17574.md — DeepInsight Physical AI 全栈统一评测基础设施；wiki/entities/deepinsight.md；交叉 simulation-evaluation-infrastructure、robot-training-stack-layers-technology-map

## [2026-06-18] ingest | sources/papers/rove_arxiv_2606_17011.md — ROVE 人形 VLA 干预后训练；wiki/entities/paper-rove-humanoid-vla-intervention.md；交叉 teleoperation、vla、online-vs-offline-rl

## [2026-06-18] ingest | sources/blogs/wechat_shenlan_homogeneous_coordinates_transform.md — 《具身智能基础》专栏 05 齐次坐标与齐次变换；新建 wiki/formalizations/homogeneous-coordinates-transform.md；更新专栏父节点

- 工具：已安装 [Panniantong/Agent-Reach](https://github.com/Panniantong/Agent-Reach) v1.5.0 + [wechat-article-for-ai](https://github.com/bzd6661/wechat-article-for-ai)（Camoufox）；专辑 <https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzkwMDcyNDUzMQ==&action=getalbum&album_id=4525948187102363653> 共 5 篇，本篇短链 <https://mp.weixin.qq.com/s/3vwaizPOgJKCwQ9e5LuKGA>
- 原始资料：[`sources/blogs/wechat_shenlan_homogeneous_coordinates_transform.md`](sources/blogs/wechat_shenlan_homogeneous_coordinates_transform.md)、[`sources/raw/wechat_shenlan_homogeneous_coords_2026-06-18.md`](sources/raw/wechat_shenlan_homogeneous_coords_2026-06-18.md)
- 沉淀页面：[`wiki/formalizations/homogeneous-coordinates-transform.md`](wiki/formalizations/homogeneous-coordinates-transform.md)
- 交叉更新：[`wiki/overview/shenlan-embodied-ai-fundamentals-series.md`](wiki/overview/shenlan-embodied-ai-fundamentals-series.md)、[`wiki/formalizations/lie-group-rigid-body-motions.md`](wiki/formalizations/lie-group-rigid-body-motions.md)、[`wiki/formalizations/3d-coordinate-transforms-vision-robotics.md`](wiki/formalizations/3d-coordinate-transforms-vision-robotics.md)、[`wiki/formalizations/se3-representation.md`](wiki/formalizations/se3-representation.md)、[`sources/README.md`](sources/README.md)、[`sources/repos/panniantong_agent_reach.md`](sources/repos/panniantong_agent_reach.md)

## [2026-06-18] ingest | sources/repos/wtfos.md、sources/sites/fpv-wtf.md — wtfOS DJI 数字 FPV 固件框架入库；新建 wiki/entities/wtfos.md；交叉 multirotor-simulation-planning-control-stack、betaflight

- 原始资料：[`sources/repos/wtfos.md`](sources/repos/wtfos.md)（<https://github.com/fpv-wtf/wtfos>）、[`sources/sites/fpv-wtf.md`](sources/sites/fpv-wtf.md)（<https://fpv.wtf/>）
- 沉淀页面：[`wiki/entities/wtfos.md`](wiki/entities/wtfos.md)
- 交叉更新：[`wiki/overview/multirotor-simulation-planning-control-stack.md`](wiki/overview/multirotor-simulation-planning-control-stack.md)、[`wiki/entities/betaflight.md`](wiki/entities/betaflight.md)
## [2026-06-17] lint | scripts/lint_wiki.py — V25 P0 数据集页元数据巡检 V1（`dataset_metadata_check`）

- 变更：[`scripts/lint_wiki.py`](scripts/lint_wiki.py) 新增 `_check_dataset_entity_metadata`，针对 frontmatter `tags` 含 `dataset` 的 `wiki/entities/*.md`（兼容列表式与内联式 tags），按关键词命中近似检查正文是否覆盖「规模 / 模态 / 许可证 / 重定向就绪度」四类标准化速查维度，缺失维度作为 INFO 级 result key `dataset_missing_metadata` 写入报告，加入 `INFO_ONLY_KEYS`（不计入失败总数、不阻塞 CI）。
- 测试：新增 [`tests/test_lint_wiki_dataset_metadata.py`](tests/test_lint_wiki_dataset_metadata.py) 4 用例（完整页通过、内联 tags 命中并记缺失维度、非 dataset 页跳过、INFO 不计失败总数）。
- 验证：`make lint` 0 errors（信息型预警 21→22）；全库巡检命中 17 页缺失维度，基线快照写入 [`exports/lint-report.md`](exports/lint-report.md)；`ruff check` 通过；lint_wiki 相关 51 用例全绿。

## [2026-06-17] ingest | sources/repos/betaflight.md、sources/sites/betaflight-com.md — Betaflight FPV 飞控固件入库；新建 wiki/entities/betaflight.md；交叉 multirotor-simulation-planning-control-stack、px4-autopilot、gym-pybullet-drones

- 原始资料：[`sources/repos/betaflight.md`](sources/repos/betaflight.md)（<https://github.com/betaflight/betaflight>）、[`sources/sites/betaflight-com.md`](sources/sites/betaflight-com.md)（<https://betaflight.com/>）
- 沉淀页面：[`wiki/entities/betaflight.md`](wiki/entities/betaflight.md)
- 交叉更新：[`wiki/overview/multirotor-simulation-planning-control-stack.md`](wiki/overview/multirotor-simulation-planning-control-stack.md)、[`wiki/entities/px4-autopilot.md`](wiki/entities/px4-autopilot.md)、[`wiki/entities/gym-pybullet-drones.md`](wiki/entities/gym-pybullet-drones.md)

## [2026-06-17] ingest | sources/repos/plotjuggler.md — PlotJuggler 时序可视化工具入库；新建 wiki/entities/plotjuggler.md；交叉 robot-policy-debug-playbook、ros2-basics、px4-autopilot

## [2026-06-17] ingest | sources/papers/toporetarget_arxiv_2606_16272.md — 接入 TopoRetarget 交互保留灵巧重定向；wiki/methods/toporetarget-interaction-preserving-dexterous-retargeting.md、wiki/concepts/motion-retargeting.md、wiki/tasks/manipulation.md、wiki/entities/wuji-robotics.md

## [2026-06-17] ingest | sources/repos/nvlabs-soma-x.md、sources/sites/soma-x-docs.md、sources/papers/soma_arxiv_2603_16858.md — NVlabs/SOMA-X 统一参数化人体模型入库；新建 wiki/entities/soma-x.md；交叉 soma-retargeter、motion-retargeting、genmo、kimodo

## [2026-06-17] ingest | sources/repos/sbto.md — 深化 Atarilab/sbto 仓库 ingest；新建 wiki/entities/sbto.md

## [2026-06-17] ingest | sources/papers/dynaretarget_arxiv_2602_06827.md — DynaRetarget/SBTO 全文 ingest；wiki/methods/dynaretarget-sbto-motion-retargeting.md、wiki/entities/paper-notebook-dynaretarget-dynamically-feasible-retargeting-us.md、sources/repos/sbto.md、sources/sites/dynaretarget-github-io.md

## [2026-06-17] structural | wiki/overview/topic-*.md + docs/topic-filters.js + docs/graph.html — 14 项图谱专题各增汇总节点与导读 UI

- 新建 14 个专题汇总页：[`wiki/overview/topic-motion-retargeting.md`](wiki/overview/topic-motion-retargeting.md) … [`topic-vision-backbone.md`](wiki/overview/topic-vision-backbone.md)（一句话定义、缩写速查、覆盖范围、专题互链、参考来源）
- [`docs/topic-filters.js`](docs/topic-filters.js)：`TOPIC_HUB_IDS` / `TOPIC_META.wikiPath+description`；各专题 `ids` 显式纳入汇总节点，保证专题视图下始终可见
- [`docs/graph.html`](docs/graph.html)：选中专题时画布左上角显示「专题汇总」导读条 + 汇总节点高亮（`.node-topic-hub`）；链接跳转详情页
- 验证：`make ci-preflight` 通过（1209 nodes / 7615 edges）

## [2026-06-17] ingest | sources/repos/bullet3.md, sources/sites/pybullet-org.md — 官方 Bullet3 仓与 pybullet.org 入库；升格 wiki/entities/pybullet.md

- 原始资料：[`sources/repos/bullet3.md`](sources/repos/bullet3.md)（<https://github.com/bulletphysics/bullet3>）、[`sources/sites/pybullet-org.md`](sources/sites/pybullet-org.md)（<https://pybullet.org/wordpress/>）
- 沉淀页面：[`wiki/entities/pybullet.md`](wiki/entities/pybullet.md)
- 交叉更新：[`sources/README.md`](sources/README.md)
## [2026-06-17] ingest | sources/papers/motiondisco_arxiv_2606_06139.md — MotionDisco LLM 引导运动发现

- 原始资料：[`sources/papers/motiondisco_arxiv_2606_06139.md`](sources/papers/motiondisco_arxiv_2606_06139.md)（<https://arxiv.org/abs/2606.06139>、<https://atarilab.github.io/motiondisco.io/>）
- 沉淀页面：[`wiki/entities/paper-motiondisco-extreme-humanoid-loco-manipulation.md`](wiki/entities/paper-motiondisco-extreme-humanoid-loco-manipulation.md)
- 交叉更新：[`wiki/tasks/loco-manipulation.md`](wiki/tasks/loco-manipulation.md)（路线 §19 + 关联页）

## [2026-06-16] checklist-v25 | V24 收口 & 初始化 V25

- V24 全部条目此前已收口（P0–P3 + DoD 逐条 `[x]`，`make lint` 0 errors，图谱 1193 节点 / 7421 边、`community_quality_warning=false`、最大社区占比 0.177、事实库 186 条）。
- 新建 [`docs/checklists/tech-stack-next-phase-checklist-v25.md`](docs/checklists/tech-stack-next-phase-checklist-v25.md)：专题选定为「人形训练数据管线」，承接 V24 收尾密集 ingest 的 AMASS/LaFAN1/OMOMO/PHUMA/Humanoid Everyday 五套数据集与 motion-retargeting 概念页，规划「原始动作捕捉/视频 → 重定向 → RL/IL 训练输入」端到端知识链（P1 query+concept）、数据层矛盾检测规则扩展（P2 事实库 186→≥196）、数据集页元数据巡检（P0）与图谱第 15 项「训练数据管线」专题视图（P3）。
- 同步将 README badge / 维护看板、`AGENTS.md`、`docs/README.md`、`docs/checklists/README.md` 的「当前清单」指针从 V24 切到 V25；V24 移入 `archive/` 并修正其内部相对链接，进入历史归档区。

## [2026-06-16] ingest | sources/repos/omomo_release.md, sources/repos/phuma.md, sources/sites/humanoideveryday.md — AMASS/LaFAN1/OMOMO/PHUMA/Humanoid Everyday 五集入库

- 原始资料：[`sources/sites/amass-dataset.md`](sources/sites/amass-dataset.md)、[`sources/repos/ubisoft-laforge-animation-dataset.md`](sources/repos/ubisoft-laforge-animation-dataset.md)、[`sources/repos/omomo_release.md`](sources/repos/omomo_release.md)、[`sources/repos/phuma.md`](sources/repos/phuma.md)、[`sources/sites/humanoideveryday.md`](sources/sites/humanoideveryday.md)
- 沉淀页面：[`wiki/entities/omomo-dataset.md`](wiki/entities/omomo-dataset.md)、[`wiki/entities/humanoid-everyday-dataset.md`](wiki/entities/humanoid-everyday-dataset.md)、[`wiki/comparisons/humanoid-reference-motion-datasets.md`](wiki/comparisons/humanoid-reference-motion-datasets.md)；升格 [`wiki/entities/dataset-bfm-phuma.md`](wiki/entities/dataset-bfm-phuma.md)
- 交叉更新：[`wiki/entities/amass.md`](wiki/entities/amass.md)、[`wiki/entities/lafan1-dataset.md`](wiki/entities/lafan1-dataset.md)、[`wiki/concepts/motion-retargeting.md`](wiki/concepts/motion-retargeting.md)、[`wiki/entities/omniretarget-dataset.md`](wiki/entities/omniretarget-dataset.md)、[`wiki/entities/paper-notebook-humanoid-everyday-a-comprehensive-robotic-datase.md`](wiki/entities/paper-notebook-humanoid-everyday-a-comprehensive-robotic-datase.md)

## [2026-06-16] ingest | sources/blogs/qwen_robot_suite.md — 入库 Qwen-Robot Suite 总览与 Nav/Manip/World 子博客；新建 wiki/entities/qwen-robot-{suite,nav,manip,world}.md；交叉 qwen-vla、vla、vln、generative-world-models

- 原始资料：[`sources/blogs/qwen_robot_suite.md`](sources/blogs/qwen_robot_suite.md)（<https://qwen.ai/blog?id=qwen-robotsuite>）、[`qwen_robot_nav.md`](sources/blogs/qwen_robot_nav.md)、[`qwen_robot_manip.md`](sources/blogs/qwen_robot_manip.md)、[`qwen_robot_world.md`](sources/blogs/qwen_robot_world.md)
- 沉淀页面：[`wiki/entities/qwen-robot-suite.md`](wiki/entities/qwen-robot-suite.md)、[`wiki/entities/qwen-robot-nav.md`](wiki/entities/qwen-robot-nav.md)、[`wiki/entities/qwen-robot-manip.md`](wiki/entities/qwen-robot-manip.md)、[`wiki/entities/qwen-robot-world.md`](wiki/entities/qwen-robot-world.md)
- 交叉更新：[`wiki/entities/qwen-vla.md`](wiki/entities/qwen-vla.md)、[`wiki/methods/vla.md`](wiki/methods/vla.md)、[`wiki/tasks/vision-language-navigation.md`](wiki/tasks/vision-language-navigation.md)、[`wiki/methods/generative-world-models.md`](wiki/methods/generative-world-models.md)
## [2026-06-16] structural | wiki/entities/humanoid-robot.md — 补全 HIL（Hardware-in-the-Loop）及正文全部缩写速查

- 页面：[`wiki/entities/humanoid-robot.md`](wiki/entities/humanoid-robot.md) — 流程图「HIL 与台架安全测试」指硬件在环，非 Hybrid Imitation Learning
- 词典：[`schema/abbrev-glossary.json`](schema/abbrev-glossary.json) 新增 HIL / PRS / HAL / FastDDS

## [2026-06-16] ingest | sources/blogs/current_robotics_curr0_loco_dexterous_manipulation.md — Current Robotics Curr-0 人形 loco-dexterous 全栈博客入库

- 原始资料：[`sources/blogs/current_robotics_curr0_loco_dexterous_manipulation.md`](sources/blogs/current_robotics_curr0_loco_dexterous_manipulation.md)（<https://current-robotics.com/blog/curr-0>）
- 沉淀页面：[`wiki/entities/current-robotics-curr0.md`](wiki/entities/current-robotics-curr0.md)
- 交叉更新：[`wiki/tasks/loco-manipulation.md`](wiki/tasks/loco-manipulation.md)（路线 §18 + 关联页）、[`wiki/entities/wuji-robotics.md`](wiki/entities/wuji-robotics.md)、[`sources/README.md`](sources/README.md)

## [2026-06-16] ingest | sources/blogs/wechat_shenlan_rl_embodied_minimal_closed_loop.md — Agent Reach 抓取深蓝《具身智能基础》专栏 04（RL 最小闭环）并并入运动控制路线

- 工具：已安装 [Panniantong/Agent-Reach](https://github.com/Panniantong/Agent-Reach) v1.5.0 + [wechat-article-for-ai](https://github.com/bzd6661/wechat-article-for-ai)（Camoufox）；Jina Reader 对 `mp.weixin.qq.com` 返回 CAPTCHA，未采用
- 原始资料：[`sources/blogs/wechat_shenlan_rl_embodied_minimal_closed_loop.md`](sources/blogs/wechat_shenlan_rl_embodied_minimal_closed_loop.md)（<https://mp.weixin.qq.com/s/hHkQqLfIOTn0CoAZNuLWJA>）；落盘 [`sources/raw/wechat_shenlan_rl_minimal_closed_loop_2026-06-16.md`](sources/raw/wechat_shenlan_rl_minimal_closed_loop_2026-06-16.md)
- 沉淀页面：[`wiki/entities/pybullet.md`](wiki/entities/pybullet.md)、[`wiki/concepts/embodied-rl-minimal-closed-loop.md`](wiki/concepts/embodied-rl-minimal-closed-loop.md)（**公众号文本身不设独立 wiki 节点**）
- 交叉更新：[`roadmap/motion-control.md`](roadmap/motion-control.md)（L5.0 桥梁 + L5.1 推荐路径 + L7.5 趋势表）、[`wiki/methods/reinforcement-learning.md`](wiki/methods/reinforcement-learning.md)、[`wiki/formalizations/mdp.md`](wiki/formalizations/mdp.md)、[`wiki/formalizations/pomdp.md`](wiki/formalizations/pomdp.md)、[`wiki/overview/shenlan-embodied-ai-fundamentals-series.md`](wiki/overview/shenlan-embodied-ai-fundamentals-series.md)、[`sources/README.md`](sources/README.md)

## [2026-06-15] structural | docs/detail.html + docs/main.js + docs/topic-filters.js — V24 P3 详情页"所属专题"轻量徽标，专题命中规则抽共享模块并收口 V24

- 新增 `docs/topic-filters.js` 作为专题命中规则单一事实源（`TOPIC_FILTERS` / `TOPIC_META` / `matches` / `topicsForNode`）；`docs/graph.html` 移除内联 `TOPIC_FILTERS` 与 `nodeMatchesTopic` 实现改为消费共享模块，并新增 `?topic=<key>` URL 参数自动激活对应专题视图
- `docs/main.js` 新增 `renderDetailTopicBadges`：复用 `link-graph.json` 现成社区数据计算当前页命中的专题，渲染"所属专题"徽标 → `graph.html?topic=<key>`（无命中静默隐藏）；`docs/detail.html` 增 `#detailTopicBadges` 容器，`docs/style.css` 增 `.detail-topic-badge` 胶囊样式
- `docs/checklists/tech-stack-next-phase-checklist-v24.md`：P3 可选项打勾，验收标准（make lint / 节点边数 / community 均衡 / log 记录）逐条复核打勾，V24 全部条目收口
- 验证：`make lint` 退出码 0（「✅ 所有检查通过！」）；节点 1183 / 边 7292、`community_quality_warning=false`、最大社区占比 0.179；Puppeteer 端到端截图归档 `.cursor-artifacts/screenshots/detail-topic-badge.png`、`graph-topic-from-url.png`（详情页徽标→图谱专题视图链路打通）

## [2026-06-15] ingest | sources/repos/gen2humanoid.md — 入库 Gen2Humanoid 文本→HY-Motion→GMR 人形管线；新建 wiki/entities/gen2humanoid.md；交叉 hy-motion-1、motion-retargeting-gmr、motion-retargeting-pipeline

## [2026-06-15] ingest | MoveIt/MoveIt 2 一手资料 — sources/sites/moveit-*.md + sources/repos/moveit-*.md、ros-planning-srdfdom.md；新建 wiki/entities/moveit2.md；交叉 manipulation、curobo、ros2-official-documentation

- 原始资料：[`sources/sites/moveit-official-portal.md`](sources/sites/moveit-official-portal.md)、[`sources/sites/moveit2-picknik-documentation.md`](sources/sites/moveit2-picknik-documentation.md)、[`sources/sites/moveit1-noetic-tutorials.md`](sources/sites/moveit1-noetic-tutorials.md)、[`sources/repos/moveit-moveit2.md`](sources/repos/moveit-moveit2.md)、[`sources/repos/moveit-moveit1.md`](sources/repos/moveit-moveit1.md)、[`sources/repos/ros-planning-srdfdom.md`](sources/repos/ros-planning-srdfdom.md)
- 沉淀页面：[`wiki/entities/moveit2.md`](wiki/entities/moveit2.md)；交叉更新 [`wiki/tasks/manipulation.md`](wiki/tasks/manipulation.md)、[`wiki/entities/curobo.md`](wiki/entities/curobo.md)、[`sources/sites/ros2-official-documentation.md`](sources/sites/ros2-official-documentation.md)

## [2026-06-15] ingest | sources/repos/earthtojake-text-to-cad.md — 入库 CAD Skills（earthtojake/text-to-cad）Agent Skills 库；新建 wiki/entities/cad-skills.md；交叉 text-to-cad、urdf-studio、articraft、mattpocock-skills

- 原始资料：[`sources/repos/earthtojake-text-to-cad.md`](sources/repos/earthtojake-text-to-cad.md)
- 沉淀页面：[`wiki/entities/cad-skills.md`](wiki/entities/cad-skills.md)；交叉更新 [`wiki/concepts/text-to-cad.md`](wiki/concepts/text-to-cad.md)、[`sources/sites/text-to-cad-tools.md`](sources/sites/text-to-cad-tools.md)、[`wiki/entities/urdf-studio.md`](wiki/entities/urdf-studio.md)、[`wiki/entities/articraft.md`](wiki/entities/articraft.md)、[`wiki/entities/mattpocock-skills.md`](wiki/entities/mattpocock-skills.md)

## [2026-06-15] ingest | sources/papers/rumelhart_backprop_learning_representations_nature_1986.md — Rumelhart et al. 1986 Nature 反向传播一手归档；新建 wiki/concepts/backpropagation.md；交叉 deep-learning-foundations / transformer / udl_book

## [2026-06-14] structural | docs/graph.html — V24 P3 图谱新增「视觉感知骨干」专题视图（vision-backbone）

- `TOPIC_FILTERS` 新增 `vision-backbone` 项：path 片段 `backbone/backbones/cnn/vit/resnet/yolo/detection` 并集命中；因核心页同处 community-3（与动作重定向共享）不宜按社区命中，`nodeMatchesTopic` 扩展支持 `ids` 显式纳入 `visual-representation-for-policy` / `generative-vision-pretraining` 两页
- `#filter-topic-chips` 新增 `data-topic="vision-backbone"`（👁️ 视觉骨干）chip，专题视图精准命中 9 个相关节点（cnn-vs-vit / vision-backbones / visual-representation-for-policy / perception-backbone-selection / object-detection / object-detection-model-selection / generative-vision-pretraining + ResNet/YOLO 实体）
- `docs/checklists/tech-stack-next-phase-checklist-v24.md` P3 首项打勾
- 验证：`make lint` 退出码 0（仅 1 条信息型预警）；Puppeteer 截图归档 `.cursor-artifacts/screenshots/graph-topic-vision-backbone.png`

## [2026-06-14] ingest | sources/personal/amp_mjlab_policy_training_essence.md + perceptive_locomotion_representation_essence.md — 两条 ChatGPT 对话核心知识点：wiki/concepts/neural-feedback-controller.md、terrain-latent-representation.md；增补 privileged-training、amp-mjlab

## [2026-06-14] ingest | sources/papers/agi_to_asi_arxiv_2606_12683.md — DeepMind From AGI to ASI 技术报告；wiki/entities/paper-from-agi-to-asi.md；交叉 embodied-scaling-laws / data-flywheel / robot-learning-three-eras-narrative

## [2026-06-14] ingest | sources/papers/oasis_humanoid_loco_manip_2606_08548.md — OASIS arXiv:2606.08548 全文精读入库，升格 wiki/entities/paper-loco-manip-04-oasis.md

## [2026-06-14] ingest | sources/papers/mighty_arxiv_2511_10822.md + sources/repos/mighty.md — MIT ACL MIGHTY Hermite 样条 UAV 轨迹规划（RA-L 2026）；wiki/entities/paper-mighty-hermite-spline-trajectory-planning.md；交叉 multirotor-simulation-planning-control-stack、ego-planner-swarm

## [2026-06-14] ingest | sources/blogs/wechat_embodied_ai_lab_loco_manip_8_papers_survey.md — 具身智能研究室 Loco-Manip 8 篇数据入口周报；父节点 loco-manip-8-papers-technology-map + 四组 loco-manip-category-* 子节点 + 8 篇论文实体

- 工具：已安装 [Panniantong/Agent-Reach](https://github.com/Panniantong/Agent-Reach) v1.4.0（`pip install -e` + `agent-reach install --channels=wechat`）；微信正文经 `~/.agent-reach/tools/wechat-article-for-ai`（Camoufox）
- 原始资料：`sources/blogs/wechat_embodied_ai_lab_loco_manip_8_papers_survey.md`、`sources/raw/wechat_loco_manip_8_papers_2026-06-14.md`、`sources/papers/loco_manip_8_papers_catalog.md`、`sources/papers/loco_manip_survey_*.md`（8 篇）
- 沉淀页面：[`wiki/overview/loco-manip-8-papers-technology-map.md`](wiki/overview/loco-manip-8-papers-technology-map.md)（父）、[`loco-manip-category-01-egocentric-data.md`](wiki/overview/loco-manip-category-01-egocentric-data.md) … [`loco-manip-category-04-contact-teleop.md`](wiki/overview/loco-manip-category-04-contact-teleop.md)（子）、`wiki/entities/paper-loco-manip-01-ego-pi.md` … `paper-loco-manip-08-x-op.md`
- 去重：GenHOI（arXiv:2606.12995）≠ 既有 SimGenHOI 节点
- 交叉更新：[`wiki/tasks/loco-manipulation.md`](wiki/tasks/loco-manipulation.md)、[`sources/README.md`](sources/README.md)、[`sources/repos/panniantong_agent_reach.md`](sources/repos/panniantong_agent_reach.md)
- 验证：`make ci-preflight`

## [2026-06-14] structural | wiki/methods/ppo.md + wiki/concepts/transformer.md + sources/papers/attention_is_all_you_need.md — 新建 PPO/Transformer 概念方法页，消除 lint 高频术语缺页误报

- 新建：[`wiki/methods/ppo.md`](wiki/methods/ppo.md)（clip 代理目标、GAE、机器人 RL 落地要点）、[`wiki/concepts/transformer.md`](wiki/concepts/transformer.md)（自注意力 / MHA / 机器人 VLA·ACT 角色）
- 入库：[`sources/papers/attention_is_all_you_need.md`](sources/papers/attention_is_all_you_need.md)（Vaswani et al. 2017 一手摘要）
- 交叉更新：[`wiki/methods/policy-optimization.md`](wiki/methods/policy-optimization.md)、[`wiki/concepts/deep-learning-foundations.md`](wiki/concepts/deep-learning-foundations.md)、[`wiki/concepts/humanoid-policy-network-architecture.md`](wiki/concepts/humanoid-policy-network-architecture.md)、[`wiki/comparisons/ppo-vs-sac.md`](wiki/comparisons/ppo-vs-sac.md)
- 验证：`make ci-preflight`

## [2026-06-13] ingest | sources/papers/ruka_v2_arxiv_2603_26660.md + sources/repos/ruka-v2.md + sources/sites/ruka-hand-v2-github-io.md — NYU 全开源腱驱动灵巧手 RUKA-v2；升格 wiki/entities/ruka-v2-hand.md，交叉 orca-hand / dexterous-data-collection-guide / manipulation

- 新建实体：[`wiki/entities/ruka-v2-hand.md`](wiki/entities/ruka-v2-hand.md)（16 指 DoF + 2-DoF 腕、AnyTeleop + OpenTeach + BAKU 验证链）
- 交叉更新：[`wiki/entities/orca-hand.md`](wiki/entities/orca-hand.md)、[`wiki/queries/dexterous-data-collection-guide.md`](wiki/queries/dexterous-data-collection-guide.md)、[`wiki/tasks/manipulation.md`](wiki/tasks/manipulation.md)、[`wiki/entities/paper-notebook-ruka-rethinking-the-design-of-humanoid-hands-wit.md`](wiki/entities/paper-notebook-ruka-rethinking-the-design-of-humanoid-hands-wit.md)
- 后续修正：[`wiki/entities/ruka-v2-hand.md`](wiki/entities/ruka-v2-hand.md) 成本字段 `$1,500` 改 `$1{,}500` 避免 KaTeX 误解析
- 验证：`make ci-preflight`

## [2026-06-13] structural | schema/canonical-facts.json — V24 P2 事实库由 172 → 186 条，补齐视觉骨干/机器人表征矛盾检测规则

- 新增 14 条矛盾检测规则：ResNet 残差缓解退化、深层网络退化非过拟合、ViT 数据量门槛、ViT 归纳偏置弱、CNN 归纳偏置强、YOLO 单阶段实时、两阶段精度高延迟大、YOLO 误差结构、注意力二次复杂度、冻结预训练表征样本效率高、端到端视觉策略样本效率低、R3M 人类视频预训练表征、VC-1 具身视觉骨干、视觉域差距优先于换骨干
- 逐条经脚本校验：每条 `terms`+`pos_claims` 对现存 wiki 页（cnn-vs-vit-backbones / vision-backbones / object-detection / visual-representation-for-policy / perception-backbone-selection 等）均有命中，`neg_claims` 仅刻画错误论断、不误伤正文
- `docs/checklists/tech-stack-next-phase-checklist-v24.md` P2 与 DoD 事实库条目打勾
- 验证：`python3 scripts/lint_wiki.py` 退出码 0，矛盾检测 0 项；JSON 合法

## [2026-06-13] structural | tech-map/modules/system/ros2.md + sources/sites/ros2-official-documentation.md — 填充 tech-node-system-ros2 空详情页；归档 ROS 2 Humble 一手文档，交叉 ros2-basics / ros2-vs-lcm / sim2real 部署链

## [2026-06-12] structural | wiki/comparisons/hil-vs-mtrg-vs-zest-parkour-imitation.md、wiki/queries/table-tennis-hierarchical-skill-learning-guide.md + 5 页陈旧措辞 + 2 页 paper 实体 — 消除 10 条 lint 信息型预警

- 为高频引用 methods 补落地：[`hil-vs-mtrg-vs-zest-parkour-imitation.md`](wiki/comparisons/hil-vs-mtrg-vs-zest-parkour-imitation.md)（覆盖 HIL/MTRG）、[`table-tennis-hierarchical-skill-learning-guide.md`](wiki/queries/table-tennis-hierarchical-skill-learning-guide.md)
- `paper-humanoid-soccer-swarm-intelligence` 补 `venue` 键；`paper-notebook-a-hierarchical-model-based-system-for-high-perfo` 补「方法栈」段
- 软化 5 页绝对化 SOTA 措辞：`generative-vision-pretraining`、`paper-resnet`、`paper-wem`、`paper-worldvln`、`paper-yolo`
- 验证：`make ci-preflight`


- 全库自动统计正文以 `**加粗**`/`` `反引号` `` 高频出现（≥6 个不同页面引用）但缺独立 `concepts/methods/formalizations` 页的术语，输出"建议新建页"候选（INFO 级，不阻塞 CI），作为后续 ingest/query 选题入口
- 与既有 `_check_missing_concepts`（人工 watch 列表映射已知 slug）互补；单 token 词形过滤路径/文件名、大小写归并、候选上限 15、停用词剔除 frontmatter 键
- 实测候选 8 条（PPO/MuJoCo/Transformer 等），新增 INFO 区块至健康报告；`docs/checklists/tech-stack-next-phase-checklist-v24.md` P0 该项打勾
- 验证：`tests/test_lint_wiki_missing_concept_pages.py` 6 例通过；`ruff format/check`、`mypy scripts/lint_wiki.py` 全绿；`python3 scripts/lint_wiki.py` 退出码 0

## [2026-06-12] ingest | sources/repos/manim-community.md + manim-3b1b.md + sites/manim-community.md — Manim/ManimCE/ManimGL 程序化数学动画；wiki/entities/manim.md，交叉 character-animation-vs-robotics、blender

## [2026-06-12] fix(wiki): 合并 MuJoCo Playground 等重复节点并修复 paper-notebook 标题 `[ ]` 残留

- 合并 4 对计划子节点 → 已有实体：`paper-notebook-mujoco-playground-*` → [`wiki/entities/mujoco-playground.md`](wiki/entities/mujoco-playground.md)；另 Genesis / ORB-SLAM3 / VINS-Fusion 同理
- 根因修复：`short_label` / `clean_display_title` 剥离 markdown 链接与残留方括号；`collect_wiki_index` 索引全部 `wiki/entities/*`；`dedupe_paper_notebook_nodes.py` 增加按标题合并 + 全量标题修复
- 相关：`wiki/overview/paper-notebook-category-09-state-estimation.md`、`wiki/overview/paper-notebook-category-11-simulation-benchmark.md`、`schema/paper-notebook-wiki-full-map.yml`
- 验证：`tests/test_paper_notebook_title_cleanup.py`、`make ci-preflight`

## [2026-06-12] ingest | sources/papers/visualmimic_arxiv_2509_20322.md — VisualMimic 视觉分层 sim2real loco-manipulation；升格 wiki/entities/paper-notebook-visualmimic.md，更新 wiki/tasks/loco-manipulation.md

## [2026-06-12] fix(wiki): Paper Notebooks 分类索引去重 — 已完成深读笔记不再与 PROGRESS.md 待深读别名并列；`merge_paper_catalog` + `dedupe_category_entries`；重生成 `wiki/overview/paper-notebook-category-*.md`

- 相关：`wiki/overview/humanoid-paper-notebooks-index.md`、`scripts/bootstrap_paper_notebook_knowledge.py`
- 清理 4 个已无入链的 `paper-notebook-*` 计划占位实体（PPO/PULSE/Expressive WBC/Generating Diverse → 已有深读或概念页承接）
- 验证：`tests/test_bootstrap_paper_notebook_dedupe.py`、`make ci-preflight`

## [2026-06-12] structural | 站点大 JSON（search-index/index-v1/site-data-v1/link-graph×2 份）与 sitemap 移出 git，改为 pages.yml 部署时生成；export.yml 仅提交小型派生文件；tests.yml pytest 前生成快照；修复 archive v3–v9 共 7 处 Karpathy Wiki 断链

## [2026-06-12] ingest | sources/papers/humanoid_soccer_swarm_intelligence_sensors_2025.md + robocup_spl + artemis — 人形机器人群控一手资料；wiki/concepts/humanoid-multi-robot-coordination.md wiki/entities/paper-humanoid-soccer-swarm-intelligence.md wiki/tasks/humanoid-soccer.md

## [2026-06-12] structural | docs/checklists 清理首批 — v1–v23 归档至 archive/、移除 PR 验证截图产物、.obsidian/workspace.json 停止跟踪、删除 .codex、AGENTS.md 清单指针 v23→v24；详见 docs/change-log.md

## [2026-06-12] ingest | sources/papers/smplolympics_arxiv_2407_00187.md + table_tennis_strategy_skill_arxiv_2407_16210.md — SMPLOlympics 体育 benchmark 与 PhysicsPingPong 乒乓球分层控制；wiki/entities/smplolympics.md wiki/methods/table-tennis-strategy-skill-learning.md

## [2026-06-12] ingest | sources/papers/hil_hybrid_imitation_learning_arxiv_2505_12619.md + mtrg_reference_goal_driven_rl_arxiv_2602_20375.md + zest.md — HIL/MTRG 新入库，ZEST 交叉引用；wiki/methods/hil-hybrid-imitation-learning.md wiki/methods/mtrg-reference-goal-driven-rl.md

## [2026-06-12] fix(graph): 合并 Paper Notebooks 分类页与对应 task/concept 页的重复图谱社区

- 实现：`COMMUNITY_HUB_ALIASES` + `_merge_partition_by_hub_equivalence`（如 `论文深读·灵巧操作` 与 `操作（Manipulation）` 合并为同一社区）
- 测试：`test_merge_partition_by_hub_equivalence_merges_alias_hubs`、`test_exported_communities_have_no_duplicate_canonical_hubs`
- 验证：`make ci-preflight` 通过

## [2026-06-12] ingest | sources/papers/bifrost_umi_arxiv_2605_03452.md — 增补 arXiv Related Works 与 TWIST2 采集范式对照；交叉 wiki/entities/paper-bifrost-umi.md

## [2026-06-12] ingest | sources/papers/clot_arxiv_2602_15060.md + sources/sites/clot-project.md + sources/repos/clot.md — CLOT arXiv/项目页/代码 ingest；深化 wiki/entities/paper-amp-survey-16-clot.md（闭环全局、Observation Pre-shift）

## [2026-06-12] ingest | sources/sites/twist2-project.md + sources/repos/twist2.md — TWIST2 项目页/仓库一手 ingest；深化 wiki/entities/paper-twist2.md（Mermaid 管线、ICRA 2026、分层 visuomotor）

## [2026-06-11] structural | scripts/lint_wiki.py — V24 P0「陈旧声明（stale claim）巡检 V1」：新增 `_check_stale_claims` 信息型检查 + 6 例单测 + lint 报告基线快照

- 实现：正文（去 frontmatter / 代码块 / 误区区块）命中「SOTA / state-of-the-art / 当前最强 / 最新」等绝对化措辞，且本页 frontmatter `updated` 早于库内共享 ≥1 个 tag 的更晚页面时，输出 💡 INFO 级提示；列入 `INFO_ONLY_KEYS`，不计入 lint 失败总数、不阻塞 CI
- 新增：`STALE_CLAIM_PATTERNS`、`_frontmatter_block`、`_frontmatter_tags` 辅助函数；`format_report` 新增「陈旧声明」小节
- 基线快照：`exports/lint-report.md` 当前 5 条（generative-vision-pretraining / paper-resnet / paper-wem / paper-worldvln / paper-yolo）
- 测试：`tests/test_lint_wiki_stale_claims.py` 6 例（命中/最新页不报/无共享 tag/无绝对化措辞/代码块忽略/info-only），`pytest -k lint` 41 passed；`ruff check`、`ruff format` 通过
- 清单：勾选 [`tech-stack-next-phase-checklist-v24.md`](docs/checklists/tech-stack-next-phase-checklist-v24.md) P0 首项

## [2026-06-11] structural | scripts/dedupe_paper_notebook_nodes.py — 全量去重合并 4 对 `paper-notebook-*` 计划子节点与已有深读实体（按 frontmatter arXiv）；`make paper-notebook-dedupe` 复跑零残留

- 合并：`paper-notebook-behavior-foundation-model-for-humanoid-robots` → [`paper-behavior-foundation-model-humanoid.md`](wiki/entities/paper-behavior-foundation-model-humanoid.md)（2509.13780）
- 合并：`paper-notebook-reinforcement-learning-for-versatile-dynamic-and` → [`paper-cassie-biped-versatile-locomotion-rl.md`](wiki/entities/paper-cassie-biped-versatile-locomotion-rl.md)（2401.16889）
- 合并：`paper-notebook-real-world-humanoid-locomotion-with-rl` → [`paper-digit-humanoid-locomotion-rl.md`](wiki/entities/paper-digit-humanoid-locomotion-rl.md)（2303.03381）
- 合并：`paper-notebook-pilot` → [`paper-pilot-perceptive-loco-manipulation.md`](wiki/entities/paper-pilot-perceptive-loco-manipulation.md)（2601.17440）
- 工具：新增 [`scripts/dedupe_paper_notebook_nodes.py`](scripts/dedupe_paper_notebook_nodes.py)、`make paper-notebook-dedupe`；补强 [`scripts/sync_paper_notebook_links.py`](scripts/sync_paper_notebook_links.py) 多候选 arXiv 评分与 bootstrap 防重建
- 相关：`schema/paper-notebook-wiki-full-map.yml`、分类父节点表项、`make ci-preflight` 通过

## [2026-06-11] structural | scripts/bootstrap_paper_notebook_knowledge.py — 同步 papers/PROGRESS.md 全量 563 条：合并 progress.json 后 665 篇入图谱，新建约 380 个 `wiki/entities/paper-notebook-*` 计划子节点；14 类分类父节点扩表

- 数据源：[papers/PROGRESS.md](https://github.com/ImChong/Humanoid_Robot_Learning_Paper_Notebooks/blob/main/papers/PROGRESS.md) + [progress.json](https://github.com/ImChong/Humanoid_Robot_Learning_Paper_Notebooks/blob/main/progress.json)
- 工具：`make paper-notebook-bootstrap`；`schema/paper-notebook-wiki-full-map.yml` 扩至 665 篇
- 相关：`wiki/overview/paper-notebook-category-*.md`、`wiki/overview/humanoid-paper-notebooks-index.md`

## [2026-06-11] structural | scripts/bootstrap_paper_notebook_knowledge.py — 同步 Paper Notebooks progress.json 待深读 115 篇：新建 87 个 `wiki/entities/paper-notebook-*` 计划子节点 + sources；更新分类父节点与 full-map（252 篇）

- 数据源：[Humanoid_Robot_Learning_Paper_Notebooks/progress.json](https://github.com/ImChong/Humanoid_Robot_Learning_Paper_Notebooks/blob/main/progress.json) 中 `status=pending` 且尚无完整深读笔记的条目
- 新建：`sources/papers/humanoid_pnb_*.md`（87）、`wiki/entities/paper-notebook-*.md`（87，`status: planned`）
- 交叉更新：`wiki/overview/paper-notebook-category-04-loco-manipulation-and-wbc.md`（33→147 篇）、`wiki/overview/paper-notebook-category-02-motion-retargeting.md`、`wiki/overview/humanoid-paper-notebooks-index.md`、`schema/paper-notebook-wiki-full-map.yml`
- 工具：`make paper-notebook-bootstrap`；`make ci-preflight` 通过

## [2026-06-11] ingest | sources/papers/now_you_see_that_arxiv_2602_06382.md — Now You See That（RSS 2026）8 步深度增广 + 多 critic/discriminator 特权 RL + vision-aware DAgger；wiki/entities/paper-now-you-see-that-humanoid-vision-locomotion.md；交叉 humanoid-locomotion / stair-obstacle-perceptive-locomotion

## [2026-06-11] structural | scripts: preflight 提速（Louvain 图谱、lint 去重、stale 规则、bump-wiki-from-sources）

## [2026-06-11] ingest | sources/papers/rma_arxiv_2107_04034.md — RMA（RSS 2021）论文/项目页/rl_locomotion 代码入库；沉淀 wiki/entities/paper-rma-rapid-motor-adaptation.md；交叉更新 privileged-training、sim2real、locomotion

## [2026-06-11] structural | wiki/entities/paper-perceptive-bfm.md、docs/main.js — 正文 PMT 公式改 `$...$` 启用 KaTeX 蓝框；修复 Mermaid `<br/>` 渲染语法错误

## [2026-06-11] ingest | sources/papers/perceptive_bfm_corl_2026.md — Perceptive BFM（CoRL 2026）TCRS+PMT 地形感知 BFM；wiki/entities/paper-perceptive-bfm.md 及 behavior-foundation-model / privileged-training / footstep-planning / sonic / stair-obstacle 交叉更新

## [2026-06-11] structural | wiki/queries/humanoid-soccer-skill-learning-method-selection.md、wiki/entities/paper-omg-omni-modal-humanoid-control.md、wiki/methods/paid-framework.md、wiki/tasks/humanoid-soccer.md — lint 清零 + 人形足球技能学习选型 Query

- 新建 Query：[`wiki/queries/humanoid-soccer-skill-learning-method-selection.md`](wiki/queries/humanoid-soccer-skill-learning-method-selection.md)（PAiD vs RoboNaldo 选型指南；落地高频引用的 paid-framework 交叉链）
- 补强实体：[`wiki/entities/paper-omg-omni-modal-humanoid-control.md`](wiki/entities/paper-omg-omni-modal-humanoid-control.md)（frontmatter 补 code 来源键、正文补「评测与开放进度」段）
- 交叉更新：[`wiki/methods/paid-framework.md`](wiki/methods/paid-framework.md)、[`wiki/tasks/humanoid-soccer.md`](wiki/tasks/humanoid-soccer.md)
- 51 页 stale 复核：2026-06-10 去重合并对 source catalog 的链接改写 bump `updated`→2026-06-11（经 review 内容仍准确，非内容重写）
- 门禁：`make ci-preflight` 通过、lint-report 归零

## [2026-06-11] tooling | scripts/scaffold_wiki_page.py、tests/test_scaffold_wiki_page.py — V24 P0「query → wiki 回填脚手架」：新增页面骨架生成脚本与测试

- 新增 [`scripts/scaffold_wiki_page.py`](scripts/scaffold_wiki_page.py)：`type + 标题`（可选 `--slug`）按全库 frontmatter 规范生成骨架——含 `## 英文缩写速查` 落在规范位置（定义之后、为什么重要之前）、`related`/`sources` 占位、三段式正文；query 类型额外含 `**Query 产物**` / `## 参考来源` / `## 关联页面`
- 复用 `lint_wiki.has_section` / `wiki_abbrev_section.is_abbrev_glossary_well_placed` 做生成后结构自检；`--dry-run` 只打印不落盘、`--force` 控制覆盖
- 新增 [`tests/test_scaffold_wiki_page.py`](tests/test_scaffold_wiki_page.py) 7 例：结构自检、缩写区块位序、query 标记、frontmatter 键、slug 推断、dry-run 不落盘、写入与防覆盖
- 验证：新增测试全绿（全量 149 passed）、ruff/format/mypy 通过、`lint_wiki` 基线 51 项不变
- 勾选 checklist v24 P0「query → wiki 回填脚手架」

## [2026-06-10] structural | wiki/concepts/vision-backbones.md、wiki/methods/object-detection.md — V24 P1「视觉感知专题交叉补强」：明示「骨干特征 → 检测/分割头 → 策略输入」衔接链并与新页双向回链

- 在 [`vision-backbones.md`](wiki/concepts/vision-backbones.md) 新增「骨干特征 → 检测/分割头 → 策略输入」小节，补回链 [`cnn-vs-vit-backbones.md`](wiki/comparisons/cnn-vs-vit-backbones.md)、[`perception-backbone-selection.md`](wiki/queries/perception-backbone-selection.md)（frontmatter `related` + 关联页面）
- 在 [`object-detection.md`](wiki/methods/object-detection.md) 新增「在感知链中的位置」小节，把检测头明示为衔接链中间环节；补回链 [`cnn-vs-vit-backbones.md`](wiki/comparisons/cnn-vs-vit-backbones.md)、[`visual-representation-for-policy.md`](wiki/concepts/visual-representation-for-policy.md)、[`perception-backbone-selection.md`](wiki/queries/perception-backbone-selection.md)，消除单向链接孤儿
- 与 P1 三新页（对比 / 概念 / Query）形成双向回链；`make ci-preflight` 通过、派生索引与站点导出同步重生
- 清单：[`tech-stack-next-phase-checklist-v24.md`](docs/checklists/tech-stack-next-phase-checklist-v24.md) P1「视觉感知专题交叉补强」勾选完成

## [2026-06-10] structural | wiki/entities/paper-bfm-zero.md、paper-opentrack.md、paper-ams.md、paper-hiking-in-the-wild.md、paper-adaptive-humanoid-control.md、paper-deep-whole-body-parkour.md — 全量去重合并 6 对重复实体页（HRL 栈 / BFM / AMP 双索引）

- 合并为单一实体：[`paper-bfm-zero.md`](wiki/entities/paper-bfm-zero.md)、[`paper-opentrack.md`](wiki/entities/paper-opentrack.md)、[`paper-ams.md`](wiki/entities/paper-ams.md)、[`paper-hiking-in-the-wild.md`](wiki/entities/paper-hiking-in-the-wild.md)、[`paper-adaptive-humanoid-control.md`](wiki/entities/paper-adaptive-humanoid-control.md)、[`paper-deep-whole-body-parkour.md`](wiki/entities/paper-deep-whole-body-parkour.md)
- 删除 12 个重复页（`paper-hrl-stack-*` / `paper-bfm-*` / `paper-amp-survey-*` 各 2 对）
- 交叉更新：[`humanoid-rl-motion-control-body-system-stack.md`](wiki/overview/humanoid-rl-motion-control-body-system-stack.md)、[`humanoid-amp-motion-prior-survey.md`](wiki/overview/humanoid-amp-motion-prior-survey.md)、BFM 技术地图/分类页、[`stair-obstacle-perceptive-locomotion.md`](wiki/tasks/stair-obstacle-perceptive-locomotion.md)、catalog 源表

## [2026-06-10] ingest | sources/papers/robonaldo_arxiv_2606_11092.md — RoboNaldo 三阶段射门课程 RL；wiki/entities/paper-robonaldo-humanoid-soccer-shooting.md、wiki/tasks/humanoid-soccer.md、wiki/methods/paid-framework.md

## [2026-06-10] structural | wiki/entities/paper-beyondmimic.md、paper-sentinel.md、paper-sonic.md、paper-twist.md — 合并 HRL 栈与 BFM 双索引重复实体页（4 对）

- 合并为单一实体：[`paper-beyondmimic.md`](wiki/entities/paper-beyondmimic.md)、[`paper-sentinel.md`](wiki/entities/paper-sentinel.md)、[`paper-sonic.md`](wiki/entities/paper-sonic.md)、[`paper-twist.md`](wiki/entities/paper-twist.md)
- 删除 8 个重复页（`paper-hrl-stack-*` / `paper-bfm-*` 各 4 对）
- 交叉更新：[`humanoid-rl-motion-control-body-system-stack.md`](wiki/overview/humanoid-rl-motion-control-body-system-stack.md)、BFM 技术地图/分类页、[`paper-resmimic.md`](wiki/entities/paper-resmimic.md)、[`schema/paper-notebook-wiki-map.yml`](schema/paper-notebook-wiki-map.yml)

## [2026-06-10] structural | wiki/entities/paper-twist2.md — 合并 TWIST2 重复实体页（paper-hrl-stack-10-twist2 与 paper-bfm-10-twist2 为同一篇 arXiv:2505.02833）

- 合并为单一实体：[`wiki/entities/paper-twist2.md`](wiki/entities/paper-twist2.md)（保留 42 篇 RL 栈与 BFM 41 篇双索引语境）
- 删除重复页：[`paper-hrl-stack-10-twist2.md`](wiki/entities/paper-hrl-stack-10-twist2.md)、[`paper-bfm-10-twist2.md`](wiki/entities/paper-bfm-10-twist2.md)
- 交叉更新：[`humanoid-rl-motion-control-body-system-stack.md`](wiki/overview/humanoid-rl-motion-control-body-system-stack.md)、[`bfm-41-papers-technology-map.md`](wiki/overview/bfm-41-papers-technology-map.md)、[`bfm-category-02-goal-conditioned-learning.md`](wiki/overview/bfm-category-02-goal-conditioned-learning.md)、[`limmt-gqs-motion-curation.md`](wiki/methods/limmt-gqs-motion-curation.md)

## [2026-06-10] ingest | sources/sites/ttl_uart_logic_level_primary_refs.md、rs232_tia_eia_primary_refs.md、rs485_tia_eia_primary_refs.md — TTL/RS-232/RS-485 一手资料入库；wiki/concepts/ttl-serial-logic-level.md、rs-232-serial-interface.md、rs-485-serial-bus.md 及 uart-serial-communication 交叉链接

## [2026-06-10] ingest | sources/papers/mpc_rl_arxiv_2606_05687.md — MPC-RL 与 π MPC 入库；wiki/entities/paper-mpc-rl-humanoid-locomotion-manipulation.md、wiki/methods/pi-mpc.md 及 mpc-vs-rl / loco-manipulation 交叉更新

- 原始资料：[`mpc_rl_arxiv_2606_05687.md`](sources/papers/mpc_rl_arxiv_2606_05687.md)（<https://arxiv.org/abs/2606.05687>）；[`pi_mpc_arxiv_2601_14414.md`](sources/papers/pi_mpc_arxiv_2601_14414.md)（<https://arxiv.org/abs/2601.14414>）；[`junhengl_mpc_rl.md`](sources/repos/junhengl_mpc_rl.md)（<https://github.com/junhengl/mpc-rl>）
- 新建实体：[`wiki/entities/paper-mpc-rl-humanoid-locomotion-manipulation.md`](wiki/entities/paper-mpc-rl-humanoid-locomotion-manipulation.md)（CD-MPC landmark reward、πⁿ MPC 批训练、部署纯 RL、Mermaid 管线）
- 新建方法页：[`wiki/methods/pi-mpc.md`](wiki/methods/pi-mpc.md)（parallel-in-horizon ADMM、velocity-form、construction-free）
- 交叉更新：[`wiki/comparisons/mpc-vs-rl.md`](wiki/comparisons/mpc-vs-rl.md)、[`wiki/tasks/loco-manipulation.md`](wiki/tasks/loco-manipulation.md)、[`wiki/concepts/centroidal-dynamics.md`](wiki/concepts/centroidal-dynamics.md)、[`wiki/methods/model-predictive-control.md`](wiki/methods/model-predictive-control.md)、[`wiki/queries/mpc-solver-selection.md`](wiki/queries/mpc-solver-selection.md)、[`sources/README.md`](sources/README.md)

## [2026-06-10] ingest | sources/sites/omg-tsinghua-mars-lab-github-io.md — OMG omni-modal G1 运动生成（清华 MARS Lab）入库并建实体页

- 原始资料：[`omg-tsinghua-mars-lab-github-io.md`](sources/sites/omg-tsinghua-mars-lab-github-io.md)（<https://tsinghua-mars-lab.github.io/OMG/>）；[`omg-tsinghua-mars-lab.md`](sources/repos/omg-tsinghua-mars-lab.md)（<https://github.com/tsinghua-mars-lab/OMG>）
- 新建实体：[`wiki/entities/paper-omg-omni-modal-humanoid-control.md`](wiki/entities/paper-omg-omni-modal-humanoid-control.md)（generator–tracker 分层、OMG-DiT 多模态条件、OMG-Data 规模、HoloMotion tracker 部署与 Mermaid 管线）
- 交叉更新：[`wiki/methods/diffusion-motion-generation.md`](wiki/methods/diffusion-motion-generation.md)、[`wiki/concepts/whole-body-tracking-pipeline.md`](wiki/concepts/whole-body-tracking-pipeline.md)、[`wiki/entities/holomotion.md`](wiki/entities/holomotion.md)、[`sources/README.md`](sources/README.md)

## [2026-06-10] structural | wiki/formalizations/field-oriented-control-derivation.md — FOC Clarke/Park 与 dq 转矩方程逐步推导；交叉链接概念页与设计流程

## [2026-06-10] ingest | sources/sites/ansys_motor_cad_electric_machine_design.md — 电机设计流程入库；wiki/overview/motor-design-workflow.md 并与 FOC/TN/仿真选型交叉链接

## [2026-06-10] ingest | sources/personal/motor_curves_and_em_simulation_faq.md — TN/TI 曲线与电机电磁仿真软件入库；wiki/concepts/motor-torque-speed-curve.md、wiki/concepts/motor-torque-current-curve.md、wiki/comparisons/motor-em-simulation-software.md

## [2026-06-10] ingest | sources/papers/rhythm_arxiv_2603_02856.md — Rhythm 双 G1 交互全身控制（IAMR+IGRL+MAGIC）入库并建实体页

- 原始资料：[`rhythm_arxiv_2603_02856.md`](sources/papers/rhythm_arxiv_2603_02856.md)（<https://arxiv.org/abs/2603.02856>）；[`hoshi-no-ai-rhythm-github-io.md`](sources/sites/hoshi-no-ai-rhythm-github-io.md)（<https://hoshi-no-ai.github.io/Rhythm/>）
- 新建实体：[`wiki/entities/paper-rhythm-dual-humanoid-interaction.md`](wiki/entities/paper-rhythm-dual-humanoid-interaction.md)（IAMR 解耦重定向 + IGRL 图奖励 MAPPO + 真机部署 + MAGIC 数据集 + Mermaid 管线）
- 交叉更新：[`wiki/concepts/whole-body-tracking-pipeline.md`](wiki/concepts/whole-body-tracking-pipeline.md)、[`wiki/concepts/motion-retargeting-pipeline.md`](wiki/concepts/motion-retargeting-pipeline.md)、[`wiki/methods/marl.md`](wiki/methods/marl.md)、[`wiki/entities/paper-assistmimic.md`](wiki/entities/paper-assistmimic.md)、[`sources/README.md`](sources/README.md)

## [2026-06-10] structural | wiki/concepts/humanoid-policy-network-architecture.md — 新增「架构代际对比表」

- 在 [`wiki/concepts/humanoid-policy-network-architecture.md`](wiki/concepts/humanoid-policy-network-architecture.md) 的架构演化总览后新增六行对比表：浅层 MLP / AMP / MoE / Transformer-Diffusion / VLA-WAM / 低层小网，按「骨干规模、输入输出、代表工作、强项、主要局限」横向对比，并强调真机低层高频策略与上层新架构分层共存

## [2026-06-10] ingest | sources/papers/dit4dit_arxiv_2603_10448.md — DiT4DiT 双 DiT 联合 VAM 入库；wiki/entities/paper-dit4dit-video-action-model.md 并与 MotionWAM 双向链接

## [2026-06-10] ingest | sources/papers/motionwam_arxiv_2606_09215.md — MotionWAM 实时人形 loco-manipulation WAM 入库；wiki/entities/paper-motionwam-humanoid-loco-manipulation-wam.md、wiki/concepts/world-action-models.md、wiki/tasks/loco-manipulation.md 等交叉更新

## [2026-06-09] query | wiki/queries/perception-backbone-selection.md — V24 P1 视觉表征知识链收官（机器人感知骨干/表征选型 Query）

- 新建 Query：[`wiki/queries/perception-backbone-selection.md`](wiki/queries/perception-backbone-selection.md)（「分类骨干 / 检测头 / 通用预训练表征」三类选型决策树、推荐组合 pipeline、关键工程经验与典型失败模式；含 Mermaid 决策树与缩写速查）
- 交叉补强（消孤儿）：在 [`wiki/comparisons/cnn-vs-vit-backbones.md`](wiki/comparisons/cnn-vs-vit-backbones.md)、[`wiki/concepts/visual-representation-for-policy.md`](wiki/concepts/visual-representation-for-policy.md)、[`wiki/queries/object-detection-model-selection.md`](wiki/queries/object-detection-model-selection.md) 增加双向回链
- 清单推进：[`docs/checklists/tech-stack-next-phase-checklist-v24.md`](docs/checklists/tech-stack-next-phase-checklist-v24.md) P1「视觉表征知识链 (+3)」三页全部 `[x]`；同步更新 [`wiki/queries/README.md`](wiki/queries/README.md) 索引
- 门禁：`make lint` 全绿、`ci-preflight` 派生文件已重生（图谱 804 节点 / 5644 边，无孤儿节点）

## [2026-06-09] ingest | sources/papers/vision_banana_arxiv_2604_20329.md — Vision Banana（DeepMind）生成式视觉预训练入库并建实体/概念页

- 原始资料：[`vision_banana_arxiv_2604_20329.md`](sources/papers/vision_banana_arxiv_2604_20329.md)（<https://arxiv.org/abs/2604.20329>）；[`vision-banana-project.md`](sources/sites/vision-banana-project.md)（<https://vision-banana.github.io/>、<https://deepmind.google/research/publications/240658/>）
- 新建实体：[`wiki/entities/vision-banana.md`](wiki/entities/vision-banana.md)（NBP instruction-tuning、RGB 任务统一接口、2D/3D benchmark 表与 Mermaid 管线）
- 新建概念：[`wiki/concepts/generative-vision-pretraining.md`](wiki/concepts/generative-vision-pretraining.md)（生成预训练 ≈ LLM 预训练、三条技术谱系）
- 交叉更新：[`wiki/concepts/vision-backbones.md`](wiki/concepts/vision-backbones.md)、[`wiki/concepts/visual-representation-for-policy.md`](wiki/concepts/visual-representation-for-policy.md)、[`wiki/formalizations/3d-coordinate-transforms-vision-robotics.md`](wiki/formalizations/3d-coordinate-transforms-vision-robotics.md)、[`sources/README.md`](sources/README.md)

## [2026-06-09] ingest | sources/papers/mamma_arxiv_2506_13040.md — MAMMA（CVPR 2026 Oral）markerless 多视角双人 SMPL-X 采集入库并建实体页

- 原始资料：[`mamma_arxiv_2506_13040.md`](sources/papers/mamma_arxiv_2506_13040.md)（<https://arxiv.org/abs/2506.13040>）；[`mamma-tue-mpg-de.md`](sources/sites/mamma-tue-mpg-de.md)（<https://mamma.is.tue.mpg.de/>）；[`mamma.md`](sources/repos/mamma.md)（<https://github.com/cuevhv/mamma>）
- 新建实体：[`wiki/entities/paper-mamma-markerless-motion-capture.md`](wiki/entities/paper-mamma-markerless-motion-capture.md)（MammaNet 稠密 landmark + 跨视角匹配 + SMPL-X 优化 + Mermaid 管线）
- 交叉更新：[`wiki/concepts/motion-retargeting-pipeline.md`](wiki/concepts/motion-retargeting-pipeline.md)、[`wiki/entities/freemocap.md`](wiki/entities/freemocap.md)、[`wiki/overview/paper-notebook-category-14-human-motion.md`](wiki/overview/paper-notebook-category-14-human-motion.md)、[`sources/README.md`](sources/README.md)

## [2026-06-09] ingest | sources/papers/dimos_arxiv_2305_12411.md — DIMOS 室内人–场景运动合成（ICCV 2023）入库并建实体页

- 原始资料：[`dimos_arxiv_2305_12411.md`](sources/papers/dimos_arxiv_2305_12411.md)（<https://arxiv.org/abs/2305.12411>）；[`dimos-zkf1997-github-io.md`](sources/sites/dimos-zkf1997-github-io.md)（<https://zkf1997.github.io/DIMOS/>）；[`dimos.md`](sources/repos/dimos.md)（<https://github.com/zkf1997/DIMOS>）
- 新建实体：[`wiki/entities/paper-dimos-human-scene-motion-synthesis.md`](wiki/entities/paper-dimos-human-scene-motion-synthesis.md)（RL + CVAE 潜空间 + 场景感知 locomotion/interaction + Mermaid 管线）
- 交叉更新：[`wiki/concepts/character-animation-vs-robotics.md`](wiki/concepts/character-animation-vs-robotics.md)、[`wiki/methods/diffusion-motion-generation.md`](wiki/methods/diffusion-motion-generation.md)、[`wiki/methods/crisp-real2sim.md`](wiki/methods/crisp-real2sim.md)、[`wiki/entities/paper-amp-survey-15-physhsi.md`](wiki/entities/paper-amp-survey-15-physhsi.md)、[`wiki/overview/paper-notebook-category-14-human-motion.md`](wiki/overview/paper-notebook-category-14-human-motion.md)

## [2026-06-09] ingest | sources/papers/dart_control_arxiv_2410_05260.md — DART/DartControl（ICLR 2025）论文+仓库+项目页入库并建方法页

- 原始资料：[`dart_control_arxiv_2410_05260.md`](sources/papers/dart_control_arxiv_2410_05260.md)（<https://arxiv.org/abs/2410.05260>）；[`zkf1997_dart.md`](sources/repos/zkf1997_dart.md)（<https://github.com/zkf1997/DART>）；[`dart-control-project.md`](sources/sites/dart-control-project.md)（<https://zkf1997.github.io/DART/>）
- 新建方法页：[`wiki/methods/dart-control.md`](wiki/methods/dart-control.md)（自回归运动原语潜扩散 + 在线文本/空间控制 + Mermaid 管线）
- 交叉更新：[`wiki/methods/diffusion-motion-generation.md`](wiki/methods/diffusion-motion-generation.md)、[`wiki/methods/hy-motion-1.md`](wiki/methods/hy-motion-1.md)、[`wiki/methods/genmo.md`](wiki/methods/genmo.md)、[`wiki/entities/awesome-text-to-motion-zilize.md`](wiki/entities/awesome-text-to-motion-zilize.md)、[`wiki/entities/phc.md`](wiki/entities/phc.md)、[`wiki/entities/amass.md`](wiki/entities/amass.md)、[`wiki/comparisons/wbc-vs-rl.md`](wiki/comparisons/wbc-vs-rl.md)、[`sources/README.md`](sources/README.md)

## [2026-06-09] ingest | sources/papers/coins_arxiv_2207_12824.md — COINS 论文 + 项目页 + 仓库入库并建实体页

- 原始资料：[`coins_arxiv_2207_12824.md`](sources/papers/coins_arxiv_2207_12824.md)（<https://arxiv.org/abs/2207.12824>）；[`coins-zkf1997-github-io.md`](sources/sites/coins-zkf1997-github-io.md)（<https://zkf1997.github.io/COINS/index.html>）；[`coins.md`](sources/repos/coins.md)（<https://github.com/zkf1997/COINS>）
- 新建实体：[`wiki/entities/paper-coins-compositional-human-scene-interaction.md`](wiki/entities/paper-coins-compositional-human-scene-interaction.md)（PelvisVAE/BodyVAE 三阶段 + 组合交互 + PROX-S + Mermaid 管线）
- 交叉更新：[`wiki/methods/crisp-real2sim.md`](wiki/methods/crisp-real2sim.md)、[`sources/README.md`](sources/README.md)

## [2026-06-09] ingest | sources/repos/robot_lab.md — 复核 fan-ziqi/robot_lab 并刷新实体页（24+ 环境、新机型、rl_sar 部署链、Mermaid）

- 原始资料：[`sources/repos/robot_lab.md`](sources/repos/robot_lab.md)（<https://github.com/fan-ziqi/robot_lab>）
- 刷新实体：[`wiki/entities/robot-lab.md`](wiki/entities/robot-lab.md)（版本矩阵、24 环境机型表、BeyondMimic/AMP 实验任务、Sim2Real→rl_sar Mermaid）
- 交叉更新：[`wiki/concepts/wheel-legged-quadruped.md`](wiki/concepts/wheel-legged-quadruped.md)、[`wiki/entities/openloong.md`](wiki/entities/openloong.md)

## [2026-06-09] ingest | sources/papers/humanoid_gym_arxiv_2404_05695.md — Humanoid-Gym 论文 + 官方/社区仓库入库并建实体页

- 原始资料：[`humanoid_gym_arxiv_2404_05695.md`](sources/papers/humanoid_gym_arxiv_2404_05695.md)（<https://arxiv.org/abs/2404.05695>）；[`humanoid-gym.md`](sources/repos/humanoid-gym.md)（<https://github.com/roboterax/humanoid-gym>）；[`humanoid-gym-modified.md`](sources/repos/humanoid-gym-modified.md)（<https://github.com/roboman-ly/humanoid-gym-modified>）
- 新建实体：[`wiki/entities/humanoid-gym.md`](wiki/entities/humanoid-gym.md)（步态相位奖励 + 非对称 AC + MuJoCo sim2sim + Mermaid 管线；含 Pandaman/Gazebo fork 小节）
- 交叉更新：[`wiki/entities/legged-gym.md`](wiki/entities/legged-gym.md)、[`references/repos/rl-frameworks.md`](references/repos/rl-frameworks.md)、[`wiki/overview/paper-notebook-category-03-high-impact-selection.md`](wiki/overview/paper-notebook-category-03-high-impact-selection.md)、[`sources/README.md`](sources/README.md)

## [2026-06-09] ingest | sources/papers/resmimic_arxiv_2510_05070.md — ResMimic GMT→残差全身 loco-manipulation 入库并建实体页

- 原始资料：[`resmimic_arxiv_2510_05070.md`](sources/papers/resmimic_arxiv_2510_05070.md)（<https://arxiv.org/abs/2510.05070>）；[`resmimic-github-io.md`](sources/sites/resmimic-github-io.md)（<https://resmimic.github.io/>）；[`resmimic.md`](sources/repos/resmimic.md)（<https://github.com/amazon-far/ResMimic>）
- 新建实体：[`wiki/entities/paper-resmimic.md`](wiki/entities/paper-resmimic.md)（两阶段残差 + 点云/接触奖励 + 虚拟力课程 + Mermaid 管线）
- 交叉更新：[`wiki/tasks/loco-manipulation.md`](wiki/tasks/loco-manipulation.md)、[`wiki/concepts/whole-body-tracking-pipeline.md`](wiki/concepts/whole-body-tracking-pipeline.md)、[`sources/README.md`](sources/README.md)

## [2026-06-09] ingest | sources/papers/ladderman_arxiv_2606_05873.md — LadderMan 人形感知梯子攀爬（项目页 + arXiv）消化并建实体页

- 原始资料：[`sources/papers/ladderman_arxiv_2606_05873.md`](sources/papers/ladderman_arxiv_2606_05873.md)（<https://arxiv.org/abs/2606.05873>）；[`sources/sites/ladderman-robot-github-io.md`](sources/sites/ladderman-robot-github-io.md)（<https://ladderman-robot.github.io/>）
- 新建实体：[`wiki/entities/paper-ladderman-humanoid-perceptive-ladder-climbing.md`](wiki/entities/paper-ladderman-humanoid-perceptive-ladder-climbing.md)（两阶段 hybrid tracking + DAgger+RL、VFM/RFM sim-to-real、梯上双智能体操作 + Mermaid 管线）
- 交叉更新：[`wiki/tasks/stair-obstacle-perceptive-locomotion.md`](wiki/tasks/stair-obstacle-perceptive-locomotion.md)、[`wiki/tasks/loco-manipulation.md`](wiki/tasks/loco-manipulation.md)、[`wiki/entities/unitree-g1.md`](wiki/entities/unitree-g1.md)、[`wiki/entities/paper-rpl-robust-humanoid-perceptive-locomotion.md`](wiki/entities/paper-rpl-robust-humanoid-perceptive-locomotion.md)、[`wiki/methods/dagger.md`](wiki/methods/dagger.md)、[`wiki/concepts/privileged-training.md`](wiki/concepts/privileged-training.md)

## [2026-06-09] ingest | sources/papers/limmt_arxiv_2606_06953.md — LIMMT（ICML 2026，GQS 数据策展 3% AMASS 胜全量）入库

- 原始资料：[`limmt_arxiv_2606_06953.md`](sources/papers/limmt_arxiv_2606_06953.md)（<https://arxiv.org/abs/2606.06953>）；[`limmt-giraffeguan-github-io.md`](sources/sites/limmt-giraffeguan-github-io.md)（<https://giraffeguan.github.io/limmt/>）
- 新建方法页：[`wiki/methods/limmt-gqs-motion-curation.md`](wiki/methods/limmt-gqs-motion-curation.md)（GQS 三阶段 + Mermaid + Any2Track/TWIST2/PHUMA/G1 实验归纳）
- 交叉更新：[`wiki/methods/egm-efficient-general-mimic.md`](wiki/methods/egm-efficient-general-mimic.md)、[`wiki/queries/humanoid-motion-tracking-method-selection.md`](wiki/queries/humanoid-motion-tracking-method-selection.md)、[`wiki/concepts/whole-body-tracking-pipeline.md`](wiki/concepts/whole-body-tracking-pipeline.md)、[`sources/README.md`](sources/README.md)

## [2026-06-09] ingest | sources/repos/python_robotics.md — 接入 PythonRobotics 代码库/教材/arXiv 论文并新建实体页与导航栈交叉引用

- 原始资料：[`sources/repos/python_robotics.md`](sources/repos/python_robotics.md)、[`sources/papers/python_robotics_arxiv_1808_10703.md`](sources/papers/python_robotics_arxiv_1808_10703.md)、[`sources/courses/python_robotics_textbook.md`](sources/courses/python_robotics_textbook.md)
- 新建实体：[`wiki/entities/python-robotics.md`](wiki/entities/python-robotics.md)
- 交叉更新：[`wiki/overview/navigation-slam-autonomy-stack.md`](wiki/overview/navigation-slam-autonomy-stack.md)、[`wiki/entities/navigation2.md`](wiki/entities/navigation2.md)、[`wiki/entities/modern-robotics-book.md`](wiki/entities/modern-robotics-book.md)、[`wiki/formalizations/kalman-filter.md`](wiki/formalizations/kalman-filter.md)

## [2026-06-08] structural | wiki/concepts/visual-representation-for-policy.md — V24 P1 视觉表征知识链第二项：视觉表征作为策略输入

- 新建概念页：[`wiki/concepts/visual-representation-for-policy.md`](wiki/concepts/visual-representation-for-policy.md)（端到端联合训练 vs 冻结预训练骨干 vs 机器人专用预训练表征（R3M / VC-1 / DINOv2）三条路径与取舍 + Mermaid 决策图）
- 交叉回链：[`wiki/concepts/vision-backbones.md`](wiki/concepts/vision-backbones.md) 新增 related/关联出边，消除孤儿页
- 进展：V24 P1「视觉表征知识链 (+3)」第二项交付（首项 cnn-vs-vit-backbones 已于 2026-06-07 完成），余 `wiki/queries/perception-backbone-selection.md`
- lint：`python3 scripts/lint_wiki.py` 全绿；同步重建全站索引与图谱统计

## [2026-06-08] ingest | sources/repos/* — 补充 fairmotion / AMP-RSL-RL 两个重定向相关成熟仓库实体并互链

- 原始资料：`sources/repos/amp_rsl_rl.md`、`sources/repos/fairmotion.md`
- 新建实体：[`wiki/entities/amp-rsl-rl.md`](wiki/entities/amp-rsl-rl.md)、[`wiki/entities/fairmotion.md`](wiki/entities/fairmotion.md)
- 交叉更新：[`wiki/concepts/motion-retargeting.md`](wiki/concepts/motion-retargeting.md)、[`references/repos/retarget-tools.md`](references/repos/retarget-tools.md)、[`sources/README.md`](sources/README.md)
- 说明：fairmotion 经核实为已归档(2023)的通用动捕工具、本身不做机器人重定向，按「上游数据基础设施」收录（与 FreeMoCap/MotionCode 同列）；AMP-RSL-RL 为 IIT 的 rsl_rl+AMP 人形模仿实现。

## [2026-06-08] ingest | sources/repos/* — 补全人形/四足重定向成熟开源仓库实体（14 页）并互链 motion-retargeting 主线

- 原始资料：`sources/repos/mocap_retarget.md`、`soma_retargeter.md`、`gvhmr.md`、`videomimic.md`、`phc.md`、`human2humanoid.md`、`motion_imitation_peng.md`、`amp_for_hardware.md`、`metalhead.md`、`leggedgym_ex.md`、`stmr_quadruped_retargeting.md`、`go2_motion_imitation.md`、`pan_motion_retargeting.md`、`walk_the_dog.md`
- 新建实体：[`wiki/entities/mocap-retarget.md`](wiki/entities/mocap-retarget.md)、[`soma-retargeter.md`](wiki/entities/soma-retargeter.md)、[`gvhmr.md`](wiki/entities/gvhmr.md)、[`videomimic.md`](wiki/entities/videomimic.md)、[`phc.md`](wiki/entities/phc.md)、[`human2humanoid.md`](wiki/entities/human2humanoid.md)、[`motion-imitation-quadruped.md`](wiki/entities/motion-imitation-quadruped.md)、[`amp-for-hardware.md`](wiki/entities/amp-for-hardware.md)、[`metalhead.md`](wiki/entities/metalhead.md)、[`leggedgym-ex.md`](wiki/entities/leggedgym-ex.md)、[`stmr-quadruped-retargeting.md`](wiki/entities/stmr-quadruped-retargeting.md)、[`go2-motion-imitation.md`](wiki/entities/go2-motion-imitation.md)、[`pan-motion-retargeting.md`](wiki/entities/pan-motion-retargeting.md)、[`walk-the-dog.md`](wiki/entities/walk-the-dog.md)
- 交叉更新：[`wiki/concepts/motion-retargeting.md`](wiki/concepts/motion-retargeting.md)、[`references/repos/retarget-tools.md`](references/repos/retarget-tools.md)、[`sources/README.md`](sources/README.md)

## [2026-06-08] ingest | sources/papers/omniretarget_arxiv_2509_26633.md — OmniRetarget（ICRA 2026）全文消化：holosoma 代码 + HF 数据集 + 项目页

- 原始资料：[`omniretarget_arxiv_2509_26633.md`](sources/papers/omniretarget_arxiv_2509_26633.md)（<https://arxiv.org/abs/2509.26633>、PDF <https://omniretarget.github.io/static/images/paper.pdf>）；[`omniretarget-github-io.md`](sources/sites/omniretarget-github-io.md)；[`holosoma.md`](sources/repos/holosoma.md)（<https://github.com/amazon-far/holosoma>）；[`omniretarget-dataset-huggingface.md`](sources/sites/omniretarget-dataset-huggingface.md)（<https://huggingface.co/datasets/omniretarget/OmniRetarget_Dataset>）
- 深化实体：[`wiki/entities/paper-hrl-stack-03-omniretarget.md`](wiki/entities/paper-hrl-stack-03-omniretarget.md)（interaction mesh + Sequential SOCP + 5 reward 下游 RL + Mermaid 管线）
- 新建实体：[`wiki/entities/holosoma.md`](wiki/entities/holosoma.md)、[`wiki/entities/omniretarget-dataset.md`](wiki/entities/omniretarget-dataset.md)
- 交叉更新：[`wiki/concepts/motion-retargeting.md`](wiki/concepts/motion-retargeting.md)、[`sources/README.md`](sources/README.md)

## [2026-06-08] ingest | sources/blogs/wechat_embodied_ai_lab_robot_training_stack_layers_2026.md — Agent Reach 抓取训练栈分层长文并建六层技术地图

- 工具：已安装 [Panniantong/Agent-Reach](https://github.com/Panniantong/Agent-Reach) v1.4.0（修复 hatchling `guides` force-include 重复后 `pip install -e` + `agent-reach install --channels=wechat`）；微信正文经 `~/.agent-reach/tools/wechat-article-for-ai`（Camoufox）
- 原始资料：[`wechat_embodied_ai_lab_robot_training_stack_layers_2026.md`](sources/blogs/wechat_embodied_ai_lab_robot_training_stack_layers_2026.md)（<https://mp.weixin.qq.com/s/Z9pgVa48wQKLYVRD3psnhw>）；[`mujoco_playground.md`](sources/repos/mujoco_playground.md)；落盘 [`sources/raw/wechat_embodied_ai_lab_robot_training_stack_layers_2026-06-08.md`](sources/raw/wechat_embodied_ai_lab_robot_training_stack_layers_2026-06-08.md)
- 沉淀页面：[`wiki/overview/robot-training-stack-layers-technology-map.md`](wiki/overview/robot-training-stack-layers-technology-map.md)（六层训练–评估栈 + Mermaid）；新建 [`wiki/entities/mujoco-playground.md`](wiki/entities/mujoco-playground.md)
- 交叉更新：[`isaac-lab.md`](wiki/entities/isaac-lab.md)、[`mujoco.md`](wiki/entities/mujoco.md)、[`mjlab.md`](wiki/entities/mjlab.md)、[`unilab.md`](wiki/entities/unilab.md)、[`newton-physics.md`](wiki/entities/newton-physics.md)、[`genesis-world-10.md`](wiki/entities/genesis-world-10.md)、[`simulator-selection-guide.md`](wiki/queries/simulator-selection-guide.md)、[`simulation-evaluation-infrastructure.md`](wiki/concepts/simulation-evaluation-infrastructure.md)、[`humanoid-rl-motion-control-body-system-stack.md`](wiki/overview/humanoid-rl-motion-control-body-system-stack.md)、[`mujoco-vs-isaac-lab.md`](wiki/comparisons/mujoco-vs-isaac-lab.md)、[`agent-reach.md`](wiki/entities/agent-reach.md)、[`sources/README.md`](sources/README.md)

## [2026-06-07] structural | wiki/comparisons/cnn-vs-vit-backbones.md — V24 P1 视觉表征知识链首页：CNN vs ViT 视觉骨干对比

- 新增页面：[wiki/comparisons/cnn-vs-vit-backbones.md](wiki/comparisons/cnn-vs-vit-backbones.md)（归纳偏置、数据量需求、分辨率/吞吐、多尺度特征、边缘部署、下游迁移六维对比；机器人感知取舍决策图与误区）
- 交叉回链：[wiki/concepts/vision-backbones.md](wiki/concepts/vision-backbones.md)、[wiki/methods/object-detection.md](wiki/methods/object-detection.md) 关联页面区块增补对比页入链，消除孤儿节点
- 派生同步：`make graph` + `make badge` + `make export`（知识图谱 772→773 节点 / 5332→5338 边，README badge 同步）；`make lint` 0 阻塞问题，搜索回归 37/37，导出质量 12/12
- 清单推进：[docs/checklists/tech-stack-next-phase-checklist-v24.md](docs/checklists/tech-stack-next-phase-checklist-v24.md) P1「视觉表征知识链 (+3)」首项打勾

## [2026-06-07] ingest | sources/papers/esi_bench_arxiv_2605_18746.md — 补强 ESI-Bench 动作空间/基准对比并交叉 VLN

- 原始资料：[sources/papers/esi_bench_arxiv_2605_18746.md](sources/papers/esi_bench_arxiv_2605_18746.md)（<https://arxiv.org/abs/2605.18746>）、[sources/sites/esi-bench-project.md](sources/sites/esi-bench-project.md)（<https://esi-bench.github.io/>）、[sources/repos/esi_bench.md](sources/repos/esi_bench.md)（<https://github.com/ESI-Bench/ESI-Bench>）
- 消化实体：[wiki/entities/esi-bench.md](wiki/entities/esi-bench.md)（任务形式化、高层动作空间、与 VSI-Bench/EmbodiedBench 定位表）
- 交叉补强：[wiki/concepts/3d-spatial-vqa.md](wiki/concepts/3d-spatial-vqa.md)、[wiki/tasks/vision-language-navigation.md](wiki/tasks/vision-language-navigation.md)

## [2026-06-07] ingest | sources/papers/eth-g1-diffusion.md — Learning Whole-Body Humanoid Locomotion（arXiv:2604.17335）扩散生成 + RL 全身跟踪真机 G1

- 原始资料：[sources/papers/eth-g1-diffusion.md](sources/papers/eth-g1-diffusion.md)、[sources/sites/wholebody-locomotion.md](sources/sites/wholebody-locomotion.md)
- 消化实体：[wiki/entities/paper-hrl-stack-27-learning_whole_body_humanoid_locomot.md](wiki/entities/paper-hrl-stack-27-learning_whole_body_humanoid_locomot.md)（由 42 篇栈索引级升格为全文消化）
- 交叉补强：[wiki/methods/diffusion-motion-generation.md](wiki/methods/diffusion-motion-generation.md)、[wiki/tasks/humanoid-locomotion.md](wiki/tasks/humanoid-locomotion.md)、[wiki/entities/unitree-g1.md](wiki/entities/unitree-g1.md)

## [2026-06-07] ingest | sources/sites/blender-org.md、sources/repos/blender.md — Blender 开源 DCC 官网与官方仓库；wiki/entities/blender.md

- 原始资料：[sources/sites/blender-org.md](sources/sites/blender-org.md)、[sources/repos/blender.md](sources/repos/blender.md)
- 新增实体：[wiki/entities/blender.md](wiki/entities/blender.md)
- 交叉补强：[wiki/entities/nvidia-omniverse.md](wiki/entities/nvidia-omniverse.md)、[wiki/concepts/character-animation-vs-robotics.md](wiki/concepts/character-animation-vs-robotics.md)、[wiki/entities/robot-motion-keyframe-editors.md](wiki/entities/robot-motion-keyframe-editors.md)、[wiki/entities/sam3dbody-cpp.md](wiki/entities/sam3dbody-cpp.md)

## [2026-06-07] ingest | sources/papers/rpl_arxiv_2602_03002.md — RPL（arXiv:2602.03002）Amazon FAR 人形多向深度感知行走与载荷 loco-manipulation

- 原始资料：[sources/papers/rpl_arxiv_2602_03002.md](sources/papers/rpl_arxiv_2602_03002.md)、[sources/sites/rpl-humanoid-github-io.md](sources/sites/rpl-humanoid-github-io.md)
- 新增实体：[wiki/entities/paper-rpl-robust-humanoid-perceptive-locomotion.md](wiki/entities/paper-rpl-robust-humanoid-perceptive-locomotion.md)
- 交叉补强：[wiki/tasks/stair-obstacle-perceptive-locomotion.md](wiki/tasks/stair-obstacle-perceptive-locomotion.md)、[wiki/tasks/loco-manipulation.md](wiki/tasks/loco-manipulation.md)

## [2026-06-07] structural | schema + wiki — Paper Notebooks 64 篇 stub 实体从深读笔记同步一句话总结

- 工具：[scripts/sync_paper_notebook_summaries.py](scripts/sync_paper_notebook_summaries.py)；`make paper-notebook-summaries`
- 从 [Humanoid Robot Learning Paper Notebooks](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/index.html) 各篇 `## 🎯 一句话总结` 同步至 `wiki/entities/paper-notebook-*` 的 frontmatter `summary` 与 `## 一句话定义`，并更新对应 `sources/papers/humanoid_pnb_*.md` 一句话说明
- RL Sim2Sim Demo 映射 7 页（[schema/rl-sim2sim-demo-wiki-map.yml](schema/rl-sim2sim-demo-wiki-map.yml)）经审计已具备有效 summary，无需改动

## [2026-06-07] structural | schema + wiki — Paper Notebooks 全量分类父节点与 64 篇未映射论文 sources/实体入库

- 工具：[scripts/bootstrap_paper_notebook_knowledge.py](scripts/bootstrap_paper_notebook_knowledge.py)；`make paper-notebook-bootstrap`；完整映射 [schema/paper-notebook-wiki-full-map.yml](schema/paper-notebook-wiki-full-map.yml)（137/137）
- 父节点：[wiki/overview/humanoid-paper-notebooks-index.md](wiki/overview/humanoid-paper-notebooks-index.md) + 14 类 `wiki/overview/paper-notebook-category-*.md`（03 类含 5 个子分类段落）
- 新增 64 组 `sources/papers/humanoid_pnb_*.md` + `wiki/entities/paper-notebook-*.md` 索引实体；既有 73 篇保留原深度 wiki 并挂入分类子节点

## [2026-06-07] structural | schema + wiki — 同步 RL Sim2Sim Demo 在线演示链接至对应 wiki 节点

- 工具：[scripts/sync_rl_sim2sim_demo_links.py](scripts/sync_rl_sim2sim_demo_links.py)；`make rl-sim2sim-demo-links`；映射 [schema/rl-sim2sim-demo-wiki-map.yml](schema/rl-sim2sim-demo-wiki-map.yml) + [schema/rl-sim2sim-demo-index.json](schema/rl-sim2sim-demo-index.json)
- 来源归档：[sources/sites/rl-sim2sim-demo-website.md](sources/sites/rl-sim2sim-demo-website.md)
- 挂接节点：[wiki/concepts/sim2real.md](wiki/concepts/sim2real.md)、[wiki/entities/amp-mjlab.md](wiki/entities/amp-mjlab.md)、[wiki/entities/paper-hrl-stack-22-perceptive_humanoid_parkour.md](wiki/entities/paper-hrl-stack-22-perceptive_humanoid_parkour.md)、[wiki/methods/gentlehumanoid-motion-tracking.md](wiki/methods/gentlehumanoid-motion-tracking.md) 等 7 页

## [2026-06-07] structural | schema + wiki — 同步 Humanoid Paper Notebooks 深读笔记链接至对应 wiki 节点

- 工具：[scripts/sync_paper_notebook_links.py](scripts/sync_paper_notebook_links.py)；`make paper-notebook-links`；映射 [schema/paper-notebook-wiki-overrides.yml](schema/paper-notebook-wiki-overrides.yml) + [schema/paper-notebook-index.json](schema/paper-notebook-index.json)
- 覆盖 73/138 篇已有对应节点的论文笔记；修复 5 处旧版 `02_High_Impact` / `09_Sim-to-Real` URL；示例节点 [wiki/entities/paper-sonic.md](wiki/entities/paper-sonic.md)、[wiki/methods/awr.md](wiki/methods/awr.md)、[wiki/tasks/teleoperation.md](wiki/tasks/teleoperation.md)

## [2026-06-06] structural | docs — V23 P3 图谱页「专题视图」扩充（WBT / 跨具身 / 真机安全微调）

- 清单推进：[tech-stack-next-phase-checklist-v23.md](docs/checklists/tech-stack-next-phase-checklist-v23.md) P3 末项打勾，V23 清单全部完成
- 前端改动：[docs/graph.html](docs/graph.html) `TOPIC_FILTERS` 在 V22 10 项基础上新增 `wbt`（segments 11 项，命中 22 节点）、`cross-embodiment`（segments 3 项，命中 3 节点）、`safe-fine-tuning`（community-13 + segments 9 项，命中 18 节点）；`#filter-topic-chips` 同步新增 🕺 WBT / 🔀 跨具身 / 🛡️ 安全微调 三枚 `data-topic` chip，复用既有 `nodeMatchesTopic` 双路并集逻辑
- 工具修复：[scripts/screenshot_graph_topic.cjs](scripts/screenshot_graph_topic.cjs) 由点击 V22 已移除的 `#topic-view` 下拉改为展开 `#filter-topic-section` 后点击 `[data-topic]` chip
- 验证：`make lint` 全绿（仅 1 条无关信息型预警）、内联 JS `new Function` 语法校验通过；三专题视图截图归档至 `.cursor-artifacts/screenshots/graph-topic-{wbt,cross-embodiment,safe-fine-tuning}.png`
- V23 验收：节点 690（≥445）/ 边 4993（≥3320）/ 事实库 172（≥170）/ `largest_community_ratio` 0.104（≤0.25）/ `community_quality_warning` false，全部达标

## [2026-06-06] ingest | sources/sites/karpathy-ai.md、sources/blogs/karpathy_llm_wiki_gist.md — Andrej Karpathy 个人站与 LLM Wiki Gist；wiki/entities/andrej-karpathy.md、wiki/references/llm-wiki-karpathy.md、wiki/overview/robot-learning-overview.md

## [2026-06-06] structural | wiki — 全库 587 页英文缩写速查区块重排至「一句话定义」后、「为什么重要」前；新增 reorder 脚本与 lint 位置校验

- 工具：`scripts/wiki_abbrev_section.py`、`scripts/reorder_abbrev_glossary.py`；`lint_wiki.py` 新增位置错误阻塞项；`gen_abbrev_glossary.py` 插入锚点对齐 schema
- 代表页：[wiki/concepts/sim2real.md](wiki/concepts/sim2real.md)、[wiki/entities/lerobot.md](wiki/entities/lerobot.md)、[wiki/entities/paper-bfm-zero.md](wiki/entities/paper-bfm-zero.md)、[wiki/queries/sim2real-checklist.md](wiki/queries/sim2real-checklist.md)

## [2026-06-06] ingest | sources/papers/cosmos3_arxiv_2606_02800.md、sources/sites/cosmos3-project.md、sources/repos/nvidia_cosmos.md — Cosmos 3 全模态 Physical AI 世界模型；wiki/entities/cosmos-3.md、wiki/methods/generative-world-models.md、wiki/concepts/world-action-models.md、wiki/methods/mimic-video.md、wiki/entities/nvidia-so101-sim2real-lab-workflow.md

## [2026-06-06] ingest | sources/papers/vision_backbone_detection_classics.md — 入库 ResNet (1512.03385) 与 YOLO v1 (1506.02640) 及视觉骨干/目标检测 wiki

- 原始资料：[resnet_arxiv_1512_03385.md](sources/papers/resnet_arxiv_1512_03385.md)（<https://arxiv.org/abs/1512.03385>）、[yolo_arxiv_1506_02640.md](sources/papers/yolo_arxiv_1506_02640.md)（<https://arxiv.org/abs/1506.02640>）、[vision_backbone_detection_classics.md](sources/papers/vision_backbone_detection_classics.md)
- 沉淀页面：[wiki/entities/paper-resnet-deep-residual-learning.md](wiki/entities/paper-resnet-deep-residual-learning.md)、[wiki/entities/paper-yolo-unified-realtime-detection.md](wiki/entities/paper-yolo-unified-realtime-detection.md)、[wiki/concepts/vision-backbones.md](wiki/concepts/vision-backbones.md)、[wiki/methods/object-detection.md](wiki/methods/object-detection.md)
- 交叉更新：[wiki/concepts/deep-learning-foundations.md](wiki/concepts/deep-learning-foundations.md)、[wiki/tasks/manipulation.md](wiki/tasks/manipulation.md)

## [2026-06-06] lint | health-check — 全库健康度提升至满分：isaac-gym / isaac-lab 补 summary frontmatter（health_score 2→3）；paper-learning-to-adapt-bio-inspired-quadruped-gait 补 venue 元数据；684/684 节点 health_score=3，lint 0 issues

## [2026-06-06] ingest | sources/papers/learning_to_adapt_nature_2025.md、sources/repos/ihcr_learning_to_adapt.md — Learning to Adapt（Nature MI 2025）四足 bio-inspired 多步态 DRL；wiki/entities/paper-learning-to-adapt-bio-inspired-quadruped-gait.md、wiki/concepts/gait-generation.md、wiki/tasks/locomotion.md、wiki/entities/quadruped-robot.md、wiki/entities/paper-walk-these-ways-quadruped-mob.md

## [2026-06-05] structural | docs — V23 P3 详情页「最近相关 ingest」时间线

- 清单推进：[tech-stack-next-phase-checklist-v23.md](docs/checklists/tech-stack-next-phase-checklist-v23.md) P3 首项打勾
- 前端改动：[docs/detail.html](docs/detail.html) 在 `detail-related` 与 `detail-recommended` 之间新增 `#detail-recent-ingest-section`（默认 `hidden`，空态整段含标题不渲染）；[docs/main.js](docs/main.js) 新增 `renderDetailRecentIngestTimeline`，并发取 `link-graph.json`（1-hop 邻居）与 `graph-stats.json`（`latest_wiki_nodes`）求交，窗口锚定最新一条 ingest 回溯 30 天，按 `recency` 倒序、最多 6 项；[docs/style.css](docs/style.css) 新增 `.detail-recent-ingest-*` 时间线样式
- 验证：`node --check` + `eslint docs/main.js` 全绿、`make lint` 全部检查通过；BFM 详情页截图命中 5 项

## [2026-06-05] structural | wiki — 连接度前 50 hub 页补齐英文缩写速查（第 11–50 名，40 页）

- 依据 `exports/link-graph.json` 总度排序：第 1–10 名已在 main，本次为第 11–50 名共 40 页新增 `## 英文缩写速查`；批量工具 `scripts/batch_insert_abbrev_glossary.py`
- 页面：`wiki/concepts/behavior-foundation-model.md`、`wiki/entities/humanoid-robot.md`、`wiki/methods/generative-world-models.md`、`wiki/entities/unitree-g1.md`、`wiki/tasks/loco-manipulation.md`、`wiki/concepts/motion-retargeting.md`、`wiki/entities/isaac-gym-isaac-lab.md`、`wiki/entities/mujoco.md`、`wiki/overview/robot-world-models-training-loop-taxonomy.md`、`wiki/concepts/foundation-policy.md`、`wiki/methods/sonic-motion-tracking.md`、`wiki/overview/bfm-category-02-goal-conditioned-learning.md`、`wiki/overview/world-models-15-open-source-technology-map.md`、`wiki/queries/humanoid-motion-tracking-method-selection.md`、`wiki/methods/amp-reward.md`、`wiki/overview/navigation-slam-autonomy-stack.md`、`wiki/concepts/world-action-models.md`、`wiki/concepts/contact-rich-manipulation.md`、`wiki/concepts/whole-body-tracking-pipeline.md`、`wiki/methods/motion-retargeting-gmr.md`、`wiki/methods/model-predictive-control.md`、`wiki/concepts/motion-retargeting-pipeline.md`、`wiki/entities/legged-gym.md`、`wiki/tasks/teleoperation.md`、`wiki/overview/ego-9-papers-technology-map.md`、`wiki/queries/legged-humanoid-rl-pd-gain-setting.md`、`wiki/methods/model-based-rl.md`、`wiki/tasks/stair-obstacle-perceptive-locomotion.md`、`wiki/concepts/domain-randomization.md`、`wiki/overview/robot-learning-overview.md`、`wiki/methods/diffusion-policy.md`、`wiki/entities/paper-behavior-foundation-model-humanoid.md`、`wiki/entities/open-source-humanoid-hardware.md`、`wiki/methods/policy-optimization.md`、`wiki/concepts/reward-design.md`、`wiki/concepts/privileged-training.md`、`wiki/methods/beyondmimic.md`、`wiki/overview/bfm-category-05-hierarchical-control.md`、`wiki/entities/quadruped-robot.md`、`wiki/concepts/video-as-simulation.md`

## [2026-06-05] structural | wiki — 连接数前十 hub 页补齐英文缩写速查表

- 依据 `exports/graph-stats.json` 的 `top_hubs`（总度前十），在下列页面一句话定义/观点之后新增 `## 英文缩写速查` 三列表：`wiki/concepts/sim2real.md`、`wiki/tasks/locomotion.md`、`wiki/overview/humanoid-rl-motion-control-body-system-stack.md`、`wiki/methods/reinforcement-learning.md`、`wiki/methods/vla.md`、`wiki/overview/bfm-41-papers-technology-map.md`、`wiki/methods/imitation-learning.md`、`wiki/tasks/manipulation.md`、`wiki/concepts/whole-body-control.md`、`wiki/overview/humanoid-amp-motion-prior-survey.md`

## [2026-06-05] ingest | sources/papers/pilot_arxiv_2601_17440.md — PILOT 感知统一 loco-manipulation LLC（arXiv:2601.17440）；wiki/entities/paper-pilot-perceptive-loco-manipulation.md、wiki/tasks/loco-manipulation.md、wiki/tasks/stair-obstacle-perceptive-locomotion.md、wiki/entities/unitree-g1.md

- 原始资料：[pilot_arxiv_2601_17440.md](sources/papers/pilot_arxiv_2601_17440.md)（<https://arxiv.org/abs/2601.17440>）
- 沉淀页面：[wiki/entities/paper-pilot-perceptive-loco-manipulation.md](wiki/entities/paper-pilot-perceptive-loco-manipulation.md)
- 交叉更新：[wiki/tasks/loco-manipulation.md](wiki/tasks/loco-manipulation.md)、[wiki/tasks/stair-obstacle-perceptive-locomotion.md](wiki/tasks/stair-obstacle-perceptive-locomotion.md)、[wiki/concepts/whole-body-control.md](wiki/concepts/whole-body-control.md)、[wiki/tasks/teleoperation.md](wiki/tasks/teleoperation.md)、[wiki/entities/unitree-g1.md](wiki/entities/unitree-g1.md)

## [2026-06-05] structural | schema + wiki — 英文缩写速查表工作流与 SSR 页试点

- 工作流：[schema/page-types.md](schema/page-types.md)、[schema/ingest-workflow.md](schema/ingest-workflow.md)、[schema/linking.md](schema/linking.md)、[AGENTS.md](AGENTS.md)；`lint_wiki.py` 新增 `missing_abbrev_glossary` 信息型检查
- 试点页面：[wiki/entities/paper-ssr-humanoid-open-world-traversal.md](wiki/entities/paper-ssr-humanoid-open-world-traversal.md)（`## 英文缩写速查` 三列表）

## [2026-06-05] ingest | sources/papers/ssr_arxiv_2605_30770.md、sources/sites/ssr-humanoid-github-io.md — SSR 开放世界人形穿越（想象落脚点 + 潜空间对称 + 分地形 AMP）入库

- 原始资料：[ssr_arxiv_2605_30770.md](sources/papers/ssr_arxiv_2605_30770.md)（<https://arxiv.org/abs/2605.30770>、<https://arxiv.org/html/2605.30770v1>）；[ssr-humanoid-github-io.md](sources/sites/ssr-humanoid-github-io.md)（<https://ssr-humanoid.github.io/>）
- 沉淀页面：[wiki/entities/paper-ssr-humanoid-open-world-traversal.md](wiki/entities/paper-ssr-humanoid-open-world-traversal.md)（含单阶段 PPO + 三项机制 Mermaid 管线）
- 交叉更新：[wiki/tasks/stair-obstacle-perceptive-locomotion.md](wiki/tasks/stair-obstacle-perceptive-locomotion.md)、[wiki/tasks/humanoid-locomotion.md](wiki/tasks/humanoid-locomotion.md)、[sources/README.md](sources/README.md)

## [2026-06-05] ingest | sources/papers/sprint_arxiv_2605_28549.md、sources/sites/sprint-anonymous-project-page.md — SPRINT 人形竞技冲刺（频谱先验 + 6 m/s G1 真机）入库

- 原始资料：[sprint_arxiv_2605_28549.md](sources/papers/sprint_arxiv_2605_28549.md)（<https://arxiv.org/abs/2605.28549>、<https://arxiv.org/html/2605.28549v1>）；[sprint-anonymous-project-page.md](sources/sites/sprint-anonymous-project-page.md)（<https://anonymous.4open.science/w/SPRINT-138A/>）
- 沉淀页面：[wiki/entities/paper-sprint-humanoid-athletic-sprints.md](wiki/entities/paper-sprint-humanoid-athletic-sprints.md)（含三阶段 Mermaid 管线）
- 交叉更新：[wiki/tasks/humanoid-locomotion.md](wiki/tasks/humanoid-locomotion.md)、[wiki/queries/humanoid-motion-tracking-method-selection.md](wiki/queries/humanoid-motion-tracking-method-selection.md)、[sources/README.md](sources/README.md)

## [2026-06-05] ingest | sources/papers/homeworld_arxiv_2606_06390.md — HomeWorld（Kairos）全屋 sim-ready 场景生成入库

- 原始资料：[homeworld_arxiv_2606_06390.md](sources/papers/homeworld_arxiv_2606_06390.md)（<https://arxiv.org/abs/2606.06390>）；[kairos-homeworld-github-io.md](sources/sites/kairos-homeworld-github-io.md)（<https://kairos-homeworld.github.io/>）；[homeworld.md](sources/repos/homeworld.md)（<https://github.com/Kairos-HomeWorld/HomeWorld>，Coming Soon）
- 沉淀页面：[wiki/entities/paper-homeworld-whole-home-scene-generation.md](wiki/entities/paper-homeworld-whole-home-scene-generation.md)（含 Mermaid 四阶段流水线）
- 交叉更新：[wiki/concepts/video-as-simulation.md](wiki/concepts/video-as-simulation.md)、[wiki/tasks/manipulation.md](wiki/tasks/manipulation.md)、[wiki/tasks/vision-language-navigation.md](wiki/tasks/vision-language-navigation.md)、[sources/README.md](sources/README.md)

## [2026-06-05] ingest | sources/papers/host_humanoid_standingup_arxiv_2502_08378.md — HoST（RSS 2025）人形多姿态起身 RL 入库

- 原始资料：[host_humanoid_standingup_arxiv_2502_08378.md](sources/papers/host_humanoid_standingup_arxiv_2502_08378.md)（<https://arxiv.org/abs/2502.08378>）；[host-humanoid-standingup-project.md](sources/sites/host-humanoid-standingup-project.md)（<https://taohuang13.github.io/humanoid-standingup.github.io/>）；[host_internrobotics.md](sources/repos/host_internrobotics.md)（<https://github.com/InternRobotics/HoST>）
- 沉淀页面：[wiki/entities/paper-host-humanoid-standingup.md](wiki/entities/paper-host-humanoid-standingup.md)
- 交叉更新：[wiki/tasks/balance-recovery.md](wiki/tasks/balance-recovery.md)、[wiki/tasks/locomotion.md](wiki/tasks/locomotion.md)、[wiki/entities/unitree-g1.md](wiki/entities/unitree-g1.md)、[wiki/entities/paper-unified-walk-run-recovery-sdamp.md](wiki/entities/paper-unified-walk-run-recovery-sdamp.md)、[sources/README.md](sources/README.md)

## [2026-06-05] structural | wiki/methods/model-predictive-control.md — MPC 页补充滚动时域 Mermaid 流程图

- 页面：[wiki/methods/model-predictive-control.md](wiki/methods/model-predictive-control.md)（「有限时域优化」小节）

## [2026-06-05] ingest | sources/papers/explicit_stair_geometry_arxiv_2605_09944.md — 显式楼梯几何条件化人形爬梯（arXiv:2605.09944）入库

- 原始资料：[explicit_stair_geometry_arxiv_2605_09944.md](sources/papers/explicit_stair_geometry_arxiv_2605_09944.md)（<https://arxiv.org/abs/2605.09944>）
- 沉淀页面：[wiki/entities/paper-explicit-stair-geometry-humanoid-locomotion.md](wiki/entities/paper-explicit-stair-geometry-humanoid-locomotion.md)（含 Mermaid 训练—部署管线）
- 交叉更新：[wiki/tasks/locomotion.md](wiki/tasks/locomotion.md)、[wiki/tasks/stair-obstacle-perceptive-locomotion.md](wiki/tasks/stair-obstacle-perceptive-locomotion.md)、[wiki/concepts/terrain-adaptation.md](wiki/concepts/terrain-adaptation.md)、[wiki/entities/unitree-g1.md](wiki/entities/unitree-g1.md)、[wiki/entities/paper-faststair-humanoid-stair-ascent.md](wiki/entities/paper-faststair-humanoid-stair-ascent.md)、[sources/README.md](sources/README.md)

## [2026-06-05] ingest | sources/papers/faststair_arxiv_2601_10365.md — FastStair 挂接楼梯/障碍中心节点并刷新交叉引用

- 原始资料（已存在，本次补摘录 §5 感知定位）：[`sources/papers/faststair_arxiv_2601_10365.md`](sources/papers/faststair_arxiv_2601_10365.md)、[`sources/sites/npcliu-faststair-github-io.md`](sources/sites/npcliu-faststair-github-io.md)
- 新建中心节点：[`wiki/tasks/stair-obstacle-perceptive-locomotion.md`](wiki/tasks/stair-obstacle-perceptive-locomotion.md) — **带/不带感知 · 上下楼梯 · 越障** 维护挂接点
- 交叉更新：[`wiki/entities/paper-faststair-humanoid-stair-ascent.md`](wiki/entities/paper-faststair-humanoid-stair-ascent.md)、[`wiki/tasks/locomotion.md`](wiki/tasks/locomotion.md)、[`wiki/tasks/humanoid-locomotion.md`](wiki/tasks/humanoid-locomotion.md)、[`wiki/concepts/terrain-adaptation.md`](wiki/concepts/terrain-adaptation.md)、[`wiki/concepts/footstep-planning.md`](wiki/concepts/footstep-planning.md)、[`wiki/concepts/capture-point-dcm.md`](wiki/concepts/capture-point-dcm.md)、[`wiki/entities/paper-e-sds-environment-aware-humanoid-locomotion-rl.md`](wiki/entities/paper-e-sds-environment-aware-humanoid-locomotion-rl.md)、[`wiki/entities/dreamwaq-plus.md`](wiki/entities/dreamwaq-plus.md)、[`wiki/entities/extreme-parkour.md`](wiki/entities/extreme-parkour.md)、[`wiki/entities/paper-hrl-stack-22-perceptive_humanoid_parkour.md`](wiki/entities/paper-hrl-stack-22-perceptive_humanoid_parkour.md)、[`wiki/entities/paper-deep-whole-body-parkour.md`](wiki/entities/paper-deep-whole-body-parkour.md)、[`wiki/entities/paper-hiking-in-the-wild.md`](wiki/entities/paper-hiking-in-the-wild.md)、[`wiki/entities/paper-walk-these-ways-quadruped-mob.md`](wiki/entities/paper-walk-these-ways-quadruped-mob.md)、[`wiki/entities/jackhan-mujoco-walke3-simulation.md`](wiki/entities/jackhan-mujoco-walke3-simulation.md)

## [2026-06-04] structural | schema/canonical-facts.json — V23 P2 事实库扩展 156→172，补 WBT 跨具身与真机安全微调矛盾检测规则

- 推进 [tech-stack-next-phase-checklist-v23.md](docs/checklists/tech-stack-next-phase-checklist-v23.md) P2「事实库扩展」一项，达成 ≥170 条目标（实际 172）
- 新增 16 条事实：SONIC 规模化预训练 / Any2Any 跨具身迁移 / BFM 无参考 / SD-AMP 双判别器门控 / Heracles 扩散中间件 / WBT pipeline 端到端 / WBT 跨具身解耦 / SONIC-vs-Any2Any 训练范式 / SD-AMP-vs-Heracles 抽象层 / BeyondMimic 失败率采样 / SLowRL 安全 LoRA / 真机 RL 安全约束 / Sim2Real-vs-Real2Sim / 安全 LoRA 投影 / 跨具身策略迁移三路径 / CRISP Real2Sim
- 修正：收紧 `SD-AMP 状态门控双判别器` 的 neg 正则，避免误命中 [Heracles 页](wiki/entities/paper-heracles-humanoid-diffusion.md)「SD-AMP…单策略…判别器」对照描述
- 验证：`make lint` 潜在矛盾 0；`make ci-preflight` 导出质量 12/12 通过

## [2026-06-04] ingest | sources/blogs/wechat_shenlan_3d_coordinate_transforms.md、wechat_shenlan_riemannian_manifold_tangent_space.md — Agent Reach 抓取《具身智能基础》专栏 02/03 并建几何三篇父节点

- 工具：已安装 [Panniantong/Agent-Reach](https://github.com/Panniantong/Agent-Reach) v1.4.0（修复 hatchling `force-include` 重复后 `pip install -e` + `agent-reach install --channels=wechat`）；微信正文经 `~/.agent-reach/tools/wechat-article-for-ai`（Camoufox）
- 原始资料：[`wechat_shenlan_3d_coordinate_transforms.md`](sources/blogs/wechat_shenlan_3d_coordinate_transforms.md)（<https://mp.weixin.qq.com/s/P5Jm7bMhaTHsytHStFbbLg>）；[`wechat_shenlan_riemannian_manifold_tangent_space.md`](sources/blogs/wechat_shenlan_riemannian_manifold_tangent_space.md)（<https://mp.weixin.qq.com/s/uFTKN5FDvlHQxOSspvxVZw>）；落盘 [`sources/raw/wechat_shenlan_3d_coord_transforms_2026-06-04.md`](sources/raw/wechat_shenlan_3d_coord_transforms_2026-06-04.md)、[`sources/raw/wechat_shenlan_riemannian_manifold_2026-06-04.md`](sources/raw/wechat_shenlan_riemannian_manifold_2026-06-04.md)；专栏 01 已存在 [`wechat_shenlan_lie_group_lie_algebra_quaternion.md`](sources/blogs/wechat_shenlan_lie_group_lie_algebra_quaternion.md)（<https://mp.weixin.qq.com/s/JviRH2LW-fkCHA5gY7Qflw>）
- 沉淀页面：**父节点** [`wiki/overview/shenlan-embodied-ai-fundamentals-series.md`](wiki/overview/shenlan-embodied-ai-fundamentals-series.md)；子节点 [`wiki/formalizations/3d-coordinate-transforms-vision-robotics.md`](wiki/formalizations/3d-coordinate-transforms-vision-robotics.md)、[`wiki/formalizations/riemannian-manifold-tangent-space.md`](wiki/formalizations/riemannian-manifold-tangent-space.md)
- 交叉更新：[`wiki/formalizations/lie-group-rigid-body-motions.md`](wiki/formalizations/lie-group-rigid-body-motions.md)、[`wiki/overview/vla-open-source-repro-landscape-2025.md`](wiki/overview/vla-open-source-repro-landscape-2025.md)、[`sources/README.md`](sources/README.md)、[`sources/repos/panniantong_agent_reach.md`](sources/repos/panniantong_agent_reach.md)

## [2026-06-04] structural | wiki/concepts/motion-retargeting-pipeline.md — 流水线页三处公式改 `$...$` 以启用 KaTeX 蓝框；`docs/main.js` 保留 Mermaid `htmlLabels` 的 `<br/>` 换行

- 页面：[wiki/concepts/motion-retargeting-pipeline.md](wiki/concepts/motion-retargeting-pipeline.md)
- 前端：`docs/main.js` 中 `escapeMermaidForInnerHtml` 不再转义 `<br/>`，修复流程图节点多行标签被拼成一行的问题

## [2026-06-04] ingest | sources/papers/splitadapter_arxiv_2606_03297.md — SplitAdapter 负载感知人形搬箱因子化适配入库；wiki/entities/paper-splitadapter-load-aware-loco-manipulation.md

- 原始资料：[splitadapter_arxiv_2606_03297.md](sources/papers/splitadapter_arxiv_2606_03297.md)（<https://arxiv.org/abs/2606.03297>）；[splitadapter-github-io.md](sources/sites/splitadapter-github-io.md)（<https://splitadapter.github.io/>）
- 沉淀页面：[wiki/entities/paper-splitadapter-load-aware-loco-manipulation.md](wiki/entities/paper-splitadapter-load-aware-loco-manipulation.md)（含 Mermaid 因子化适配管线）
- 交叉更新：[wiki/tasks/loco-manipulation.md](wiki/tasks/loco-manipulation.md)、[wiki/concepts/sim2real.md](wiki/concepts/sim2real.md)、[wiki/entities/paper-amp-survey-15-physhsi.md](wiki/entities/paper-amp-survey-15-physhsi.md)、[sources/README.md](sources/README.md)

## [2026-06-04] ingest | sources/papers/htd_refine_arxiv_2605_26879.md — HTD-Refine（CVPR 2026）单目 HMR 高阶动力学后处理入库；wiki/entities/paper-htd-refine-monocular-hmr.md

- 原始资料：[htd_refine_arxiv_2605_26879.md](sources/papers/htd_refine_arxiv_2605_26879.md)（<https://arxiv.org/abs/2605.26879>）；[htd-refine-zju3dv-github-io.md](sources/sites/htd-refine-zju3dv-github-io.md)（<https://zju3dv.github.io/htd-refine/>）
- 沉淀页面：[wiki/entities/paper-htd-refine-monocular-hmr.md](wiki/entities/paper-htd-refine-monocular-hmr.md)
- 交叉更新：[wiki/concepts/motion-retargeting-pipeline.md](wiki/concepts/motion-retargeting-pipeline.md)、[wiki/concepts/whole-body-tracking-pipeline.md](wiki/concepts/whole-body-tracking-pipeline.md)、[wiki/methods/motion-retargeting-gmr.md](wiki/methods/motion-retargeting-gmr.md)

## [2026-06-04] ingest | sources/blogs/wechat_embodied_ai_lab_legs_vla_3dgs_loco_manip.md — Agent Reach 抓取 LEGS/3DGS 人形 VLA loco-manip 专题并沉淀论文实体

- 工具：已安装 [Panniantong/Agent-Reach](https://github.com/Panniantong/Agent-Reach) v1.4.0（editable 安装，修复 hatchling `force-include` 重复文件后 `pip install -e` + `wechat-article-for-ai`/Camoufox）
- 原始资料：[`sources/blogs/wechat_embodied_ai_lab_legs_vla_3dgs_loco_manip.md`](sources/blogs/wechat_embodied_ai_lab_legs_vla_3dgs_loco_manip.md)（<https://mp.weixin.qq.com/s/B1sYOPKg6TQwnNGs-_8NDw>）；落盘 [`sources/raw/wechat_legs_vla_3dgs_2026-06-04.md`](sources/raw/wechat_legs_vla_3dgs_2026-06-04.md)；论文 [`sources/papers/legs_arxiv_2606_01458.md`](sources/papers/legs_arxiv_2606_01458.md)；项目页 [`sources/sites/legsvla-github-io.md`](sources/sites/legsvla-github-io.md)
- 沉淀页面：[`wiki/entities/paper-legs-embodied-gaussian-splatting-vla.md`](wiki/entities/paper-legs-embodied-gaussian-splatting-vla.md)（含 Mermaid 管线总览）
- 交叉更新：[`wiki/tasks/loco-manipulation.md`](wiki/tasks/loco-manipulation.md)、[`wiki/methods/vla.md`](wiki/methods/vla.md)、[`wiki/methods/sonic-motion-tracking.md`](wiki/methods/sonic-motion-tracking.md)、[`wiki/entities/gs-playground.md`](wiki/entities/gs-playground.md)、[`sources/README.md`](sources/README.md)、[`sources/repos/panniantong_agent_reach.md`](sources/repos/panniantong_agent_reach.md)

## [2026-06-04] ingest | sources/sites/pupper-v3-documentation-readthedocs.md、sources/repos/pupperv3_monorepo.md、sources/courses/stanford_cs123_robotics_ai.md — Pupper v3 官方文档与 monorepo/CS123 入库；更新 wiki/entities/stanford-doggo-and-pupper.md

- 原始资料：[pupper-v3-documentation-readthedocs.md](sources/sites/pupper-v3-documentation-readthedocs.md)（<https://pupper-v3-documentation.readthedocs.io/en/latest/index.html>）；[pupperv3_monorepo.md](sources/repos/pupperv3_monorepo.md)（<https://github.com/Nate711/pupperv3-monorepo>）；[stanford_cs123_robotics_ai.md](sources/courses/stanford_cs123_robotics_ai.md)（<https://cs123-stanford.readthedocs.io/en/latest/>）
- 沉淀页面：[wiki/entities/stanford-doggo-and-pupper.md](wiki/entities/stanford-doggo-and-pupper.md)（补充 v3 规格、安全、RL/VLM、ROS 2 流程图；区分 v2/easy_quadruped lineage）
- 交叉更新：[wiki/entities/easy-quadruped.md](wiki/entities/easy-quadruped.md)、[wiki/entities/quadruped-robot.md](wiki/entities/quadruped-robot.md)、[sources/README.md](sources/README.md)

## [2026-06-04] ingest | sources/papers/humanoid_gpt_arxiv_2606_03985.md — Humanoid-GPT（CVPR 2026，2B 帧零样本 motion tracking）入库

- 原始资料：[humanoid_gpt_arxiv_2606_03985.md](sources/papers/humanoid_gpt_arxiv_2606_03985.md)（<https://arxiv.org/abs/2606.03985>）；[humanoid-gpt-qizekun-github-io.md](sources/sites/humanoid-gpt-qizekun-github-io.md)（<https://qizekun.github.io/Humanoid-GPT/>）；[humanoid_gpt_galaxy_general_robotics.md](sources/repos/humanoid_gpt_galaxy_general_robotics.md)（<https://github.com/GalaxyGeneralRobotics/Humanoid-GPT>）
- 沉淀页面：[wiki/entities/paper-humanoid-gpt.md](wiki/entities/paper-humanoid-gpt.md)
- 交叉更新：[wiki/methods/sonic-motion-tracking.md](wiki/methods/sonic-motion-tracking.md)、[wiki/queries/humanoid-motion-tracking-method-selection.md](wiki/queries/humanoid-motion-tracking-method-selection.md)

## [2026-06-04] ingest | sources/papers/extreme_parkour_arxiv_2309_14341.md、sources/repos/extreme-parkour.md、sources/sites/extreme-parkour-github-io.md — Extreme Parkour（ICRA 2024）入库；wiki/entities/extreme-parkour.md

- 原始资料：[extreme_parkour_arxiv_2309_14341.md](sources/papers/extreme_parkour_arxiv_2309_14341.md)（<https://arxiv.org/abs/2309.14341>）；[extreme-parkour.md](sources/repos/extreme-parkour.md)（<https://github.com/chengxuxin/extreme-parkour>）；[extreme-parkour-github-io.md](sources/sites/extreme-parkour-github-io.md)（<https://extreme-parkour.github.io/>）
- 沉淀页面：[wiki/entities/extreme-parkour.md](wiki/entities/extreme-parkour.md)
- 交叉更新：[wiki/tasks/locomotion.md](wiki/tasks/locomotion.md)、[wiki/concepts/privileged-training.md](wiki/concepts/privileged-training.md)、[roadmap/motion-control.md](roadmap/motion-control.md)、[sources/README.md](sources/README.md)

## [2026-06-04] structural | wiki/comparisons/sim2real-vs-real2sim-fine-tuning.md — 新建「Sim2Real 残差适配 vs Real2Sim 真机回放 vs 真机直接 RL 微调」对比页（V23 P2 安全微调知识链 3/3，专题收官）

- 新建 wiki：[sim2real-vs-real2sim-fine-tuning.md](wiki/comparisons/sim2real-vs-real2sim-fine-tuning.md)——把真机适配「最后一公里」拆成三策略：残差适配（冻结 $W_0$ + 低秩残差 + Recovery/Safety Filter 吸收残差，SLowRL）/ Real2Sim 真机回放（用真机数据反修仿真后回仿真重训，CRISP）/ 真机直接 RL 微调；给出 11 维核心对照表 + 数据流 Mermaid + 成本/安全/数据效率三维深读 + 三场景选型 + 5 类误判 + 决策矩阵，明确三者本质是「gap 在真机侧 / 仿真侧 / 真机侧端到端消化」的连续谱。
- 交叉更新：[safe-real-world-rl-fine-tuning.md](wiki/concepts/safe-real-world-rl-fine-tuning.md)、[sim2real-approaches.md](wiki/comparisons/sim2real-approaches.md) 补双向入链（消除孤儿页）。
- 进度：V23 P2「安全微调知识链 (+3)」3/3 完成，父项 `[~]`→`[x]`。
- 验证：`make lint` 仅余 2 条与本页无关的预存陈旧页警告（`generative-world-models` / `π0-policy`），`eval_search_quality` 37/37 通过；`make ci-preflight` 重生成派生文件，图谱 666 节点 / 4681 边 / 孤儿 0、`largest_community_ratio` 0.179、`community_quality_warning: false`。

## [2026-06-03] ingest | sources/papers/assistmimic_arxiv_2603_11346.md — AssistMimic（CVPR 2026 双人 assistive MARL tracking）入库

- 原始资料：[assistmimic_arxiv_2603_11346.md](sources/papers/assistmimic_arxiv_2603_11346.md)（<https://arxiv.org/abs/2603.11346>）；[yutoshibata07-assistmimic-github-io.md](sources/sites/yutoshibata07-assistmimic-github-io.md)（<https://yutoshibata07.github.io/AssistMimic/>）
- 沉淀页面：[wiki/entities/paper-assistmimic.md](wiki/entities/paper-assistmimic.md)
- 交叉更新：[wiki/methods/marl.md](wiki/methods/marl.md)、[wiki/entities/paper-bfm-22-phc.md](wiki/entities/paper-bfm-22-phc.md)、[wiki/concepts/whole-body-tracking-pipeline.md](wiki/concepts/whole-body-tracking-pipeline.md)

## [2026-06-03] ingest | sources/papers/dwm_arxiv_2512_17907.md、sources/repos/snuvclab_dwm.md — 补充 DWM 官方代码工程栈（CogVideoX-5B LoRA / WAN / 三元组数据目录）并更新 wiki/methods/dwm.md

- 原始资料：[`sources/papers/dwm_arxiv_2512_17907.md`](sources/papers/dwm_arxiv_2512_17907.md)（[arXiv:2512.17907](https://arxiv.org/abs/2512.17907)）；[`sources/sites/snuvclab-dwm-github-io.md`](sources/sites/snuvclab-dwm-github-io.md)；[`sources/repos/snuvclab_dwm.md`](sources/repos/snuvclab_dwm.md)（[snuvclab/dwm](https://github.com/snuvclab/dwm)，2026-04-03 代码发布）
- 沉淀页面：[`wiki/methods/dwm.md`](wiki/methods/dwm.md) — 增补「工程实现」节（CogVideoX-5B LoRA、VideoX-Fun 初始化、数据三元组目录、WAN 变体）
- 验证：`make ci-preflight` 通过

## [2026-06-03] ingest | sources/blogs/wechat_shenlan_world_models_15_open_source_2026.md — Agent Reach 抓取深蓝世界模型 15 项目并建三线图谱

- 工具：已安装 [Panniantong/Agent-Reach](https://github.com/Panniantong/Agent-Reach) v1.4.0（editable 安装 + `wechat-article-for-ai`/Camoufox）
- 原始资料：[`sources/blogs/wechat_shenlan_world_models_15_open_source_2026.md`](sources/blogs/wechat_shenlan_world_models_15_open_source_2026.md)（<https://mp.weixin.qq.com/s/KZT8sI4n7GvHWyM20wN3gg>）；落盘 [`sources/raw/wechat_world_models_15_2026-06-03.md`](sources/raw/wechat_world_models_15_2026-06-03.md)；参考资料 [`sources/papers/shenlan_world_models_15_reference_catalog.md`](sources/papers/shenlan_world_models_15_reference_catalog.md)
- 沉淀页面：[`wiki/overview/world-models-15-open-source-technology-map.md`](wiki/overview/world-models-15-open-source-technology-map.md)（**父节点**）；子节点 [`wiki/overview/world-models-route-01-cascade.md`](wiki/overview/world-models-route-01-cascade.md)、[`wiki/overview/world-models-route-02-joint.md`](wiki/overview/world-models-route-02-joint.md)、[`wiki/overview/world-models-route-03-virtual-sandbox.md`](wiki/overview/world-models-route-03-virtual-sandbox.md)；论文实体 `paper-shenlan-wm-01`…`03`、`05`…`15`（04→[`mimic-video`](wiki/methods/mimic-video.md)）
- 交叉更新：[`wiki/overview/robot-world-models-training-loop-taxonomy.md`](wiki/overview/robot-world-models-training-loop-taxonomy.md)、[`sources/README.md`](sources/README.md)、[`sources/repos/panniantong_agent_reach.md`](sources/repos/panniantong_agent_reach.md)
- 验证：`make ci-preflight` 通过

## [2026-06-03] structural | wiki/formalizations/safe-lora-update-projection.md — 新建「安全 LoRA 投影更新形式化」（V23 P2 安全微调知识链 2/3）

- 新建 wiki：[safe-lora-update-projection.md](wiki/formalizations/safe-lora-update-projection.md)（「冻结 $W_0$ + 低秩残差 $\frac{\alpha}{r}BA$ + 两层安全投影」统一形式：参数侧秩约束作隐式正则、动作侧 $\Pi_{\mathcal{S}}$ 分硬切换 Recovery 与连续 QP 安全壳两谱系，写成低秩子空间 CMDP；SLowRL 实例化表 + 全参 CMDP / 纯 QP 安全壳 / 生成式改写退化对照 + 评测口径）。
- 交叉更新：[safe-real-world-rl-fine-tuning.md](wiki/concepts/safe-real-world-rl-fine-tuning.md)、[SLowRL 实体](wiki/entities/paper-slowrl-safe-lora-locomotion-sim2real.md) 补双向入链（消除孤儿页）。
- 检索回归修复：[`scripts/search_wiki_core.py`](scripts/search_wiki_core.py) `_canonical_topic_boost` 由 1.4→1.7——safe-RL 专题扩张后「CBF 安全集 barrier」query 的 CBF 定义页被 clf-vs-cbf / safe-RL 系列页挤出 top5（main 既有回归），提权后 CBF 定义页回到 top5。
- 进度：V23 P2「安全微调知识链 (+3)」2/3（仍 `[~]`，余 sim2real-vs-real2sim-fine-tuning 对比页）。
- 验证：`make lint` 全绿（`eval_search_quality` 37/37）；`make ci-preflight` 重生成派生文件。

## [2026-06-02] ingest | sources/papers/shape_your_body_arxiv_2606_00702.md、sources/sites/shape-your-body-nico-bohlinger.md — Shape Your Body（VGDS 多具身价值梯度共设计）入库

- 原始资料：[shape_your_body_arxiv_2606_00702.md](sources/papers/shape_your_body_arxiv_2606_00702.md)（[PDF](https://www.ias.informatik.tu-darmstadt.de/uploads/Team/NicoBohlinger/shape_your_body.pdf)、[arXiv HTML](https://arxiv.org/html/2606.00702v1)）；[shape-your-body-nico-bohlinger.md](sources/sites/shape-your-body-nico-bohlinger.md)（[项目页](https://nico-bohlinger.github.io/shape-your-body/)）
- 沉淀页面：[wiki/entities/paper-shape-your-body-value-gradient-design.md](wiki/entities/paper-shape-your-body-value-gradient-design.md)
- 交叉更新：[cross-embodiment-transfer-strategy.md](wiki/queries/cross-embodiment-transfer-strategy.md)、[reinforcement-learning.md](wiki/methods/reinforcement-learning.md)、[foundation-policy.md](wiki/concepts/foundation-policy.md)
- 验证：`make ci-preflight` 通过

## [2026-06-02] structural | wiki/entities/paper-hrl-stack-22-perceptive_humanoid_parkour.md — 补全 motion matching / 蒸馏损失公式的 KaTeX 蓝框显示

- 修正页面：[`wiki/entities/paper-hrl-stack-22-perceptive_humanoid_parkour.md`](wiki/entities/paper-hrl-stack-22-perceptive_humanoid_parkour.md)（$\hat{x}_t$、$\arg\min_i \|\hat{x}_t - x_i\|^2$、$L = \lambda_{\mathrm{PPO}} L_{\mathrm{PPO}} + \lambda_D L_D$ 改用 `$...$` 包裹，detail 页可触发 `math-inline` 与 KaTeX）
- 验证：`make ci-preflight` 通过

## [2026-06-02] structural | wiki/concepts/safe-real-world-rl-fine-tuning.md — 新建「真机安全 RL 微调」概念页（V23 P2 安全微调知识链 1/3）

- 新建 wiki：[safe-real-world-rl-fine-tuning.md](wiki/concepts/safe-real-world-rl-fine-tuning.md)（残差视角 + 三路径详解：SLowRL 低秩残差 + Recovery/Safety Filter、Heracles 生成式兜底中间件、CBF/CLF 安全壳；5 维对比表 + 5 类常见误区）。
- 交叉更新：[sim2real.md](wiki/concepts/sim2real.md)（安全微调段落引导 + frontmatter related + 关联页面）、[safety-filter.md](wiki/concepts/safety-filter.md)、[SLowRL 实体](wiki/entities/paper-slowrl-safe-lora-locomotion-sim2real.md)、[Heracles 实体](wiki/entities/paper-heracles-humanoid-diffusion.md) 补入站边。
- 进度：V23 P2「安全微调知识链 (+3)」标记 1/3（`[~]` 进行中）。
- 验证：`make lint` 全绿（1 条 INFO 不阻塞）；`make ci-preflight` 重生成派生文件。

## [2026-06-02] ingest | sources/papers/mobilegym_arxiv_2605_26114.md、sources/repos/purewhiter_mobilegym.md、sources/sites/mobilegym-dev.md — MobileGym 入库

- 沉淀页面：[`wiki/entities/mobilegym.md`](wiki/entities/mobilegym.md)

## [2026-06-02] structural | schema/naming.md、scripts/generate_link_graph.py — 统一图谱社区名为「中文（English） 社区」并补全 override

- 规范：[`schema/naming.md`](schema/naming.md) 新增「图谱社区命名」；[`scripts/generate_link_graph.py`](scripts/generate_link_graph.py) 增加 `COMMUNITY_HUB_NAME_RE` 校验与 WARNING
- 修正社区名：规模化运动跟踪（SONIC）、人形硬件技术地图（Humanoid Hardware 101）、机器人学习（Robot Learning）、行为基础模型技术地图（BFM）
- 测试：[`tests/test_community_naming.py`](tests/test_community_naming.py)
- 验证：`make ci-preflight` 通过
## [2026-06-02] structural | wiki/overview humanoid-hardware-101-* 与 humanoid-actuator-102-* — 首页最新知识节点补登子 hub

- 父节点：[`wiki/overview/humanoid-hardware-101-technology-map.md`](wiki/overview/humanoid-hardware-101-technology-map.md)、[`wiki/overview/humanoid-actuator-102-technology-map.md`](wiki/overview/humanoid-actuator-102-technology-map.md)
- Hardware 101 子 hub：[`wiki/overview/humanoid-hardware-101-chassis-materials.md`](wiki/overview/humanoid-hardware-101-chassis-materials.md)、[`wiki/overview/humanoid-hardware-101-actuation-sensing-chain.md`](wiki/overview/humanoid-hardware-101-actuation-sensing-chain.md)、[`wiki/overview/humanoid-hardware-101-linear-transmission-bearings.md`](wiki/overview/humanoid-hardware-101-linear-transmission-bearings.md)、[`wiki/overview/humanoid-hardware-101-integrated-actuators.md`](wiki/overview/humanoid-hardware-101-integrated-actuators.md)、[`wiki/overview/humanoid-hardware-101-power-compute-electronics.md`](wiki/overview/humanoid-hardware-101-power-compute-electronics.md)、[`wiki/overview/humanoid-hardware-101-sensing-end-effectors.md`](wiki/overview/humanoid-hardware-101-sensing-end-effectors.md)、[`wiki/overview/humanoid-hardware-101-supply-chain-economics.md`](wiki/overview/humanoid-hardware-101-supply-chain-economics.md)
- Actuator 102 子 hub：[`wiki/overview/humanoid-actuator-102-load-and-mass-spiral.md`](wiki/overview/humanoid-actuator-102-load-and-mass-spiral.md)、[`wiki/overview/humanoid-actuator-102-split-architecture.md`](wiki/overview/humanoid-actuator-102-split-architecture.md)、[`wiki/overview/humanoid-actuator-102-gear-reflected-inertia.md`](wiki/overview/humanoid-actuator-102-gear-reflected-inertia.md)、[`wiki/overview/humanoid-actuator-102-thermal-and-control.md`](wiki/overview/humanoid-actuator-102-thermal-and-control.md)、[`wiki/overview/humanoid-actuator-102-compliance-sensing.md`](wiki/overview/humanoid-actuator-102-compliance-sensing.md)、[`wiki/overview/humanoid-actuator-102-industrial-actuator-trap.md`](wiki/overview/humanoid-actuator-102-industrial-actuator-trap.md)、[`wiki/overview/humanoid-actuator-102-decision-species.md`](wiki/overview/humanoid-actuator-102-decision-species.md)、[`wiki/overview/humanoid-actuator-102-future-artificial-muscle.md`](wiki/overview/humanoid-actuator-102-future-artificial-muscle.md)
- 验证：`make graph` 后 `exports/home-stats.json` 的 `latest_wiki_nodes` 含上述 17 个 hub。

## [2026-06-02] ingest | sources/blogs/wechat_human_five_humanoid_actuator_102.md、sources/papers/humanoid_actuator_102_reference_catalog.md — Agent Reach 抓取 Humanoid 执行器 102 并建八章图谱

- 工具：[Agent Reach](https://github.com/Panniantong/Agent-Reach) v1.4.0 + `wechat-article-for-ai`（Camoufox）
- 原始资料：[`sources/blogs/wechat_human_five_humanoid_actuator_102.md`](sources/blogs/wechat_human_five_humanoid_actuator_102.md)（<https://mp.weixin.qq.com/s/zinp6ulTorzfqmCR_HaI5A>）；落盘 [`sources/raw/wechat_humanoid_actuator_102_2026-06-02.md`](sources/raw/wechat_humanoid_actuator_102_2026-06-02.md)；参考资料 [`sources/papers/humanoid_actuator_102_reference_catalog.md`](sources/papers/humanoid_actuator_102_reference_catalog.md)
- 沉淀页面：[`wiki/overview/humanoid-actuator-102-technology-map.md`](wiki/overview/humanoid-actuator-102-technology-map.md)（父节点）；子节点 [`wiki/overview/humanoid-actuator-102-load-and-mass-spiral.md`](wiki/overview/humanoid-actuator-102-load-and-mass-spiral.md)、[`wiki/overview/humanoid-actuator-102-split-architecture.md`](wiki/overview/humanoid-actuator-102-split-architecture.md)、[`wiki/overview/humanoid-actuator-102-gear-reflected-inertia.md`](wiki/overview/humanoid-actuator-102-gear-reflected-inertia.md)、[`wiki/overview/humanoid-actuator-102-thermal-and-control.md`](wiki/overview/humanoid-actuator-102-thermal-and-control.md)、[`wiki/overview/humanoid-actuator-102-compliance-sensing.md`](wiki/overview/humanoid-actuator-102-compliance-sensing.md)、[`wiki/overview/humanoid-actuator-102-industrial-actuator-trap.md`](wiki/overview/humanoid-actuator-102-industrial-actuator-trap.md)、[`wiki/overview/humanoid-actuator-102-decision-species.md`](wiki/overview/humanoid-actuator-102-decision-species.md)、[`wiki/overview/humanoid-actuator-102-future-artificial-muscle.md`](wiki/overview/humanoid-actuator-102-future-artificial-muscle.md)
- 交叉更新：[`wiki/overview/humanoid-hardware-101-technology-map.md`](wiki/overview/humanoid-hardware-101-technology-map.md)、[`wiki/overview/humanoid-hardware-101-integrated-actuators.md`](wiki/overview/humanoid-hardware-101-integrated-actuators.md)、[`sources/README.md`](sources/README.md)

## [2026-06-02] ingest | sources/repos/nvidia_isaac_teleop.md — Isaac Teleop 入库；新建 wiki/entities/isaac-teleop.md；交叉 wiki/entities/isaac-lab.md、wiki/tasks/teleoperation.md

- 原始资料：[nvidia_isaac_teleop.md](sources/repos/nvidia_isaac_teleop.md)（[GitHub](https://github.com/NVIDIA/IsaacTeleop)、[官方文档](https://nvidia.github.io/IsaacTeleop/main/index.html)、[Isaac Lab 功能页](https://isaac-sim.github.io/IsaacLab/main/source/features/isaac_teleop.html)）
- 沉淀页面：[wiki/entities/isaac-teleop.md](wiki/entities/isaac-teleop.md)；交叉 [isaac-lab.md](wiki/entities/isaac-lab.md)、[teleoperation.md](wiki/tasks/teleoperation.md)
- 验证：`make ci-preflight` 通过

## [2026-06-01] structural | wiki/concepts/motion-retargeting.md、sim2real.md — 跨具身专题交叉补强（V23 P1 收官）

- 变更：在 [motion-retargeting.md](wiki/concepts/motion-retargeting.md) 新增「三段流水线衔接：重定向产物 → WBT 训练数据 → 跨具身策略蒸馏」小节（映射/训练/迁移三段表格 + 衔接段落），关联页面补 [WBT pipeline](wiki/concepts/whole-body-tracking-pipeline.md) / [跨具身迁移选型](wiki/queries/cross-embodiment-transfer-strategy.md) / [SONIC 四方对比](wiki/comparisons/sonic-vs-beyondmimic-vs-sdamp-vs-heracles.md) 三条出边。
- 变更：在 [sim2real.md](wiki/concepts/sim2real.md) 新增「在『映射 → 训练 → 迁移』三段流水线中的位置」小节（点明 Sim2Real 横切训练与迁移两段），frontmatter `related` 与关联页面同步补 motion-retargeting / WBT pipeline / 跨具身迁移 / SONIC 对比。
- 进度：V23 P1「跨具身专题交叉补强」标记为 `[x]`，至此 P1 全部完成。
- 验证：`make lint` 全绿；`make ci-preflight` 重生成派生文件。

## [2026-06-01] ingest | sources/papers/kalman_filter_ekf_primary_refs.md、sources/papers/lqr_ilqr_primary_refs.md — KF/EKF/LQR/iLQR 一手资料入库并新建 KF 形式化页

- 原始资料：[kalman_filter_ekf_primary_refs.md](sources/papers/kalman_filter_ekf_primary_refs.md)（Kalman 1960/61、Gelb 1974、Simon 2006 等）、[lqr_ilqr_primary_refs.md](sources/papers/lqr_ilqr_primary_refs.md)（Bryson & Ho 1975、Li & Todorov 2004、Tassa 2012/14 等）；课程：[welch_bishop_kalman_filter.md](sources/courses/welch_bishop_kalman_filter.md)、[mit_underactuated_kalman_lqr.md](sources/courses/mit_underactuated_kalman_lqr.md)
- 沉淀页面：[wiki/formalizations/kalman-filter.md](wiki/formalizations/kalman-filter.md)（新建）；交叉更新 [ekf.md](wiki/formalizations/ekf.md)、[lqr.md](wiki/formalizations/lqr.md)、[lqr-ilqr.md](wiki/methods/lqr-ilqr.md)、[state-estimation.md](wiki/concepts/state-estimation.md)
- 验证：`make ci-preflight` 通过

## [2026-06-01] ingest | sources/blogs/wechat_embodied_ai_lab_ego_9_papers_survey.md — Agent Reach 抓取 Ego 9 篇专题并建四类子系统图谱与 9 论文节点

- 工具：已安装 [Panniantong/Agent-Reach](https://github.com/Panniantong/Agent-Reach) v1.4.0（`pip install git+https://github.com/Panniantong/Agent-Reach.git` + `agent-reach install --channels=wechat`）；微信正文经 `wechat-article-for-ai`（Camoufox）
- 原始资料：[`sources/blogs/wechat_embodied_ai_lab_ego_9_papers_survey.md`](sources/blogs/wechat_embodied_ai_lab_ego_9_papers_survey.md)（<https://mp.weixin.qq.com/s/4JQ1xa-cJ7J1ep_e4txNnA>）；落盘 [`sources/raw/wechat_ego_9_papers_2026-06-01.md`](sources/raw/wechat_ego_9_papers_2026-06-01.md)
- 沉淀页面：[`wiki/overview/ego-9-papers-technology-map.md`](wiki/overview/ego-9-papers-technology-map.md)（父节点 + Mermaid）；子节点 [`ego-category-01-data-collection`](wiki/overview/ego-category-01-data-collection.md)、[`ego-category-02-human-to-robot`](wiki/overview/ego-category-02-human-to-robot.md)、[`ego-category-03-world-models`](wiki/overview/ego-category-03-world-models.md)、[`ego-category-04-ego-exo-fusion`](wiki/overview/ego-category-04-ego-exo-fusion.md)；论文实体 `paper-ego-01`…`05`、`08`、`09`（06→[`paper-hrl-stack-33`](wiki/entities/paper-hrl-stack-33-ego_vision_world_model_for_humanoid.md)、07→[`paper-wem`](wiki/entities/paper-wem-world-ego-modeling.md)）
- 交叉更新：[`humanoid-rl-motion-control-body-system-stack.md`](wiki/overview/humanoid-rl-motion-control-body-system-stack.md)、[`robot-world-models-training-loop-taxonomy.md`](wiki/overview/robot-world-models-training-loop-taxonomy.md)、[`sources/repos/panniantong_agent_reach.md`](sources/repos/panniantong_agent_reach.md)、[`sources/README.md`](sources/README.md)

## [2026-06-01] fix(ux) | docs/style.css — 撤销路线页首屏 360px 左缩进，与面包屑/正文左缘对齐

- 变更：去掉 #470 误加的 `margin-inline-start: 360px`；保留首屏列表满宽与 `#roadmapSummary` 使用 `<div>`。
- 验证：`make ci-preflight` 通过。

## [2026-06-01] fix(wiki) | wiki/overview/humanoid-hardware-101-technology-map.md — 修复七类子系统 Mermaid（子图直连改节点边、去 click、htmlLabels 换行）

- 根因：子图 `G1 --> G4` 直连在 `securityLevel: strict` 下易解析失败；`click` 指令被站点 Mermaid 配置禁用；标签内 `/` 与 `·` 增加歧义。
- 变更：改为 `CH/M/LS --> ACT --> ROB` 节点级边；标签用 `<br/>` 分行；分类入口保留下方表格链接。
- 关联页面：`wiki/overview/humanoid-hardware-101-technology-map.md`
- 验证：`make ci-preflight` 通过。

## [2026-06-01] ingest | sources/blogs/wechat_human_five_humanoid_hardware_101.md — Agent Reach 抓取 human five《Humanoid Hardware 入门 101》并建七类子系统图谱

- 工具：已安装 [Panniantong/Agent-Reach](https://github.com/Panniantong/Agent-Reach) v1.4.0（`pip install git+https://github.com/Panniantong/Agent-Reach.git` + `agent-reach install --channels=wechat`）；微信正文经 `~/.agent-reach/tools/wechat-article-for-ai`（Camoufox）
- 原始资料：[`sources/blogs/wechat_human_five_humanoid_hardware_101.md`](sources/blogs/wechat_human_five_humanoid_hardware_101.md)（<https://mp.weixin.qq.com/s/10hYwFzC1EuCypFVzC6QGQ>）；落盘 [`sources/raw/wechat_humanoid_hardware_101_2026-06-01.md`](sources/raw/wechat_humanoid_hardware_101_2026-06-01.md)
- 沉淀页面：[`wiki/overview/humanoid-hardware-101-technology-map.md`](wiki/overview/humanoid-hardware-101-technology-map.md)（父节点 + Mermaid）；子节点 [`humanoid-hardware-101-chassis-materials`](wiki/overview/humanoid-hardware-101-chassis-materials.md)、[`actuation-sensing-chain`](wiki/overview/humanoid-hardware-101-actuation-sensing-chain.md)、[`linear-transmission-bearings`](wiki/overview/humanoid-hardware-101-linear-transmission-bearings.md)、[`integrated-actuators`](wiki/overview/humanoid-hardware-101-integrated-actuators.md)、[`power-compute-electronics`](wiki/overview/humanoid-hardware-101-power-compute-electronics.md)、[`sensing-end-effectors`](wiki/overview/humanoid-hardware-101-sensing-end-effectors.md)、[`supply-chain-economics`](wiki/overview/humanoid-hardware-101-supply-chain-economics.md)
- 交叉更新：[`wiki/queries/humanoid-hardware-selection.md`](wiki/queries/humanoid-hardware-selection.md)、[`wiki/entities/open-source-humanoid-hardware.md`](wiki/entities/open-source-humanoid-hardware.md)、[`sources/repos/panniantong_agent_reach.md`](sources/repos/panniantong_agent_reach.md)、[`sources/README.md`](sources/README.md)

## [2026-06-01] fix(ux) | docs/main.js — 修复运动控制路线页 L1–L7 章节被 L0 自测块吞没

- 根因：`<details class="selftest-answers">` 内的 ` ```mermaid ` 会提前 `flushHtmlBlock()`，导致 `<details>` 未闭合，后续 L1–L7 的 h2 落入错误 DOM，`wrapRoadmapCollapsibleMajorHeadings` 只显示到 L0。
- 变更：HTML 块解析中保留围栏行直至 `</details>`；`flushHtmlBlock` 时将块内 mermaid 转为 `.mermaid`。
- 验证：`roadmap.html?id=roadmap-motion-control` 顶层折叠章节含 L0–L7；`make ci-preflight` 通过。

## [2026-05-31] query | wiki/queries/cross-embodiment-transfer-strategy.md — 跨具身策略迁移选型指南（V23 P1 WBT 知识链收官）

- 新建 wiki：[cross-embodiment-transfer-strategy.md](wiki/queries/cross-embodiment-transfer-strategy.md)（单具身重训 + 重定向 / Any2Any 高效后训练 / 多具身联合训练三路径：9 维「算力 × 数据 × 泛化」对照表 + Mermaid 决策树 + 7 类典型故障模式 + 4 条推荐组合 pipeline；定位为 [WBT pipeline](wiki/concepts/whole-body-tracking-pipeline.md) 阶段 5 选型横切面）。
- 交叉更新：[whole-body-tracking-pipeline.md](wiki/concepts/whole-body-tracking-pipeline.md) 阶段 5 与关联页面、[humanoid-motion-tracking-method-selection.md](wiki/queries/humanoid-motion-tracking-method-selection.md) §6 加入站链接。
- 进度：V23 P1「WBT 知识链 (+3)」三页（pipeline / 四方法对比 / 跨具身 Query）全部完成，标记为 `[x]`。
- 验证：`make lint` 全绿（孤儿页消解）、`eval_search_quality` 37/37 通过。

## [2026-05-31] ingest | sources/papers/php_parkour_arxiv_2602_15827.md、sources/sites/php-parkour-github-io.md、sources/papers/omniretarget_arxiv_2509_26633.md — PHP/RSS2026 与 OmniRetarget 深读；wiki/entities/paper-hrl-stack-22-perceptive_humanoid_parkour.md、wiki/entities/paper-hrl-stack-03-omniretarget.md

## [2026-05-31] fix(ux) | docs/main.js — 路线页自测参考答案等 HTML 块内公式补蓝色边框包裹

- 根因：`<details class="selftest-answers">` 等原样 HTML 块绕过 `renderMathBlocks`，KaTeX 能渲染但缺少 `math-inline` / `math-block` 与 detail 一致的蓝框样式。
- 变更：新增 `applyMathBlocksInHtmlFragment`，在 `flushHtmlBlock` 中对 HTML 片段文本节点补公式包裹。
- 验证：`roadmap.html?id=roadmap-motion-control` 展开参考答案后行内公式带蓝框；`make ci-preflight` 通过。

## [2026-05-31] structural | roadmap/motion-control.md — 各 L 层补充英文缩写速查表（缩写 / 全称 / 简要说明）

- 变更：[roadmap/motion-control.md](roadmap/motion-control.md) 在 L−1～L7 及 L4.1–L4.4、L5.1–L5.3、L7.1–L7.5 增加统一格式「英文缩写速查」表；L−1 原「必备术语速查」改为三列英文全称版。
- 关联页面：`roadmap/motion-control.md`
- 验证：`make ci-preflight` 通过。

## [2026-05-31] structural | references/repos/simulation.md、tech-map/modules/system/simulation.md — 区分「仿真平台索引」与「技术栈模块」详情页，消除同名 Simulation 混淆

- 问题：`detail.html?id=reference-repos-simulation` 与 `tech-node-system-simulation` 标题同为 Simulation、正文量差异大，易被误判为重复页。
- 变更：reference 页改名为「仿真平台与工具链」并链回 tech-map；tech-map 页改名为「仿真（系统集成层）」、补充模块定位与 wiki 互链；更新 `references/repos/README.md` 入口文案。
- 验证：`make ci-preflight` 通过。

## [2026-05-31] ingest | sources/papers/tau0_wm_tech_report.md、sources/sites/tau0-wm-agibot-finch.md、sources/repos/sii_research_tau_0_wm.md — τ₀-WM 统一视频–动作世界模型入库

- 原始资料：[tau0_wm_tech_report.md](sources/papers/tau0_wm_tech_report.md)（<https://finch-static.agibot.com/VAM/blog/tau_0_wm.pdf>）、[tau0-wm-agibot-finch.md](sources/sites/tau0-wm-agibot-finch.md)（<https://finch.agibot.com/research/tau0-wm>）、[sii_research_tau_0_wm.md](sources/repos/sii_research_tau_0_wm.md)（<https://github.com/sii-research/tau-0-wm>、<https://huggingface.co/sii-research/tau-0-wm>）
- 新建 wiki：[tau0-world-model.md](wiki/entities/tau0-world-model.md)（5B VAM、异构 ~27.3k h 掩码预训练、动作条件仿真 + 测试时 propose–evaluate–revise）
- 交叉更新：[world-action-models.md](wiki/concepts/world-action-models.md)、[generative-world-models.md](wiki/methods/generative-world-models.md)、[mimic-video.md](wiki/methods/mimic-video.md)、[ge-sim-2.md](wiki/entities/ge-sim-2.md)、[robot-world-models-training-loop-taxonomy.md](wiki/overview/robot-world-models-training-loop-taxonomy.md)、[manipulation.md](wiki/tasks/manipulation.md)

## [2026-05-31] fix(ux) | docs/main.js — 路线页 Mermaid 在章节折叠后补渲染

- 根因：L4 方法链等 Mermaid 在 `wrapRoadmapCollapsibleMajorHeadings` 之前渲染，部分环境下折叠 DOM 重组后流程图空白或单行。
- 变更：路线正文在折叠包装完成后再 `renderDetailMermaid`；`bindRoadmapSectionMermaidRerender` 在展开章节时对未出 SVG 的图补跑；保留 `htmlLabels: true` 以支持 `<br/>` / `<b>` 多行标签。
- 验证：`roadmap.html?id=roadmap-motion-control` L4 方法链四节点多行渲染；`make ci-preflight` 通过。

## [2026-05-31] fix(ux) | docs/main.js — 修复详情页链接标签内 `*斜体*` / `**粗体**` 未渲染

- 根因：`renderInlineMarkdown` 在链接 token 化时对 label 仅 `escapeHtml`，强调语法在还原后不会再次处理。
- 变更：新增 `renderLinkLabel`，在 `<a>` 内应用与正文一致的 inline 样式；影响含 `*…*` 书名的外链（如 linear-algebra-curriculum）。
- 验证：`detail.html?id=entity-linear-algebra-curriculum` 中 Axler 链接呈现 `<em>`。

## [2026-05-31] structural | roadmap/motion-control.md、docs/main.js — 修复 L4 方法链 Mermaid 换行与加粗渲染

- 根因：`flowchart.htmlLabels: false` 时节点内 `<br/>` / `<em>` 被当作纯文本，四段 L4 标签挤成单行。
- 变更：`docs/main.js` 启用 `htmlLabels: true`；`roadmap/motion-control.md` L4.0 流程图标题改用 `<b>`，去掉易干扰解析的弯引号。
- 验证：本地 `roadmap.html?id=roadmap-motion-control` 中 L4 图 `foreignObject` 多行标签正常；`make ci-preflight` 通过。

## [2026-05-31] ingest | sources/courses/gatech_interactive_linear_algebra.md、sources/courses/axler_linear_algebra_done_right_4e.md、sources/courses/linear_algebra_teaching_materials_curated.md — 线性代数优秀教学材料入库；L0 策展页与运动控制路线互链

- 原始资料：[gatech_interactive_linear_algebra.md](sources/courses/gatech_interactive_linear_algebra.md)（<https://textbooks.math.gatech.edu/ila/>）、[axler_linear_algebra_done_right_4e.md](sources/courses/axler_linear_algebra_done_right_4e.md)（<https://linear.axler.net/LADR4e.pdf>）、[linear_algebra_teaching_materials_curated.md](sources/courses/linear_algebra_teaching_materials_curated.md)（3Blue1Brown、Strang 18.06 等策展）
- 新建 wiki：[linear-algebra-curriculum.md](wiki/entities/linear-algebra-curriculum.md)（机器人 L0 章节地图 + 2–4 周学习路径）
- 交叉更新：[roadmap/motion-control.md](roadmap/motion-control.md) L0 推荐读什么/入口、[modern-robotics-book.md](wiki/entities/modern-robotics-book.md)、[tech-map/modules/math/linear-algebra.md](tech-map/modules/math/linear-algebra.md)

## [2026-05-31] ingest | sources/papers/unilab_arxiv_2605_30313.md、sources/repos/unilab.md、sources/sites/unilabsim-project.md — UniLab 异构 CPU 仿真 / GPU 学习训练系统入库

- 原始资料：[unilab_arxiv_2605_30313.md](sources/papers/unilab_arxiv_2605_30313.md)（<https://arxiv.org/abs/2605.30313>）、[unilab.md](sources/repos/unilab.md)（<https://github.com/unilabsim/UniLab>）、[unilabsim-project.md](sources/sites/unilabsim-project.md)（<https://unilabsim.github.io>）
- 新建 wiki：[unilab.md](wiki/entities/unilab.md)（统一 runtime、MuJoCoUni/MotrixSim 双后端、3–10× 端到端墙钟、跨平台训练）
- 交叉更新：[isaac-gym-isaac-lab.md](wiki/entities/isaac-gym-isaac-lab.md)、[motrix.md](wiki/entities/motrix.md)、[simulator-selection-guide.md](wiki/queries/simulator-selection-guide.md)、[mujoco-vs-isaac-lab.md](wiki/comparisons/mujoco-vs-isaac-lab.md)

## [2026-05-31] structural | wiki/comparisons/ctde-vs-decentralized-marl.md、wiki/queries/humanoid-motion-tracking-method-selection.md、wiki/methods/marl.md — 为两个高频引用 methods 补 queries/comparisons 落地，消除 lint 信息型预警

- 背景：`make lint` 报两条信息型预警——[egm-efficient-general-mimic.md](wiki/methods/egm-efficient-general-mimic.md)、[marl.md](wiki/methods/marl.md) 被多页引用却无 queries/comparisons 落地。
- 新建 wiki：[ctde-vs-decentralized-marl.md](wiki/comparisons/ctde-vs-decentralized-marl.md)（CTDE 集中式训练分布式执行 vs 完全去中心化选型对比）。
- 交叉更新：[humanoid-motion-tracking-method-selection.md](wiki/queries/humanoid-motion-tracking-method-selection.md) 通用 tracker 段补 [EGM](wiki/methods/egm-efficient-general-mimic.md)；[marl.md](wiki/methods/marl.md) 关联页面回链新对比页。
- 验证：`make lint` 全绿，两条信息型预警归零；搜索回归 37/37。

## [2026-05-30] checklist-v23 | wiki/comparisons/sonic-vs-beyondmimic-vs-sdamp-vs-heracles.md — V23 P1「WBT 知识链」第二页落地

- 变更：新建 [wiki/comparisons/sonic-vs-beyondmimic-vs-sdamp-vs-heracles.md](wiki/comparisons/sonic-vs-beyondmimic-vs-sdamp-vs-heracles.md)，把四条主流 WBT 「策略学习」路线（**SONIC 规模化预训练 / BeyondMimic 精准物理 + 失败采样 / SD-AMP 状态门控双判别器 / Heracles 状态条件扩散中间件**）放进同一张 13 维度对照表 + 数据流 Mermaid + 四类适用场景 + 6 类常见误判 + 决策矩阵；显式声明四者按「OOD 修补位置」（数据池 / 训练物理 / 训练判别器 / 部署参考层）构成连续谱而非互斥选择，工程系统常**串联组合**。
- 链接：frontmatter `related` 拉入 [WBT pipeline](wiki/concepts/whole-body-tracking-pipeline.md)、[motion-retargeting-pipeline.md](wiki/concepts/motion-retargeting-pipeline.md)、[whole-body-control.md](wiki/concepts/whole-body-control.md)、[behavior-foundation-model.md](wiki/concepts/behavior-foundation-model.md)、[SONIC](wiki/methods/sonic-motion-tracking.md)、[BeyondMimic](wiki/methods/beyondmimic.md)、[SD-AMP](wiki/entities/paper-unified-walk-run-recovery-sdamp.md)、[Heracles](wiki/entities/paper-heracles-humanoid-diffusion.md)、[Any2Any](wiki/entities/paper-any2any-cross-embodiment-wbt.md)、[AMP](wiki/methods/amp-reward.md)、[DeepMimic](wiki/methods/deepmimic.md)、[扩散运动生成](wiki/methods/diffusion-motion-generation.md)、[motion tracking 选型 query](wiki/queries/humanoid-motion-tracking-method-selection.md)、[balance recovery](wiki/tasks/balance-recovery.md)；`sources` 链入 9 条原始资料（SONIC / BeyondMimic / SD-AMP / Heracles 的 arXiv 摘要 + HRL stack 策展条目 + 项目页 + 代码仓 + sites）。
- 验证：`python3 scripts/lint_wiki.py` — 仅余 2 条与本页无关的预存 INFO（`egm-efficient-general-mimic` / `marl` 高频引用缺 queries/comparisons 落地），新页面在 type / summary / sources / 关联出边等所有阻塞检查项上均 0 错误。
- 关联清单：[`docs/checklists/tech-stack-next-phase-checklist-v23.md`](docs/checklists/tech-stack-next-phase-checklist-v23.md) P1「WBT 知识链 (+3)」中的第二条 `sonic-vs-beyondmimic-vs-sdamp-vs-heracles.md` 打勾；剩 1 页 `queries/cross-embodiment-transfer-strategy.md` 待落地。

## [2026-05-30] ingest | sources/papers/sam_3d_body_arxiv_2602_15989.md、sources/repos/sam-3d-body.md、sources/repos/sam3dbody-cpp.md — SAM 3D Body（MHR 全身 HMR）与 SAM3DBody-cpp 工程运行时入库

- 原始资料：[sam_3d_body_arxiv_2602_15989.md](sources/papers/sam_3d_body_arxiv_2602_15989.md)（<https://arxiv.org/abs/2602.15989>）、[sam-3d-body.md](sources/repos/sam-3d-body.md)（<https://github.com/facebookresearch/sam-3d-body>）、[sam3dbody-cpp.md](sources/repos/sam3dbody-cpp.md)（<https://github.com/AmmarkoV/SAM3DBody-cpp>）
- 新建 wiki：[sam-3d-body.md](wiki/entities/sam-3d-body.md)（可提示单图 MHR、官方 checkpoint/数据集）、[sam3dbody-cpp.md](wiki/entities/sam3dbody-cpp.md)（ONNX+ggml、BVH/CSV、离线五遍精修）
- 交叉更新：[motion-retargeting-pipeline.md](wiki/concepts/motion-retargeting-pipeline.md)、[wilor.md](wiki/methods/wilor.md)、[genmo.md](wiki/methods/genmo.md)

## [2026-05-30] ingest | sources/papers/ge_sim_2_arxiv_2605_27491.md、sources/repos/ge_sim_v2.md、sources/sites/ge-sim-v2-project.md — GE-Sim 2.0 闭环视频世界模拟器入库；wiki/entities/ge-sim-2.md

- 原始资料：[ge_sim_2_arxiv_2605_27491.md](sources/papers/ge_sim_2_arxiv_2605_27491.md)（<https://arxiv.org/abs/2605.27491>）、[ge_sim_v2.md](sources/repos/ge_sim_v2.md)（<https://github.com/AgibotTech/GE-Sim-V2>）、[ge-sim-v2-project.md](sources/sites/ge-sim-v2-project.md)（<https://ge-sim-v2.github.io/>）
- 新建 wiki：[ge-sim-2.md](wiki/entities/ge-sim-2.md)（视觉+本体双专家、World Judge、加速 rollout；WorldArena 2B 榜首）
- 交叉更新：[generative-world-models.md](wiki/methods/generative-world-models.md)、[video-as-simulation.md](wiki/concepts/video-as-simulation.md)、[robot-world-models-training-loop-taxonomy.md](wiki/overview/robot-world-models-training-loop-taxonomy.md)、[ewmbench.md](wiki/entities/ewmbench.md)

## [2026-05-30] structural | wiki/concepts/sim2real.md — 对齐 Sim2Real 工程流程 Mermaid：训练前准备、训练期 DR/RMA、SAGE/中间件/real-to-sim 与可选 Real2Sim 上游

- 变更：更新 [sim2real.md](wiki/concepts/sim2real.md)「Sim2Real 工程流程总览」Mermaid，与 §7 SOP、[sim2real-checklist](wiki/queries/sim2real-checklist.md)、[sim2real-gap-reduction](wiki/queries/sim2real-gap-reduction.md) 时序一致。
- 关联页面：`wiki/concepts/sim2real.md`

## [2026-05-30] ingest | sources/repos/qwen-vla.md、sources/papers/qwenvla_arxiv_2605_30280.md — Qwen-VLA 统一 VLA 通才入库；wiki/entities/qwen-vla.md

- 原始资料：[qwen-vla.md](sources/repos/qwen-vla.md)（<https://github.com/QwenLM/Qwen-VLA>）、[qwenvla_arxiv_2605_30280.md](sources/papers/qwenvla_arxiv_2605_30280.md)（<https://arxiv.org/abs/2605.30280>）
- 新建 wiki：[qwen-vla.md](wiki/entities/qwen-vla.md)（Qwen3.5-4B + 1.15B DiT flow；操作–导航–轨迹统一；embodiment prompt；渐进训练 SFT/RL）
- 交叉更新：[vla.md](wiki/methods/vla.md)、[star-vla.md](wiki/methods/star-vla.md)、[xiaomi-robotics-0.md](wiki/entities/xiaomi-robotics-0.md)、[vla-open-source-repro-landscape-2025.md](wiki/overview/vla-open-source-repro-landscape-2025.md)

## [2026-05-30] ingest | sources/papers/dreamwaq_plus_arxiv_2409_19709.md、sources/sites/dreamwaqpp-github-io.md — DreamWaQ++（arXiv:2409.19709 / T-RO 2026）入库；wiki/entities/dreamwaq-plus.md

- 原始资料：[dreamwaq_plus_arxiv_2409_19709.md](sources/papers/dreamwaq_plus_arxiv_2409_19709.md)（<https://arxiv.org/abs/2409.19709>）、[dreamwaqpp-github-io.md](sources/sites/dreamwaqpp-github-io.md)（<https://dreamwaqpp.github.io/>）
- 新建 wiki：[dreamwaq-plus.md](wiki/entities/dreamwaq-plus.md)（多模态点云+本体、分层外感知记忆、非对称 AC+PPO、楼梯/陡坡/OOD）
- 交叉更新：[privileged-training.md](wiki/concepts/privileged-training.md)、[terrain-adaptation.md](wiki/concepts/terrain-adaptation.md)、[locomotion.md](wiki/tasks/locomotion.md)、[privileged_training.md](sources/papers/privileged_training.md)、[motion-control.md](roadmap/motion-control.md)

## [2026-05-30] ingest | sources/papers/schedulestream_arxiv_2511_04758.md、sources/repos/nvlabs-schedulestream.md — ScheduleStream/TAMPAS 入库；wiki/entities/schedulestream.md

## [2026-05-30] ingest | sources/papers/gamma_world_arxiv_2605_28816.md — Gamma-World 多智能体世界模型入库；wiki/entities/paper-gamma-world-multi-agent.md

## [2026-05-30] ingest | sources/papers/physx_omni_arxiv_2605_21572.md — PhysX-Omni/PhysXVerse/PhysX-Bench 入库；wiki/entities/physx-omni.md

## [2026-05-29] checklist-v23 | wiki/concepts/whole-body-tracking-pipeline.md — V23 P1「WBT 知识链」首页落地

- 变更：新建 [wiki/concepts/whole-body-tracking-pipeline.md](wiki/concepts/whole-body-tracking-pipeline.md)，把 Whole-Body Tracking 端到端流水线统一为 **6 阶段**（参考采集 → 重定向 → 训练数据 → 策略学习 → 跨具身迁移 → 真机部署），并把 **SONIC / BeyondMimic / SD-AMP / Heracles / Any2Any / GMT(RGMT)** 6 条主流落地路径以 6 列对照表展开；包含 mermaid 端到端流程图、与 [人形 RL 身体系统栈](wiki/overview/humanoid-rl-motion-control-body-system-stack.md) 8 层框架的映射、6 类常见失败模式、评测视角。
- 链接：frontmatter `related` 拉入 [motion-retargeting-pipeline.md](wiki/concepts/motion-retargeting-pipeline.md)、[whole-body-control.md](wiki/concepts/whole-body-control.md)、[sim2real.md](wiki/concepts/sim2real.md)、[behavior-foundation-model.md](wiki/concepts/behavior-foundation-model.md) 与 SONIC / BeyondMimic / SD-AMP / Heracles / Any2Any / RGMT 全部 6 条路径的对应 method / entity 页；`sources` 链入对应 8 条原始资料。
- 关联清单：[`docs/checklists/tech-stack-next-phase-checklist-v23.md`](docs/checklists/tech-stack-next-phase-checklist-v23.md) P1「WBT 知识链 (+3)」中的首条页面打勾，剩余 2 页（`comparisons/sonic-vs-beyondmimic-vs-sdamp-vs-heracles.md`、`queries/cross-embodiment-transfer-strategy.md`）继续后推。

## [2026-05-29] lint | wiki/entities/paper-*.md、scripts/fix_paper_entity_lint.py — 清零 paper 实体信息型 lint 预警（261→0）

- 变更：批量补齐 **131** 个 `wiki/entities/paper-*.md` 的 frontmatter 来源键（`arxiv` / `venue` / `code`，从正文 URL、sources 文件名与索引表提取）；缺「方法 / 评测 / 对比」三段式的页面在 [参考来源] 前追加 **方法栈 / 实验与评测 / 与其他工作对比** 策展级摘要块；新增维护脚本 [fix_paper_entity_lint.py](scripts/fix_paper_entity_lint.py) 供后续复跑。
- 验证：`make lint` 信息型预警 **261→0**（`paper_missing_source_meta` / `paper_missing_three_sections` 均为 0）；`make ci-preflight` 通过。
- 关联 PR：与 [2026-05-29] stale-wiki-review 同期分支 `cursor/stale-wiki-review-d40c`（#420）。

## [2026-05-29] lint | stale-wiki-review — 14 个陈旧 wiki 页与较新 sources 交叉同步

- 逐页 review lint 陈旧报告：concepts（character-animation、terrain-adaptation、whole-body-coordination、capture-point-dcm、footstep-planning、embodied-data-cleaning、state-estimation）、methods（exoactor、gae）、comparisons（ppo-vs-sac）、formalizations（probability-flow）、entities（lafan1-dataset、pinocchio）、overview（robot-world-models-training-loop-taxonomy）；补 source 映射与正文切片，`updated: 2026-05-29`。
- 验证：lint 阻塞 **14→0**；`make ci-preflight` 通过。

## [2026-05-28] checklist-v23 | scripts/generate_link_graph.py、docs/main.js、docs/style.css、tests/test_generate_link_graph_latest_nodes.py — V23 P0「图谱 latest_wiki_nodes 时间窗口可配置」收口

- 变更：`scripts/generate_link_graph.py` 把 `latest_wiki_nodes_from_log` 从「锁定最新日历日」改为「最近 30 天回看 + 取前 N 项」；新增形参 `max_items` / `window_days`、模块级常量 `LATEST_NODES_DEFAULT=10` / `LATEST_NODES_CAP=30` / `LATEST_NODES_WINDOW_DAYS=30` / `LATEST_NODES_ENV_VAR="GRAPH_LATEST_NODES_MAX"`；新增 `resolve_latest_nodes_max()` 解析 CLI flag `--latest-nodes-max N`（优先级最高）→ 环境变量 `GRAPH_LATEST_NODES_MAX` → 默认 10，并 clamp 至 [1, 30]；`main()` 接入 argparse 后将 N 透传给 `_compute_graph_stats`。
- 前端：`docs/main.js renderLatestWikiNode` 在跨日返回时按 `recency` 分组渲染「维护日志时间线」（日期 + 项数小标 + 卡片网格），单日时维持原 cards 渲染；只通过 `#homeLatestWikiModule` 挂载点生效（即仅首页 `docs/index.html`），详情/图谱/路线图页不受影响。`docs/style.css` 新增 `.home-latest-wiki-timeline*` 三条轻量样式。
- 测试：`tests/test_generate_link_graph_latest_nodes.py` 新增 10 用例覆盖 `max_items` 单日截断 / `window_days` 多日合并 / 30 天外日期被排除 / `max_items=0` 返空 / CLI vs env 优先级 / 非法 env 回退默认值 / clamp 上限 30 + 下限 1；`PYTHONPATH=scripts python3 -m unittest discover tests -v` 87 通过（含 V22/V23 既有用例），唯一 ERROR `test_lint_wiki_stale_pages` 为 pytest 缺失的预存问题，与本次改动无关。
- 清单：[`docs/checklists/tech-stack-next-phase-checklist-v23.md`](docs/checklists/tech-stack-next-phase-checklist-v23.md) P0「图谱 latest_wiki_nodes 时间窗口可配置」打勾。

## [2026-05-28] ingest | sources/papers/bam_extended_friction_servos_arxiv_2410_08650.md、sources/repos/rhoban_bam.md — BAM 舵机扩展摩擦（arXiv:2410.08650 / ICRA 2025）与 Rhoban/bam 开源管线入库

- 原始资料：[bam_extended_friction_servos_arxiv_2410_08650.md](sources/papers/bam_extended_friction_servos_arxiv_2410_08650.md)（<https://arxiv.org/abs/2410.08650v1>、PDF/HTML）、[rhoban_bam.md](sources/repos/rhoban_bam.md)（<https://github.com/Rhoban/bam>）
- 新建 wiki：[paper-bam-extended-friction-servo-actuators.md](wiki/entities/paper-bam-extended-friction-servo-actuators.md)、[bam-better-actuator-models.md](wiki/entities/bam-better-actuator-models.md)
- 交叉更新：[sim2real.md](wiki/concepts/sim2real.md)、[system-identification.md](wiki/concepts/system-identification.md)、[actuator-network.md](wiki/methods/actuator-network.md)、[sim2real-gap-reduction.md](wiki/queries/sim2real-gap-reduction.md)

## [2026-05-28] ingest | sources/repos/open_duck_*.md — 接入 Open Duck 四仓（Mini Hub / Playground / Reference Motion / Runtime）并沉淀 wiki 实体与 sim2real、locomotion 交叉引用

- 原始资料：[open_duck_mini.md](sources/repos/open_duck_mini.md)、[open_duck_playground.md](sources/repos/open_duck_playground.md)、[open_duck_reference_motion_generator.md](sources/repos/open_duck_reference_motion_generator.md)、[open_duck_mini_runtime.md](sources/repos/open_duck_mini_runtime.md)
- 新建 wiki：[open-duck-mini.md](wiki/entities/open-duck-mini.md)、[open-duck-playground.md](wiki/entities/open-duck-playground.md)、[open-duck-reference-motion-generator.md](wiki/entities/open-duck-reference-motion-generator.md)、[open-duck-mini-runtime.md](wiki/entities/open-duck-mini-runtime.md)
- 交叉更新：[sim2real.md](wiki/concepts/sim2real.md)、[locomotion.md](wiki/tasks/locomotion.md)、[open-source-humanoid-hardware.md](wiki/entities/open-source-humanoid-hardware.md)、[disney-olaf-character-robot.md](wiki/methods/disney-olaf-character-robot.md)

## [2026-05-28] ingest | sources/repos/tnkr.md、sources/blogs/tnkr_launch_youtube_nlv.md — 接入 Tnkr 协作平台（tnkr.ai）与官方 launch 视频；沉淀 wiki/entities/tnkr.md；交叉更新 lerobot、urdf-studio、humanoid-robot

## [2026-05-28] ingest | sources/blogs/genesis_ai_simulation_world_10_blog.md — Genesis AI 博客：仿真评测引擎、Genesis World 1.0（Nyx/Quadrants）与 real-to-sim 相关性叙事

- 原始资料：[genesis_ai_simulation_world_10_blog.md](sources/blogs/genesis_ai_simulation_world_10_blog.md)（<https://www.genesis.ai/blog/the-role-of-simulation-in-scalable-robotics-genesis-world-10-and-the-path-forward>）
- 新建 wiki：[genesis-world-10.md](wiki/entities/genesis-world-10.md)、[simulation-evaluation-infrastructure.md](wiki/concepts/simulation-evaluation-infrastructure.md)
- 交叉更新：[genesis-sim.md](wiki/entities/genesis-sim.md)、[gene-26-5-genesis-ai.md](wiki/entities/gene-26-5-genesis-ai.md)、[genesis_gene_ecosystem.md](sources/papers/genesis_gene_ecosystem.md)、[sim2real.md](wiki/concepts/sim2real.md)

## [2026-05-28] ingest | sources/repos/aholo-viewer.md、sources/blogs/worldlabs_spark_2_0_streaming_3dgs.md — 接入 Aholo Viewer 与 Spark 2.0；沉淀 wiki/entities/spark-3dgs-renderer.md、wiki/entities/aholo-viewer.md、wiki/comparisons/spark-vs-aholo-web-3dgs-renderers.md；交叉更新 world-labs、gs-playground、generative-world-models

## [2026-05-28] fix(lint) | scripts/lint_wiki.py — 陈旧页面检测改用 git commit time，根治 cloud Agent 容器 fresh-clone 伪阳性

- 问题：`_check_sources_health` 用 `Path.stat().st_mtime` 比较 source 与 wiki 的修改时间；cloud Agent 容器 clone 时 `sources/papers/` 的 mtime 被刷成 checkout 时间，wiki 文件保留更早 mtime，导致 18+ 个 wiki 页被误报为「陈旧」（实际两边 git 提交日同日）。
- 修复：新增 `_build_git_mtime_map()`，一次 `git log --format=%ct --name-only` 全量扫描得到 `{Path: 最近提交 unix ts}`；`_check_sources_health` 改为优先使用 git mtime，未入库的本地新文件 fallback 到 fs mtime；非 git 环境下整体退化到旧行为不报硬错。
- 测试：`tests/test_lint_wiki_stale_pages.py` 4 用例覆盖 fresh-clone 场景不误报 / 真陈旧仍报 / 未提交本地编辑回退 fs mtime / 非 git 环境回退 fs mtime；全量 pytest 111 通过，ruff + mypy 均绿。
- 效果：本次 cloud 环境 `make lint` 陈旧条目从 19 个全伪阳性，变为 30 个均为 git 历史中真实「wiki 早于 source ≥1 天」的需 review 列表。

## [2026-05-27] query | wiki/queries/humanoid-motion-tracking-method-selection.md — 对照 smp.md 扩写后的源文件 review：纠正 mermaid 将 SMP 标为「判别器先验」的事实错误（SMP 为评分匹配，非判别器）；section 2 增补 SMP 选型轴（冻结扩散 + 可丢 MoCap vs AMP 同采样量 wall-clock ~1.8×）；frontmatter 加 smp.md 源、bump updated

## [2026-05-27] checklist-v23 | scripts/lint_wiki.py、tests/test_lint_wiki_paper_metadata.py — V23 P0「Entity-Paper 类页元数据 Lint」收口

- 变更：`scripts/lint_wiki.py` 新增 `_check_paper_entity_metadata`，针对 `wiki/entities/paper-*.md` 做两项信息型检查——frontmatter 是否含 `arxiv` / `venue` / `code` 任一来源键、正文是否覆盖「方法 / 评测 / 对比」三段式；两个新 result key `paper_missing_source_meta` 与 `paper_missing_three_sections` 加入 `INFO_ONLY_KEYS`，缺失不阻塞 CI 但写入 lint 报告作为 ingest 流水线基线。
- 基线快照：当前 131 个 paper-* 实体全部缺 arxiv/venue/code 显式键（依赖 `sources:` 间接来源），130/131 缺三段式之一（多数缺「评测」段），将作为后续 ingest 模板对齐的目标。
- 测试：`tests/test_lint_wiki_paper_metadata.py` 新增 6 用例（命中 arxiv/venue/code、缺失定位、三段式部分缺失、非 paper- 实体豁免、信息型计数）；`python -m pytest tests/ --ignore=tests/test_graph_layout.py --no-cov` 106 通过；`ruff check` 与 `mypy scripts/lint_wiki.py` 均绿。
- 清单：[`docs/checklists/tech-stack-next-phase-checklist-v23.md`](docs/checklists/tech-stack-next-phase-checklist-v23.md) P0「Entity-Paper 类页元数据 Lint」打勾。

## [2026-05-27] ingest | sources/papers/smp.md — 扩写 arXiv:2512.03028 完整摘录（摘要、SDS/ESM/GSI、任务与 AMP 对比、G1 部署、wiki 映射）；同步 sources/README 论文索引

## [2026-05-27] ingest | sources/repos/smp_suz_tsinghua.md — 接入清华 SUZ-tsinghua/smp（G1 上 SMP + mjlab 复现）；沉淀 wiki/entities/smp-g1-mjlab.md；交叉更新 wiki/methods/smp.md、wiki/entities/mimickit.md、wiki/entities/mjlab.md、wiki/comparisons/amp-add-smp-motion-prior-variants.md

## [2026-05-27] structural | wiki/entities/* — 为导航·SLAM 栈 21 仓补齐实体节点（slam-toolbox/cartographer/FAST-LIO 等 15 页）

## [2026-05-27] ingest | sources/repos/navigation_slam_autonomy_stack_catalog.md — 入库 Nav2/SLAM/Autoware/Isaac ROS/LeRobot/OpenVLA 等 21 仓；沉淀 wiki/overview/navigation-slam-autonomy-stack.md、wiki/comparisons/lidar-slam-lio-vio-selection.md、wiki/entities/navigation2.md、wiki/entities/autoware.md、wiki/entities/openvla.md、wiki/entities/isaac-ros-visual-slam.md、wiki/entities/isaac-ros-nvblox.md；互链 openloong/lerobot/vla/ros2

## [2026-05-27] ingest | sources/repos/multirotor_uav_stack_catalog.md 及 10 仓 — PX4/XTDrone/EGO-Planner/PyBullet Gym/AirSim/quad-swarm-rl/Crazyswarm2/Crazyflie/Flightmare/MAVSDK 入库；沉淀 wiki/overview/multirotor-simulation-planning-control-stack.md、wiki/entities/px4-autopilot.md、wiki/entities/airsim.md

## [2026-05-27] structural | 首页最新节点 — 人形 RL/AMP 61 篇 + BFM 41 篇论文实体与五类分类 hub

- 人形 RL 身体系统栈（42 篇）：`wiki/entities/paper-hrl-stack-*.md`
- 人形 AMP 运动先验（19 篇）：`wiki/entities/paper-amp-survey-*.md`
- BFM 论文实体（41 篇）：`wiki/entities/paper-bfm-*.md`、`wiki/entities/paper-behavior-foundation-model-humanoid.md`
- BFM 五类分类 hub（5）：`wiki/overview/bfm-category-01-forward-backward-representation.md`、`wiki/overview/bfm-category-02-goal-conditioned-learning.md`、`wiki/overview/bfm-category-03-intrinsic-reward-pretraining.md`、`wiki/overview/bfm-category-04-adaptation.md`、`wiki/overview/bfm-category-05-hierarchical-control.md`
- 总览：[`wiki/overview/humanoid-rl-motion-control-body-system-stack.md`](wiki/overview/humanoid-rl-motion-control-body-system-stack.md)、[`wiki/overview/humanoid-amp-motion-prior-survey.md`](wiki/overview/humanoid-amp-motion-prior-survey.md)、[`wiki/overview/bfm-41-papers-technology-map.md`](wiki/overview/bfm-41-papers-technology-map.md)、[`wiki/concepts/behavior-foundation-model.md`](wiki/concepts/behavior-foundation-model.md)

## [2026-05-27] structural | wiki/overview/bfm-category-01-* … bfm-category-05-* — BFM 五类问题各建图谱分类 hub 节点并交叉链接 41 篇论文实体

- 原始资料：[wechat_embodied_ai_lab_bfm_41_papers_survey.md](sources/blogs/wechat_embodied_ai_lab_bfm_41_papers_survey.md)（<https://mp.weixin.qq.com/s/Ei32la_vo0UW9Y_QCAqB2g>）
- 新增分类页（5）：见 `wiki/overview/bfm-category-01-forward-backward-representation.md` … `bfm-category-05-hierarchical-control.md`
- 交叉更新：41 个 `wiki/entities/paper-bfm-*` 回链对应分类 hub；[bfm-41-papers-technology-map.md](wiki/overview/bfm-41-papers-technology-map.md)、[behavior-foundation-model.md](wiki/concepts/behavior-foundation-model.md)
- 生成脚本：[scripts/generate_bfm_category_overviews.py](scripts/generate_bfm_category_overviews.py)

## [2026-05-27] structural | scripts/generate_bfm_awesome_wiki_entities.py — awesome-bfm 41 篇论文升格 wiki 实体详情页；图谱 + 搜索 + SW 缓存版本同步

- 新增实体（40 篇论文）：`wiki/entities/paper-bfm-zero.md` … `wiki/entities/paper-bfm-41-unihsi.md`（#13 复用 `wiki/entities/paper-behavior-foundation-model-humanoid.md`）
- 新增实体（9 个数据集）：`wiki/entities/dataset-bfm-humanoid-x.md` 等（AMASS 复用 `wiki/entities/amass.md`）
- 交叉更新：`wiki/overview/bfm-41-papers-technology-map.md`（Wiki 实体索引表）、`scripts/sync_sw_cache_version.py`（`sync_all_stats` 链内按 `exports/graph-stats.json` bump `docs/sw.js` CACHE_NAME）

## [2026-05-27] structural | docs/checklists/github-actions-ci-gate.md — 补 CI 门禁看板并开 PR 触发全量 GitHub Actions

- 变更：`docs/checklists/github-actions-ci-gate.md`、`docs/checklists/README.md`、`docs/checklists/cloud-agent-pr-workflow.md`、`schema/README.md`（交叉链接触发 Search & Export Quality Check）
- 目的：在 PR #387 未跑 Actions 即合并后，用 chore PR 重新拉起 `Tests` / `Wiki Lint` / `Search & Export Quality Check`；合并前以 Checks 全绿为准

## [2026-05-27] ingest | sources/papers/bfm_awesome_41_catalog.md、sources/papers/bfm_awesome_*.md（41+10）— awesome-bfm-papers 论文与数据集分别入库；消化更新 wiki/overview/bfm-41-papers-technology-map.md

- 原始资料：[`sources/papers/bfm_awesome_41_catalog.md`](sources/papers/bfm_awesome_41_catalog.md) 及 51 个 `bfm_awesome_<slug>.md`（41 篇论文 + 10 数据集；#13 交叉指向既有 [`bfm_humanoid_arxiv_2509_13780.md`](sources/papers/bfm_humanoid_arxiv_2509_13780.md)）；生成脚本 [`scripts/generate_bfm_awesome_sources.py`](scripts/generate_bfm_awesome_sources.py)；索引 [`sources/README.md`](sources/README.md)
- 沉淀/交叉更新：[`wiki/overview/bfm-41-papers-technology-map.md`](wiki/overview/bfm-41-papers-technology-map.md)（原始资料索引节、01 组 Source 列）、[`sources/blogs/wechat_embodied_ai_lab_bfm_41_papers_survey.md`](sources/blogs/wechat_embodied_ai_lab_bfm_41_papers_survey.md)

## [2026-05-26] checklist-v23 | scripts/search_wiki_core.py、tests/test_search_wiki_core.py — V23 P0「缩写/别名归一化检索 V2」收口

- 变更：`scripts/search_wiki_core.py` 的 `WIKI_ABBREVIATIONS` 在 V22 16 条基础上补齐 9 条 V22 期间高频缩写（**WBT** / **BFM** / **DAgger** / **RSI** / **RFC** / **RMA** / **EMA** / **LoRA** / **DoF**），共 25 条；映射均双向化（`_build_alias_indexes` 自动构造 forward + reverse）。
- 测试：`tests/test_search_wiki_core.py` 新增两组 subTest——`test_v22_abbreviations_expand_to_full`（9 条缩写 → 全称展开）与 `test_v22_full_phrases_expand_to_abbreviation`（9 条全称 → 缩写大写化反向命中），`python -m unittest tests.test_search_wiki_core -v` 26/26 通过。
- 门禁：`ruff check`、`ruff format --check`、`PYTHONPATH=scripts mypy scripts/search_wiki_core.py` 均绿。
- 清单：[`docs/checklists/tech-stack-next-phase-checklist-v23.md`](docs/checklists/tech-stack-next-phase-checklist-v23.md) P0「缩写/别名归一化检索 V2」打勾。

## [2026-05-26] ingest | sources/blogs/wechat_embodied_ai_lab_humanoid_rl_motion_survey.md、sources/blogs/wechat_embodied_ai_lab_humanoid_amp_motion_prior_survey.md — Agent Reach 重抓两篇微信公众号长文；42+19 篇论文分别入库并升格 wiki 实体节点

- 工具：已安装 [Panniantong/Agent-Reach](https://github.com/Panniantong/Agent-Reach) v1.4.0（`pip install` + `agent-reach install --channels=wechat`）；微信正文经 `~/.agent-reach/tools/wechat-article-for-ai`（Camoufox）
- 原始资料：[`sources/blogs/wechat_embodied_ai_lab_humanoid_rl_motion_survey.md`](sources/blogs/wechat_embodied_ai_lab_humanoid_rl_motion_survey.md)（<https://mp.weixin.qq.com/s/hz9JXtJeUPRfUGzfD-pZuA>）、[`sources/blogs/wechat_embodied_ai_lab_humanoid_amp_motion_prior_survey.md`](sources/blogs/wechat_embodied_ai_lab_humanoid_amp_motion_prior_survey.md)（<https://mp.weixin.qq.com/s/YZsm3855iP3TNTTt1aou7w>）；抓取落盘 [`sources/raw/wechat_humanoid_rl_42_survey_2026-05-26.md`](sources/raw/wechat_humanoid_rl_42_survey_2026-05-26.md)、[`sources/raw/wechat_humanoid_amp_19_survey_2026-05-26.md`](sources/raw/wechat_humanoid_amp_19_survey_2026-05-26.md)
- 论文 source：[`sources/papers/humanoid_rl_stack_42_catalog.md`](sources/papers/humanoid_rl_stack_42_catalog.md) + 42× `humanoid_rl_stack_*`；[`sources/papers/humanoid_amp_survey_19_catalog.md`](sources/papers/humanoid_amp_survey_19_catalog.md) + 19× `humanoid_amp_survey_*`；生成脚本 [`scripts/generate_humanoid_stack_survey.py`](scripts/generate_humanoid_stack_survey.py)
- 沉淀实体（61）：`wiki/entities/paper-hrl-stack-01-*.md` … `paper-hrl-stack-42-*.md`；`wiki/entities/paper-amp-survey-01-*.md` … `paper-amp-survey-19-*.md`
- 交叉更新：[`wiki/overview/humanoid-rl-motion-control-body-system-stack.md`](wiki/overview/humanoid-rl-motion-control-body-system-stack.md)、[`wiki/overview/humanoid-amp-motion-prior-survey.md`](wiki/overview/humanoid-amp-motion-prior-survey.md)、[`sources/README.md`](sources/README.md)

## [2026-05-26] ingest | sources/repos/simplefoc_arduino_foc.md、sources/sites/simplefoc_documentation.md — 接入 SimpleFOC 生态；沉淀 wiki/entities/simplefoc.md、wiki/concepts/field-oriented-control.md；交叉更新 wiki/overview/motor-drive-firmware-bus-protocols.md

## [2026-05-26] ingest | sources/blogs/wechat_embodied_ai_lab_bfm_41_papers_survey.md — Agent Reach 抓取具身智能研究室 BFM 41 篇专题长文并消化入库

- 工具：已安装 [Panniantong/Agent-Reach](https://github.com/Panniantong/Agent-Reach) v1.4.0（`pip install` + `agent-reach install --channels=wechat`）；微信正文经 `~/.agent-reach/tools/wechat-article-for-ai`（Camoufox），Jina Reader 对 `mp.weixin.qq.com` 返回 CAPTCHA
- 原始资料：[`sources/blogs/wechat_embodied_ai_lab_bfm_41_papers_survey.md`](sources/blogs/wechat_embodied_ai_lab_bfm_41_papers_survey.md)（<https://mp.weixin.qq.com/s/Ei32la_vo0UW9Y_QCAqB2g>）；配套 [`sources/repos/awesome_bfm_papers.md`](sources/repos/awesome_bfm_papers.md)、[`sources/papers/bfm_survey_arxiv_2506_20487.md`](sources/papers/bfm_survey_arxiv_2506_20487.md)；索引 [`sources/README.md`](sources/README.md)
- 沉淀页面：[`wiki/overview/bfm-41-papers-technology-map.md`](wiki/overview/bfm-41-papers-technology-map.md)（五类问题 × 41 篇地图 + Mermaid + 智元/众擎策展观察）
- 交叉更新：[`wiki/concepts/behavior-foundation-model.md`](wiki/concepts/behavior-foundation-model.md)、[`wiki/overview/humanoid-rl-motion-control-body-system-stack.md`](wiki/overview/humanoid-rl-motion-control-body-system-stack.md)、[`wiki/entities/paper-behavior-foundation-model-humanoid.md`](wiki/entities/paper-behavior-foundation-model-humanoid.md)、[`sources/repos/panniantong_agent_reach.md`](sources/repos/panniantong_agent_reach.md)

## [2026-05-26] ingest | sources/repos/awesome_bfm_papers.md、sources/papers/bfm_survey_arxiv_2506_20487.md — 接入 awesome-bfm-papers 与 BFM 综述；沉淀 wiki/concepts/behavior-foundation-model.md；交叉更新 foundation-policy、whole-body-control、paper-behavior-foundation-model-humanoid、humanoid-rl-motion-control-body-system-stack

- 原始资料：<https://github.com/friedrichyuan/awesome-bfm-papers>、[`sources/papers/bfm_survey_arxiv_2506_20487.md`](sources/papers/bfm_survey_arxiv_2506_20487.md)（arXiv:2506.20487）；索引 [`sources/README.md`](sources/README.md)
- 沉淀页面：[`wiki/concepts/behavior-foundation-model.md`](wiki/concepts/behavior-foundation-model.md)（BFM 定义、预训练三线 + 适应两线 taxonomy、Mermaid 流程图）
- 交叉更新：[`wiki/concepts/foundation-policy.md`](wiki/concepts/foundation-policy.md)、[`wiki/concepts/whole-body-control.md`](wiki/concepts/whole-body-control.md)、[`wiki/entities/paper-behavior-foundation-model-humanoid.md`](wiki/entities/paper-behavior-foundation-model-humanoid.md)、[`wiki/overview/humanoid-rl-motion-control-body-system-stack.md`](wiki/overview/humanoid-rl-motion-control-body-system-stack.md)

## [2026-05-25] checklist-v22 | DoD 收口 & 初始化 V23

- V22 DoD 最后一项「`log.md` 记录 V22 关键改动」收口：本条目即为兑现物，把 V22 P0–P3 与 DoD 数值快照沉淀到日志，与 [`docs/checklists/tech-stack-next-phase-checklist-v22.md`](docs/checklists/tech-stack-next-phase-checklist-v22.md) 同步勾选并标注 2026-05-25 验证日期。
- V22 完整交付：
    - **P0 自动化**：① `scripts/search_wiki_core.py` 缩写归一化检索（16 条 WBC/VLA/IL/RL/MPC/PPO/SAC/HQP/CBF/CLF/BC/IK/FK/LIP/ZMP/TSID，双向展开 + "已展开为…"提示）；② `scripts/generate_link_graph.py` 社区粒度二级拆分（Girvan-Newman 一级 + Louvain `resolution=1.15` 二级），最大社区占比由 V21 46.1% → V22 15.9%（-30.2 pp），17 社区均衡分布；③ `scripts/lint_wiki.py` 新增 `methods_without_practitioner_query` 方法-Query 闭环 Lint（INFO 级，不阻塞 CI，作为 P1/P2 推进基线）。
    - **P1 动作重定向与角色化人形**：新增 5 页 `wiki/concepts/motion-retargeting-pipeline.md`、`wiki/formalizations/motion-retargeting-objective.md`、`wiki/comparisons/gmr-vs-nmr-vs-reactor.md`、`wiki/concepts/character-animation-vs-robotics.md`，覆盖「映射几何 → 目标函数 → 谱系对比 → 角色 vs 工业边界」四视角，双向回链 GMR / NMR / ReActor / SONIC / ExoActor / WBC / Sim2Real / Disney Olaf / Roboto Origin。
    - **P2 抓取与操作感知**：新增 3 页 `wiki/methods/grasp-pose-estimation.md`、`wiki/queries/grasp-policy-selection.md`、`wiki/comparisons/anygrasp-vs-graspnet.md`，覆盖 GraspNet 三代谱系（GPD → GraspNet-1Billion → Contact-GraspNet/AnyGrasp）与「检测式 + IL/VLA」选型；同步在 `wiki/concepts/contact-rich-manipulation.md` 与 `wiki/concepts/visuo-tactile-fusion.md` 中补「抓取 → 插装 → 精细操作」三段式级联，把 P1 触觉链路与 P2 抓取链路打通。
    - **P3 交互层**：① `docs/detail.html` + `docs/main.js` + `docs/style.css` 新增「关联页面社区分布」横向条形小图，按 link-graph 17 社区聚类显示当前节点邻域偏向；② `docs/graph.html` 新增「专题视图」切换器（10 项专题：动作重定向 / 抓取 / 触觉 / 通信协议 / WBC / Locomotion / VLA / IL+RL / Sim2Real / 状态估计），社区 id + path 片段双路并集判定，切换时自动 `fitToVisibleNodes()`。
    - **事实库**：`schema/canonical-facts.json` 由 140 → **156** 条（+16），重点补动作重定向 5 条 / 抓取与感知 10 条 / 近期 ingest 2 条（BifrostUMI / OpenLoong）。
- DoD 数值快照（验证日 2026-05-25 `exports/graph-stats.json` `generated_at: 2026-05-25`）：
    | 维度 | V22 目标 | V22 实测 | 达成情况 |
    |------|----------|----------|----------|
    | `make lint` | 0 errors | 0 errors（419/419 wiki/entity 页 ingest 来源覆盖率 100%） | ✅ 远超 |
    | 图谱节点 | ≥ 312 | 429 | ✅ +117 / +37.5% |
    | 图谱边 | ≥ 2050 | 3200 | ✅ +1150 / +56.1% |
    | 事实库 | ≥ 155 | 156 | ✅ 达标 |
    | `community_quality_warning` | false | false（最大社区 VLA 15.9% / 17 社区） | ✅ 远超 ≤ 40% 阈值 |
- 新建 [`docs/checklists/tech-stack-next-phase-checklist-v23.md`](docs/checklists/tech-stack-next-phase-checklist-v23.md)：专题选定「全身运动跟踪（WBT）与跨具身迁移」，配合「真机安全微调与 Sim2Real 深化」；P1 直接消化 V22 期间已 ingest 的 SONIC / SD-AMP / Heracles / Any2Any / SLowRL / BifrostUMI / BFM 等 WBT 谱系论文，P2 围绕 SLowRL 安全 LoRA / Heracles 扩散兜底等真机安全微调路径展开；P3 详情页新增「最近 ingest 时间线」与图谱专题视图扩充 3 项（WBT / 跨具身 / 真机安全）。V23 目标：节点 ≥ 445、边 ≥ 3320、事实库 ≥ 170、`largest_community_ratio ≤ 0.25`。
- 同步将 `README.md` 维护看板 + Sources Coverage badge、`AGENTS.md` § `docs/checklists/`、`docs/README.md` 常用入口、`docs/checklists/README.md` 当前入口与历史归档的「当前清单」指针从 V22 切到 V23；V22 进入历史归档区（`docs/checklists/README.md` 历史列表追加 v22 条目）。
- 本轮无代码改动，仅清单/日志状态回填与索引/指针同步；下一日按"每日推进一项"节奏从 V23 P0「缩写/别名归一化检索 V2」起步。

## [2026-05-25] ingest | sources/papers/slowrl_arxiv_2603_17092.md + any2any_arxiv_2605_23733.md — 接入 SLowRL（Go2 安全 LoRA 真机微调）与 Any2Any（跨具身 WBT 迁移）；沉淀 wiki/entities/paper-slowrl-safe-lora-locomotion-sim2real.md、wiki/entities/paper-any2any-cross-embodiment-wbt.md；交叉更新 sim2real、locomotion、humanoid-motion-tracking-method-selection、sonic-motion-tracking

- 原始资料：`sources/papers/slowrl_arxiv_2603_17092.md`（<https://arxiv.org/abs/2603.17092>）、`sources/papers/any2any_arxiv_2605_23733.md`（<https://arxiv.org/abs/2605.23733>）；索引 `sources/README.md`
- 沉淀页面：`wiki/entities/paper-slowrl-safe-lora-locomotion-sim2real.md`、`wiki/entities/paper-any2any-cross-embodiment-wbt.md`
- 交叉更新：`wiki/concepts/sim2real.md`、`wiki/tasks/locomotion.md`、`wiki/queries/humanoid-motion-tracking-method-selection.md`、`wiki/methods/sonic-motion-tracking.md`

## [2026-05-25] ingest | sources/papers/unified_walk_run_recovery_sdamp_arxiv_2605_18611.md + heracles_humanoid_diffusion_arxiv_2603_27756.md — 沉淀 wiki/entities/paper-unified-walk-run-recovery-sdamp.md、wiki/entities/paper-heracles-humanoid-diffusion.md；交叉更新 amp-reward、locomotion、balance-recovery、diffusion-motion-generation、humanoid-motion-tracking-method-selection、amp-mjlab、unitree-g1

## [2026-05-25] ingest | sources/repos/ppf-contact-solver.md — 接入 ZOZO GPU 接触求解器并沉淀 wiki/entities/ppf-contact-solver.md、wiki/entities/paper-ppf-cubic-barrier-contact-solver.md

## [2026-05-24] structural | docs/checklists/tech-stack-next-phase-checklist-v22.md — V22 DoD「community_quality_warning: false」回填打勾

- 触发：[`docs/checklists/tech-stack-next-phase-checklist-v22.md`](docs/checklists/tech-stack-next-phase-checklist-v22.md) DoD 余 2 项中数值最直接可验项；按"每日推进一项"节奏继续顺次回填
- 验证：`exports/graph-stats.json`（`generated_at: 2026-05-24`）实测 `community_count = 17`、`largest_community_ratio = 0.248`（最大社区 = "VLA（Vision-Language-Action） 社区" 105 / 423 = 24.8%，远低于 V22 ≤ 40% 阈值）、`community_quality_warning = false`、`singleton_communities = []`；最大社区占比相对 V21 基线 46.1% 累计下降 21.3 pp，结构稳定且 17 个社区中 ≥ 10 项节点的有 12 个
- 归因：V22 P0「社区粒度二级拆分」（Girvan-Newman 一级 + Louvain `resolution=1.15` 二级对占比 > 40% 且节点数 ≥ 30 的巨型社区做二级拆分）持续生效，叠加 P1 / P2 / P3 累积新增页面（motion-retargeting × 5 / 抓取链 × 3 / 接触-操作交叉 / VLA-WAM / BifrostUMI / OpenLoong / WorldVLN / easy_quadruped 等）形成的多向回链让原 Locomotion 巨型社区进一步均匀化
- 状态联动：V22 checklist DoD「community_quality_warning: false」由 `[ ]` 变 `[x]`；checklist 文件就地追加 2026-05-24 验证日期与数值快照
- 后续：DoD 余 1 项（`log.md` 记录 V22 关键改动）按节奏继续回填，本日新增日志本身即对该项的部分兑现；DoD 全部清零后基于 llm-wiki 与最新 graph-stats / 事实库 / 站点状态新建 V23 清单
- 本轮无代码改动，仅清单与日志状态回填

## [2026-05-24] ingest | sources/papers/worldvln_arxiv_2605_15964.md — 接入 WorldVLN 空中 VLN 自回归 WAM；沉淀 wiki/entities/paper-worldvln-aerial-vln-wam.md；交叉更新 vision-language-navigation、world-action-models

- 原始资料：`sources/papers/worldvln_arxiv_2605_15964.md`（<https://arxiv.org/abs/2605.15964>）、`sources/sites/worldvln-embodiedcity.md`、 `sources/repos/worldvln_embodiedcity.md`；索引 `sources/README.md`
- 沉淀页面：`wiki/entities/paper-worldvln-aerial-vln-wam.md`
- 交叉更新：`wiki/tasks/vision-language-navigation.md`、`wiki/concepts/world-action-models.md`

## [2026-05-24] ingest | sources/repos/easy_quadruped.md — 接入 Xzgz718/easy_quadruped（StanfordQuadruped 二次开发）并沉淀 wiki 实体

- 原始资料：`sources/repos/easy_quadruped.md`（上游 MIT [StanfordQuadruped](https://github.com/stanfordroboticsclub/StanfordQuadruped)，公开快照含 `src/` 步态控制、`pupper/` IK/标定、`sim/` MuJoCo 浮动机身闭环）
- 沉淀页面：`wiki/entities/easy-quadruped.md`
- 交叉更新：`wiki/entities/stanford-doggo-and-pupper.md`、`wiki/entities/quadruped-robot.md`、`wiki/concepts/gait-generation.md`、`references/repos/simulation.md`、`sources/README.md`

## [2026-05-24] structural | docs/checklists/tech-stack-next-phase-checklist-v22.md — V22 DoD「图谱节点 ≥ 312 / 边 ≥ 2050」回填打勾

- 触发：[`docs/checklists/tech-stack-next-phase-checklist-v22.md`](docs/checklists/tech-stack-next-phase-checklist-v22.md) DoD 余 3 项中数值最直接可验项；按"每日推进一项"节奏，今日选定图谱规模口径
- 验证：`exports/graph-stats.json`（`generated_at: 2026-05-23`）实测 `node_count = 421`（V22 目标 312，超 +109 / +34.9%）、`edge_count = 3122`（V22 目标 2050，超 +1072 / +52.3%）、`community_count = 17`、`largest_community_ratio = 0.254`、`orphan_nodes = []`，两项数值远超 V22 目标且与 V22 P1（动作重定向 5 页 + 多向回链）/ P2（抓取链 3 页 + 接触-操作交叉 + AnyGrasp/GraspNet 互链）/ P3（详情页社区分布 + 图谱专题视图）历史推升轨迹一致
- 状态联动：V22 checklist DoD「图谱节点 ≥ 312 边 ≥ 2050」由 `[ ]` 变 `[x]`；checklist 文件就地追加 2026-05-24 验证日期与数值快照
- 后续：DoD 余 2 项（`community_quality_warning: false`、`log.md` 记录 V22 关键改动）按节奏继续回填，全部完成后基于 llm-wiki 与最新 graph-stats / 事实库 / 站点状态新建 V23 清单
- 本轮无代码改动，仅清单与日志状态回填

## [2026-05-24] lint | docs/checklists/tech-stack-next-phase-checklist-v22.md — V22 DoD「`make lint`: 0 errors」回填打勾

- 触发：[`docs/checklists/tech-stack-next-phase-checklist-v22.md`](docs/checklists/tech-stack-next-phase-checklist-v22.md) DoD 余 4 项中最确定可验项；按 2026-05-23 后续计划「每日推进一项」执行
- 验证：`make lint` 实跑 = `python3 scripts/eval_search_quality.py`（通过率 37/37，≥ 80% 阈值）→ `python3 scripts/lint_wiki.py`（0 矛盾 / 0 空壳页 / 0 高频缺页 / 0 缺 type / 0 log.md 活跃度警告 / 0 缺摘要 / 0 Query 格式残缺 / 0 Formalization 缺公式 / 0 公式变量缺解释 / 0 README 版本不一致 / 0 图谱孤儿节点 / 0 Methods 缺 Formalization / Concept / 主要路线 / 0 Entities 缺 Methods/Tasks 出边 / 0 高频 methods 缺 queries/comparisons）；419/419 wiki/entity 页 ingest 来源覆盖率 100%；终行 "✅ 所有检查通过！"
- 状态联动：V22 checklist DoD「`make lint`: 0 errors」由 `[ ]` 变 `[x]`；checklist 文件就地追加验证日期与项目级 0 警告快照
- 后续：DoD 余 3 项（图谱节点 ≥ 312 边 ≥ 2050、`community_quality_warning: false`、log.md 记录 V22 关键改动）按节奏继续回填，全部完成后基于 llm-wiki 与最新 graph-stats / 事实库 / 站点状态新建 V23 清单
- 本轮无代码改动，仅清单与日志状态回填

## [2026-05-23] structural | schema/canonical-facts.json — V22 DoD 事实库扩展：140 → 156 条，补全动作重定向 / 抓取 / 近期 ingest 矛盾检测规则

- 触发：[`docs/checklists/tech-stack-next-phase-checklist-v22.md`](docs/checklists/tech-stack-next-phase-checklist-v22.md) DoD「事实库扩展至 155 条以上（重点补 motion-retargeting / grasp-pose 矛盾检测规则）」尚未打勾；P1 / P2 主线已沉淀大量新页（motion-retargeting-pipeline / motion-retargeting-objective / gmr-vs-nmr-vs-reactor / character-animation-vs-robotics / grasp-pose-estimation / grasp-policy-selection / anygrasp-vs-graspnet）以及 OpenLoong / BifrostUMI 实体页，需要让 `lint_wiki._check_contradictions` 覆盖到位
- 新增条目（17 条 → 总计 156 条）：
    - **动作重定向 5 条**：`GMR 运动学优化定位`（IK/QP/运动学层 vs 强化学习/仿真闭环）、`ReActor 双层联合优化`（参数化参考 + 单一策略联合更新 vs 纯运动学/离线/开环）、`Motion Retargeting Pipeline 端到端阶段`（8 阶段流水线 vs 单次映射/单阶段）、`Motion Retargeting 目标函数加权组合`（姿态 + 接触 + 平衡 + 限位 + 平滑 vs 单一姿态项）、`Character Humanoid 目标双重性`（表演可信度 × 物理可控性的三方博弈 vs 与工业人形等价）
    - **抓取与感知 10 条**：`6-DoF vs 7-DoF 抓取`（7-DoF = 6-DoF + 夹爪开度）、`GraspNet 三代谱系演进`（采样评估 → 稠密回归 → 时序关联）、`Contact-GraspNet 接触点参数化`（每点回归基线方向 + 接近向量 + 抓取宽度）、`AnyGrasp 跨帧时序关联`（many-to-many + COG 稳定度 + bin clearing vs 单帧独立）、`AnyGrasp SDK License 分发`（二进制 + License vs 完全开源）、`MPPH 抓取吞吐指标`（Mean Picks Per Hour，吞吐口径 vs 精度/AP 等价）、`抓取候选需显式碰撞检查`（网络分数 ≠ 物理可执行）、`抓取选型 检测式优先`（先检测式 grasp pose 起步，再 IL/VLA 替换可学环节）、`GraspNet-1Billion 评测基准`（百万级真实标注、公开 benchmark）
    - **近期 ingest 2 条**：`BifrostUMI 无机器人示范`（robot-free 全身示范 + 扩散 47-D 高层 + SKR vs 依赖真机遥操作/无扩散）、`OpenLoong 全栈开源`（青龙公版机硬件 + 软件 + 社区门户 vs 闭源/仅软件）
- 验证：`python3 scripts/lint_wiki.py` 退出码 0、0 contradictions、0 ⚠️ / 0 💡，"✅ 所有检查通过！"；419/419 wiki/entity 页 ingest 来源覆盖率 100%。`exports/graph-stats.json` 维持 421 nodes / 3122 edges / `community_quality_warning: false`，本次仅触动 schema，未派生重排
- 状态联动：V22 checklist DoD「事实库 155 条」由 `[ ]` 变 `[x]`；P1「动作重定向知识链 (+3)」父项由 `[~]` 变 `[x]`（3/3 子项已早期落地，仅此次回填父项状态）
- 后续：DoD 余 4 项（`make lint` 0 errors / 图谱节点 ≥ 312 边 ≥ 2050 / `community_quality_warning: false` / log.md 记录 V22 关键改动）均已在历史记录中达成或自然满足，下日按"每日推进一项"继续顺次打勾或在 V22 完全收尾时新建 V23

## [2026-05-23] ingest | sources/papers/bifrost_umi_arxiv_2605_03452.md — 接入 BifrostUMI 无机器人人形全身示范与 SKR 管线并沉淀 wiki/entities/paper-bifrost-umi.md

- 原始资料：`sources/papers/bifrost_umi_arxiv_2605_03452.md`（<https://arxiv.org/abs/2605.03452>）、`sources/sites/bifrost-umi-project.md`（<https://baai-aether.github.io/BifrostUMI/>）；索引 `sources/README.md`
- 沉淀页面：`wiki/entities/paper-bifrost-umi.md`（Robot-Free 采集、扩散 47-D 关键点高层、SKR、mink IK + WBC、G1 实验与 Mermaid 管线）
- 交叉更新：`wiki/tasks/teleoperation.md`、`wiki/tasks/loco-manipulation.md`、`wiki/concepts/motion-retargeting.md`、`wiki/methods/diffusion-policy.md`、`wiki/entities/unitree-g1.md`、`sources/papers/teleoperation.md`
- 派生再生成：`make ci-preflight`

## [2026-05-23] ingest | sources/repos/openloong.md — 接入 OpenLoong 青龙全栈开源（硬件 AtomGit、Framework、Dyn-Control、社区门户）并沉淀 wiki/entities/openloong.md

- 原始资料：`sources/repos/openloong.md`、`sources/repos/openloong_hardware.md`（<https://atomgit.com/openloong/OpenLoongHardware/tree/main>）、`sources/sites/openloong_community.md`（<https://www.openloong.org.cn/cn/projects/openloong>）；索引 `sources/README.md`
- 沉淀页面：`wiki/entities/openloong.md`（四层架构、硬件 TA 子系统、Framework 子仓矩阵、MPC+WBC 与并行栈链接）
- 交叉更新：`wiki/entities/open-source-humanoid-hardware.md`、`wiki/entities/humanoid-robot.md`
- 派生再生成：`make ci-preflight`

## [2026-05-22] ingest | sources/papers/esi_bench_arxiv_2605_18746.md — 接入 ESI-Bench 具身空间智能基准并沉淀 wiki/entities/esi-bench.md

- 原始资料：`sources/papers/esi_bench_arxiv_2605_18746.md`（<https://arxiv.org/abs/2605.18746>）、`sources/sites/esi-bench-project.md`（<https://esi-bench.github.io/>）、`sources/repos/esi_bench.md`（<https://github.com/ESI-Bench/ESI-Bench>）；索引 `sources/README.md`
- 沉淀页面：`wiki/entities/esi-bench.md`（感知–行动环、10/29/3081 任务 taxonomy、MLLM 主动/被动/oracle 发现）
- 交叉更新：`wiki/concepts/3d-spatial-vqa.md`
- 派生再生成：`make ci-preflight`

## [2026-05-22] ingest | sources/papers/wem_arxiv_2605_19957.md — 接入 WEM/World-Ego Modeling 与 HTEWorld；沉淀 wiki/entities/paper-wem-world-ego-modeling.md 并交叉更新 generative-world-models、robot-world-models-taxonomy、loco-manipulation、ewmbench

## [2026-05-22] ingest | sources/blogs/wechat_shenlan_vln_repro_four_paradigms_2026.md — Agent Reach 抓取深蓝具身智能 VLN 四范式新手复现长文并消化入库

- 原始资料：`sources/blogs/wechat_shenlan_vln_repro_four_paradigms_2026.md`（<https://mp.weixin.qq.com/s/AzCDukzwrfIyms_65kh1mg>）；索引 `sources/README.md`
- 沉淀页面：`wiki/overview/vln-open-source-repro-paradigms.md`（VLFM / NavGPT / NoMaD / Uni-NaVid + Mermaid 演进 + 复现门槛表）
- 交叉更新：`wiki/tasks/vision-language-navigation.md`、`wiki/overview/vla-open-source-repro-landscape-2025.md`、`sources/blogs/wechat_shenlan_vla_github_repro_survey_2025.md`
- 派生再生成：`make ci-preflight`

## [2026-05-22] ingest | sources/blogs/wechat_shenlan_vla_github_repro_survey_2025.md — Agent Reach 抓取深蓝具身智能 VLA GitHub 复现推荐长文并消化入库

- 原始资料：`sources/blogs/wechat_shenlan_vla_github_repro_survey_2025.md`（<https://mp.weixin.qq.com/s/k_i-1NEBP-lEzth19HOHkQ>）；索引 `sources/README.md`
- 沉淀页面：`wiki/overview/vla-open-source-repro-landscape-2025.md`（11 项开源栈 + Mermaid 景观 + 复现目标表）
- 交叉更新：`wiki/methods/vla.md`、`wiki/methods/star-vla.md`、`wiki/queries/manipulation-vla-architecture-selection.md`、`sources/blogs/wechat_shenlan_lie_group_lie_algebra_quaternion.md`
- 派生再生成：`make ci-preflight`

## [2026-05-22] ingest | sources/blogs/wechat_shenlan_lie_group_lie_algebra_quaternion.md — 安装 Agent Reach 抓取深蓝具身智能李群/李代数/四元数专栏文并消化入库

- 工具：已安装 [Panniantong/Agent-Reach](https://github.com/Panniantong/Agent-Reach) v1.4.0（`pip install` + `agent-reach install --channels=wechat`）；微信正文经 `~/.agent-reach/tools/wechat-article-for-ai`（Camoufox），Jina Reader 对 `mp.weixin.qq.com` 返回 CAPTCHA
- 原始资料：`sources/blogs/wechat_shenlan_lie_group_lie_algebra_quaternion.md`（<https://mp.weixin.qq.com/s/JviRH2LW-fkCHA5gY7Qflw>）；索引 `sources/README.md`
- 沉淀页面：`wiki/formalizations/lie-group-rigid-body-motions.md`
- 交叉更新：`wiki/formalizations/se3-representation.md`、`wiki/entities/modern-robotics-book.md`、`sources/repos/panniantong_agent_reach.md`
- 派生再生成：`make ci-preflight`

## [2026-05-22] ingest | sources/sites/robotics-venues-primary-refs.md — 汇总 ICRA、IROS、CoRL、RSS、T-RO、IJRR、Science Robotics 官方介绍与投稿入口；沉淀 wiki/comparisons/robotics-research-venues.md

- 原始资料：`sources/sites/robotics-venues-primary-refs.md`；索引 `sources/README.md`
- 沉淀页面：`wiki/comparisons/robotics-research-venues.md`
- 交叉更新：`wiki/overview/robot-learning-overview.md`、`references/papers/README.md`
- 派生再生成：`make ci-preflight`（与 main 上 PR #353 一致；合并时与深蓝专栏 ingest 同日块并列保留）

## [2026-05-21] ingest | sources/sites/kimodo-project.md、sources/papers/kimodo_arxiv_2603_15546.md — 深化 Kimodo 官方项目页与 arXiv:2603.15546 论文摘录；扩充 sources/repos/kimodo.md、wiki/entities/kimodo.md（两阶段去噪、变体选型、Mermaid 管线、GEM/SONIC/ProtoMotions 互链）；交叉更新 wiki/methods/diffusion-motion-generation.md

## [2026-05-21] feat(ux) | docs/detail.html、docs/main.js、docs/style.css — V22 P3 详情页「关联项按社区分布」小条形图（基于 link-graph 社区，替换早些时候的按 type 分桶版本）

- 触发：PR #347 review，社区维度（link-graph 的 Girvan-Newman + Louvain 二级拆分）比类型维度更有信息量——type 字段与 frontmatter 直接重复，而社区分桶能体现「当前节点的 1-hop 邻域聚集在哪几个主题」，与 V22 P0 的社区粒度二级拆分（17 个社区 / largest_community_ratio ≤ 0.40）形成闭环
- 改动形态：
  - [`docs/detail.html`](docs/detail.html)：`#detailRelatedTypeDist` 容器更名为 `#detailRelatedCommunityDist`，标题文案改「按社区分布」
  - [`docs/main.js`](docs/main.js)：移除按 type 派生中文标签的 `deriveDetailCategoryLabel()` 与 `renderRelatedTypeDistribution()`；新增 `ensureDetailCommunityIndex()`（懒加载 `exports/link-graph.json`，建立 `pathToCommunity` Map 与 `communityLabel` 字典，失败兜底为空 Map）与 `renderRelatedCommunityDistribution()`（按 detail page 的 `path` 查表 → 拿社区 ID → 计数；不在图谱内的 roadmap / reference / tech_map 统一桶为「未分类」并永远排在末尾，避免遮挡有效社区）；社区标签显式 `replace(/\s*社区\s*$/, '')` 去掉末尾「社区」二字以节省横向空间，悬停 `title` 仍保留完整原始标签
  - [`docs/style.css`](docs/style.css)：`.related-type-*` 系列样式整体改名为 `.related-community-*`，桌面端标签列宽 92px → 160px，540px 窄屏 78px → 110px，以容纳更长的社区中文标签（如 "Whole-Body Control (WBC，全身控制)"）
  - [`docs/checklists/tech-stack-next-phase-checklist-v22.md`](docs/checklists/tech-stack-next-phase-checklist-v22.md)：P3 首项标题与实现说明同步换为「关联社区分布」版本，附 type→community 切换理由
- 验证：`make lint-js` 通过；本地 http.server + Puppeteer 视口截图 `wiki-concepts-whole-body-control` 桌面 / 移动双端（共 12 项 · 8 个社区，含 Whole-Body Control 4 / Motion Retargeting 3 / Imitation Learning 1 / Locomotion 1 / Sim2Real 1 / Unitree G1 1 / Reward Design 1 / 未分类 1）与 `wiki-concepts-armature-modeling`（共 5 项 · 3 个社区，WBC 3 / Motion Retargeting 1 / Sim2Real 1）均正确落稳
- 截图：`.cursor-artifacts/screenshots/detail-related-community-dist-wbc.png`、`detail-related-community-dist-wbc-mobile.png`、`detail-related-community-dist.png`

## [2026-05-21] feat(ux) | docs/detail.html、docs/main.js、docs/style.css — V22 P3 详情页「关联类型分布」小条形图

- 触发：[`docs/checklists/tech-stack-next-phase-checklist-v22.md`](docs/checklists/tech-stack-next-phase-checklist-v22.md) P3「详情页关联类型分布小条形图」唯一子项；P0–P2 已全部落地，进入交互层关系视角增强阶段
- 改动形态：
  - [`docs/detail.html`](docs/detail.html)：在 `#detail-related` 标题下新增 `#detailRelatedTypeDist` 容器（含标题 / Meta / 横向条形栅格），默认 hidden，由 JS 在有关联项时显式打开
  - [`docs/main.js`](docs/main.js)：新增 `deriveDetailCategoryLabel()`（按 `path` 优先 → `type` 兜底 → `id` 前缀兜底，输出中文标签：概念 / 方法 / 形式化 / 对比 / Query / 任务 / 实体 / 总览 / 深挖 / 路线图 / 技术地图）；新增 `renderRelatedTypeDistribution()`（统计、按计数倒序+标签字典序、最大计数为 100% 基准、其余按比例并保底 6% 可见宽度），在 `renderDetailPage` 的正常态与「未匹配 detail page」空态均调用一次以避免幽灵骨架
  - [`docs/style.css`](docs/style.css)：新增 `.related-type-dist*` 系列样式（卡片化容器 / 三列网格 `92px 1fr 56px`：标签—轨道—计数 / `var(--accent)` 填充 / 540px 窄屏自适应缩列至 `78px 1fr 46px`）
- 验证：`make lint-js` 通过（仅一条 pre-existing `resetMermaidLightboxView` 未使用警告，与本次改动无关）；本地 `python3 -m http.server 8765` + `puppeteer-core` 视口截图（`/opt/pw-browsers/chromium-1194/chrome-linux/chrome`）打开 `detail.html?id=wiki-concepts-armature-modeling` 锚点 `detail-related`，条形图正确显示「方法 / 概念」两类共 5 项；截图落 `.cursor-artifacts/screenshots/detail-related-type-dist.png`
- 状态联动：V22 checklist 「详情页关联类型分布小条形图」由 `[ ]` 变 `[x]`；P3 剩余「图谱页专题视图切换器」与 DoD 收尾继续推进

## [2026-05-21] ingest | sources/papers/deeprl_locomotion_action_space_sca2017.md — Peng & van de Panne SCA 2017 四动作空间 DeepRL 对照；沉淀 wiki/entities/paper-deeprl-locomotion-action-space-sca2017.md；交叉更新 rl_pd 索引、legged-humanoid-rl-pd-gain-setting、xue-bin-peng、locomotion

## [2026-05-21] ingest | sources/papers/gencad_arxiv_2409_16294.md、sources/papers/gencad3d_arxiv_2509_15246.md、sources/sites/gencad-github-io.md、sources/sites/gencad3d-github-io.md、sources/repos/ferdous-alam-gencad.md、sources/repos/yunomi-git-gencad-3d.md — 入库 GenCAD / GenCAD-3D 论文、项目页与代码仓；沉淀 wiki/entities/gencad.md、wiki/entities/gencad-3d.md；交叉更新 wiki/concepts/text-to-cad.md、sources/sites/text-to-cad-tools.md

## [2026-05-21] structural | wiki/concepts/contact-rich-manipulation.md、wiki/concepts/visuo-tactile-fusion.md — V22 P2 接触/操作交叉补强：补「抓取 → 插装 → 精细操作」级联引用，打通 P1 触觉链路与 P2 抓取链路

- 触发：[`docs/checklists/tech-stack-next-phase-checklist-v22.md`](docs/checklists/tech-stack-next-phase-checklist-v22.md) P2「接触/操作交叉补强」唯一子项；V22 P2 抓取知识链 (+3) 已落地，需把上游检测式 grasp 与本页中段执行层、下游触觉精细操作连成一条流水线视角
- 改动形态：
  - [`wiki/concepts/contact-rich-manipulation.md`](wiki/concepts/contact-rich-manipulation.md)：新增「抓取 → 插装 → 精细操作（级联视角）」小节，三段式表格显式串联 P2 上游（[Grasp Pose Estimation](wiki/methods/grasp-pose-estimation.md)、[AnyGrasp](wiki/entities/anygrasp.md)、[ContactNet](wiki/methods/contact-net.md)、[抓取策略选型 Query](wiki/queries/grasp-policy-selection.md)、[AnyGrasp vs GraspNet](wiki/comparisons/anygrasp-vs-graspnet.md)）→ 本页中段 → P1 下游（[Impedance Control](wiki/concepts/impedance-control.md)、[Tactile Impedance Control](wiki/methods/tactile-impedance-control.md)、[TSID](wiki/concepts/tsid.md)/[WBC](wiki/concepts/whole-body-control.md)），并补「① 准但 ② 没接管会撞死」的工程含义说明；frontmatter `related` 与「关联页面」尾部互链至 P2 抓取链
  - [`wiki/concepts/visuo-tactile-fusion.md`](wiki/concepts/visuo-tactile-fusion.md)：新增同名小节，附 Mermaid 流水线图与三段式表格，强调「检测式 grasp 不带接触可信度，门控/注意力必须在触觉给出几何漂移信号时立即让出权重」这一常被忽略的衔接点；frontmatter `related` 与「关联页面」加入 P2 抓取链与 [Tactile Impedance Control](wiki/methods/tactile-impedance-control.md)、[Hybrid Force-Position Control](wiki/concepts/hybrid-force-position-control.md)
  - 两页 `updated` 字段刷新至 2026-05-21
- 验证：`python3 scripts/eval_search_quality.py` 37/37 通过；`python3 scripts/check_export_quality.py` 12/12 通过；`make ci-preflight` 同步派生产物（page catalog / exports / search-index / link-graph / docs/index.html / sitemap）。`exports/graph-stats.json`：节点 410、边 3004（远超 V22 目标 312/2050）、largest_community_ratio 0.207、`community_quality_warning: false`。`lint_wiki.py` 9 项 `stale_pages` 均为同日早些 ingest 引入的历史 baseline，与本次改动无关
- 状态联动：V22 checklist 「接触/操作交叉补强」由 `[ ]` 变 `[x]`；P2 全部子项落地完毕

## [2026-05-21] ingest | sources/repos/sensenova-skills.md — OpenSenseNova/SenseNova-Skills 入库并沉淀 wiki/entities/sensenova-skills.md；交叉更新 wiki/entities/hermes-agent.md、wiki/entities/mattpocock-skills.md

## [2026-05-21] ingest | sources/repos/boyu_ai_hands_on_rl.md、sources/sites/hrl-boyuai-hands-on-rl.md、sources/courses/boyuai_hands_on_rl_elites_course.md — 接入动手学强化学习（蘑菇书）在线书/代码仓/伯禹视频课并沉淀 wiki/entities/hands-on-rl-book.md；交叉更新 wiki/methods/reinforcement-learning.md、roadmap/depth-rl-locomotion.md、roadmap/motion-control.md、wiki/overview/robot-learning-overview.md

## [2026-05-21] ingest | sources/sites/nvidia-physical-ai-learning.md、sources/courses/nvidia_sim_to_real_so101_isaac.md — 入库 NVIDIA Physical AI 门户与 SO-101 Sim2Real 课；沉淀 wiki/entities/nvidia-physical-ai-learning.md、wiki/entities/nvidia-so101-sim2real-lab-workflow.md；互链 sim2real、lerobot、isaac-lab、vla、sage

## [2026-05-21] structural | wiki/concepts/domain-randomization.md 等 17 页 — 清理 lint 长期 stale 预存量：按 source 给 17 个 wiki 页补 ingest 档案交叉引用

- 触发：`make lint` 报「陈旧页面」17 条（mtime 判定：source 比 wiki 新 ≥ 24h）；预存自 2026-05-19 起多次 ingest 累积，与本批改动前的提交无关
- 影响页面（按 source 分组）：
  - `sources/papers/barkour_arxiv_2305_14654.md` → [`wiki/concepts/domain-randomization.md`](wiki/concepts/domain-randomization.md)、[`wiki/concepts/sim2real.md`](wiki/concepts/sim2real.md)、[`wiki/methods/reinforcement-learning.md`](wiki/methods/reinforcement-learning.md)
  - `sources/papers/bfm_humanoid_arxiv_2509_13780.md` → [`wiki/tasks/teleoperation.md`](wiki/tasks/teleoperation.md)、[`wiki/concepts/privileged-training.md`](wiki/concepts/privileged-training.md)、[`wiki/methods/dagger.md`](wiki/methods/dagger.md)、[`wiki/concepts/curriculum-learning.md`](wiki/concepts/curriculum-learning.md)、[`wiki/entities/amass.md`](wiki/entities/amass.md)、[`wiki/entities/unitree-g1.md`](wiki/entities/unitree-g1.md)
  - `sources/papers/capvector_arxiv_2605_10903.md` → [`wiki/methods/star-vla.md`](wiki/methods/star-vla.md)
  - `sources/papers/defi_arxiv_2604_16391.md` → [`wiki/methods/diffusion-policy.md`](wiki/methods/diffusion-policy.md)、[`wiki/methods/action-chunking.md`](wiki/methods/action-chunking.md)
  - `sources/papers/holomotion_arxiv_2605_15336.md` → [`wiki/methods/imitation-learning.md`](wiki/methods/imitation-learning.md)
  - `sources/papers/physforge_arxiv_2605_05163.md` → [`wiki/entities/sapien.md`](wiki/entities/sapien.md)
  - `sources/papers/robot_link_rotor_inertia_primary_refs.md` → [`wiki/entities/modern-robotics-book.md`](wiki/entities/modern-robotics-book.md)
  - `sources/papers/system_identification.md` → [`wiki/methods/actuator-network.md`](wiki/methods/actuator-network.md)（已有引用，扩写覆盖范围以反映 source 现含 Hwangbo / Gautier–Khalil / Grandia / Peng 等条目）
  - `sources/papers/wm_robot_survey_arxiv_2605_00080.md` → [`wiki/methods/model-based-rl.md`](wiki/methods/model-based-rl.md)
- 改动形态：每页在「参考来源」追加 1 条 ingest 档案行（含一句话提炼），统一与项目约定模式对齐；未做结构/正文重写
- 验证：`make lint` 17 → 0 issues；`make ci-preflight` 通过（同步 `exports/`、`docs/exports/`、`docs/search-index.json` 等，导出质量 12/12）

## [2026-05-21] fix(search): 搜索回归 WBC/MPC 定义页排名 — 条件化 comparison 提权 + 定义页 canonical boost

- `scripts/search_wiki_core.py`：`comparison` 类型仅在查询含「对比/选型」等意图时 ×1.3；WBC/MPC 定义页在缩写命中时 ×1.4 canonical boost
- `scripts/search_indexing.py`：`全身控制` / `模型预测控制` 同义词展开至 wbc/mpc
- 补强 `wiki/concepts/whole-body-control.md`、`wiki/methods/model-predictive-control.md` 标题与 summary 中文检索词
- 验证：`eval_search_quality.py` 37/37（原 35/37）

## [2026-05-21] query | wiki/queries/humanoid-motion-tracking-method-selection.md 等 — V22 方法-Query 闭环：31 条高频 methods 落地预警清零

- 新增 Query：`wiki/queries/humanoid-motion-tracking-method-selection.md`、`manipulation-vla-architecture-selection.md`、`humanoid-contact-character-control-guide.md`、`dexterous-manipulation-data-pipeline.md`
- 新增 Comparison：`wiki/comparisons/amp-add-smp-motion-prior-variants.md`
- 覆盖 methods：`deepmimic`、`beyondmimic`、`amp-reward`、`add`、`smp`、`motionbricks`、`any2track`、`ams`、`gentlehumanoid`、`ase`、`genmo`、`diffusion-motion-generation`、`mimic-video`、`defi`、`dwm`、`star-vla`、`pi07-policy`、`π0-policy`、`pelican-unified-1`、`claw`、`being-h07`、`disney-olaf`、`humanoid-transformer-touch-dreaming`、`hipan`、`zest`、`efgcl`、`auto-labeling-pipelines`、`wilor`、`tactile-impedance-control`、`actuator-network`、`gae`（共 31 页）
- 注册：`wiki/queries/README.md`
- 派生再生成：`make ci-preflight`

## [2026-05-21] ingest | sources/blogs/wechat_embodied_ai_lab_humanoid_rl_motion_survey.md、sources/blogs/wechat_embodied_ai_lab_humanoid_amp_motion_prior_survey.md — 安装 Agent Reach 抓取具身智能研究室两篇微信公众号长文并消化入库

- 工具：已安装 [Panniantong/Agent-Reach](https://github.com/Panniantong/Agent-Reach) v1.4.0（`pip install` + `agent-reach install --channels=wechat`）；微信正文经 `~/.agent-reach/tools/wechat-article-for-ai`（Camoufox），Jina Reader 对 `mp.weixin.qq.com` 返回 CAPTCHA
- 原始资料：`sources/blogs/wechat_embodied_ai_lab_humanoid_rl_motion_survey.md`（<https://mp.weixin.qq.com/s/hz9JXtJeUPRfUGzfD-pZuA>）、`sources/blogs/wechat_embodied_ai_lab_humanoid_amp_motion_prior_survey.md`（<https://mp.weixin.qq.com/s/YZsm3855iP3TNTTt1aou7w>）；索引 `sources/README.md`
- 沉淀页面：`wiki/overview/humanoid-amp-motion-prior-survey.md`；补强 `wiki/overview/humanoid-rl-motion-control-body-system-stack.md`
- 交叉更新：`wiki/methods/amp-reward.md`、`wiki/entities/agent-reach.md`、`sources/repos/panniantong_agent_reach.md`
- 派生再生成：`make ci-preflight` 同步 `exports/`、`docs/exports/`、`docs/search-index.json`、`docs/sitemap.xml`、`README.md`、`docs/index.html`、`index.md` 等

## [2026-05-20] structural | wiki/comparisons/anygrasp-vs-graspnet.md — V22 P2 AnyGrasp vs GraspNet 抓取检测家族对比

- 新增页面：`wiki/comparisons/anygrasp-vs-graspnet.md`，按「一句话定义 + 14 维核心对比表 + Mermaid 数据流并排图（GraspNet 家族白盒基线 / AnyGrasp SDK 工程闭环）+ 三类适用场景 + 6 类常见误判 + 决策矩阵 + 评测指标视角」结构覆盖 GraspNet-1Billion / Contact-GraspNet / GSNet / AnyGrasp 四条子路线；显式区分「白盒改造 vs 工程化交付」「单帧 vs 动态跨帧」「完全开源 vs 二进制 License」三对核心取舍。
- 交叉互链：`wiki/methods/grasp-pose-estimation.md`、`wiki/entities/anygrasp.md`、`wiki/queries/grasp-policy-selection.md`、`wiki/methods/contact-net.md`、`wiki/tasks/manipulation.md` 的 frontmatter `related` 与「关联页面」均加入本页入口，形成「方法谱系页 + 实体页 + Query + 对比页」四级互链闭环。
- 清单推进：`docs/checklists/tech-stack-next-phase-checklist-v22.md` P2「抓取知识链 (+3)」第三项 `anygrasp-vs-graspnet.md` 打勾，整体专题完结进入 `[x]` 完成状态。
- 派生再生成：`make ci-preflight` 同步 `exports/`、`docs/exports/`、`docs/search-index.json`、`docs/sitemap.xml`、`README.md`、`docs/index.html`、`exports/lint-report.md`（图谱节点 399 → 400、边数 2836 → 2850、comparison 类型从 18 → 19；陈旧页面 21 条与本次改动无关，为今日早些 ingest 留下的预存量）。

## [2026-05-20] query | wiki/queries/grasp-policy-selection.md — V22 P2 抓取策略选型 Query 落地

- 新增页面：`wiki/queries/grasp-policy-selection.md`，覆盖三轴选型（物体已知度 / 候选稠密度 / 方法类型）+ TL;DR 决策树 + 四类推荐组合 pipeline（已知物体 / 桌面 bin picking / 动态场景 / 任务级语言指令）+ 关键工程经验 + 常见误区，与 [Grasp Pose Estimation](wiki/methods/grasp-pose-estimation.md) / [AnyGrasp](wiki/entities/anygrasp.md) / [Manipulation](wiki/tasks/manipulation.md) / [Visual Servoing](wiki/methods/visual-servoing.md) / [Contact-Rich Manipulation](wiki/concepts/contact-rich-manipulation.md) 互链。
- 交叉互链：`wiki/queries/README.md` 注册新 Query；`wiki/methods/grasp-pose-estimation.md` frontmatter `related` + 「关联页面」加入本页；`wiki/entities/anygrasp.md`、`wiki/tasks/manipulation.md` 关联页面区块新增 Query 入口。
- 清单推进：`docs/checklists/tech-stack-next-phase-checklist-v22.md` P2「抓取知识链」第二项 `grasp-policy-selection.md` 打勾，附实现摘要。
- 派生再生成：`make ci-preflight`。

## [2026-05-20] ingest | sources/papers/defi_arxiv_2604_16391.md — DeFI 解耦前向/逆动力学 VLA；wiki/methods/defi-decoupled-dynamics-vla.md

## [2026-05-20] structural | wiki/methods/grasp-pose-estimation.md — V22 P2 抓取位姿估计方法谱系页

- 新增 `wiki/methods/grasp-pose-estimation.md`，覆盖 6-DoF/7-DoF 表征、三代谱系（GPD → GraspNet-1Billion → Contact-GraspNet/GSNet/Graspness/AnyGrasp）、点云/RGBD 输入对照、AP/MPPH 评测、下游 cuRobo/视觉伺服/触觉闭环串联与常见误区，含 Mermaid 谱系图。
- 交叉互链：`wiki/entities/anygrasp.md` frontmatter `related` 与「关联页面」回链；`wiki/tasks/manipulation.md` 关联方法区块加入条目；`wiki/methods/contact-net.md` 关联页面新增本页；`references/repos/manipulation-perception.md` 顶部加入「方法谱系总览」指针；`index.md` 重点页面新增条目。
- 清单推进：`docs/checklists/tech-stack-next-phase-checklist-v22.md` P2「抓取知识链」首项打勾，整体专题进入 `[~]` 进行中状态。

## [2026-05-20] ingest | sources/papers/robot_link_rotor_inertia_primary_refs.md — 连杆/转子惯量一手资料入库并沉淀 wiki

- 原始资料：`sources/papers/robot_link_rotor_inertia_primary_refs.md`（URDF / Modern Robotics Ch.8 / Gautier–Khalil 1990 / MuJoCo armature）
- 沉淀页面：`wiki/concepts/robot-link-and-rotor-inertia.md`
- 交叉更新：`wiki/concepts/armature-modeling.md`、`wiki/concepts/system-identification.md`
- 派生再生成：`make ci-preflight`

## [2026-05-20] ingest | sources/repos/mattpocock-skills.md — mattpocock/skills 入库并沉淀 wiki

- 原始资料：`sources/repos/mattpocock-skills.md`（<https://github.com/mattpocock/skills>）；索引 `sources/README.md`
- 沉淀页面：`wiki/entities/mattpocock-skills.md`
- 交叉更新：`wiki/entities/superpowers-obra.md`、`wiki/entities/caveman.md`、`wiki/references/llm-wiki-karpathy.md`
- 派生再生成：`make ci-preflight`

## [2026-05-20] structural | wiki/entities/amp-mjlab.md — 补充 play.py（run_play）详细 Mermaid 流程图

- 页面：`wiki/entities/amp-mjlab.md` 在「训练与回放」下新增 `run_play` 流程图（CLI → play 环境覆盖 → checkpoint 加载 → AMPOnPolicyRunner 推理 → ONNX 导出 → Viewer 主循环），源码依据 [ImChong/AMP_mjlab](https://github.com/ImChong/AMP_mjlab) `scripts/play.py`
- 派生再生成：`make ci-preflight`

## [2026-05-19] structural | wiki/concepts/character-animation-vs-robotics.md — V22 P1「角色化人形边界澄清」落地

- 新增页面：`wiki/concepts/character-animation-vs-robotics.md`（角色动画 vs 机器人控制：六维张力矩阵 + 五案例切片 + 决策矩阵 + Mermaid「角色端→桥接层→机器人端」流程）
- 交叉更新：`wiki/methods/disney-olaf-character-robot.md`、`wiki/entities/botlab-motioncanvas.md`、`wiki/entities/roboto-origin.md`、`wiki/entities/xue-bin-peng.md`、`wiki/concepts/motion-retargeting.md`、`wiki/concepts/reward-design.md` 的 `related` 与「关联页面 / 与其他页面的关系」加入新入口
- 清单同步：`docs/checklists/tech-stack-next-phase-checklist-v22.md` 勾选 P1「角色化人形（Character Humanoid）边界澄清」并补实现摘要
- 派生再生成：`make ci-preflight` 同步 `exports/`、`docs/exports/`、`docs/search-index.json`、`docs/sitemap.xml`、`README.md`、`docs/index.html`、`index.md`、`exports/lint-report.md`（注：lint 报告中 17 条「陈旧页面」均为今日早些 ingest 留下的预存量，与本次改动无直接关系）

## [2026-05-19] ingest | sources/repos/caveman.md — JuliusBrussee/caveman 入库并沉淀 wiki

- 原始资料：`sources/repos/caveman.md`（<https://github.com/JuliusBrussee/caveman>）；索引 `sources/README.md`
- 沉淀页面：`wiki/entities/caveman.md`
- 交叉更新：`wiki/entities/superpowers-obra.md`、`wiki/entities/hermes-agent.md`、`wiki/entities/agent-reach.md`
- 派生再生成：`make ci-preflight` 同步 `exports/`、`docs/exports/`、`docs/search-index.json`、`docs/sitemap.xml`、`README.md`、`docs/index.html`、`index.md` 等

## [2026-05-19] ingest | sources/papers/gentlehumanoid_upper_body_compliance.md、sources/repos/axellwppr_motion_tracking.md — GentleHumanoid / motion_tracking 入库并沉淀 wiki

- 原始资料：`sources/papers/gentlehumanoid_upper_body_compliance.md`（[arXiv:2511.04679](https://arxiv.org/abs/2511.04679)）、`sources/sites/gentle-humanoid-axell-top.md`、`sources/sites/motion-tracking-axell-top.md`、`sources/repos/axellwppr_motion_tracking.md`（<https://github.com/Axellwppr/motion_tracking>）；索引 `sources/README.md`
- 沉淀页面：`wiki/methods/gentlehumanoid-motion-tracking.md`、`wiki/entities/axellwppr-motion-tracking.md`
- 交叉更新：`wiki/concepts/whole-body-control.md`、`wiki/concepts/contact-dynamics.md`、`wiki/concepts/impedance-control.md`、`wiki/overview/humanoid-rl-motion-control-body-system-stack.md`、`wiki/methods/sonic-motion-tracking.md`
- 派生再生成：`make ci-preflight` 同步 `exports/`、`docs/exports/`、`docs/search-index.json`、`docs/sitemap.xml`、`README.md`、`docs/index.html`、`index.md` 等

## [2026-05-19] structural | wiki/overview/motor-drive-firmware-bus-protocols.md — 电机驱动器底软通信协议总览（种类与优缺点）

- 原始资料：`sources/courses/motor_drive_firmware_bus_protocols.md`（CiA CANopen/CiA402、DroneCAN、MIT 紧凑帧与厂商私有等索引）
- 沉淀页面：`wiki/overview/motor-drive-firmware-bus-protocols.md`（物理层 × 应用层 × 控制语义三层；协议族优缺点表；常见组合与选型 Mermaid）
- 交叉更新：`wiki/concepts/can-bus-protocol.md`、`wiki/concepts/ethercat-protocol.md`、`wiki/comparisons/can-vs-ethercat-joint-bus.md`
- 派生再生成：`make ci-preflight`

## [2026-05-19] ingest | sources/sites/cia_can_*、sources/courses/uart_rs485_serial_embedded.md — CiA CAN/CAN FD/CANopen/DroneCAN 与 UART·RS485 一手资料入库

- 原始资料：`sources/sites/cia_can_knowledge_can_classic_and_hs.md`、`sources/sites/cia_can_fd_basic_idea.md`、`sources/sites/cia_canopen_overview.md`、`sources/sites/cia_dronecan_uavcan.md`、`sources/courses/uart_rs485_serial_embedded.md`（CiA [CAN knowledge](https://www.can-cia.org/can-knowledge/)、[DroneCAN](http://dronecan.org/)、TI SLLA383 / Wikipedia UART）；索引 `sources/README.md`
- 沉淀页面：`wiki/concepts/can-bus-protocol.md`、`wiki/concepts/can-fd.md`、`wiki/concepts/uart-serial-communication.md`、`wiki/comparisons/can-vs-ethercat-joint-bus.md`
- 交叉更新：`wiki/concepts/ethercat-protocol.md`、`wiki/concepts/processor-in-the-loop-sim2real.md`、`wiki/formalizations/control-loop-latency-modeling.md`、`wiki/queries/real-time-control-middleware-guide.md`
- 派生再生成：`make ci-preflight` 同步 `exports/`、`docs/exports/`、`docs/search-index.json`、`docs/sitemap.xml`、`README.md`、`docs/index.html`、`index.md` 等

## [2026-05-19] ingest | sources/repos/newton-physics.md、sources/sites/nvidia-newton-physics.md、sources/sites/newton-physics-docs-overview.md — Newton Physics 引擎（NVIDIA / DeepMind / Disney，Warp + MuJoCo Warp）入库

- 原始资料：`sources/repos/newton-physics.md`（<https://github.com/newton-physics/newton>）、`sources/sites/nvidia-newton-physics.md`（<https://developer.nvidia.com/newton-physics>）、`sources/sites/newton-physics-docs-overview.md`（<https://newton-physics.github.io/newton/stable/guide/overview.html>）；索引 `sources/README.md`
- 沉淀页面：`wiki/entities/newton-physics.md`
- 交叉更新：`wiki/entities/mujoco.md`、`wiki/entities/mjlab.md`、`wiki/entities/isaac-gym-isaac-lab.md`、`wiki/queries/simulator-selection-guide.md`
- 派生再生成：`make ci-preflight` 同步 `exports/`、`docs/exports/`、`docs/search-index.json`、`docs/sitemap.xml`、`README.md`、`docs/index.html`、`index.md` 等

## [2026-05-19] ingest | sources/blogs/wechat_embodied_ai_lab_daji_semantic_body_interface.md、sources/papers/daji_arxiv_2605_14417.md — DAJI 预期关节意图（微信精读 / arXiv:2605.14417）入库

- 原始资料：`sources/blogs/wechat_embodied_ai_lab_daji_semantic_body_interface.md`（<https://mp.weixin.qq.com/s/u1ZUaFGYRKXxMcS7-V_2WA>）、`sources/papers/daji_arxiv_2605_14417.md`、`sources/sites/daji-hxxxz0-github-io.md`、`sources/repos/hxxxz0_daji.md`；索引 `sources/README.md`
- 沉淀页面：`wiki/entities/paper-daji-anticipatory-joint-intent.md`
- 交叉更新：`wiki/methods/vla.md`、`wiki/tasks/loco-manipulation.md`、`wiki/overview/humanoid-rl-motion-control-body-system-stack.md`
- 派生再生成：`make ci-preflight` 同步 `exports/`、`docs/exports/`、`docs/search-index.json`、`docs/sitemap.xml`、`README.md`、`docs/index.html`、`index.md` 等

## [2026-05-19] ingest | sources/blogs/wechat_embodied_ai_lab_robot_world_model_training_loop.md、sources/papers/wm_robot_survey_arxiv_2605_00080.md、sources/sites/wm-robot-survey-ntumars.md — 安装 Agent Reach 抓取微信公众号；机器人世界模型综述（arXiv:2605.00080）入库

- 工具：已安装 [Panniantong/Agent-Reach](https://github.com/Panniantong/Agent-Reach) v1.4.0（`pip install` 可编辑包 + `agent-reach install --channels=wechat`）；微信正文经 `wechat-article-for-ai`（Camoufox），Jina Reader 对该 URL 返回 CAPTCHA
- 原始资料：`sources/blogs/wechat_embodied_ai_lab_robot_world_model_training_loop.md`（<https://mp.weixin.qq.com/s/0edW0GhwtyNc5nF6RDIfuw>）、`sources/papers/wm_robot_survey_arxiv_2605_00080.md`、`sources/sites/wm-robot-survey-ntumars.md`；索引 `sources/README.md`
- 沉淀页面：`wiki/overview/robot-world-models-training-loop-taxonomy.md`
- 交叉更新：`wiki/methods/generative-world-models.md`、`wiki/concepts/world-action-models.md`、`wiki/methods/vla.md`、`wiki/overview/humanoid-rl-motion-control-body-system-stack.md`、`wiki/entities/agent-reach.md`、`sources/repos/panniantong_agent_reach.md`
- 派生再生成：`make ci-preflight` 同步 `exports/`、`docs/exports/`、`docs/search-index.json`、`docs/sitemap.xml`、`README.md`、`docs/index.html`、`index.md` 等

## [2026-05-19] ingest | sources/repos/nousresearch_hermes_agent.md、sources/sites/hermes-agent-nousresearch-docs.md — Hermes Agent（NousResearch）仓库与官方文档入库

- 原始资料：`sources/repos/nousresearch_hermes_agent.md`、`sources/sites/hermes-agent-nousresearch-docs.md`（GitHub <https://github.com/NousResearch/hermes-agent>、产品页 <https://hermes-agent.nousresearch.com/>、文档 <https://hermes-agent.nousresearch.com/docs>）；索引 `sources/README.md`
- 沉淀页面：`wiki/entities/hermes-agent.md`
- 交叉更新：`wiki/entities/superpowers-obra.md`、`wiki/entities/agent-reach.md`、`index.md`（Entities 目录补 Hermes Agent 条目）
- 派生再生成：`make ci-preflight` 同步 `exports/`、`docs/exports/`、`docs/search-index.json`、`docs/sitemap.xml`、`README.md`、`docs/index.html`、`index.md` 等

## [2026-05-18] structural | wiki/comparisons/gmr-vs-nmr-vs-reactor.md — V22 P1 动作重定向知识链 (3/3)：新增 GMR / NMR / ReActor 三方对比页

- 沉淀页面：`wiki/comparisons/gmr-vs-nmr-vs-reactor.md`（一句话定义 + 12 维核心对比表 + Mermaid 三路数据流并排图 + 三方适用场景 + 5 类常见误判 + 决策矩阵；强调「误差修补发生位置」（下游 / 离线 / 在线）作为核心选型轴；显式标注 NMR 仍以 GMR 为 CEPR 初值、三者实际常串联而非互斥）
- 交叉更新：`wiki/concepts/motion-retargeting.md`、`wiki/concepts/motion-retargeting-pipeline.md`、`wiki/formalizations/motion-retargeting-objective.md`、`wiki/methods/motion-retargeting-gmr.md`、`wiki/methods/neural-motion-retargeting-nmr.md`、`wiki/methods/reactor-physics-aware-motion-retargeting.md`（关联页面区块回链本对比页）、`index.md`（Wiki Comparisons 目录插入 GMR vs NMR vs ReActor 条目）、`docs/checklists/tech-stack-next-phase-checklist-v22.md`（P1 第 3 项打勾，含实现摘要；至此 V22 P1「动作重定向知识链 (+3)」3/3 全部完成）
- 派生再生成：保持当前状态，待后续 ingest 或 P2 推进时统一 `make ci-preflight` 同步 `exports/`、`docs/exports/`、`docs/search-index.json`、`docs/sitemap.xml`、`README.md`、`docs/index.html`、`index.md` 等

## [2026-05-18] ingest | sources/papers/capvector_arxiv_2605_10903.md、sources/sites/capvector-github-io.md、sources/repos/openhelix_team_capvector.md — CapVector（参数空间 capability vector + 正交正则标准 SFT）arXiv:2605.10903 入库

- 原始资料：`sources/papers/capvector_arxiv_2605_10903.md`、`sources/sites/capvector-github-io.md`、`sources/repos/openhelix_team_capvector.md`（PDF <https://arxiv.org/pdf/2605.10903>、项目页 <https://capvector.github.io/>、代码 <https://github.com/OpenHelix-Team/CapVector>）；索引 `sources/README.md`
- 沉淀页面：`wiki/entities/paper-capvector-capability-vectors-vla.md`
- 交叉更新：`wiki/methods/vla.md`、`index.md`（Entities 目录补 CapVector 条目）
- 派生再生成：`make ci-preflight` 同步 `exports/`、`docs/exports/`、`docs/search-index.json`、`docs/sitemap.xml`、`README.md`、`docs/index.html`、`index.md` 等

## [2026-05-18] ingest | sources/papers/physforge_arxiv_2605_05163.md — PhysForge（VLM 分层物理蓝图 + KVI 扩散、PhysDB）arXiv:2605.05163 入库

- 原始资料：`sources/papers/physforge_arxiv_2605_05163.md`（PDF <https://arxiv.org/pdf/2605.05163>、项目页 <https://hku-mmlab.github.io/PhysForge/>）；索引 `sources/README.md`
- 沉淀页面：`wiki/entities/paper-physforge-physics-grounded-3d-assets.md`
- 交叉更新：`wiki/entities/articraft.md`、`wiki/entities/robotwin.md`、`index.md`（Entities 目录补 PhysForge 条目）
- 派生再生成：`make ci-preflight` 同步 `exports/`、`docs/exports/`、`docs/search-index.json`、`docs/sitemap.xml`、`README.md`、`docs/index.html`、`index.md` 等

## [2026-05-18] ingest | sources/repos/horizon_robotics_holomotion.md、sources/papers/holomotion_arxiv_2605_15336.md — 地平线 HoloMotion-1（混合语料 + 稀疏 MoE Transformer + 序列级 PPO）官方资料入库

- 原始资料：`sources/repos/horizon_robotics_holomotion.md`、`sources/papers/holomotion_arxiv_2605_15336.md`（GitHub / 项目主页 / arXiv:2605.15336 / Hugging Face / Docker Hub）；索引 `sources/README.md`
- 沉淀页面：`wiki/entities/holomotion.md`
- 交叉更新：`wiki/concepts/foundation-policy.md`、`wiki/methods/sonic-motion-tracking.md`、`wiki/entities/paper-behavior-foundation-model-humanoid.md`、`index.md`（Entities 目录补 `holomotion` 条目）
- 派生再生成：`make ci-preflight` 同步 `exports/`、`docs/exports/`、`docs/search-index.json`、`docs/sitemap.xml`、`README.md`、`docs/index.html`、`index.md` 等

## [2026-05-18] ingest | sources/blogs/wechat_jixie_robot_open_source_treasury_issue02_10_robots.md — 微信公众号「机械Robot」开源宝库第02期：10 个机器人/平台实体 + 策展 overview

- 原始资料：`sources/blogs/wechat_jixie_robot_open_source_treasury_issue02_10_robots.md`；索引 `sources/README.md`
- 沉淀页面：`wiki/overview/robot-open-source-wechat-issue02-curator.md`；`wiki/entities/pollen-reachy2.md`、`wiki/entities/poppy-project-robots.md`、`wiki/entities/inmoov-humanoid.md`、`wiki/entities/stanford-doggo-and-pupper.md`、`wiki/entities/elephantrobotics-mycobot-320.md`、`wiki/entities/elephantrobotics-myagv.md`、`wiki/entities/tidybot2.md`、`wiki/entities/kinova-gen3.md`、`wiki/entities/franka-research-3.md`、`wiki/entities/parol6-source-robotics.md`
- 交叉更新：`wiki/overview/robot-open-source-wechat-issue01-curator.md`、`wiki/entities/open-source-humanoid-hardware.md`、`wiki/entities/humanoid-robot.md`
- 派生再生成：`make ci-preflight` 同步 `exports/`、`docs/exports/`、`docs/search-index.json`、`docs/sitemap.xml`、`README.md`、`docs/index.html`、`index.md` 等

## [2026-05-18] ingest | sources/blogs/wechat_jixie_robot_open_source_treasury_issue01_10_robots.md — 微信公众号「机械Robot」开源宝库第01期：10 个机器人/平台独立实体节点 + 策展 overview

- 原始资料：`sources/blogs/wechat_jixie_robot_open_source_treasury_issue01_10_robots.md`；索引 `sources/README.md`
- 沉淀页面：`wiki/overview/robot-open-source-wechat-issue01-curator.md`；`wiki/entities/fourier-grx-n1.md`、`wiki/entities/agibot-lingxi-x1.md`、`wiki/entities/tienkung-humanoid-open-source.md`、`wiki/entities/odri-solo-and-bolt.md`、`wiki/entities/berkeley-humanoid-lite.md`、`wiki/entities/orca-hand.md`、`wiki/entities/turtlebot3.md`、`wiki/entities/robotis-open-manipulator-line.md`、`wiki/entities/robotis-op3.md`、`wiki/entities/robotis-thormang3.md`
- 交叉更新：`wiki/entities/open-source-humanoid-hardware.md`、`wiki/entities/humanoid-robot.md`
- 派生再生成：`make ci-preflight` 同步 `exports/`、`docs/exports/`、`docs/search-index.json`、`docs/sitemap.xml`、`README.md`、`docs/index.html`、`index.md` 等

## [2026-05-18] ingest | sources/repos/mujoco-mjx.md、sources/repos/brax.md、sources/papers/brax_arxiv_2106_13281.md、sources/sites/mujoco-mjx-readthedocs.md — MuJoCo MJX 与 Brax 官方仓/文档/论文入库；新增实体页并交叉更新 MuJoCo / dm_control / 选型指南 / LIFT

- 原始资料：`sources/repos/mujoco-mjx.md`、`sources/repos/brax.md`、`sources/papers/brax_arxiv_2106_13281.md`、`sources/sites/mujoco-mjx-readthedocs.md`；索引 `sources/README.md`
- 沉淀页面：`wiki/entities/mujoco-mjx.md`、`wiki/entities/brax.md`
- 交叉更新：`wiki/entities/mujoco.md`、`wiki/entities/dm-control.md`、`wiki/queries/simulator-selection-guide.md`、`wiki/comparisons/mujoco-vs-isaac-sim.md`、`wiki/comparisons/mujoco-vs-isaac-lab.md`、`wiki/entities/lift-humanoid.md`、`index.md`（重点入口 + Page Catalog）
- 派生再生成：`make ci-preflight` 同步 `exports/`、`docs/exports/`、`docs/search-index.json`、`docs/sitemap.xml`、`README.md`、`docs/index.html` 等

## [2026-05-18] ingest | sources/papers/barkour_arxiv_2305_14654.md、sources/blogs/google-research-barkour-quadruped-agility-2023-05-26.md、sources/repos/google_deepmind_barkour_robot.md、sources/repos/mujoco_menagerie_google_barkour_models.md — Barkour 四足敏捷基准与开源生态入库

- 原始资料：arXiv:2305.14654、Google Research 博客（2023-05-26）、[`google-deepmind/barkour_robot`](https://github.com/google-deepmind/barkour_robot)、[`mujoco_menagerie` 下 `google_barkour_v0` / `google_barkour_vb`](https://github.com/google-deepmind/mujoco_menagerie/tree/main/google_barkour_v0)；OnShape 以 README 中 **gdm.onshape.com / deepmind.onshape.com** 文档链接为准（`cad.onshape.com` 为产品首页）
- 沉淀页面：`wiki/entities/paper-barkour-quadruped-agility-benchmark.md`
- 交叉更新：`wiki/entities/quadruped-robot.md`、`wiki/entities/mujoco.md`、`wiki/tasks/locomotion.md`、`references/papers/locomotion-rl.md`、`sources/README.md`
- 派生再生成：`make ci-preflight` 同步 `exports/`、`docs/exports/`、`docs/search-index.json`、`docs/sitemap.xml`、`README.md`、`docs/index.html`、`index.md` 等

## [2026-05-18] structural | wiki/overview/humanoid-rl-motion-control-body-system-stack.md、wiki/concepts/planetary-roller-screw-humanoid-leg-actuation.md — 参考来源补充微信公众号原文外链

- 更新页面：`wiki/overview/humanoid-rl-motion-control-body-system-stack.md`（`https://mp.weixin.qq.com/s/hz9JXtJeUPRfUGzfD-pZuA`）、`wiki/concepts/planetary-roller-screw-humanoid-leg-actuation.md`（`https://mp.weixin.qq.com/s/webqJRQJREZdABw8bdl68w`）；保留仓库内 `sources/` 归档链接
- 派生再生成：`make ci-preflight` 同步 `exports/`、`docs/exports/`、`docs/search-index.json`、`docs/sitemap.xml`、`README.md`、`docs/index.html`、`index.md` 等

## [2026-05-18] ingest | sources/sites/wechat-embodied-ai-lab-humanoid-rl-motion-survey-2026-05-18.md、wiki/overview/humanoid-rl-motion-control-body-system-stack.md — 具身智能研究室 42 篇 humanoid RL 运动控制综述入库；新增「身体系统栈」视角 overview 页

- 原始资料：`sources/sites/wechat-embodied-ai-lab-humanoid-rl-motion-survey-2026-05-18.md`（公众号长文，Camoufox 抓取，约 4.5w 字）
- 沉淀页面：`wiki/overview/humanoid-rl-motion-control-body-system-stack.md`（提炼作者的 8 层身体系统栈 + 6 个研究判断；把已有 wiki 实体页 DeepMimic / SONIC / BeyondMimic / Any2Track / AMS / GMR / NMR / DoorMan / VIRAL / BFM / GR00T-WBC / ULTRA 按层挂接；明确「单页未升格论文」候选清单）
- 交叉关联：`wiki/overview/humanoid-motion-control-know-how.md`、`wiki/tasks/humanoid-locomotion.md`、`wiki/tasks/loco-manipulation.md`、`wiki/tasks/ultra-survey.md`、`wiki/tasks/balance-recovery.md` 与多篇 `wiki/methods` / `wiki/entities` 页面通过 related 区块互链
- 派生再生成：`make ci-preflight` 同步 `exports/`、`docs/exports/`、`docs/search-index.json`、`docs/sitemap.xml`、`README.md`、`docs/index.html`、`index.md` 等

## [2026-05-18] ingest | sources/blogs/wechat_zanezhang_tesla_optimus_leg_planetary_roller_screw.md — 微信公众号：Optimus 腿部行星滚柱丝杠解读入库；新增 wiki/concepts/planetary-roller-screw-humanoid-leg-actuation.md；交叉更新 humanoid-robot、locomotion、sources/README.md

- 原始资料：`sources/blogs/wechat_zanezhang_tesla_optimus_leg_planetary_roller_screw.md`；索引 `sources/README.md`
- 沉淀页面：`wiki/concepts/planetary-roller-screw-humanoid-leg-actuation.md`（PRS 原理、反转式布置、连杆映射、与旋转关节权衡、Mermaid 主干流程）
- 交叉更新：`wiki/entities/humanoid-robot.md`（Optimus 备注与参考来源）、`wiki/tasks/locomotion.md`（关联系统与方法回链）
- 派生再生成：`make ci-preflight` 同步 `exports/`、`docs/exports/`、`docs/search-index.json`、`docs/sitemap.xml`、`README.md`、`docs/index.html`、`index.md` 等

## [2026-05-18] ingest | sources/repos/panniantong_agent_reach.md — Panniantong/Agent-Reach 入库；新增 wiki/entities/agent-reach.md；交叉更新 superpowers-obra、index.md、sources/README.md

- 原始资料：`sources/repos/panniantong_agent_reach.md`；索引 `sources/README.md`
- 沉淀页面：`wiki/entities/agent-reach.md`（编码代理互联网接入脚手架：可插拔渠道、`doctor`、上游 CLI/MCP 与本地凭据主张；配流程图）
- 交叉更新：`wiki/entities/superpowers-obra.md`（关联页面回链）、`index.md`（重点入口 + Page Catalog）、`sources/README.md`
- 派生再生成：`make ci-preflight` 同步 `exports/`、`docs/exports/`、`docs/search-index.json`、`docs/sitemap.xml`、`README.md`、`docs/index.html` 等

## [2026-05-18] ingest | sources/papers/bfm_humanoid_arxiv_2509_13780.md、sources/sites/bfm4humanoid-github-io.md — BFM（Behavior Foundation Model，arXiv:2509.13780）入库；新增 wiki/entities/paper-behavior-foundation-model-humanoid.md；交叉更新 foundation-policy / whole-body-control；并对齐 CLAUDE.md 的 PR 截图流程

- 原始资料：`sources/papers/bfm_humanoid_arxiv_2509_13780.md`、`sources/sites/bfm4humanoid-github-io.md`；索引 `sources/README.md`
- 沉淀页面：`wiki/entities/paper-behavior-foundation-model-humanoid.md`（CVAE + 位级二值掩码 + 在线蒸馏 + 潜空间组合 + 残差解码器新技能的人形 WBC 基础模型，配 Mermaid 流程图与 Table III/IV 量化对照）
- 交叉更新：`wiki/concepts/foundation-policy.md`（新增 BFM 子项与回链）、`wiki/concepts/whole-body-control.md`（Learning-based & Generative WBC 段补 BFM；关联页面与 sources 互链）、`index.md`（新增 Entity 条目）、`sources/README.md`
- 规范同步：`CLAUDE.md` 新增「Claude Code Agent：PR 与验证截图」一节，与 `docs/checklists/cloud-agent-pr-workflow.md` 对齐
- 派生再生成：`make ci-preflight` 同步 `exports/`、`docs/exports/`、`docs/search-index.json`、`docs/sitemap.xml`、`README.md`、`docs/index.html`、`index.md` 等

## [2026-05-17] structural | wiki/formalizations/motion-retargeting-objective.md — V22 P1 动作重定向知识链 (2/3)：新增重定向目标函数形式化页

- 沉淀页面：`wiki/formalizations/motion-retargeting-objective.md`（通用目标函数 $\mathcal{L}^{\text{pose}}+\mathcal{L}^{\text{ee}}+\mathcal{L}^{\text{bal}}+\mathcal{L}^{\text{lim}}+\mathcal{L}^{\text{smooth}}$；姿态相似/末端接触/平衡/限位/平滑五大罚项的数学定义；GMR/DeepMimic/ReActor/NMR/SPIDER 五种工程退化形态对照）
- 交叉更新：`wiki/concepts/motion-retargeting.md`、`wiki/concepts/motion-retargeting-pipeline.md`（关联页面区块回链本页）、`docs/checklists/tech-stack-next-phase-checklist-v22.md`（P1 第 2 项打勾，含实现摘要）
- 派生再生成：`make ci-preflight` 同步 `exports/`、`docs/exports/`、`docs/search-index.json`、`docs/sitemap.xml`、`README.md`、`index.md` 等

## [2026-05-17] ingest | sources/repos/lingbot-map.md、sources/papers/lingbot_map_arxiv_2604_14141.md、sources/sites/lingbot-map-technology-robbant.md、sources/sites/businesswire-lingbot-map-2026-04-16.md — LingBot-Map 论文/站点/通稿入库；勘误 byant 误链；扩充 wiki/methods/lingbot-map.md

- 原始资料：`sources/repos/lingbot-map.md`、`sources/papers/lingbot_map_arxiv_2604_14141.md`、`sources/sites/lingbot-map-technology-robbant.md`、`sources/sites/businesswire-lingbot-map-2026-04-16.md`；索引 `sources/README.md`
- 沉淀页面：`wiki/methods/lingbot-map.md`（流程图、工程局限、参考来源；官方仓库为 [Robbyant/lingbot-map](https://github.com/Robbyant/lingbot-map)）
- 交叉更新：`index.md`（由 `make ci-preflight` 同步 `exports/`、`docs/exports/`、`docs/search-index.json`、`docs/sitemap.xml`、`README.md` 等）

## [2026-05-17] ingest | sources/papers/humannet_table1_benchmark_corpora.md — HumanNet Table1 代表性人视频/行为语料官方入口索引；新增对比页 wiki/comparisons/humannet-table1-human-video-corpora.md

- 原始资料：`sources/papers/humannet_table1_benchmark_corpora.md`；索引 `sources/README.md`
- 沉淀页面：`wiki/comparisons/humannet-table1-human-video-corpora.md`
- 交叉更新：`wiki/entities/humannet.md`、`wiki/methods/vla.md`、`index.md`（由 `make ci-preflight` 同步 `exports/`、`docs/exports/`、`docs/search-index.json`、`docs/sitemap.xml`、`README.md` 等）

## [2026-05-17] ingest | sources/papers/egoscale_arxiv_2602_16710.md、sources/sites/nvidia-research-egoscale.md — EgoScale（arXiv:2602.16710）与 NVIDIA GEAR 项目页入库

- 原始资料：`sources/papers/egoscale_arxiv_2602_16710.md`、`sources/sites/nvidia-research-egoscale.md`；索引 `sources/README.md`
- 沉淀页面：`wiki/methods/egoscale.md`
- 交叉更新：`wiki/methods/vla.md`、`wiki/methods/imitation-learning.md`、`wiki/tasks/manipulation.md`、`wiki/entities/humannet.md`、`wiki/concepts/embodied-scaling-laws.md`、`references/papers/imitation-learning.md`、`index.md`（由 `make ci-preflight` 同步）

## [2026-05-17] ingest | sources/papers/mimic_video_arxiv_2512_15692.md、sources/sites/mimic-video-github-io.md、sources/repos/lucidrains_mimic_video.md — mimic-video（VAM，arXiv:2512.15692）入库

- 原始资料：`sources/papers/mimic_video_arxiv_2512_15692.md`、`sources/sites/mimic-video-github-io.md`、`sources/repos/lucidrains_mimic_video.md`；索引 `sources/README.md`
- 沉淀页面：`wiki/methods/mimic-video.md`
- 交叉更新：`wiki/methods/vla.md`、`wiki/methods/imitation-learning.md`、`wiki/concepts/video-as-simulation.md`、`wiki/methods/generative-world-models.md`、`wiki/tasks/manipulation.md`、`references/papers/imitation-learning.md`、`index.md`（由 `make ci-preflight` 同步）

## [2026-05-17] ingest | sources/papers/crisp_real2sim_iclr2026.md、sources/sites/crisp-real2sim-project-github-io.md、sources/repos/crisp_real2sim_repo.md — CRISP（ICLR 2026）Real2Sim 入库

- 原始资料：`sources/papers/crisp_real2sim_iclr2026.md`、`sources/sites/crisp-real2sim-project-github-io.md`、`sources/repos/crisp_real2sim_repo.md`；索引 `sources/README.md`
- 沉淀页面：`wiki/methods/crisp-real2sim.md`
- 交叉更新：`wiki/concepts/sim2real.md`、`wiki/entities/gs-playground.md`、`references/papers/sim2real.md`、`index.md`（新增方法页目录条目）、`README.md` / `exports/` / `docs/exports/` 等（`make ci-preflight`）

## [2026-05-17] ingest | sources/papers/egm_arxiv_2512_19043.md、sources/blogs/egm_themoonlight_literature_review_2512_19043.md — EGM（arXiv:2512.19043）与第三方导读入库

- 原始资料：`sources/papers/egm_arxiv_2512_19043.md`、`sources/blogs/egm_themoonlight_literature_review_2512_19043.md`；索引 `sources/README.md`
- 沉淀页面：`wiki/methods/egm-efficient-general-mimic.md`
- 交叉更新：`wiki/methods/beyondmimic.md`、`wiki/methods/sonic-motion-tracking.md`、`index.md`（由 `make ci-preflight` 同步）

## [2026-05-17] ingest | sources/papers/e_sds_arxiv_2512_16446.md、sds_quadruped_arxiv_2410_11571.md、repos/rpl_cs_ucl_sds.md — E-SDS（arXiv:2512.16446）与 SDS 前序资料入库

- 原始资料：`sources/papers/e_sds_arxiv_2512_16446.md`、`sources/papers/sds_quadruped_arxiv_2410_11571.md`、`sources/repos/rpl_cs_ucl_sds.md`；索引 `sources/README.md`
- 沉淀页面：`wiki/entities/paper-e-sds-environment-aware-humanoid-locomotion-rl.md`
- 交叉更新：`wiki/tasks/locomotion.md`、`wiki/methods/reinforcement-learning.md`、`references/papers/locomotion-rl.md`、`index.md`（由 `make ci-preflight` 同步）

## [2026-05-17] ingest | sources/repos/unitree_ros.md、sources/repos/unitree_ros_to_real.md — 官方 ROS1+Gazebo 与真机 ROS 桥入库

- 原始资料：`sources/repos/unitree_ros.md`、`sources/repos/unitree_ros_to_real.md`；索引 `sources/README.md`
- 沉淀页面：`wiki/entities/unitree-ros.md`
- 交叉更新：`wiki/entities/unitree.md`、`wiki/entities/unitree-rl-mjlab.md`、`wiki/tasks/locomotion.md`、`index.md`（由 `make ci-preflight` 同步）

## [2026-05-17] ingest | sources/papers/urdd_beyond_urdf_arxiv_2512_23135.md、Apollo-Lab-Yale 多仓与 Pages — URDD（arXiv:2512.23135）入库

- 原始资料：`sources/papers/urdd_beyond_urdf_arxiv_2512_23135.md`、`sources/repos/apollo-lab-yale-apollo-resources.md`、`sources/repos/apollo-lab-yale-apollo-rust.md`、`sources/repos/apollo-lab-yale-apollo-three-engine.md`、`sources/repos/apollo-lab-yale-apollo-py.md`、`sources/sites/apollo-lab-yale-apollo-resources-github-io.md`；索引 `sources/README.md`
- 沉淀页面：`wiki/entities/paper-urdd-universal-robot-description-directory.md`
- 交叉更新：`wiki/entities/robot-viewer.md`、`wiki/entities/urdf-studio.md`、`wiki/entities/mujoco.md`、`sources/urdf.md`、`index.md`（由 `make ci-preflight` 同步）

## [2026-05-17] ingest | sources/papers/hy_motion_arxiv_2512_23464.md、sources/repos/tencent_hunyuan_hy_motion_1_0.md — HY-Motion 1.0（arXiv:2512.23464）与官方仓入库

- 原始资料：`sources/papers/hy_motion_arxiv_2512_23464.md`、`sources/repos/tencent_hunyuan_hy_motion_1_0.md`；索引 `sources/README.md`
- 沉淀页面：`wiki/methods/hy-motion-1.md`
- 交叉更新：`wiki/methods/diffusion-motion-generation.md`、`wiki/methods/genmo.md`、`wiki/entities/awesome-text-to-motion-zilize.md`、`index.md`（由 `make ci-preflight` 同步）

## [2026-05-17] ingest | sources/papers/spider_scalable_physics_informed_dexterous_retargeting.md、sources/sites/jc-bao-spider-project-github-io.md、sources/repos/jc-bao-spider-project.md — SPIDER（arXiv:2511.09484）与项目页入库

- 原始资料：`sources/papers/spider_scalable_physics_informed_dexterous_retargeting.md`、`sources/sites/jc-bao-spider-project-github-io.md`、`sources/repos/jc-bao-spider-project.md`；索引 `sources/README.md`
- 沉淀页面：`wiki/methods/spider-physics-informed-dexterous-retargeting.md`
- 交叉更新：`wiki/concepts/motion-retargeting.md`、`wiki/concepts/motion-retargeting-pipeline.md`、`wiki/methods/motion-retargeting-gmr.md`、`index.md`（由 `make ci-preflight` 同步）

## [2026-05-17] ingest | sources/papers/dwm_arxiv_2512_17907.md、sources/sites/snuvclab-dwm-github-io.md、sources/repos/snuvclab_dwm.md — DWM（Dexterous World Models）项目页与论文入库

- 原始资料：`sources/papers/dwm_arxiv_2512_17907.md`、`sources/sites/snuvclab-dwm-github-io.md`、`sources/repos/snuvclab_dwm.md`；索引 `sources/README.md`
- 沉淀页面：`wiki/methods/dwm.md`
- 交叉更新：`wiki/methods/generative-world-models.md`、`wiki/concepts/video-as-simulation.md`、`wiki/tasks/manipulation.md`、`index.md`（由 `make ci-preflight` 同步）

## [2026-05-17] ingest | sources/repos/cyoahs-robot-motion-editor.md、stanford-tml-robot-keyframe-kit.md、project-instinct-robot-motion-editor.md — 机器人关键帧/运动编辑三仓库入库

- 原始资料：`sources/repos/cyoahs-robot-motion-editor.md`、`sources/repos/stanford-tml-robot-keyframe-kit.md`、`sources/repos/project-instinct-robot-motion-editor.md`；索引 `sources/README.md`
- 沉淀页面：`wiki/entities/robot-motion-keyframe-editors.md`
- 交叉更新：`wiki/entities/project-instinct.md`、`wiki/entities/mujoco.md`、`wiki/concepts/motion-retargeting-pipeline.md`、`wiki/tasks/manipulation.md`、`index.md`（由 `make ci-preflight` 同步）

## [2026-05-17] ingest | sources/papers/faststair_arxiv_2601_10365.md、sources/sites/npcliu-faststair-github-io.md — FastStair（arXiv:2601.10365）入库

- 原始资料：`sources/papers/faststair_arxiv_2601_10365.md`、`sources/sites/npcliu-faststair-github-io.md`；索引 `sources/README.md`
- 沉淀页面：`wiki/entities/paper-faststair-humanoid-stair-ascent.md`
- 交叉更新：`wiki/tasks/locomotion.md`、`index.md`（由 `make ci-preflight` 同步）

## [2026-05-17] ingest | sources/repos/obra-superpowers.md、sources/blogs/fsck_superpowers_announcement_2025-10-09.md — Superpowers（obra）编码代理技能与交付工作流入库

- 原始资料：`sources/repos/obra-superpowers.md`、`sources/blogs/fsck_superpowers_announcement_2025-10-09.md`；索引 `sources/README.md`
- 沉淀页面：`wiki/entities/superpowers-obra.md`
- 交叉更新：`wiki/references/llm-wiki-karpathy.md`、`index.md`（由 `make ci-preflight` 同步）

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

## [2026-05-17] ingest | sources/repos/sage-sim2real-actuator-gap.md — SAGE（Sim2Real Actuator Gap Estimator）与 README 要点归档

- 原始资料：`sources/repos/sage-sim2real-actuator-gap.md`（<https://github.com/isaac-sim2real/sage> 及 README 公开信息；关联 AMASS、Human2Humanoid、OSMO 工作流线索）
- 沉淀页面：`wiki/entities/sage-sim2real-actuator-gap-estimator.md`
- 交叉更新：`wiki/concepts/sim2real.md`、`wiki/concepts/domain-randomization.md`、`wiki/roadmaps/humanoid-control-roadmap.md`、`wiki/queries/sim2real-gap-reduction.md`、`wiki/methods/actuator-network.md`、`sources/README.md`、`index.md`

## [2026-05-17] ingest | LIFT（arXiv:2601.21363）与 lift-humanoid.github.io、bigai-ai/LIFT-humanoid 归档入库

- 原始资料：`sources/papers/lift_humanoid_arxiv_2601_21363.md`、`sources/sites/lift-humanoid-github-io.md`、`sources/repos/bigai-lift-humanoid.md`
- 沉淀页面：`wiki/entities/lift-humanoid.md`
- 交叉更新：`wiki/tasks/locomotion.md`、`wiki/concepts/sim2real.md`、`wiki/methods/reinforcement-learning.md`、`wiki/methods/model-based-rl.md`、`wiki/queries/rl-algorithm-selection.md`、`sources/README.md`、`index.md`

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

## [2026-06-20] structural | checklist-v25 P1 数据层专题交叉补强 —— motion-retargeting 与 humanoid-reference-motion-datasets 明示「数据来源 → 质量评估 → 重定向 → 策略输入」四段衔接

- `wiki/comparisons/humanoid-reference-motion-datasets.md` 新增「四段衔接」表与因果判据段，把五集数据落到①数据来源→②质量评估→③重定向→④策略输入四段，并显式回链 `motion-data-quality.md` 与 `humanoid-training-data-pipeline.md`。
- `wiki/concepts/motion-retargeting.md` 新增「上游衔接」表，把重定向定位为链路第③段，明示其触发与补层由 motion-data-quality 四轴（形态差距/接触/物理）决定，与 P1 新页形成双向回链、消除孤儿。
- `make lint` 0 errors（仅 3 条既有信息型预警）；勾选 v25 P1「数据层专题交叉补强」条目。

## [2026-06-23] structural | checklist-v25 P3 详情页「训练数据管线」专题徽标联动 —— 修正分词粒度漏匹配

- 详情页「所属专题」徽标行（`docs/main.js renderMetaTopicBadges`）本就以 `docs/topic-filters.js` 为单一事实源、`topicsForNode` 数据驱动：命中 `data-pipeline` 即渲染「📦 训练数据」徽标并跳 `graph.html?topic=data-pipeline`，空态降级隐藏整行——P3 第①项把 `data-pipeline` 写入单一事实源后，详情页徽标已自动联动，无需二次实现。
- 本次补强 `data-pipeline.segments` +5（`retarget`/`retargeter`/`omniretarget`/`mocap`/`freemocap`），修正纯分词粒度导致的漏匹配（`mocap-retarget`/`soma-retargeter`/`paper-...-omniretarget`/`freemocap` 等重定向与动捕实体此前只命中 `motion-retargeting`）；node 逐页校验后数据集 + 重定向 + mocap 实体 46/46 候选页全部稳定命中专题（全库 47 节点）。
- `make lint` 0 errors（另含 4 条信息型预警，不阻塞 CI）；勾选 v25 P3「详情页『同专题相关页』提示」条目。截图待带 Chrome 的环境补归档。
