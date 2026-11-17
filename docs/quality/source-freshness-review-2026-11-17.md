# 来源时效性与缺页概念复核（2026-11-17）

本轮对应 PR #3824，先将 main `e4b88a3c4` 合入，复核基线为 `a1e9e68ffec882c866cadddca1011d0ec2a4e475`。来源时效性提示从上一轮的 168 条变为 169 条，新增项来自 RoboLab v4 / v0.3.3 的来源更新。

## 核对方法与边界

- 对本地 `_check_sources_health` 的 169 条实际报告，逐条定位触发来源、来源最新提交和 Wiki 最近提交/frontmatter `updated` 日期之前的可比来源快照；用 `git log --follow` 追溯来源迁移前的路径。
- 分流只比较快照差异的位置：Wiki 链接目标及路径式标签单独归导航，`通读` 标记单列；正文、外部资源、来源身份等其他差异保持待核对。导航分类不证明每条重定向的实体身份均已独立审计。
- 表中每行是该条报告的触发来源，**不是该 Wiki 页全部来源的完整审计**；“新增来源或无可比历史”也不等于已证明页面内容过时。
- 本轮保持来源时效性检查规则与级别，未批量修改更新时间、未将这类提示静音。只有补充了实际内容的八页更新 `updated`；其余提示继续保留。
- 实质修复基于仓库既有核读归档及其明确版本。未新增原论文全文通读、外部源码运行或真机复现的声明。

## 基线差异分流

| 差异位置 | 报告条数 | 本轮处理 |
|---|---:|---|
| 仅 Wiki 导航差异 | 116 | 保留提示，记录可比提交；不以改日期替代核对 |
| 导航与通读标记差异 | 2 | 保留提示，记录可比提交；不以改日期替代核对 |
| 内容或来源身份变化，需逐页核对 | 48 | 优先核对会改变选型、评测或部署结论的内容；八项修复见下 |
| 新增来源或无可比历史 | 3 | 优先核对会改变选型、评测或部署结论的内容；八项修复见下 |

## 已同步的八个页面

| 页面 | 实质更新 |
|---|---|
| [wiki/entities/articraft.md](../../wiki/entities/articraft.md) | 同步 Agentic Real2Sim v5 对照，区分资产与 episode 的评测单位 |
| [wiki/queries/embodied-eval-benchmark-selection-loop.md](../../wiki/queries/embodied-eval-benchmark-selection-loop.md) | 补 v5 三种评测口径及 RoboLab 样本量/版本边界 |
| [wiki/overview/embodied-station-2026-11-05-43-papers-technology-map.md](../../wiki/overview/embodied-station-2026-11-05-43-papers-technology-map.md) | 同步 ARP 原文复核、RoboLab/动态任务与开放状态 |
| [wiki/overview/embodied-station-2026-11-04-57-papers-technology-map.md](../../wiki/overview/embodied-station-2026-11-04-57-papers-technology-map.md) | 纠正 W2A 展示站仓库与可运行算法代码的区别 |
| [wiki/entities/paper-vlaquantbench.md](../../wiki/entities/paper-vlaquantbench.md) | 补 MACaQ 通用 PTQ 与机器人闭环验收的证据边界 |
| [wiki/entities/paper-ctrl-world.md](../../wiki/entities/paper-ctrl-world.md) | 纠正共享仓库不等于完整 VLAW 流程开源 |
| [wiki/entities/paper-progress-reward-modeling-survey.md](../../wiki/entities/paper-progress-reward-modeling-survey.md) | 补历史依赖、在线/事后接口、标定与部署方向边界 |
| [wiki/concepts/sim-vs-real-eval-gap.md](../../wiki/concepts/sim-vs-real-eval-gap.md) | 同步 RoboLab v4/v0.3.3，修正复现/代表性及同分布/泄漏混同 |

其余内容/身份差异仍需逐页核对，例如遥操作来源的新核读记录、二阶优化来源历史变更、UCAG-P 对照页、VBot→EgoWorld 的关联语境等；本报告不把这些项标为已完成。

## 两个缺页概念建议

- **`complete`**：六页命中是节点状态说明、完成档位或 demo 场景标签。只为此标签增加缺页候选停用词；回归测试同时保留真正新概念 `NewControl` 的告警，不改变缺页频次阈值和告警级别。
- **`embodiment`**：六页语境跨越数据元数据、框架身体接口、平台属性及 SayCan 的测试类别。既有 URDF/OXE 等页分别覆盖部分含义，缺少统一概念入口，因此新增[机器人本体](../../wiki/concepts/embodiment.md)，区分身体结构、观测、动作与控制契约、任务可行性，并由 OXE、URDF、Inspect Robots、SayCan 回链。不是将该词加入豁免清单。

## 专项复检

同一脚本在修改后实测：来源时效性提示 **169 → 161**，消除的八项与上述内容更新一致，没有新增来源时效性提示；缺页概念建议 **2 → 0**。缺页概念测试文件的 **45 项测试通过**，包括真实新概念继续告警的回归用例。全量预检结果另见 PR 正文，不能把本节专项结果当作全库 info 清零。

全量复检另发现 ARP 既有结论“避免把局部收益夸大成全面 SOTA”被当作陈旧性能声明。仅为紧邻命中词的这一完整否定句式补充识别；测试覆盖同一句后半段、分号后的真实断言，以及“避免碰撞，达到 SOTA”等非否定语境，确保仍保留真正的时效性告警。未改动 ARP 的技术结论或更新时间来消除提示。

## 169 条基线报告的可追溯记录

“可比旧提交”指来源在对应 Wiki 最近提交/更新日之前的版本；缺失则记 `—`。导航分流只定位差异，不作为删除 lint 告警的依据。

| Wiki 页面 | 触发来源 | 可比旧提交 → 来源最新提交 | 差异分流 | 本轮状态 |
|---|---|---|---|---|
| [wiki/methods/behavior-cloning.md](../../wiki/methods/behavior-cloning.md) | [adapt_arxiv_2608_20087.md](../../sources/papers/adapt_arxiv_2608_20087.md) | `7913ff923` → `2600489c3` | 仅 Wiki 导航差异 | 保留待核对 |
| [wiki/entities/crazyswarm2.md](../../wiki/entities/crazyswarm2.md) | [aerial_aquatic_remora_scirobotics_2022.md](../../sources/papers/aerial_aquatic_remora_scirobotics_2022.md) | `8bc9b765f` → `4368dbdf3` | 仅 Wiki 导航差异 | 保留待核对 |
| [wiki/entities/paper-aerial-aquatic-remora-hitchhiking-robot.md](../../wiki/entities/paper-aerial-aquatic-remora-hitchhiking-robot.md) | [aerial_aquatic_remora_scirobotics_2022.md](../../sources/papers/aerial_aquatic_remora_scirobotics_2022.md) | `8bc9b765f` → `4368dbdf3` | 仅 Wiki 导航差异 | 保留待核对 |
| [wiki/entities/paper-octopus-inspired-esoam-soft-arm.md](../../wiki/entities/paper-octopus-inspired-esoam-soft-arm.md) | [aerial_aquatic_remora_scirobotics_2022.md](../../sources/papers/aerial_aquatic_remora_scirobotics_2022.md) | `8bc9b765f` → `4368dbdf3` | 仅 Wiki 导航差异 | 保留待核对 |
| [wiki/entities/paper-miniature-deep-sea-morphable-robot.md](../../wiki/entities/paper-miniature-deep-sea-morphable-robot.md) | [aerial_aquatic_remora_scirobotics_2022.md](../../sources/papers/aerial_aquatic_remora_scirobotics_2022.md) | `8bc9b765f` → `4368dbdf3` | 仅 Wiki 导航差异 | 保留待核对 |
| [wiki/entities/articraft.md](../../wiki/entities/articraft.md) | [agentic_real2sim_arxiv_2607_19190.md](../../sources/papers/agentic_real2sim_arxiv_2607_19190.md) | `dcc85d3bb` → `efa624578` | 内容或来源身份变化，需逐页核对 | 已同步内容 |
| [wiki/queries/embodied-eval-benchmark-selection-loop.md](../../wiki/queries/embodied-eval-benchmark-selection-loop.md) | [agentic_real2sim_arxiv_2607_19190.md](../../sources/papers/agentic_real2sim_arxiv_2607_19190.md) | `dcc85d3bb` → `efa624578` | 内容或来源身份变化，需逐页核对 | 已同步内容 |
| [wiki/entities/paper-notebook-ame-2-agile-and-generalized-legged-locomotion-vi.md](../../wiki/entities/paper-notebook-ame-2-agile-and-generalized-legged-locomotion-vi.md) | [agile_perceptive_traversal_arxiv_2608_29769.md](../../sources/papers/agile_perceptive_traversal_arxiv_2608_29769.md) | `91b67e4a2` → `1f32bc297` | 仅 Wiki 导航差异 | 保留待核对 |
| [wiki/entities/paper-attention-is-all-you-need.md](../../wiki/entities/paper-attention-is-all-you-need.md) | [ai_architecture_foundations.md](../../sources/papers/ai_architecture_foundations.md) | `a468cc693` → `2600489c3` | 仅 Wiki 导航差异 | 保留待核对 |
| [wiki/concepts/mlp.md](../../wiki/concepts/mlp.md) | [ai_architecture_foundations.md](../../sources/papers/ai_architecture_foundations.md) | `a468cc693` → `2600489c3` | 仅 Wiki 导航差异 | 保留待核对 |
| [wiki/concepts/mixture-of-experts.md](../../wiki/concepts/mixture-of-experts.md) | [ai_architecture_foundations.md](../../sources/papers/ai_architecture_foundations.md) | `a468cc693` → `2600489c3` | 仅 Wiki 导航差异 | 保留待核对 |
| [wiki/concepts/recurrent-neural-network.md](../../wiki/concepts/recurrent-neural-network.md) | [ai_architecture_foundations.md](../../sources/papers/ai_architecture_foundations.md) | `a468cc693` → `2600489c3` | 仅 Wiki 导航差异 | 保留待核对 |
| [wiki/concepts/mamba.md](../../wiki/concepts/mamba.md) | [ai_architecture_foundations.md](../../sources/papers/ai_architecture_foundations.md) | `a468cc693` → `2600489c3` | 仅 Wiki 导航差异 | 保留待核对 |
| [wiki/concepts/autoencoder.md](../../wiki/concepts/autoencoder.md) | [ai_architecture_foundations.md](../../sources/papers/ai_architecture_foundations.md) | `a468cc693` → `2600489c3` | 仅 Wiki 导航差异 | 保留待核对 |
| [wiki/concepts/generative-adversarial-network.md](../../wiki/concepts/generative-adversarial-network.md) | [ai_architecture_foundations.md](../../sources/papers/ai_architecture_foundations.md) | `a468cc693` → `2600489c3` | 仅 Wiki 导航差异 | 保留待核对 |
| [wiki/concepts/diffusion-model.md](../../wiki/concepts/diffusion-model.md) | [ai_architecture_foundations.md](../../sources/papers/ai_architecture_foundations.md) | `a468cc693` → `2600489c3` | 仅 Wiki 导航差异 | 保留待核对 |
| [wiki/methods/diffusion-policy.md](../../wiki/methods/diffusion-policy.md) | [ai_architecture_foundations.md](../../sources/papers/ai_architecture_foundations.md) | `a468cc693` → `2600489c3` | 仅 Wiki 导航差异 | 保留待核对 |
| [wiki/concepts/vision-transformer.md](../../wiki/concepts/vision-transformer.md) | [ai_architecture_foundations.md](../../sources/papers/ai_architecture_foundations.md) | `a468cc693` → `2600489c3` | 仅 Wiki 导航差异 | 保留待核对 |
| [wiki/concepts/state-space-model-ssm.md](../../wiki/concepts/state-space-model-ssm.md) | [ai_architecture_foundations.md](../../sources/papers/ai_architecture_foundations.md) | `a468cc693` → `2600489c3` | 仅 Wiki 导航差异 | 保留待核对 |
| [wiki/concepts/graph-neural-network.md](../../wiki/concepts/graph-neural-network.md) | [ai_architecture_foundations.md](../../sources/papers/ai_architecture_foundations.md) | `a468cc693` → `2600489c3` | 仅 Wiki 导航差异 | 保留待核对 |
| [wiki/entities/paper-hiking-in-the-wild.md](../../wiki/entities/paper-hiking-in-the-wild.md) | [almi_adversarial_locomotion_motion_imitation_arxiv_2504_14305.md](../../sources/papers/almi_adversarial_locomotion_motion_imitation_arxiv_2504_14305.md) | `0a5e2b920` → `1f32bc297` | 仅 Wiki 导航差异 | 保留待核对 |
| [wiki/overview/embodied-station-2026-11-05-43-papers-technology-map.md](../../wiki/overview/embodied-station-2026-11-05-43-papers-technology-map.md) | [asynchronous_robot_policies_arxiv_2611_05155.md](../../sources/papers/asynchronous_robot_policies_arxiv_2611_05155.md) | `daacc6d17` → `0f7c792cd` | 内容或来源身份变化，需逐页核对 | 已同步内容 |
| [wiki/entities/deepinsight.md](../../wiki/entities/deepinsight.md) | [athena_wbc_arxiv_2607_04837.md](../../sources/papers/athena_wbc_arxiv_2607_04837.md) | `17f75c317` → `2600489c3` | 仅 Wiki 导航差异 | 保留待核对 |
| [wiki/entities/paper-bam-extended-friction-servo-actuators.md](../../wiki/entities/paper-bam-extended-friction-servo-actuators.md) | [bam_extended_friction_servos_arxiv_2410_08650.md](../../sources/papers/bam_extended_friction_servos_arxiv_2410_08650.md) | `b32518837` → `dcc85d3bb` | 导航与通读标记差异 | 保留待核对 |
| [wiki/methods/being-h07.md](../../wiki/methods/being-h07.md) | [being_m07.md](../../sources/papers/being_m07.md) | `c9a0f6bf6` → `2600489c3` | 仅 Wiki 导航差异 | 保留待核对 |
| [wiki/methods/action-chunking.md](../../wiki/methods/action-chunking.md) | [being_m07.md](../../sources/papers/being_m07.md) | `c9a0f6bf6` → `2600489c3` | 仅 Wiki 导航差异 | 保留待核对 |
| [wiki/concepts/privileged-training.md](../../wiki/concepts/privileged-training.md) | [bfm_humanoid_arxiv_2509_13780.md](../../sources/papers/bfm_humanoid_arxiv_2509_13780.md) | `cacdb6ab4` → `2600489c3` | 仅 Wiki 导航差异 | 保留待核对 |
| [wiki/concepts/curriculum-learning.md](../../wiki/concepts/curriculum-learning.md) | [bfm_humanoid_arxiv_2509_13780.md](../../sources/papers/bfm_humanoid_arxiv_2509_13780.md) | `cacdb6ab4` → `2600489c3` | 仅 Wiki 导航差异 | 保留待核对 |
| [wiki/entities/amass.md](../../wiki/entities/amass.md) | [bfm_humanoid_arxiv_2509_13780.md](../../sources/papers/bfm_humanoid_arxiv_2509_13780.md) | `cacdb6ab4` → `2600489c3` | 仅 Wiki 导航差异 | 保留待核对 |
| [wiki/methods/awr.md](../../wiki/methods/awr.md) | [chi0_kai0_arxiv_2602_09021.md](../../sources/papers/chi0_kai0_arxiv_2602_09021.md) | `44fea808b` → `1f32bc297` | 仅 Wiki 导航差异 | 保留待核对 |
| [wiki/entities/paper-lehome-learning-to-fold.md](../../wiki/entities/paper-lehome-learning-to-fold.md) | [chi0_kai0_arxiv_2602_09021.md](../../sources/papers/chi0_kai0_arxiv_2602_09021.md) | `44fea808b` → `1f32bc297` | 仅 Wiki 导航差异 | 保留待核对 |
| [wiki/tasks/manipulation.md](../../wiki/tasks/manipulation.md) | [chi0_kai0_arxiv_2602_09021.md](../../sources/papers/chi0_kai0_arxiv_2602_09021.md) | `44fea808b` → `1f32bc297` | 仅 Wiki 导航差异 | 保留待核对 |
| [wiki/concepts/humanoid-multi-robot-coordination.md](../../wiki/concepts/humanoid-multi-robot-coordination.md) | [chorus_arxiv_2606_12352.md](../../sources/papers/chorus_arxiv_2606_12352.md) | `5f8cae175` → `1f32bc297` | 仅 Wiki 导航差异 | 保留待核对 |
| [wiki/entities/paper-resmimic.md](../../wiki/entities/paper-resmimic.md) | [coordex_arxiv_2606_23680.md](../../sources/papers/coordex_arxiv_2606_23680.md) | `6135363b4` → `1b24e1603` | 内容或来源身份变化，需逐页核对 | 保留待核对 |
| [wiki/entities/wuji-robotics.md](../../wiki/entities/wuji-robotics.md) | [coordex_arxiv_2606_23680.md](../../sources/papers/coordex_arxiv_2606_23680.md) | `6135363b4` → `1b24e1603` | 内容或来源身份变化，需逐页核对 | 保留待核对 |
| [wiki/entities/paper-twist.md](../../wiki/entities/paper-twist.md) | [cwi_arxiv_2606_27676.md](../../sources/papers/cwi_arxiv_2606_27676.md) | `cc9a85c20` → `1b24e1603` | 内容或来源身份变化，需逐页核对 | 保留待核对 |
| [wiki/overview/loco-manip-contact-category-02-contact-representation.md](../../wiki/overview/loco-manip-contact-category-02-contact-representation.md) | [cwi_arxiv_2606_27676.md](../../sources/papers/cwi_arxiv_2606_27676.md) | `cc9a85c20` → `1b24e1603` | 内容或来源身份变化，需逐页核对 | 保留待核对 |
| [wiki/entities/mimic-hand-m1.md](../../wiki/entities/mimic-hand-m1.md) | [data_pyramid_embodied_manipulation_arxiv_2607_24744.md](../../sources/papers/data_pyramid_embodied_manipulation_arxiv_2607_24744.md) | `02246801f` → `1f32bc297` | 仅 Wiki 导航差异 | 保留待核对 |
| [wiki/concepts/physics-fidelity-sim2real-gap.md](../../wiki/concepts/physics-fidelity-sim2real-gap.md) | [data_pyramid_embodied_manipulation_arxiv_2607_24744.md](../../sources/papers/data_pyramid_embodied_manipulation_arxiv_2607_24744.md) | `02246801f` → `1f32bc297` | 仅 Wiki 导航差异 | 保留待核对 |
| [wiki/concepts/embodied-scaling-laws.md](../../wiki/concepts/embodied-scaling-laws.md) | [data_pyramid_embodied_manipulation_arxiv_2607_24744.md](../../sources/papers/data_pyramid_embodied_manipulation_arxiv_2607_24744.md) | `02246801f` → `1f32bc297` | 仅 Wiki 导航差异 | 保留待核对 |
| [wiki/methods/egoscale.md](../../wiki/methods/egoscale.md) | [data_pyramid_embodied_manipulation_arxiv_2607_24744.md](../../sources/papers/data_pyramid_embodied_manipulation_arxiv_2607_24744.md) | `02246801f` → `1f32bc297` | 仅 Wiki 导航差异 | 保留待核对 |
| [wiki/entities/xiaomi-robotics-1.md](../../wiki/entities/xiaomi-robotics-1.md) | [data_pyramid_embodied_manipulation_arxiv_2607_24744.md](../../sources/papers/data_pyramid_embodied_manipulation_arxiv_2607_24744.md) | `02246801f` → `1f32bc297` | 仅 Wiki 导航差异 | 保留待核对 |
| [wiki/entities/paper-trex-tactile-reactive-dexterous-manipulation.md](../../wiki/entities/paper-trex-tactile-reactive-dexterous-manipulation.md) | [data_pyramid_embodied_manipulation_arxiv_2607_24744.md](../../sources/papers/data_pyramid_embodied_manipulation_arxiv_2607_24744.md) | `02246801f` → `1f32bc297` | 仅 Wiki 导航差异 | 保留待核对 |
| [wiki/concepts/contact-rich-manipulation.md](../../wiki/concepts/contact-rich-manipulation.md) | [dexholdem_arxiv_2605_18727.md](../../sources/papers/dexholdem_arxiv_2605_18727.md) | `cc9ff1c10` → `1f32bc297` | 仅 Wiki 导航差异 | 保留待核对 |
| [wiki/entities/dexbench.md](../../wiki/entities/dexbench.md) | [dexholdem_arxiv_2605_18727.md](../../sources/papers/dexholdem_arxiv_2605_18727.md) | `cc9ff1c10` → `1f32bc297` | 仅 Wiki 导航差异 | 保留待核对 |
| [wiki/overview/hub-embodied-eval-benchmark.md](../../wiki/overview/hub-embodied-eval-benchmark.md) | [dexholdem_arxiv_2605_18727.md](../../sources/papers/dexholdem_arxiv_2605_18727.md) | `cc9ff1c10` → `1f32bc297` | 仅 Wiki 导航差异 | 保留待核对 |
| [wiki/methods/disney-olaf-character-robot.md](../../wiki/methods/disney-olaf-character-robot.md) | [disney_olaf_character_robot.md](../../sources/papers/disney_olaf_character_robot.md) | `b6c0bd3a5` → `3e80c2f74` | 内容或来源身份变化，需逐页核对 | 保留待核对 |
| [wiki/concepts/control-barrier-function.md](../../wiki/concepts/control-barrier-function.md) | [disney_olaf_character_robot.md](../../sources/papers/disney_olaf_character_robot.md) | `b6c0bd3a5` → `3e80c2f74` | 内容或来源身份变化，需逐页核对 | 保留待核对 |
| [wiki/methods/mimic-video.md](../../wiki/methods/mimic-video.md) | [dit4dit_arxiv_2603_10448.md](../../sources/papers/dit4dit_arxiv_2603_10448.md) | `c9a0f6bf6` → `2600489c3` | 仅 Wiki 导航差异 | 保留待核对 |
| [wiki/entities/paper-notebook-dynaretarget-dynamically-feasible-retargeting-us.md](../../wiki/entities/paper-notebook-dynaretarget-dynamically-feasible-retargeting-us.md) | [dynaretarget_arxiv_2602_06827.md](../../sources/papers/dynaretarget_arxiv_2602_06827.md) | `df1c9a5c2` → `3e80c2f74` | 内容或来源身份变化，需逐页核对 | 保留待核对 |
| [wiki/methods/model-based-rl.md](../../wiki/methods/model-based-rl.md) | [ego_vcp_arxiv_2510_11682.md](../../sources/papers/ego_vcp_arxiv_2510_11682.md) | `bb663dadc` → `1f32bc297` | 仅 Wiki 导航差异 | 保留待核对 |
| [wiki/entities/paper-exe-cross-embodiment.md](../../wiki/entities/paper-exe-cross-embodiment.md) | [egox-embodiment-dreaming.md](../../sources/papers/egox-embodiment-dreaming.md) | `72d27750d` → `87f59cfc7` | 内容或来源身份变化，需逐页核对 | 保留待核对 |
| [wiki/concepts/motion-retargeting.md](../../wiki/concepts/motion-retargeting.md) | [exoactor.md](../../sources/papers/exoactor.md) | `f863d6063` → `2600489c3` | 仅 Wiki 导航差异 | 保留待核对 |
| [wiki/concepts/whole-body-coordination.md](../../wiki/concepts/whole-body-coordination.md) | [exoactor.md](../../sources/papers/exoactor.md) | `f863d6063` → `2600489c3` | 仅 Wiki 导航差异 | 保留待核对 |
| [wiki/concepts/capture-point-dcm.md](../../wiki/concepts/capture-point-dcm.md) | [fddc_arxiv_2608_00500.md](../../sources/papers/fddc_arxiv_2608_00500.md) | `0e1c8feef` → `2600489c3` | 仅 Wiki 导航差异 | 保留待核对 |
| [wiki/tasks/balance-recovery.md](../../wiki/tasks/balance-recovery.md) | [fddc_arxiv_2608_00500.md](../../sources/papers/fddc_arxiv_2608_00500.md) | `0e1c8feef` → `2600489c3` | 仅 Wiki 导航差异 | 保留待核对 |
| [wiki/methods/ams.md](../../wiki/methods/ams.md) | [fddc_arxiv_2608_00500.md](../../sources/papers/fddc_arxiv_2608_00500.md) | `0e1c8feef` → `2600489c3` | 仅 Wiki 导航差异 | 保留待核对 |
| [wiki/concepts/reward-design.md](../../wiki/concepts/reward-design.md) | [fddc_arxiv_2608_00500.md](../../sources/papers/fddc_arxiv_2608_00500.md) | `0e1c8feef` → `2600489c3` | 仅 Wiki 导航差异 | 保留待核对 |
| [wiki/entities/paper-notebook-hub.md](../../wiki/entities/paper-notebook-hub.md) | [fddc_arxiv_2608_00500.md](../../sources/papers/fddc_arxiv_2608_00500.md) | `0e1c8feef` → `2600489c3` | 仅 Wiki 导航差异 | 保留待核对 |
| [wiki/entities/paper-notebook-learning-sim-to-real-humanoid-locomotion-in-15-m.md](../../wiki/entities/paper-notebook-learning-sim-to-real-humanoid-locomotion-in-15-m.md) | [fddc_arxiv_2608_00500.md](../../sources/papers/fddc_arxiv_2608_00500.md) | `0e1c8feef` → `2600489c3` | 仅 Wiki 导航差异 | 保留待核对 |
| [wiki/entities/paper-eventvla-visual-evidence-memory.md](../../wiki/entities/paper-eventvla-visual-evidence-memory.md) | [fm_vla_arxiv_2607_18231.md](../../sources/papers/fm_vla_arxiv_2607_18231.md) | `e8bdb9267` → `1f32bc297` | 仅 Wiki 导航差异 | 保留待核对 |
| [wiki/tasks/vision-language-navigation.md](../../wiki/tasks/vision-language-navigation.md) | [fsd_vln_arxiv_2607_08359.md](../../sources/papers/fsd_vln_arxiv_2607_08359.md) | `d9bb21d4b` → `1f32bc297` | 仅 Wiki 导航差异 | 保留待核对 |
| [wiki/concepts/embodied-fm-latency-generalization-tradeoff.md](../../wiki/concepts/embodied-fm-latency-generalization-tradeoff.md) | [fsd_vln_arxiv_2607_08359.md](../../sources/papers/fsd_vln_arxiv_2607_08359.md) | `d9bb21d4b` → `1f32bc297` | 仅 Wiki 导航差异 | 保留待核对 |
| [wiki/entities/paper-uni-lavira.md](../../wiki/entities/paper-uni-lavira.md) | [fsd_vln_arxiv_2607_08359.md](../../sources/papers/fsd_vln_arxiv_2607_08359.md) | `d9bb21d4b` → `1f32bc297` | 仅 Wiki 导航差异 | 保留待核对 |
| [wiki/entities/paper-da-nav.md](../../wiki/entities/paper-da-nav.md) | [fsd_vln_arxiv_2607_08359.md](../../sources/papers/fsd_vln_arxiv_2607_08359.md) | `d9bb21d4b` → `1f32bc297` | 仅 Wiki 导航差异 | 保留待核对 |
| [wiki/overview/vln-open-source-repro-paradigms.md](../../wiki/overview/vln-open-source-repro-paradigms.md) | [fsd_vln_arxiv_2607_08359.md](../../sources/papers/fsd_vln_arxiv_2607_08359.md) | `d9bb21d4b` → `1f32bc297` | 仅 Wiki 导航差异 | 保留待核对 |
| [wiki/overview/multirotor-simulation-planning-control-stack.md](../../wiki/overview/multirotor-simulation-planning-control-stack.md) | [fsd_vln_arxiv_2607_08359.md](../../sources/papers/fsd_vln_arxiv_2607_08359.md) | `d9bb21d4b` → `1f32bc297` | 仅 Wiki 导航差异 | 保留待核对 |
| [wiki/entities/paper-ge-act-2.md](../../wiki/entities/paper-ge-act-2.md) | [ge_act_2_arxiv_2609_05588.md](../../sources/papers/ge_act_2_arxiv_2609_05588.md) | `72c800bc1` → `4368dbdf3` | 内容或来源身份变化，需逐页核对 | 保留待核对 |
| [wiki/entities/paper-sa-2508-05635-genie-envisioner-a-unified-world-foundation-plat.md](../../wiki/entities/paper-sa-2508-05635-genie-envisioner-a-unified-world-foundation-plat.md) | [ge_act_2_arxiv_2609_05588.md](../../sources/papers/ge_act_2_arxiv_2609_05588.md) | `72c800bc1` → `4368dbdf3` | 内容或来源身份变化，需逐页核对 | 保留待核对 |
| [wiki/entities/protomotions.md](../../wiki/entities/protomotions.md) | [genmo.md](../../sources/papers/genmo.md) | `3100ffee2` → `2600489c3` | 仅 Wiki 导航差异 | 保留待核对 |
| [wiki/methods/hy-motion-1.md](../../wiki/methods/hy-motion-1.md) | [gentrack_arxiv_2608_01410.md](../../sources/papers/gentrack_arxiv_2608_01410.md) | `906020006` → `2600489c3` | 仅 Wiki 导航差异 | 保留待核对 |
| [wiki/entities/paper-notebook-parc-physics-based-augmentation-with-reinforceme.md](../../wiki/entities/paper-notebook-parc-physics-based-augmentation-with-reinforceme.md) | [gentrack_arxiv_2608_01410.md](../../sources/papers/gentrack_arxiv_2608_01410.md) | `906020006` → `2600489c3` | 仅 Wiki 导航差异 | 保留待核对 |
| [wiki/entities/paper-notebook-rl-from-physical-feedback-aligning-large-motion.md](../../wiki/entities/paper-notebook-rl-from-physical-feedback-aligning-large-motion.md) | [gentrack_arxiv_2608_01410.md](../../sources/papers/gentrack_arxiv_2608_01410.md) | `906020006` → `2600489c3` | 仅 Wiki 导航差异 | 保留待核对 |
| [wiki/entities/paper-twist2.md](../../wiki/entities/paper-twist2.md) | [gigabrain_wbc_0_5_arxiv_2608_18234.md](../../sources/papers/gigabrain_wbc_0_5_arxiv_2608_18234.md) | `c898a6007` → `2600489c3` | 仅 Wiki 导航差异 | 保留待核对 |
| [wiki/entities/paper-cmp.md](../../wiki/entities/paper-cmp.md) | [gigabrain_wbc_0_5_arxiv_2608_18234.md](../../sources/papers/gigabrain_wbc_0_5_arxiv_2608_18234.md) | `c898a6007` → `2600489c3` | 仅 Wiki 导航差异 | 保留待核对 |
| [wiki/methods/amp-reward.md](../../wiki/methods/amp-reward.md) | [gmp_generative_motion_prior_arxiv_2503_09015.md](../../sources/papers/gmp_generative_motion_prior_arxiv_2503_09015.md) | `0a5e2b920` → `1f32bc297` | 仅 Wiki 导航差异 | 保留待核对 |
| [wiki/methods/deepmimic.md](../../wiki/methods/deepmimic.md) | [gpc_arxiv_2606_29148.md](../../sources/papers/gpc_arxiv_2606_29148.md) | `bf542a2bc` → `2600489c3` | 仅 Wiki 导航差异 | 保留待核对 |
| [wiki/methods/ase.md](../../wiki/methods/ase.md) | [gpc_arxiv_2606_29148.md](../../sources/papers/gpc_arxiv_2606_29148.md) | `bf542a2bc` → `2600489c3` | 仅 Wiki 导航差异 | 保留待核对 |
| [wiki/entities/paper-viral-humanoid-visual-sim2real.md](../../wiki/entities/paper-viral-humanoid-visual-sim2real.md) | [grail_arxiv_2606_05160.md](../../sources/papers/grail_arxiv_2606_05160.md) | `f7d9241c4` → `1f32bc297` | 仅 Wiki 导航差异 | 保留待核对 |
| [wiki/overview/loco-manip-contact-category-03-generative-data.md](../../wiki/overview/loco-manip-contact-category-03-generative-data.md) | [grail_arxiv_2606_05160.md](../../sources/papers/grail_arxiv_2606_05160.md) | `f7d9241c4` → `1f32bc297` | 仅 Wiki 导航差异 | 保留待核对 |
| [wiki/entities/paper-htd-refine-monocular-hmr.md](../../wiki/entities/paper-htd-refine-monocular-hmr.md) | [gvhmr_arxiv_2409_06662.md](../../sources/papers/gvhmr_arxiv_2409_06662.md) | `8fa98f986` → `3e80c2f74` | 内容或来源身份变化，需逐页核对 | 保留待核对 |
| [wiki/entities/paper-notebook-egomi-learning-active-vision-and-whole-body-mani.md](../../wiki/entities/paper-notebook-egomi-learning-active-vision-and-whole-body-mani.md) | [halomi_arxiv_2606_18772.md](../../sources/papers/halomi_arxiv_2606_18772.md) | `4929faaac` → `1b24e1603` | 内容或来源身份变化，需逐页核对 | 保留待核对 |
| [wiki/entities/paper-loco-manip-161-141-toddlerbot.md](../../wiki/entities/paper-loco-manip-161-141-toddlerbot.md) | [handroid_arxiv_2607_16187.md](../../sources/papers/handroid_arxiv_2607_16187.md) | `197a579f2` → `1b24e1603` | 内容或来源身份变化，需逐页核对 | 保留待核对 |
| [wiki/entities/open-source-humanoid-hardware.md](../../wiki/entities/open-source-humanoid-hardware.md) | [handroid_arxiv_2607_16187.md](../../sources/papers/handroid_arxiv_2607_16187.md) | `197a579f2` → `1b24e1603` | 内容或来源身份变化，需逐页核对 | 保留待核对 |
| [wiki/overview/hub-tactile.md](../../wiki/overview/hub-tactile.md) | [hapmorph_arxiv_2509_05433.md](../../sources/papers/hapmorph_arxiv_2509_05433.md) | `d2cce7179` → `1b24e1603` | 内容或来源身份变化，需逐页核对 | 保留待核对 |
| [wiki/entities/paper-bfm-17-maskedmimic.md](../../wiki/entities/paper-bfm-17-maskedmimic.md) | [harmon_arxiv_2410_12773.md](../../sources/papers/harmon_arxiv_2410_12773.md) | `69931f2a1` → `4368dbdf3` | 仅 Wiki 导航差异 | 保留待核对 |
| [wiki/entities/paper-homebody.md](../../wiki/entities/paper-homebody.md) | [homebody_tml_stanford_2026.md](../../sources/papers/homebody_tml_stanford_2026.md) | `7a8bb183f` → `4368dbdf3` | 仅 Wiki 导航差异 | 保留待核对 |
| [wiki/entities/paper-gpt-policy.md](../../wiki/entities/paper-gpt-policy.md) | [homebody_tml_stanford_2026.md](../../sources/papers/homebody_tml_stanford_2026.md) | `7a8bb183f` → `4368dbdf3` | 仅 Wiki 导航差异 | 保留待核对 |
| [wiki/entities/paper-loco-manip-161-135-amo.md](../../wiki/entities/paper-loco-manip-161-135-amo.md) | [homebody_tml_stanford_2026.md](../../sources/papers/homebody_tml_stanford_2026.md) | `7a8bb183f` → `4368dbdf3` | 仅 Wiki 导航差异 | 保留待核对 |
| [wiki/entities/paper-amp-survey-15-physhsi.md](../../wiki/entities/paper-amp-survey-15-physhsi.md) | [humanoid_amp_survey_13_humanoid_goalkeeper_learning_from_position_condi.md](../../sources/papers/humanoid_amp_survey_13_humanoid_goalkeeper_learning_from_position_condi.md) | `02dc4ce0c` → `1f32bc297` | 仅 Wiki 导航差异 | 保留待核对 |
| [wiki/overview/paper-notebook-category-07-teleoperation.md](../../wiki/overview/paper-notebook-category-07-teleoperation.md) | [humanoid_pnb_ace.md](../../sources/papers/humanoid_pnb_ace.md) | `6ef313f1c` → `1b24e1603` | 内容或来源身份变化，需逐页核对 | 保留待核对 |
| [wiki/overview/paper-notebook-category-06-manipulation.md](../../wiki/overview/paper-notebook-category-06-manipulation.md) | [humanoid_pnb_dextercap.md](../../sources/papers/humanoid_pnb_dextercap.md) | `6ef313f1c` → `1b24e1603` | 内容或来源身份变化，需逐页核对 | 保留待核对 |
| [wiki/overview/paper-notebook-category-12-hardware-design.md](../../wiki/overview/paper-notebook-category-12-hardware-design.md) | [humanoid_pnb_dexterous-teleoperation-of-20-dof-bytedexter-han.md](../../sources/papers/humanoid_pnb_dexterous-teleoperation-of-20-dof-bytedexter-han.md) | `6ef313f1c` → `1b24e1603` | 内容或来源身份变化，需逐页核对 | 保留待核对 |
| [wiki/entities/paper-notebook-lapsurgie-humanoid-robots-performing-surgery-via.md](../../wiki/entities/paper-notebook-lapsurgie-humanoid-robots-performing-surgery-via.md) | [humanoid_surgeon_nature_2026.md](../../sources/papers/humanoid_surgeon_nature_2026.md) | `a472c2d60` → `7333a6ec1` | 内容或来源身份变化，需逐页核对 | 保留待核对 |
| [wiki/entities/paper-notebook-humanoids-in-hospitals-a-technical-study-of-huma.md](../../wiki/entities/paper-notebook-humanoids-in-hospitals-a-technical-study-of-huma.md) | [humanoid_surgeon_nature_2026.md](../../sources/papers/humanoid_surgeon_nature_2026.md) | `a472c2d60` → `7333a6ec1` | 内容或来源身份变化，需逐页核对 | 保留待核对 |
| [wiki/comparisons/humanoid-reference-motion-datasets.md](../../wiki/comparisons/humanoid-reference-motion-datasets.md) | [humantracker_arxiv_2608_13555.md](../../sources/papers/humantracker_arxiv_2608_13555.md) | `b5ea77d6e` → `2600489c3` | 仅 Wiki 导航差异 | 保留待核对 |
| [wiki/entities/paper-gaussian-lic2.md](../../wiki/entities/paper-gaussian-lic2.md) | [i3dgs_arxiv_2607_14481.md](../../sources/papers/i3dgs_arxiv_2607_14481.md) | `2e796fa72` → `dcc85d3bb` | 导航与通读标记差异 | 保留待核对 |
| [wiki/entities/paper-loco-manip-161-131-open-television.md](../../wiki/entities/paper-loco-manip-161-131-open-television.md) | [immersive_social_vr_llm_humanoids_arxiv_2607_07430.md](../../sources/papers/immersive_social_vr_llm_humanoids_arxiv_2607_07430.md) | `ab7cdec14` → `1f32bc297` | 仅 Wiki 导航差异 | 保留待核对 |
| [wiki/entities/paper-loco-manip-161-012-humanplus.md](../../wiki/entities/paper-loco-manip-161-012-humanplus.md) | [immersive_social_vr_llm_humanoids_arxiv_2607_07430.md](../../sources/papers/immersive_social_vr_llm_humanoids_arxiv_2607_07430.md) | `ab7cdec14` → `1f32bc297` | 仅 Wiki 导航差异 | 保留待核对 |
| [wiki/entities/paper-loco-manip-161-062-h2-compact.md](../../wiki/entities/paper-loco-manip-161-062-h2-compact.md) | [immersive_social_vr_llm_humanoids_arxiv_2607_07430.md](../../sources/papers/immersive_social_vr_llm_humanoids_arxiv_2607_07430.md) | `ab7cdec14` → `1f32bc297` | 仅 Wiki 导航差异 | 保留待核对 |
| [wiki/entities/unitree.md](../../wiki/entities/unitree.md) | [immersive_social_vr_llm_humanoids_arxiv_2607_07430.md](../../sources/papers/immersive_social_vr_llm_humanoids_arxiv_2607_07430.md) | `ab7cdec14` → `1f32bc297` | 仅 Wiki 导航差异 | 保留待核对 |
| [wiki/entities/gs-playground.md](../../wiki/entities/gs-playground.md) | [legs_arxiv_2606_01458.md](../../sources/papers/legs_arxiv_2606_01458.md) | `b4e3c60d6` → `2600489c3` | 仅 Wiki 导航差异 | 保留待核对 |
| [wiki/entities/paper-notebook-safehumanoid-vlm-rag-driven-control-of-upper-bod.md](../../wiki/entities/paper-notebook-safehumanoid-vlm-rag-driven-control-of-upper-bod.md) | [lewis_rag_neurips_2020.md](../../sources/papers/lewis_rag_neurips_2020.md) | `69481706c` → `1f32bc297` | 仅 Wiki 导航差异 | 保留待核对 |
| [wiki/entities/lingbot-vla-v2.md](../../wiki/entities/lingbot-vla-v2.md) | [lingbot_vla_v2_tech_report.md](../../sources/papers/lingbot_vla_v2_tech_report.md) | `fe4d2c24c` → `4368dbdf3` | 仅 Wiki 导航差异 | 保留待核对 |
| [wiki/entities/humannet.md](../../wiki/entities/humannet.md) | [lingbot_vla_v2_tech_report.md](../../sources/papers/lingbot_vla_v2_tech_report.md) | `fe4d2c24c` → `4368dbdf3` | 仅 Wiki 导航差异 | 保留待核对 |
| [wiki/entities/paper-greenvla-staged-vla-humanoid.md](../../wiki/entities/paper-greenvla-staged-vla-humanoid.md) | [lingbot_vla_v2_tech_report.md](../../sources/papers/lingbot_vla_v2_tech_report.md) | `fe4d2c24c` → `4368dbdf3` | 仅 Wiki 导航差异 | 保留待核对 |
| [wiki/entities/paper-vlaquantbench.md](../../wiki/entities/paper-vlaquantbench.md) | [macaq_arxiv_2611_10649.md](../../sources/papers/macaq_arxiv_2611_10649.md) | `—` → `673b3188e` | 新增来源或无可比历史 | 已同步内容 |
| [wiki/concepts/terrain-adaptation.md](../../wiki/concepts/terrain-adaptation.md) | [more_mixture_residual_experts_arxiv_2506_08840.md](../../sources/papers/more_mixture_residual_experts_arxiv_2506_08840.md) | `aa34da260` → `1f32bc297` | 仅 Wiki 导航差异 | 保留待核对 |
| [wiki/entities/paper-explicit-stair-geometry-humanoid-locomotion.md](../../wiki/entities/paper-explicit-stair-geometry-humanoid-locomotion.md) | [more_mixture_residual_experts_arxiv_2506_08840.md](../../sources/papers/more_mixture_residual_experts_arxiv_2506_08840.md) | `aa34da260` → `1f32bc297` | 仅 Wiki 导航差异 | 保留待核对 |
| [wiki/entities/paper-ross-dagger.md](../../wiki/entities/paper-ross-dagger.md) | [multi_expert_distillation_locomotion.md](../../sources/papers/multi_expert_distillation_locomotion.md) | `53e239b05` → `1f32bc297` | 仅 Wiki 导航差异 | 保留待核对 |
| [wiki/entities/paper-robot-parkour-learning.md](../../wiki/entities/paper-robot-parkour-learning.md) | [multi_expert_distillation_locomotion.md](../../sources/papers/multi_expert_distillation_locomotion.md) | `53e239b05` → `1f32bc297` | 仅 Wiki 导航差异 | 保留待核对 |
| [wiki/methods/teacher-student-dagger-training.md](../../wiki/methods/teacher-student-dagger-training.md) | [multi_expert_distillation_locomotion.md](../../sources/papers/multi_expert_distillation_locomotion.md) | `53e239b05` → `1f32bc297` | 仅 Wiki 导航差异 | 保留待核对 |
| [wiki/entities/omniretarget-dataset.md](../../wiki/entities/omniretarget-dataset.md) | [omniretarget_arxiv_2509_26633.md](../../sources/papers/omniretarget_arxiv_2509_26633.md) | `df1c9a5c2` → `1f32bc297` | 仅 Wiki 导航差异 | 保留待核对 |
| [wiki/entities/awesome-physical-ai-natnew.md](../../wiki/entities/awesome-physical-ai-natnew.md) | [pai_awesome_2403_04436_h2o-human-to-humanoid-real-time-whol.md](../../sources/papers/pai_awesome_2403_04436_h2o-human-to-humanoid-real-time-whol.md) | `30212f572` → `1f32bc297` | 仅 Wiki 导航差异 | 保留待核对 |
| [wiki/entities/awesome-physical-ai-aichr.md](../../wiki/entities/awesome-physical-ai-aichr.md) | [pai_awesome_2403_04436_h2o-human-to-humanoid-real-time-whol.md](../../sources/papers/pai_awesome_2403_04436_h2o-human-to-humanoid-real-time-whol.md) | `30212f572` → `1f32bc297` | 仅 Wiki 导航差异 | 保留待核对 |
| [wiki/entities/paper-dexemg.md](../../wiki/entities/paper-dexemg.md) | [parallel_attention_myoelectric_gesture_recognition_jbe_2026.md](../../sources/papers/parallel_attention_myoelectric_gesture_recognition_jbe_2026.md) | `cbe9f09b9` → `09f5cd962` | 内容或来源身份变化，需逐页核对 | 保留待核对 |
| [wiki/entities/paper-notebook-humanoid-locomotion-as-next-token-prediction.md](../../wiki/entities/paper-notebook-humanoid-locomotion-as-next-token-prediction.md) | [parkourformer_arxiv_2605_25782.md](../../sources/papers/parkourformer_arxiv_2605_25782.md) | `67eb809de` → `1f32bc297` | 仅 Wiki 导航差异 | 保留待核对 |
| [wiki/concepts/footstep-planning.md](../../wiki/concepts/footstep-planning.md) | [perceptive_bfm_corl_2026.md](../../sources/papers/perceptive_bfm_corl_2026.md) | `de3c07120` → `2600489c3` | 仅 Wiki 导航差异 | 保留待核对 |
| [wiki/queries/vla-deployment-guide.md](../../wiki/queries/vla-deployment-guide.md) | [pi_r2_arxiv_2607_26055.md](../../sources/papers/pi_r2_arxiv_2607_26055.md) | `1b1b12ed3` → `1f32bc297` | 仅 Wiki 导航差异 | 保留待核对 |
| [wiki/entities/paper-loco-manip-161-057-being-0.md](../../wiki/entities/paper-loco-manip-161-057-being-0.md) | [pot_vla_arxiv_2607_18016.md](../../sources/papers/pot_vla_arxiv_2607_18016.md) | `493811013` → `1f32bc297` | 仅 Wiki 导航差异 | 保留待核对 |
| [wiki/entities/paper-progress-reward-modeling-survey.md](../../wiki/entities/paper-progress-reward-modeling-survey.md) | [progress_reward_modeling_survey_arxiv_2607_21655.md](../../sources/papers/progress_reward_modeling_survey_arxiv_2607_21655.md) | `6d96f945f` → `5f9ba6457` | 内容或来源身份变化，需逐页核对 | 已同步内容 |
| [wiki/concepts/sim-vs-real-eval-gap.md](../../wiki/concepts/sim-vs-real-eval-gap.md) | [robolab_arxiv_2604_09860.md](../../sources/papers/robolab_arxiv_2604_09860.md) | `27a8e981d` → `c3edd6a04` | 内容或来源身份变化，需逐页核对 | 已同步内容 |
| [wiki/entities/paper-notebook-ttt-parkour.md](../../wiki/entities/paper-notebook-ttt-parkour.md) | [robottt_nvidia_gear.md](../../sources/papers/robottt_nvidia_gear.md) | `ce45c5675` → `1f32bc297` | 仅 Wiki 导航差异 | 保留待核对 |
| [wiki/comparisons/online-vs-offline-rl.md](../../wiki/comparisons/online-vs-offline-rl.md) | [rove_arxiv_2606_17011.md](../../sources/papers/rove_arxiv_2606_17011.md) | `19fdfd21f` → `1b24e1603` | 内容或来源身份变化，需逐页核对 | 保留待核对 |
| [wiki/entities/paper-rynnworld-4d-rgb-depth-flow.md](../../wiki/entities/paper-rynnworld-4d-rgb-depth-flow.md) | [rynnbrain_1_1_arxiv_2607_17977.md](../../sources/papers/rynnbrain_1_1_arxiv_2607_17977.md) | `a8d79d14c` → `2600489c3` | 仅 Wiki 导航差异 | 保留待核对 |
| [wiki/entities/qwen-vla.md](../../wiki/entities/qwen-vla.md) | [rynnbrain_1_1_arxiv_2607_17977.md](../../sources/papers/rynnbrain_1_1_arxiv_2607_17977.md) | `a8d79d14c` → `2600489c3` | 仅 Wiki 导航差异 | 保留待核对 |
| [wiki/entities/paper-loco-manip-161-022-textop.md](../../wiki/entities/paper-loco-manip-161-022-textop.md) | [safeflow_arxiv_2603_23983.md](../../sources/papers/safeflow_arxiv_2603_23983.md) | `499ffdb7c` → `1f32bc297` | 仅 Wiki 导航差异 | 保留待核对 |
| [wiki/entities/sam3dbody-cpp.md](../../wiki/entities/sam3dbody-cpp.md) | [sam_3d_body_arxiv_2602_15989.md](../../sources/papers/sam_3d_body_arxiv_2602_15989.md) | `e0ec7fdf2` → `3e80c2f74` | 内容或来源身份变化，需逐页核对 | 保留待核对 |
| [wiki/methods/newtons-method.md](../../wiki/methods/newtons-method.md) | [second_order_optimizers.md](../../sources/papers/second_order_optimizers.md) | `aebf9536c` → `3e80c2f74` | 内容或来源身份变化，需逐页核对 | 保留待核对 |
| [wiki/comparisons/second-order-optimizers.md](../../wiki/comparisons/second-order-optimizers.md) | [second_order_optimizers.md](../../sources/papers/second_order_optimizers.md) | `aebf9536c` → `3e80c2f74` | 内容或来源身份变化，需逐页核对 | 保留待核对 |
| [wiki/methods/trajectory-optimization.md](../../wiki/methods/trajectory-optimization.md) | [second_order_optimizers.md](../../sources/papers/second_order_optimizers.md) | `aebf9536c` → `3e80c2f74` | 内容或来源身份变化，需逐页核对 | 保留待核对 |
| [wiki/methods/bfgs.md](../../wiki/methods/bfgs.md) | [second_order_optimizers.md](../../sources/papers/second_order_optimizers.md) | `aebf9536c` → `3e80c2f74` | 内容或来源身份变化，需逐页核对 | 保留待核对 |
| [wiki/entities/curobo.md](../../wiki/entities/curobo.md) | [second_order_optimizers.md](../../sources/papers/second_order_optimizers.md) | `aebf9536c` → `3e80c2f74` | 内容或来源身份变化，需逐页核对 | 保留待核对 |
| [wiki/methods/quasi-newton-bfgs.md](../../wiki/methods/quasi-newton-bfgs.md) | [second_order_optimizers.md](../../sources/papers/second_order_optimizers.md) | `aebf9536c` → `3e80c2f74` | 内容或来源身份变化，需逐页核对 | 保留待核对 |
| [wiki/methods/conjugate-gradient-method.md](../../wiki/methods/conjugate-gradient-method.md) | [second_order_optimizers.md](../../sources/papers/second_order_optimizers.md) | `aebf9536c` → `3e80c2f74` | 内容或来源身份变化，需逐页核对 | 保留待核对 |
| [wiki/methods/truncated-newton.md](../../wiki/methods/truncated-newton.md) | [second_order_optimizers.md](../../sources/papers/second_order_optimizers.md) | `aebf9536c` → `3e80c2f74` | 内容或来源身份变化，需逐页核对 | 保留待核对 |
| [wiki/methods/dsms-contact-implicit-multiple-shooting.md](../../wiki/methods/dsms-contact-implicit-multiple-shooting.md) | [shooting_for_contact_arxiv_2608_03116.md](../../sources/papers/shooting_for_contact_arxiv_2608_03116.md) | `343aebb6a` → `3e80c2f74` | 内容或来源身份变化，需逐页核对 | 保留待核对 |
| [wiki/concepts/lora.md](../../wiki/concepts/lora.md) | [sonic_transfer_frozen_wbc_codec_lora.md](../../sources/papers/sonic_transfer_frozen_wbc_codec_lora.md) | `def6fcbbc` → `2600489c3` | 仅 Wiki 导航差异 | 保留待核对 |
| [wiki/entities/dataset-bfm-phuma.md](../../wiki/entities/dataset-bfm-phuma.md) | [sonic_transfer_frozen_wbc_codec_lora.md](../../sources/papers/sonic_transfer_frozen_wbc_codec_lora.md) | `def6fcbbc` → `2600489c3` | 仅 Wiki 导航差异 | 保留待核对 |
| [wiki/entities/paper-notebook-spark.md](../../wiki/entities/paper-notebook-spark.md) | [spark_skeleton_aligned_retargeting_arxiv_2603_11480.md](../../sources/papers/spark_skeleton_aligned_retargeting_arxiv_2603_11480.md) | `a30db1da4` → `3e80c2f74` | 内容或来源身份变化，需逐页核对 | 保留待核对 |
| [wiki/entities/paper-unified-walk-run-recovery-sdamp.md](../../wiki/entities/paper-unified-walk-run-recovery-sdamp.md) | [stubborn_arxiv_2606_12814.md](../../sources/papers/stubborn_arxiv_2606_12814.md) | `0aaec22b2` → `1f32bc297` | 仅 Wiki 导航差异 | 保留待核对 |
| [wiki/methods/limmt-gqs-motion-curation.md](../../wiki/methods/limmt-gqs-motion-curation.md) | [stubborn_arxiv_2606_12814.md](../../sources/papers/stubborn_arxiv_2606_12814.md) | `0aaec22b2` → `1f32bc297` | 仅 Wiki 导航差异 | 保留待核对 |
| [wiki/entities/paper-tramp-vision-assisted-bipedal-locomotion.md](../../wiki/entities/paper-tramp-vision-assisted-bipedal-locomotion.md) | [t_gmp_terrain_conditioned_generative_motion_priors_arxiv_2606_06944.md](../../sources/papers/t_gmp_terrain_conditioned_generative_motion_priors_arxiv_2606_06944.md) | `e517bd61d` → `1f32bc297` | 仅 Wiki 导航差异 | 保留待核对 |
| [wiki/entities/leju-robotics.md](../../wiki/entities/leju-robotics.md) | [t_gmp_terrain_conditioned_generative_motion_priors_arxiv_2606_06944.md](../../sources/papers/t_gmp_terrain_conditioned_generative_motion_priors_arxiv_2606_06944.md) | `e517bd61d` → `1f32bc297` | 仅 Wiki 导航差异 | 保留待核对 |
| [wiki/entities/paper-quietwalk-humanoid-locomotion.md](../../wiki/entities/paper-quietwalk-humanoid-locomotion.md) | [tactilestep_arxiv_2609_28959.md](../../sources/papers/tactilestep_arxiv_2609_28959.md) | `7bc6fa91f` → `4368dbdf3` | 仅 Wiki 导航差异 | 保留待核对 |
| [wiki/concepts/humanoid-policy-reward-functions.md](../../wiki/concepts/humanoid-policy-reward-functions.md) | [tactilestep_arxiv_2609_28959.md](../../sources/papers/tactilestep_arxiv_2609_28959.md) | `7bc6fa91f` → `4368dbdf3` | 仅 Wiki 导航差异 | 保留待核对 |
| [wiki/entities/paper-tactilestep.md](../../wiki/entities/paper-tactilestep.md) | [tactilestep_arxiv_2609_28959.md](../../sources/papers/tactilestep_arxiv_2609_28959.md) | `7bc6fa91f` → `4368dbdf3` | 仅 Wiki 导航差异 | 保留待核对 |
| [wiki/entities/paper-hrl-stack-06-hdmi.md](../../wiki/entities/paper-hrl-stack-06-hdmi.md) | [teledexter_arxiv_2607_11481.md](../../sources/papers/teledexter_arxiv_2607_11481.md) | `8e467866f` → `1b24e1603` | 内容或来源身份变化，需逐页核对 | 保留待核对 |
| [wiki/methods/in-hand-reorientation.md](../../wiki/methods/in-hand-reorientation.md) | [teledexter_arxiv_2607_11481.md](../../sources/papers/teledexter_arxiv_2607_11481.md) | `8e467866f` → `1b24e1603` | 内容或来源身份变化，需逐页核对 | 保留待核对 |
| [wiki/methods/auto-labeling-pipelines.md](../../wiki/methods/auto-labeling-pipelines.md) | [teleocr_arxiv_2608_12898.md](../../sources/papers/teleocr_arxiv_2608_12898.md) | `ccbc0a1c4` → `1f32bc297` | 仅 Wiki 导航差异 | 保留待核对 |
| [wiki/entities/paper-loco-manip-04-oasis.md](../../wiki/entities/paper-loco-manip-04-oasis.md) | [teleopit_arxiv_2608_01834.md](../../sources/papers/teleopit_arxiv_2608_01834.md) | `228bbf673` → `2600489c3` | 内容或来源身份变化，需逐页核对 | 保留待核对 |
| [wiki/entities/paper-ucag-p.md](../../wiki/entities/paper-ucag-p.md) | [ucag_p_arxiv_2608_26058.md](../../sources/papers/ucag_p_arxiv_2608_26058.md) | `90bd09fd0` → `3e80c2f74` | 内容或来源身份变化，需逐页核对 | 保留待核对 |
| [wiki/entities/paper-egoverse.md](../../wiki/entities/paper-egoverse.md) | [ucag_p_arxiv_2608_26058.md](../../sources/papers/ucag_p_arxiv_2608_26058.md) | `90bd09fd0` → `3e80c2f74` | 内容或来源身份变化，需逐页核对 | 保留待核对 |
| [wiki/entities/paper-dypes-vla.md](../../wiki/entities/paper-dypes-vla.md) | [ucag_p_arxiv_2608_26058.md](../../sources/papers/ucag_p_arxiv_2608_26058.md) | `90bd09fd0` → `3e80c2f74` | 内容或来源身份变化，需逐页核对 | 保留待核对 |
| [wiki/overview/wam-vla-cross-embodiment-9-papers-technology-map.md](../../wiki/overview/wam-vla-cross-embodiment-9-papers-technology-map.md) | [ucag_p_arxiv_2608_26058.md](../../sources/papers/ucag_p_arxiv_2608_26058.md) | `90bd09fd0` → `3e80c2f74` | 内容或来源身份变化，需逐页核对 | 保留待核对 |
| [wiki/overview/hub-motion-retargeting.md](../../wiki/overview/hub-motion-retargeting.md) | [umr_unified_motion_retargeting_arxiv_2609_02134.md](../../sources/papers/umr_unified_motion_retargeting_arxiv_2609_02134.md) | `3e80c2f74` → `2600489c3` | 仅 Wiki 导航差异 | 保留待核对 |
| [wiki/entities/paper-omnicontact-humanoid-loco-manipulation.md](../../wiki/entities/paper-omnicontact-humanoid-loco-manipulation.md) | [umr_unified_motion_retargeting_arxiv_2609_02134.md](../../sources/papers/umr_unified_motion_retargeting_arxiv_2609_02134.md) | `96326a1cb` → `2600489c3` | 内容或来源身份变化，需逐页核对 | 保留待核对 |
| [wiki/entities/paper-loco-manip-161-024-unitracker.md](../../wiki/entities/paper-loco-manip-161-024-unitracker.md) | [unitracker_arxiv_2507_07356.md](../../sources/papers/unitracker_arxiv_2507_07356.md) | `8dfc99e09` → `4368dbdf3` | 仅 Wiki 导航差异 | 保留待核对 |
| [wiki/entities/paper-egoworld.md](../../wiki/entities/paper-egoworld.md) | [vbot_exo2ego_arxiv_2611_16417.md](../../sources/papers/vbot_exo2ego_arxiv_2611_16417.md) | `—` → `dbbf75327` | 新增来源或无可比历史 | 保留待核对 |
| [wiki/concepts/generative-vision-pretraining.md](../../wiki/concepts/generative-vision-pretraining.md) | [vision_banana_arxiv_2604_20329.md](../../sources/papers/vision_banana_arxiv_2604_20329.md) | `c8454056f` → `4368dbdf3` | 仅 Wiki 导航差异 | 保留待核对 |
| [wiki/entities/vision-banana.md](../../wiki/entities/vision-banana.md) | [vision_banana_arxiv_2604_20329.md](../../sources/papers/vision_banana_arxiv_2604_20329.md) | `c8454056f` → `4368dbdf3` | 仅 Wiki 导航差异 | 保留待核对 |
| [wiki/methods/object-detection.md](../../wiki/methods/object-detection.md) | [vision_banana_arxiv_2604_20329.md](../../sources/papers/vision_banana_arxiv_2604_20329.md) | `c8454056f` → `4368dbdf3` | 仅 Wiki 导航差异 | 保留待核对 |
| [wiki/concepts/visual-representation-for-policy.md](../../wiki/concepts/visual-representation-for-policy.md) | [vision_banana_arxiv_2604_20329.md](../../sources/papers/vision_banana_arxiv_2604_20329.md) | `c8454056f` → `4368dbdf3` | 仅 Wiki 导航差异 | 保留待核对 |
| [wiki/entities/paper-ctrl-world.md](../../wiki/entities/paper-ctrl-world.md) | [vlaw_arxiv_2602_12063.md](../../sources/papers/vlaw_arxiv_2602_12063.md) | `—` → `deb41c7e0` | 新增来源或无可比历史 | 已同步内容 |
| [wiki/overview/embodied-station-2026-11-04-57-papers-technology-map.md](../../wiki/overview/embodied-station-2026-11-04-57-papers-technology-map.md) | [w2a_world_priors_action_models_arxiv_2611_02600.md](../../sources/papers/w2a_world_priors_action_models_arxiv_2611_02600.md) | `925e038bb` → `a4f51fa3c` | 内容或来源身份变化，需逐页核对 | 已同步内容 |
| [wiki/entities/paper-openwam.md](../../wiki/entities/paper-openwam.md) | [wb_wam_arxiv_2609_34199.md](../../sources/papers/wb_wam_arxiv_2609_34199.md) | `5ce8a34ac` → `2600489c3` | 仅 Wiki 导航差异 | 保留待核对 |
| [wiki/entities/paper-sa-2606-22136-wh0-generative-world-models-as-scalable-sources.md](../../wiki/entities/paper-sa-2606-22136-wh0-generative-world-models-as-scalable-sources.md) | [wh0_arxiv_2606_22136.md](../../sources/papers/wh0_arxiv_2606_22136.md) | `49eac61e9` → `4368dbdf3` | 仅 Wiki 导航差异 | 保留待核对 |
| [wiki/entities/paper-wan-video.md](../../wiki/entities/paper-wan-video.md) | [wh0_arxiv_2606_22136.md](../../sources/papers/wh0_arxiv_2606_22136.md) | `49eac61e9` → `4368dbdf3` | 仅 Wiki 导航差异 | 保留待核对 |
| [wiki/methods/zest.md](../../wiki/methods/zest.md) | [xmorph_arxiv_2606_30290.md](../../sources/papers/xmorph_arxiv_2606_30290.md) | `330618032` → `3e80c2f74` | 内容或来源身份变化，需逐页核对 | 保留待核对 |
