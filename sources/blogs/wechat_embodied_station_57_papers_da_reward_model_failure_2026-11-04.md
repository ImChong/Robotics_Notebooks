# 机器人做错了，奖励模型为什么还在加分？｜附代码与评测入口

> 来源归档（blog / 微信公众号）

- **标题：** 机器人做错了，奖励模型为什么还在加分？｜附代码与评测入口
- **类型：** blog
- **作者：** 具身智能小站（微信公众号）
- **原始链接：** https://mp.weixin.qq.com/s/Em5CuCvWGpaWc_3R97YnYg
- **发表日期：** 2026-11-04
- **入库日期：** 2026-11-04
- **抓取方式：** `scripts/fetch_wechat_article.py`（Chromium + 微信 UA）
- **原始抓取落盘：** [`sources/raw/wechat_embodied_station_57_papers_da_reward_model_failure_2026-11-04.md`](../raw/wechat_embodied_station_57_papers_da_reward_model_failure_2026-11-04.md)
- **一句话说明：** 本期主线是别让奖励与评测把失败看成进步：D&A 先描述视频中实际变化再按成功标准评分，ODE Policy 直接学习动作变化的向量场，uni-robot-agent 用统一接口让开放权重模型参与机器人操作；其余覆盖策略后训练、触觉与灵巧、移动操作、数据平台与评测。深读 D&A，跟进 ODE Policy 与 uni-robot-agent。 共 57 篇；**57/57 独立详情节点**（本 ingest **新建 54**、**复用 3**）。

## 57 篇 → 本库节点

| # | 组 | 论文 | arXiv | 开源结论 | wiki |
|---|----|------|-------|----------|------|
| 01 | 奖励学习与策略后训练 | D&A | [2611.03622](https://arxiv.org/abs/2611.03622) | **已开源** | [paper-dna](../../wiki/entities/paper-dna.md) |
| 02 | 奖励学习与策略后训练 | ODE Policy | [2611.01729](https://arxiv.org/abs/2611.01729) | **已开源** | [paper-ode-policy-action-dynamics](../../wiki/entities/paper-ode-policy-action-dynamics.md) |
| 03 | 奖励学习与策略后训练 | Democratizing Robot-Use Agents | [2611.01560](https://arxiv.org/abs/2611.01560) | **已开源** | [paper-uni-robot-agent](../../wiki/entities/paper-uni-robot-agent.md)（**复用**） |
| 04 | 世界模型与感知 | SyncHeC | [2611.02999](https://arxiv.org/abs/2611.02999) | **已开源** | [paper-synchec](../../wiki/entities/paper-synchec.md) |
| 05 | 触觉与灵巧操作 | ForceLDM | [2611.03294](https://arxiv.org/abs/2611.03294) | **已开源** | [paper-forceldm](../../wiki/entities/paper-forceldm.md) |
| 06 | 评测、数据与基准 | Beyond Validation Loss | [2611.01565](https://arxiv.org/abs/2611.01565) | **已开源** | [paper-policy-coverage-reliability](../../wiki/entities/paper-policy-coverage-reliability.md)（**复用**） |
| 07 | 评测、数据与基准 | RoboColosseum | [2611.02497](https://arxiv.org/abs/2611.02497) | **项目页未列 GitHub** | [paper-robocolosseum](../../wiki/entities/paper-robocolosseum.md) |
| 08 | 数据采集与跨形态迁移 | MABEL | [2611.04083](https://arxiv.org/abs/2611.04083) | **部分开源** | [paper-mabel](../../wiki/entities/paper-mabel.md) |
| 09 | 人形运动与教程 | VerifLM | [2611.03268](https://arxiv.org/abs/2611.03268) | **项目页未列 GitHub** | [paper-veriflm](../../wiki/entities/paper-veriflm.md) |
| 10 | 移动操作与导航 | OpenNWM | [2611.01844](https://arxiv.org/abs/2611.01844) | **已开源** | [paper-opennwm](../../wiki/entities/paper-opennwm.md) |
| 11 | 奖励学习与策略后训练 | Route-Guided CBF-QP Repair | [2611.01713](https://arxiv.org/abs/2611.01713) | **已开源** | [paper-route-guided-cbf-qp-repair](../../wiki/entities/paper-route-guided-cbf-qp-repair.md) |
| 12 | 评测、数据与基准 | Rollout then Recycle | [2611.03168](https://arxiv.org/abs/2611.03168) | **项目页未列 GitHub** | [paper-rollout-then-recycle](../../wiki/entities/paper-rollout-then-recycle.md) |
| 13 | 评测、数据与基准 | KlaskTron | [2611.03138](https://arxiv.org/abs/2611.03138) | **项目页未列 GitHub** | [paper-klasktron](../../wiki/entities/paper-klasktron.md) |
| 14 | 评测、数据与基准 | KlaskTron-IL | [2611.03091](https://arxiv.org/abs/2611.03091) | **已开源** | [paper-klasktron-il-adversarial-benchmark](../../wiki/entities/paper-klasktron-il-adversarial-benchmark.md) |
| 15 | 奖励学习与策略后训练 | Iterative Gumbel Planning | [2611.02236](https://arxiv.org/abs/2611.02236) | **已开源** | [paper-iterative-gumbel-planning](../../wiki/entities/paper-iterative-gumbel-planning.md) |
| 16 | 奖励学习与策略后训练 | Value Forcing | [2611.02160](https://arxiv.org/abs/2611.02160) | **已开源** | [paper-value-forcing](../../wiki/entities/paper-value-forcing.md) |
| 17 | 奖励学习与策略后训练 | RAFT | [2611.04368](https://arxiv.org/abs/2611.04368) | **部分开源** | [paper-raft-reasoning-aligned-finetuning](../../wiki/entities/paper-raft-reasoning-aligned-finetuning.md) |
| 18 | 移动操作与导航 | Feijoa | [2611.04080](https://arxiv.org/abs/2611.04080) | **项目页未列 GitHub** | [paper-feijoa](../../wiki/entities/paper-feijoa.md) |
| 19 | 评测、数据与基准 | RoboScope Kit | [2611.03850](https://arxiv.org/abs/2611.03850) | **项目页未列 GitHub** | [paper-roboscope-kit](../../wiki/entities/paper-roboscope-kit.md) |
| 20 | 移动操作与导航 | TAMPAgent | [2611.03584](https://arxiv.org/abs/2611.03584) | **项目页未列 GitHub** | [paper-tampagent](../../wiki/entities/paper-tampagent.md) |
| 21 | 奖励学习与策略后训练 | Energy-Based Policy Orchestration | [2611.02546](https://arxiv.org/abs/2611.02546) | **项目页未列 GitHub** | [paper-energy-based-policy-orchestration](../../wiki/entities/paper-energy-based-policy-orchestration.md) |
| 22 | 触觉与灵巧操作 | ReflexArc | [2611.04424](https://arxiv.org/abs/2611.04424) | **项目页未列 GitHub** | [paper-reflexarc](../../wiki/entities/paper-reflexarc.md) |
| 23 | 触觉与灵巧操作 | ReContact | [2611.02968](https://arxiv.org/abs/2611.02968) | **项目页未列 GitHub** | [paper-recontact](../../wiki/entities/paper-recontact.md) |
| 24 | 触觉与灵巧操作 | Self-Directed Dynamics Gap Closing | [2611.02691](https://arxiv.org/abs/2611.02691) | **项目页未列 GitHub** | [paper-self-directed-dynamics-gap-closing](../../wiki/entities/paper-self-directed-dynamics-gap-closing.md) |
| 25 | 评测、数据与基准 | iMED | [2611.04755](https://arxiv.org/abs/2611.04755) | **项目页未列 GitHub** | [paper-imed-multi-endoscope-dataset](../../wiki/entities/paper-imed-multi-endoscope-dataset.md) |
| 26 | 世界模型与感知 | Hybrid Particle-Voxel Dynamics | [2611.04492](https://arxiv.org/abs/2611.04492) | **已开源** | [paper-hybrid-particle-voxel-dynamics](../../wiki/entities/paper-hybrid-particle-voxel-dynamics.md) |
| 27 | 移动操作与导航 | Practical Causal Framework for Mobile Robots | [2611.04027](https://arxiv.org/abs/2611.04027) | **已开源** | [paper-practical-causal-framework-for-mobile-robots](../../wiki/entities/paper-practical-causal-framework-for-mobile-robots.md) |
| 28 | 评测、数据与基准 | FlowBench | [2611.03937](https://arxiv.org/abs/2611.03937) | **已开源** | [paper-flowbench](../../wiki/entities/paper-flowbench.md) |
| 29 | 触觉与灵巧操作 | BeyondContact | [2611.03059](https://arxiv.org/abs/2611.03059) | **已开源** | [paper-beyondcontact](../../wiki/entities/paper-beyondcontact.md) |
| 30 | 触觉与灵巧操作 | SomaAlign | [2611.02794](https://arxiv.org/abs/2611.02794) | **已开源** | [paper-somaalign](../../wiki/entities/paper-somaalign.md) |
| 31 | 数据采集与跨形态迁移 | Human-to-Robot Transfer via VLA Bridging | [2611.04391](https://arxiv.org/abs/2611.04391) | **已开源** | [paper-human-to-robot-transfer-via-vla-bridging](../../wiki/entities/paper-human-to-robot-transfer-via-vla-bridging.md) |
| 32 | 数据采集与跨形态迁移 | AIRoA-0 | [2611.03815](https://arxiv.org/abs/2611.03815) | **部分开源** | [paper-airoa-0](../../wiki/entities/paper-airoa-0.md) |
| 33 | 数据采集与跨形态迁移 | Pixel | [2611.03648](https://arxiv.org/abs/2611.03648) | **待发布** | [paper-pixel](../../wiki/entities/paper-pixel.md) |
| 34 | 数据采集与跨形态迁移 | Pixel-UMI | [2611.03647](https://arxiv.org/abs/2611.03647) | **待发布** | [paper-pixel-umi](../../wiki/entities/paper-pixel-umi.md) |
| 35 | 奖励学习与策略后训练 | Beyond Policy Patches | [2611.02777](https://arxiv.org/abs/2611.02777) | **项目页未列 GitHub** | [paper-beyond-policy-patches](../../wiki/entities/paper-beyond-policy-patches.md) |
| 36 | 奖励学习与策略后训练 | W2A | [2611.02600](https://arxiv.org/abs/2611.02600) | **项目页未列 GitHub** | [paper-w2a-world-priors-action-models](../../wiki/entities/paper-w2a-world-priors-action-models.md) |
| 37 | 触觉与灵巧操作 | TF²-Steering | [2611.01483](https://arxiv.org/abs/2611.01483) | **待发布** | [paper-tf2-steering-tactile-forecast](../../wiki/entities/paper-tf2-steering-tactile-forecast.md) |
| 38 | 人形运动与教程 | Shape Servoing Tutorial Part II | [2611.03825](https://arxiv.org/abs/2611.03825) | **已开源** | [paper-shape-servoing-tutorial-part-ii](../../wiki/entities/paper-shape-servoing-tutorial-part-ii.md) |
| 39 | 触觉与灵巧操作 | OpenZilo | [2611.03410](https://arxiv.org/abs/2611.03410) | **已开源** | [paper-openzilo](../../wiki/entities/paper-openzilo.md) |
| 40 | 触觉与灵巧操作 | Robot-Matched Soft Fingers for UMI | [2611.03779](https://arxiv.org/abs/2611.03779) | **已开源** | [paper-robot-matched-soft-fingers-for-umi](../../wiki/entities/paper-robot-matched-soft-fingers-for-umi.md) |
| 41 | 触觉与灵巧操作 | PanoTac | [2611.03520](https://arxiv.org/abs/2611.03520) | **项目页未列 GitHub** | [paper-panotac](../../wiki/entities/paper-panotac.md) |
| 42 | 移动操作与导航 | SNAP | [2611.03498](https://arxiv.org/abs/2611.03498) | **项目页未列 GitHub** | [paper-snap](../../wiki/entities/paper-snap.md) |
| 43 | 世界模型与感知 | AnchorHEC | [2611.02884](https://arxiv.org/abs/2611.02884) | **待发布** | [paper-anchorhec](../../wiki/entities/paper-anchorhec.md) |
| 44 | 世界模型与感知 | Gnomon4D | [2611.02474](https://arxiv.org/abs/2611.02474) | **已开源** | [paper-gnomon4d](../../wiki/entities/paper-gnomon4d.md) |
| 45 | 评测、数据与基准 | DexArena | [2611.02629](https://arxiv.org/abs/2611.02629) | **已开源** | [paper-dexarena](../../wiki/entities/paper-dexarena.md) |
| 46 | 评测、数据与基准 | VagueBench | [2611.03765](https://arxiv.org/abs/2611.03765) | **已开源** | [paper-vaguebench](../../wiki/entities/paper-vaguebench.md) |
| 47 | 评测、数据与基准 | When Robots Outweigh Objects | [2611.03501](https://arxiv.org/abs/2611.03501) | **项目页未列 GitHub** | [paper-inertia-mismatch](../../wiki/entities/paper-inertia-mismatch.md) |
| 48 | 奖励学习与策略后训练 | CANDO | [2611.04543](https://arxiv.org/abs/2611.04543) | **项目页未列 GitHub** | [paper-cando-residual-rl](../../wiki/entities/paper-cando-residual-rl.md) |
| 49 | 世界模型与感知 | CoReMatch | [2611.03932](https://arxiv.org/abs/2611.03932) | **已开源** | [paper-corematch](../../wiki/entities/paper-corematch.md) |
| 50 | 移动操作与导航 | Safe-Ballbot | [2611.03307](https://arxiv.org/abs/2611.03307) | **部分开源** | [paper-safe-ballbot](../../wiki/entities/paper-safe-ballbot.md) |
| 51 | 人形运动与教程 | MAPO | [2611.02613](https://arxiv.org/abs/2611.02613) | **已开源** | [paper-mapo-motion-tracking](../../wiki/entities/paper-mapo-motion-tracking.md)（**复用**） |
| 52 | 移动操作与导航 | TRAILS | [2611.03254](https://arxiv.org/abs/2611.03254) | **已开源** | [paper-trails](../../wiki/entities/paper-trails.md) |
| 53 | 移动操作与导航 | Local Solution Repair MAPD | [2611.02090](https://arxiv.org/abs/2611.02090) | **已开源** | [paper-local-solution-repair-mapd](../../wiki/entities/paper-local-solution-repair-mapd.md) |
| 54 | 世界模型与感知 | DynaSolver | [2611.03595](https://arxiv.org/abs/2611.03595) | **项目页未列 GitHub** | [paper-dynasolver](../../wiki/entities/paper-dynasolver.md) |
| 55 | 人形运动与教程 | Blockchain DAO Swarm Access Control | [2611.03975](https://arxiv.org/abs/2611.03975) | **部分开源** | [paper-blockchain-dao-swarm-access-control](../../wiki/entities/paper-blockchain-dao-swarm-access-control.md) |
| 56 | 评测、数据与基准 | Source-Resolved Collective Cognition | [2611.02693](https://arxiv.org/abs/2611.02693) | **项目页未列 GitHub** | [paper-source-resolved-collective-cognition](../../wiki/entities/paper-source-resolved-collective-cognition.md) |
| 57 | 人形运动与教程 | Just Between Us | [2611.04692](https://arxiv.org/abs/2611.04692) | **已开源** | [paper-just-between-us](../../wiki/entities/paper-just-between-us.md) |

## 对 wiki 的映射

- **57/57 独立详情节点**；同 arXiv 不重复造页（复用节点已补本文来源与地图关联）。
- 阅读坐标：[VLM 奖励评估、动作动力学与开放权重机器人智能体：具身智能小站 11-04 期 57 篇阅读坐标](../../wiki/overview/embodied-station-2026-11-04-57-papers-technology-map.md)
- 开源结论口径：GitHub 仓库用 `git ls-remote` 核实可达；项目页用 curl 读取扫描 GitHub 链接与 "coming soon"；「项目页未列 GitHub」表示入库日页面无代码链接且未见开源承诺字样。

## 当前提炼状态

- [x] 公众号正文抓取（2026-11-04）
- [x] 57 篇独立节点（54 新建 / 3 复用）
- [x] 项目页 / 仓库开源状态核查（步骤 2.5）
- [ ] 原文实验表格核验（节点内数字仅为公众号转述）
