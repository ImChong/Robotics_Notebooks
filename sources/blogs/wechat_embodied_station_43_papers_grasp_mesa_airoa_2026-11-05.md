# 人能稳稳提起的罐子，换一双机器手怎么提？

> 来源归档（blog / 微信公众号）

- **标题：** 人能稳稳提起的罐子，换一双机器手怎么提？
- **类型：** blog
- **作者：** 具身智能小站（微信公众号）
- **原始链接：** https://mp.weixin.qq.com/s/VS8xbDtQbQTJM3f_WuB9Og
- **发表日期：** 2026-11-05
- **入库日期：** 2026-11-05
- **抓取方式：** `scripts/fetch_wechat_article.py`（Chromium + 微信 UA）
- **原始抓取落盘：** [`sources/raw/wechat_embodied_station_43_papers_grasp_mesa_airoa_2026-11-05.md`](../raw/wechat_embodied_station_43_papers_grasp_mesa_airoa_2026-11-05.md)
- **一句话说明：** 本期主线是把人手示范变成机器手真能执行的动作，并把泛化与评测拆开看：GRASP 在物理仿真里用梯度引导采样重定向手物接触，MESA 分项测摆位 / 物体 / 类别 / 子任务组合泛化，AIRoA-0 用手持夹爪采集长程双臂示范；其余覆盖世界模型与控制、移动导航与人机交互。深读 GRASP，跟进 MESA 与 AIRoA-0。 共 43 篇；**43/43 独立详情节点**（本 ingest **新建 28**、**复用 15**）。

## 43 篇 → 本库节点

| # | 组 | 论文 | arXiv | 开源结论 | wiki |
|---|----|------|-------|----------|------|
| 01 | 灵巧操作与示范学习 | GRASP | [2611.05754](https://arxiv.org/abs/2611.05754) | **已开源** | [paper-grasp](../../wiki/entities/paper-grasp.md) |
| 02 | 策略泛化与评测 | MESA | [2611.05567](https://arxiv.org/abs/2611.05567) | **已开源** | [paper-mesa](../../wiki/entities/paper-mesa.md) |
| 03 | 灵巧操作与示范学习 | AIRoA-0 | [2611.03815](https://arxiv.org/abs/2611.03815) | **已开源** | [paper-airoa-0](../../wiki/entities/paper-airoa-0.md)（**复用**） |
| 04 | 策略泛化与评测 | iMED | [2611.04755](https://arxiv.org/abs/2611.04755) | **已开源** | [paper-imed-multi-endoscope-dataset](../../wiki/entities/paper-imed-multi-endoscope-dataset.md)（**复用**） |
| 05 | 策略泛化与评测 | RoboScope Kit | [2611.03850](https://arxiv.org/abs/2611.03850) | **已开源** | [paper-roboscope-kit](../../wiki/entities/paper-roboscope-kit.md)（**复用**） |
| 06 | 灵巧操作与示范学习 | PyRUA-DataFarm | [2611.05320](https://arxiv.org/abs/2611.05320) | **已开源** | [paper-pyrua-datafarm](../../wiki/entities/paper-pyrua-datafarm.md) |
| 07 | 世界模型与控制 | Action-Conditioned Latent Foresight | [2611.04839](https://arxiv.org/abs/2611.04839) | **待发布** | [paper-action-conditioned-latent-foresight](../../wiki/entities/paper-action-conditioned-latent-foresight.md) |
| 08 | 灵巧操作与示范学习 | MABEL | [2611.04083](https://arxiv.org/abs/2611.04083) | **已开源** | [paper-mabel](../../wiki/entities/paper-mabel.md)（**复用**） |
| 09 | 世界模型与控制 | FloMo | [2611.05062](https://arxiv.org/abs/2611.05062) | **已开源** | [paper-flomo](../../wiki/entities/paper-flomo.md) |
| 10 | 策略泛化与评测 | FlowBench | [2611.03937](https://arxiv.org/abs/2611.03937) | **已开源** | [paper-flowbench](../../wiki/entities/paper-flowbench.md)（**复用**） |
| 11 | 策略泛化与评测 | PolicyIK | [2611.05372](https://arxiv.org/abs/2611.05372) | **已开源** | [paper-policyik](../../wiki/entities/paper-policyik.md) |
| 12 | 策略泛化与评测 | RACE | [2611.05838](https://arxiv.org/abs/2611.05838) | **待发布** | [paper-race](../../wiki/entities/paper-race.md) |
| 13 | 灵巧操作与示范学习 | Robot-Matched Soft Fingers for UMI | [2611.03779](https://arxiv.org/abs/2611.03779) | **已开源** | [paper-robot-matched-soft-fingers-for-umi](../../wiki/entities/paper-robot-matched-soft-fingers-for-umi.md)（**复用**） |
| 14 | 策略泛化与评测 | VagueBench | [2611.03765](https://arxiv.org/abs/2611.03765) | **已开源** | [paper-vaguebench](../../wiki/entities/paper-vaguebench.md)（**复用**） |
| 15 | 灵巧操作与示范学习 | RoboTryOn | [2611.05163](https://arxiv.org/abs/2611.05163) | **已开源** | [paper-robotryon](../../wiki/entities/paper-robotryon.md) |
| 16 | 世界模型与控制 | Hybrid Particle-Voxel Dynamics | [2611.04492](https://arxiv.org/abs/2611.04492) | **已开源** | [paper-hybrid-particle-voxel-dynamics](../../wiki/entities/paper-hybrid-particle-voxel-dynamics.md)（**复用**） |
| 17 | 策略泛化与评测 | Proactive Human-in-the-Loop Correction | [2611.05667](https://arxiv.org/abs/2611.05667) | **项目页未列 GitHub** | [paper-proactive-human-in-the-loop-correction](../../wiki/entities/paper-proactive-human-in-the-loop-correction.md) |
| 18 | 策略泛化与评测 | GSBR | [2611.05295](https://arxiv.org/abs/2611.05295) | **已开源** | [paper-gsbr](../../wiki/entities/paper-gsbr.md) |
| 19 | 世界模型与控制 | HiLo-WAM | [2611.05023](https://arxiv.org/abs/2611.05023) | **已开源** | [paper-hilo-wam](../../wiki/entities/paper-hilo-wam.md) |
| 20 | 灵巧操作与示范学习 | Human-to-Robot Transfer via VLA Bridging | [2611.04391](https://arxiv.org/abs/2611.04391) | **已开源** | [paper-human-to-robot-transfer-via-vla-bridging](../../wiki/entities/paper-human-to-robot-transfer-via-vla-bridging.md)（**复用**） |
| 21 | 灵巧操作与示范学习 | DexHOT | [2611.05119](https://arxiv.org/abs/2611.05119) | **部分开源** | [paper-dexhot](../../wiki/entities/paper-dexhot.md) |
| 22 | 世界模型与控制 | Future Representation Distillation | [2611.04527](https://arxiv.org/abs/2611.04527) | **待发布** | [paper-future-representation-distillation](../../wiki/entities/paper-future-representation-distillation.md) |
| 23 | 灵巧操作与示范学习 | ReflexArc | [2611.04424](https://arxiv.org/abs/2611.04424) | **已开源** | [paper-reflexarc](../../wiki/entities/paper-reflexarc.md)（**复用**） |
| 24 | 世界模型与控制 | Closed-Loop Ocean Sampling | [2611.05776](https://arxiv.org/abs/2611.05776) | **已开源** | [paper-closed-loop-ocean-sampling](../../wiki/entities/paper-closed-loop-ocean-sampling.md) |
| 25 | 移动导航与人机交互 | WayTrace | [2611.05129](https://arxiv.org/abs/2611.05129) | **项目页未列 GitHub** | [paper-waytrace](../../wiki/entities/paper-waytrace.md) |
| 26 | 世界模型与控制 | Barrier-State Safe Navigation | [2611.05617](https://arxiv.org/abs/2611.05617) | **已开源** | [paper-barrier-state-safe-navigation](../../wiki/entities/paper-barrier-state-safe-navigation.md) |
| 27 | 世界模型与控制 | RAIPAL | [2611.04334](https://arxiv.org/abs/2611.04334) | **已开源** | [paper-raipal](../../wiki/entities/paper-raipal.md) |
| 28 | 移动导航与人机交互 | Practical Causal Framework for Mobile Robots | [2611.04027](https://arxiv.org/abs/2611.04027) | **已开源** | [paper-practical-causal-framework-for-mobile-robots](../../wiki/entities/paper-practical-causal-framework-for-mobile-robots.md)（**复用**） |
| 29 | 灵巧操作与示范学习 | GUILD | [2611.05368](https://arxiv.org/abs/2611.05368) | **项目页未列 GitHub** | [paper-guild](../../wiki/entities/paper-guild.md) |
| 30 | 策略泛化与评测 | Action-Token Supervision | [2611.05249](https://arxiv.org/abs/2611.05249) | **项目页未列 GitHub** | [paper-action-token-supervision](../../wiki/entities/paper-action-token-supervision.md) |
| 31 | 策略泛化与评测 | Asynchronous Robot Policies | [2611.05155](https://arxiv.org/abs/2611.05155) | **待发布** | [paper-asynchronous-robot-policies](../../wiki/entities/paper-asynchronous-robot-policies.md) |
| 32 | 世界模型与控制 | PrimeSim | [2611.04870](https://arxiv.org/abs/2611.04870) | **部分开源** | [paper-primesim](../../wiki/entities/paper-primesim.md) |
| 33 | 世界模型与控制 | Geometric Abstraction for Agentic Policies | [2611.05058](https://arxiv.org/abs/2611.05058) | **已开源** | [paper-geometric-abstraction-for-agentic-policies](../../wiki/entities/paper-geometric-abstraction-for-agentic-policies.md) |
| 34 | 灵巧操作与示范学习 | ELA | [2611.05571](https://arxiv.org/abs/2611.05571) | **项目页未列 GitHub** | [paper-ela](../../wiki/entities/paper-ela.md) |
| 35 | 策略泛化与评测 | CRONOS | [2611.05080](https://arxiv.org/abs/2611.05080) | **项目页未列 GitHub** | [paper-cronos](../../wiki/entities/paper-cronos.md) |
| 36 | 策略泛化与评测 | CANDO | [2611.04543](https://arxiv.org/abs/2611.04543) | **已开源** | [paper-cando-residual-rl](../../wiki/entities/paper-cando-residual-rl.md)（**复用**） |
| 37 | 移动导航与人机交互 | FEIJOA | [2611.04080](https://arxiv.org/abs/2611.04080) | **已开源** | [paper-feijoa](../../wiki/entities/paper-feijoa.md)（**复用**） |
| 38 | 移动导航与人机交互 | Expressive Swarm Perceptual Gaps | [2611.05244](https://arxiv.org/abs/2611.05244) | **部分开源** | [paper-expressive-swarm-perceptual-gaps](../../wiki/entities/paper-expressive-swarm-perceptual-gaps.md) |
| 39 | 灵巧操作与示范学习 | Shape Servoing Tutorial Part II | [2611.03825](https://arxiv.org/abs/2611.03825) | **已开源** | [paper-shape-servoing-tutorial-part-ii](../../wiki/entities/paper-shape-servoing-tutorial-part-ii.md)（**复用**） |
| 40 | 灵巧操作与示范学习 | DreamX | [2611.05377](https://arxiv.org/abs/2611.05377) | **待发布** | [paper-dreamx](../../wiki/entities/paper-dreamx.md) |
| 41 | 移动导航与人机交互 | OdorSpaces | [2611.05738](https://arxiv.org/abs/2611.05738) | **项目页未列 GitHub** | [paper-odorspaces](../../wiki/entities/paper-odorspaces.md) |
| 42 | 移动导航与人机交互 | For You, Not for Everyone | [2611.05001](https://arxiv.org/abs/2611.05001) | **已开源** | [paper-for-you-not-for-everyone](../../wiki/entities/paper-for-you-not-for-everyone.md) |
| 43 | 移动导航与人机交互 | Just Between Us | [2611.04692](https://arxiv.org/abs/2611.04692) | **已开源** | [paper-just-between-us](../../wiki/entities/paper-just-between-us.md)（**复用**） |

## 对 wiki 的映射

- **43/43 独立详情节点**；同 arXiv 不重复造页（复用节点已补本文来源与地图关联）。
- 阅读坐标：[接触丰富的手物重定向、分项泛化评测与手持夹爪示范：具身智能小站 11-05 期 43 篇阅读坐标](../../wiki/overview/embodied-station-2026-11-05-43-papers-technology-map.md)
- 开源结论口径：GitHub 仓库用 `git ls-remote` 核实可达；项目页用 curl 读取扫描 GitHub 链接与 "coming soon"；「项目页未列 GitHub」表示入库日页面无代码链接且未见开源承诺字样。

## 当前提炼状态

- [x] 公众号正文抓取（2026-11-05）
- [x] 43 篇独立节点（28 新建 / 15 复用）
- [x] 项目页 / 仓库开源状态核查（步骤 2.5）
- [ ] 原文实验表格核验（节点内数字仅为公众号转述）
