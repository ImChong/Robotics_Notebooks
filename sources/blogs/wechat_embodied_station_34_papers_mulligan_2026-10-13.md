# 机器人越熟练，采集数据越要盯住失败｜附代码与项目入口

> 来源归档（blog / 微信公众号）

- **标题：** 机器人越熟练，采集数据越要盯住失败｜附代码与项目入口
- **类型：** blog
- **作者：** 具身智能小站（微信公众号）
- **原始链接：** https://mp.weixin.qq.com/s/mGe8FNbW8HXnr5TUb7PHGw
- **发表日期：** 2026-10-13
- **入库日期：** 2026-10-14
- **抓取方式：** `scripts/fetch_wechat_article.py`（Chromium + 微信 UA）
- **原始抓取落盘：** [`sources/raw/wechat_embodied_station_34_papers_mulligan_2026-10-13.md`](../raw/wechat_embodied_station_34_papers_mulligan_2026-10-13.md)
- **一句话说明：** 真机策略越熟练，均匀采集越浪费预算：把采集起点、价值函数与人工介入作为可优化变量，是部署后持续学习的主线。本期深读 Mulligan，跟进 OpenWAM 与 EigenDEXplore。 共 34 篇；**34/34 独立详情节点**（本 ingest **新建 25**、**复用 9**）。

## 34 篇 → 本库节点

| # | 组 | 论文 | arXiv | 开源结论 | wiki |
|---|----|------|-------|----------|------|
| 01 | 强化学习与数据采集 | Mulligan | [2610.05882](https://arxiv.org/abs/2610.05882) | **已开源** | [paper-mulligan-performance-guided-data-collection](../../wiki/entities/paper-mulligan-performance-guided-data-collection.md) |
| 02 | 世界模型与仿真 | WAMJET | [2610.03797](https://arxiv.org/abs/2610.03797) | **已开源** | [paper-wamjet-wam-acceleration-harness](../../wiki/entities/paper-wamjet-wam-acceleration-harness.md) |
| 03 | 世界模型与仿真 | AffordCraft | [2610.06643](https://arxiv.org/abs/2610.06643) | **已开源** | [paper-affordcraft-task-ready-simulation-assets](../../wiki/entities/paper-affordcraft-task-ready-simulation-assets.md) |
| 04 | 策略与后训练 | RobotUse | [2610.04929](https://arxiv.org/abs/2610.04929) | **已开源** | [paper-robotuse-allocating-computation-context-decisions](../../wiki/entities/paper-robotuse-allocating-computation-context-decisions.md) |
| 05 | 世界模型与仿真 | Video2World | [2610.04432](https://arxiv.org/abs/2610.04432) | **已开源** | [paper-video2world-coding-agents-benchmark](../../wiki/entities/paper-video2world-coding-agents-benchmark.md) |
| 06 | 灵巧操作 | EyeRobot 2.0 | [2610.03710](https://arxiv.org/abs/2610.03710) | **已开源** | [paper-eyerobot-2-active-gaze-manipulation](../../wiki/entities/paper-eyerobot-2-active-gaze-manipulation.md) |
| 07 | 人形与移动操作 | MobiAgent | [2610.03476](https://arxiv.org/abs/2610.03476) | **部分开源** | [paper-mobiagent-dual-loop-self-improvement](../../wiki/entities/paper-mobiagent-dual-loop-self-improvement.md) |
| 08 | 强化学习与数据采集 | UniIntervene++ | [2610.03620](https://arxiv.org/abs/2610.03620) | **已开源** | [paper-uniintervene-plus-plus-adaptive-intervention](../../wiki/entities/paper-uniintervene-plus-plus-adaptive-intervention.md) |
| 09 | 策略与后训练 | VLA 后训练数据增强 | [2610.05994](https://arxiv.org/abs/2610.05994) | **已开源** | [paper-vla-augm-data-augmentation-post-training](../../wiki/entities/paper-vla-augm-data-augmentation-post-training.md) |
| 10 | 灵巧操作 | PEARS | [2610.08784](https://arxiv.org/abs/2610.08784) | **项目页未列 GitHub** | [paper-pears-tactile-failure-reasoning-diffusion-steering](../../wiki/entities/paper-pears-tactile-failure-reasoning-diffusion-steering.md) |
| 11 | 导航、足式与规划 | CUSP | [2610.07882](https://arxiv.org/abs/2610.07882) | **待发布** | [paper-cusp-survival-hazard-alarms-offroad](../../wiki/entities/paper-cusp-survival-hazard-alarms-offroad.md) |
| 12 | VLA 与动作生成 | StairVLA | [2610.07756](https://arxiv.org/abs/2610.07756) | **已开源** | [paper-stairvla-stage-aware-hierarchical-action](../../wiki/entities/paper-stairvla-stage-aware-hierarchical-action.md) |
| 13 | 灵巧操作 | GOTT | [2610.03861](https://arxiv.org/abs/2610.03861) | **待发布** | [paper-gott-cross-embodiment-contact-primitive](../../wiki/entities/paper-gott-cross-embodiment-contact-primitive.md) |
| 14 | 世界模型与仿真 | SimForcing | [2610.06598](https://arxiv.org/abs/2610.06598) | **已开源** | [paper-simforcing-simulation-motion-priors](../../wiki/entities/paper-simforcing-simulation-motion-priors.md) |
| 15 | VLA 与动作生成 | EAPN | [2610.06090](https://arxiv.org/abs/2610.06090) | **项目页未列 GitHub** | [paper-eapn-execution-aligned-progressive-noise](../../wiki/entities/paper-eapn-execution-aligned-progressive-noise.md) |
| 16 | 世界模型与仿真 | SDOB | [2610.04896](https://arxiv.org/abs/2610.04896) | **待发布** | [paper-sdob-sampling-based-disturbance-observers](../../wiki/entities/paper-sdob-sampling-based-disturbance-observers.md) |
| 17 | 世界模型与仿真 | TAPDreamer | [2610.06814](https://arxiv.org/abs/2610.06814) | **待发布** | [paper-tapdreamer-adversarial-patches-wam](../../wiki/entities/paper-tapdreamer-adversarial-patches-wam.md) |
| 18 | VLA 与动作生成 | CNP-Flow | [2610.05349](https://arxiv.org/abs/2610.05349) | **项目页未列 GitHub** | [paper-cnp-flow-conditional-source-distribution](../../wiki/entities/paper-cnp-flow-conditional-source-distribution.md) |
| 19 | 世界模型与仿真 | TTT-RM | [2610.04701](https://arxiv.org/abs/2610.04701) | **项目页未列 GitHub** | [paper-ttt-rm-residual-memory](../../wiki/entities/paper-ttt-rm-residual-memory.md) |
| 20 | 世界模型与仿真 | DepthWorld | [2610.08780](https://arxiv.org/abs/2610.08780) | **已开源** | [paper-depthworld-3d-world-model](../../wiki/entities/paper-depthworld-3d-world-model.md) |
| 21 | 策略与后训练 | VeriFine | [2610.08761](https://arxiv.org/abs/2610.08761) | **待发布** | [paper-verifine-scaling-verification-self-improvement](../../wiki/entities/paper-verifine-scaling-verification-self-improvement.md) |
| 22 | 世界模型与仿真 | KineWorld | [2610.06349](https://arxiv.org/abs/2610.06349) | **已开源** | [paper-kineworld-action-induced-transport-fields](../../wiki/entities/paper-kineworld-action-induced-transport-fields.md) |
| 23 | 世界模型与仿真 | 技能级世界模型的流策略动作 | [2610.04767](https://arxiv.org/abs/2610.04767) | **项目页未列 GitHub** | [paper-flow-policies-skill-level-world-models](../../wiki/entities/paper-flow-policies-skill-level-world-models.md) |
| 24 | 灵巧操作 | PatternDex | [2610.04765](https://arxiv.org/abs/2610.04765) | **待发布** | [paper-patterndex-interaction-patterns-bimanual-dexterous](../../wiki/entities/paper-patterndex-interaction-patterns-bimanual-dexterous.md) |
| 25 | 多机器人与评测 | QUEEN | [2610.03695](https://arxiv.org/abs/2610.03695) | **已开源** | [paper-queen-chess-language-model-explanations](../../wiki/entities/paper-queen-chess-language-model-explanations.md) |
| 26 | 世界模型与仿真 | OpenWAM | [2610.07922](https://arxiv.org/abs/2610.07922) | **已开源** | [paper-openwam-composable-world-action-models](../../wiki/entities/paper-openwam-composable-world-action-models.md)（**复用**） |
| 27 | 灵巧操作 | EigenDEXplore | [2610.07681](https://arxiv.org/abs/2610.07681) | **已开源** | [paper-eigendexplore](../../wiki/entities/paper-eigendexplore.md)（**复用**） |
| 28 | 策略与后训练 | FreeSpeed | [2610.05734](https://arxiv.org/abs/2610.05734) | **项目页未列 GitHub** | [paper-freespeed](../../wiki/entities/paper-freespeed.md)（**复用**） |
| 29 | 策略与后训练 | QF3 | [2610.08789](https://arxiv.org/abs/2610.08789) | **项目页未列 GitHub** | [paper-qf3-filtered-q-gradients](../../wiki/entities/paper-qf3-filtered-q-gradients.md)（**复用**） |
| 30 | 策略与后训练 | EgoLAP | [2610.08726](https://arxiv.org/abs/2610.08726) | **待发布** | [paper-egolap-egocentric-language-action-reasoning](../../wiki/entities/paper-egolap-egocentric-language-action-reasoning.md)（**复用**） |
| 31 | 人形与移动操作 | Humanoid Horizon | [2610.08320](https://arxiv.org/abs/2610.08320) | **项目页未列 GitHub** | [paper-humanoid-horizon-long-horizon-loco-manipulation](../../wiki/entities/paper-humanoid-horizon-long-horizon-loco-manipulation.md)（**复用**） |
| 32 | 人形与移动操作 | InterMimicGen | [2610.06850](https://arxiv.org/abs/2610.06850) | **项目页未列 GitHub** | [paper-intermimicgen](../../wiki/entities/paper-intermimicgen.md)（**复用**） |
| 33 | 人形与移动操作 | CoDance | [2610.05324](https://arxiv.org/abs/2610.05324) | **项目页未列 GitHub** | [paper-codance](../../wiki/entities/paper-codance.md)（**复用**） |
| 34 | 多机器人与评测 | CoRE | [2610.07752](https://arxiv.org/abs/2610.07752) | **待发布** | [paper-core-collaboration-role-experts](../../wiki/entities/paper-core-collaboration-role-experts.md)（**复用**） |

## 对 wiki 的映射

- **34/34 独立详情节点**；同 arXiv 不重复造页（复用节点已补本文来源与地图关联）。
- 阅读坐标：[失败导向采集与可组合世界动作模型：具身智能小站 10-13 期 34 篇阅读坐标](../../wiki/overview/embodied-station-2026-10-13-34-papers-technology-map.md)
- 开源结论口径：GitHub 仓库用 `git ls-remote` 核实可达；项目页用 curl 读取扫描 GitHub 链接与 "coming soon"；「项目页未列 GitHub」表示入库日页面无代码链接且未见开源承诺字样。

## 当前提炼状态

- [x] 公众号正文抓取（2026-10-14）
- [x] 34 篇独立节点（25 新建 / 9 复用）
- [x] 项目页 / 仓库开源状态核查（步骤 2.5）
- [ ] 原文实验表格核验（节点内数字仅为公众号转述）
