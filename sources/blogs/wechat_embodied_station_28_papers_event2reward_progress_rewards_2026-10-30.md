# 机器人做错了，奖励为什么还在增加？｜附代码与项目入口

> 来源归档（blog / 微信公众号）

- **标题：** 机器人做错了，奖励为什么还在增加？｜附代码与项目入口
- **类型：** blog
- **作者：** 具身智能小站（微信公众号）
- **原始链接：** https://mp.weixin.qq.com/s/dxeEQnbdviUi4D4LuTfEhw
- **发表日期：** 2026-10-30
- **入库日期：** 2026-10-30
- **抓取方式：** `scripts/fetch_wechat_article.py`（Chromium + 微信 UA）
- **原始抓取落盘：** [`sources/raw/wechat_embodied_station_28_papers_event2reward_progress_rewards_2026-10-30.md`](../raw/wechat_embodied_station_28_papers_event2reward_progress_rewards_2026-10-30.md)
- **一句话说明：** 奖励做错了还在涨，往往是把示范时间当成了进度：本期深读 Event2Reward 用停滞、倒退与恢复事件校正进度奖励；跟进 TACOS 的接触残差监督与 Kairos 的规划语义调度；其余篇幅覆盖 VLA 量化与推理设置、示范评测与采集、世界模型、人形与灵巧操作、建图与多机器人协作。 共 28 篇；**28/28 独立详情节点**（本 ingest **新建 28**、**复用 0**）。

## 28 篇 → 本库节点

| # | 组 | 论文 | arXiv | 开源结论 | wiki |
|---|----|------|-------|----------|------|
| 01 | 奖励学习与 VLA 策略 | Event2Reward | [2610.39212](https://arxiv.org/abs/2610.39212) | **已开源** | [paper-event2reward](../../wiki/entities/paper-event2reward.md) |
| 02 | 运动控制与落脚 | TACOS | [2610.37806](https://arxiv.org/abs/2610.37806) | **已开源** | [paper-tacos](../../wiki/entities/paper-tacos.md) |
| 03 | 规划、感知与建图 | Kairos | [2610.39454](https://arxiv.org/abs/2610.39454) | **已开源** | [paper-kairos-pddl-planning-engine-orchestration](../../wiki/entities/paper-kairos-pddl-planning-engine-orchestration.md) |
| 04 | 世界模型与预测 | TERRA | [2610.38657](https://arxiv.org/abs/2610.38657) | **已开源** | [paper-terra-targeted-exploration-rover-adaptation](../../wiki/entities/paper-terra-targeted-exploration-rover-adaptation.md) |
| 05 | 规划、感知与建图 | Continuum Robot Embedded Shape Sensing | [2610.39055](https://arxiv.org/abs/2610.39055) | **已开源** | [paper-continuum-robot-embedded-shape-sensing](../../wiki/entities/paper-continuum-robot-embedded-shape-sensing.md) |
| 06 | 奖励学习与 VLA 策略 | QbD-VLA | [2610.38661](https://arxiv.org/abs/2610.38661) | **已开源** | [paper-qbd-vla](../../wiki/entities/paper-qbd-vla.md) |
| 07 | 示范数据与评测 | CuraEval | [2610.38221](https://arxiv.org/abs/2610.38221) | **已开源** | [paper-curaeval](../../wiki/entities/paper-curaeval.md) |
| 08 | 多机器人协作与人机交互 | SKiD-SLAM | [2610.38141](https://arxiv.org/abs/2610.38141) | **已开源** | [paper-skid-slam](../../wiki/entities/paper-skid-slam.md) |
| 09 | 规划、感知与建图 | HERON | [2610.38233](https://arxiv.org/abs/2610.38233) | **已开源** | [paper-heron-obstacle-aware-reachability-maps](../../wiki/entities/paper-heron-obstacle-aware-reachability-maps.md) |
| 10 | 规划、感知与建图 | LOGIV | [2610.40016](https://arxiv.org/abs/2610.40016) | **已开源** | [paper-logiv](../../wiki/entities/paper-logiv.md) |
| 11 | 多机器人协作与人机交互 | Minimum Sufficient Communication | [2610.38176](https://arxiv.org/abs/2610.38176) | **已开源** | [paper-minimum-sufficient-communication](../../wiki/entities/paper-minimum-sufficient-communication.md) |
| 12 | 示范数据与评测 | ScheduleGen | [2610.39429](https://arxiv.org/abs/2610.39429) | **项目页未列 GitHub** | [paper-schedulegen](../../wiki/entities/paper-schedulegen.md) |
| 13 | 奖励学习与 VLA 策略 | Cross-Controller Transfer | [2610.39280](https://arxiv.org/abs/2610.39280) | **已开源** | [paper-cross-controller-transfer](../../wiki/entities/paper-cross-controller-transfer.md) |
| 14 | 奖励学习与 VLA 策略 | Teaching a Tree to Recover | [2610.39775](https://arxiv.org/abs/2610.39775) | **已开源** | [paper-teaching-a-tree-to-recover](../../wiki/entities/paper-teaching-a-tree-to-recover.md) |
| 15 | 示范数据与评测 | RoboSeed | [2610.40026](https://arxiv.org/abs/2610.40026) | **已开源** | [paper-roboseed](../../wiki/entities/paper-roboseed.md) |
| 16 | 示范数据与评测 | UMI-Live | [2610.39385](https://arxiv.org/abs/2610.39385) | **已开源** | [paper-umi-live](../../wiki/entities/paper-umi-live.md) |
| 17 | 规划、感知与建图 | OpenFlyV2 | [2610.39287](https://arxiv.org/abs/2610.39287) | **已开源** | [paper-openflyv2](../../wiki/entities/paper-openflyv2.md) |
| 18 | 灵巧操作与抓取 | Catch It All | [2610.39044](https://arxiv.org/abs/2610.39044) | **部分开源** | [paper-catch-it-all](../../wiki/entities/paper-catch-it-all.md) |
| 19 | 世界模型与预测 | WorldFlow | [2610.39278](https://arxiv.org/abs/2610.39278) | **已开源** | [paper-worldflow](../../wiki/entities/paper-worldflow.md) |
| 20 | 多机器人协作与人机交互 | EARS-Info | [2610.38040](https://arxiv.org/abs/2610.38040) | **已开源** | [paper-ears-info](../../wiki/entities/paper-ears-info.md) |
| 21 | 运动控制与落脚 | SONIC-K1 | [2610.39327](https://arxiv.org/abs/2610.39327) | **已开源** | [paper-sonic-k1](../../wiki/entities/paper-sonic-k1.md) |
| 22 | 灵巧操作与抓取 | Skeleton-Guided Grasp Auto-Labeling | [2610.38443](https://arxiv.org/abs/2610.38443) | **已开源** | [paper-skeleton-guided-grasp-auto-labeling](../../wiki/entities/paper-skeleton-guided-grasp-auto-labeling.md) |
| 23 | 灵巧操作与抓取 | DexSuite | [2610.37799](https://arxiv.org/abs/2610.37799) | **待发布** | [paper-dexsuite](../../wiki/entities/paper-dexsuite.md) |
| 24 | 世界模型与预测 | GenSight | [2610.39232](https://arxiv.org/abs/2610.39232) | **待发布** | [paper-gensight](../../wiki/entities/paper-gensight.md) |
| 25 | 运动控制与落脚 | ELEVATE | [2610.38388](https://arxiv.org/abs/2610.38388) | **项目页未列 GitHub** | [paper-elevate](../../wiki/entities/paper-elevate.md) |
| 26 | 多机器人协作与人机交互 | When No Worries Went Wrong | [2610.38400](https://arxiv.org/abs/2610.38400) | **项目页未列 GitHub** | [paper-when-no-worries-went-wrong](../../wiki/entities/paper-when-no-worries-went-wrong.md) |
| 27 | 奖励学习与 VLA 策略 | pi0.5 Inference Settings on BEHAVIOR-1K | [2610.37917](https://arxiv.org/abs/2610.37917) | **项目页未列 GitHub** | [paper-pi05-inference-settings-behavior-1k](../../wiki/entities/paper-pi05-inference-settings-behavior-1k.md) |
| 28 | 示范数据与评测 | Embodied at Sea | [2610.38199](https://arxiv.org/abs/2610.38199) | **待发布** | [paper-embodied-at-sea](../../wiki/entities/paper-embodied-at-sea.md) |

## 对 wiki 的映射

- **28/28 独立详情节点**；同 arXiv 不重复造页（复用节点已补本文来源与地图关联）。
- 阅读坐标：[进度奖励、接触监督与规划验证：具身智能小站 10-30 期 28 篇阅读坐标](../../wiki/overview/embodied-station-2026-10-30-28-papers-technology-map.md)
- 开源结论口径：GitHub 仓库用 `git ls-remote` 核实可达；项目页用 curl 读取扫描 GitHub 链接与 "coming soon"；「项目页未列 GitHub」表示入库日页面无代码链接且未见开源承诺字样。

## 当前提炼状态

- [x] 公众号正文抓取（2026-10-30）
- [x] 28 篇独立节点（28 新建 / 0 复用）
- [x] 项目页 / 仓库开源状态核查（步骤 2.5）
- [ ] 原文实验表格核验（节点内数字仅为公众号转述）
