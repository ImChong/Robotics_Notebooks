# 机器人缺的，是更多示范，还是更多成功解法？｜附代码与项目入口

> 来源归档（blog / 微信公众号）

- **标题：** 机器人缺的，是更多示范，还是更多成功解法？｜附代码与项目入口
- **类型：** blog
- **作者：** 具身智能小站（微信公众号）
- **原始链接：** https://mp.weixin.qq.com/s/s23oOBojr5YbchuFmQhcwA
- **发表日期：** 2026-10-29
- **入库日期：** 2026-10-29
- **抓取方式：** `scripts/fetch_wechat_article.py`（Chromium + 微信 UA）
- **原始抓取落盘：** [`sources/raw/wechat_embodied_station_58_papers_diverse_demonstrations_2026-10-29.md`](../raw/wechat_embodied_station_58_papers_diverse_demonstrations_2026-10-29.md)
- **一句话说明：** 模仿学习缺的往往不是更多相似轨迹，而是多种可成功的解法：本期深读 DiscoDemo 把强化学习当示范生成器；跟进 OpenTac 的开放触觉工具链与 Watch Your Step 的预见式落脚感知；其余篇幅覆盖世界模型表征、VLA 后训练与不确定性、评测与安全、人形控制和导航。 共 58 篇；**58/58 独立详情节点**（本 ingest **新建 35**、**复用 23**）。

## 58 篇 → 本库节点

| # | 组 | 论文 | arXiv | 开源结论 | wiki |
|---|----|------|-------|----------|------|
| 01 | 示范数据与数据引擎 | DiscoDemo | [2610.32093](https://arxiv.org/abs/2610.32093) | **已开源** | [paper-discodemo](../../wiki/entities/paper-discodemo.md) |
| 02 | 触觉与灵巧操作 | OpenTac | [2610.31661](https://arxiv.org/abs/2610.31661) | **已开源** | [paper-opentac-open-3d-tactile-sensing-ecosystem](../../wiki/entities/paper-opentac-open-3d-tactile-sensing-ecosystem.md)（**复用**） |
| 03 | 人形运动与奖励设计 | Watch Your Step | [2610.29612](https://arxiv.org/abs/2610.29612) | **已开源** | [paper-watch-your-step](../../wiki/entities/paper-watch-your-step.md)（**复用**） |
| 04 | 世界模型与表征 | FRED | [2610.30320](https://arxiv.org/abs/2610.30320) | **已开源** | [paper-self-distilling-future-video-representations-wam](../../wiki/entities/paper-self-distilling-future-video-representations-wam.md)（**复用**） |
| 05 | 世界模型与表征 | DeXplicit | [2610.30781](https://arxiv.org/abs/2610.30781) | **已开源** | [paper-dexplicit](../../wiki/entities/paper-dexplicit.md) |
| 06 | VLA、智能体与策略后训练 | AgenticRoboRL | [2610.28589](https://arxiv.org/abs/2610.28589) | **已开源** | [paper-agentic-roborl](../../wiki/entities/paper-agentic-roborl.md)（**复用**） |
| 07 | VLA、智能体与策略后训练 | AW-0 | [2610.30136](https://arxiv.org/abs/2610.30136) | **已开源** | [paper-aw-0-agentic-world-action-models](../../wiki/entities/paper-aw-0-agentic-world-action-models.md)（**复用**） |
| 08 | 示范数据与数据引擎 | EgoForesight | [2610.31845](https://arxiv.org/abs/2610.31845) | **已开源** | [paper-egoforesight](../../wiki/entities/paper-egoforesight.md) |
| 09 | 示范数据与数据引擎 | FLORA | [2610.31779](https://arxiv.org/abs/2610.31779) | **已开源** | [flora-robotics-data-engine](../../wiki/entities/flora-robotics-data-engine.md)（**复用**） |
| 10 | 评测、安全与不确定性 | Sensorimotor Alignment | [2610.31670](https://arxiv.org/abs/2610.31670) | **已开源** | [paper-sensorimotor-alignment](../../wiki/entities/paper-sensorimotor-alignment.md) |
| 11 | VLA、智能体与策略后训练 | GeoMem-VLA | [2610.29866](https://arxiv.org/abs/2610.29866) | **已开源** | [paper-geomem-vla](../../wiki/entities/paper-geomem-vla.md) |
| 12 | 评测、安全与不确定性 | Post-hoc VLA Epistemic Uncertainty | [2610.29730](https://arxiv.org/abs/2610.29730) | **已开源** | [paper-post-hoc-vla-epistemic-uncertainty](../../wiki/entities/paper-post-hoc-vla-epistemic-uncertainty.md) |
| 13 | 导航、规划与定位 | GlobMP | [2610.30607](https://arxiv.org/abs/2610.30607) | **已开源** | [paper-globmp](../../wiki/entities/paper-globmp.md) |
| 14 | VLA、智能体与策略后训练 | RynnUnify | [2610.32965](https://arxiv.org/abs/2610.32965) | **已开源** | [paper-rynnunify](../../wiki/entities/paper-rynnunify.md) |
| 15 | 示范数据与数据引擎 | Learned Speed Prior | [2610.32676](https://arxiv.org/abs/2610.32676) | **项目页未列 GitHub** | [paper-learned-speed-prior](../../wiki/entities/paper-learned-speed-prior.md) |
| 16 | 导航、规划与定位 | Self-Constructed Navigation Space | [2610.31762](https://arxiv.org/abs/2610.31762) | **待发布** | [paper-self-constructed-navigation-space](../../wiki/entities/paper-self-constructed-navigation-space.md) |
| 17 | 人形运动与奖励设计 | Memory before Goal | [2610.31166](https://arxiv.org/abs/2610.31166) | **已开源** | [paper-wholebodymem](../../wiki/entities/paper-wholebodymem.md)（**复用**） |
| 18 | 评测、安全与不确定性 | AWARE | [2610.32742](https://arxiv.org/abs/2610.32742) | **已开源** | [paper-aware](../../wiki/entities/paper-aware.md) |
| 19 | 导航、规划与定位 | Natural-Language to Nav2 | [2610.32633](https://arxiv.org/abs/2610.32633) | **已开源** | [paper-natural-language-to-nav2](../../wiki/entities/paper-natural-language-to-nav2.md) |
| 20 | VLA、智能体与策略后训练 | When One Step Is Enough | [2610.30735](https://arxiv.org/abs/2610.30735) | **已开源** | [paper-one-step-euler-integration-flow-matching-policies](../../wiki/entities/paper-one-step-euler-integration-flow-matching-policies.md)（**复用**） |
| 21 | VLA、智能体与策略后训练 | FastHarness | [2610.31427](https://arxiv.org/abs/2610.31427) | **已开源** | [paper-fastharness-experience-driven-review-scheduling](../../wiki/entities/paper-fastharness-experience-driven-review-scheduling.md)（**复用**） |
| 22 | 评测、安全与不确定性 | AstroBench | [2610.31064](https://arxiv.org/abs/2610.31064) | **已开源** | [paper-astrobench](../../wiki/entities/paper-astrobench.md) |
| 23 | VLA、智能体与策略后训练 | TARSphere | [2610.30043](https://arxiv.org/abs/2610.30043) | **已开源** | [paper-tarsphere](../../wiki/entities/paper-tarsphere.md) |
| 24 | 导航、规划与定位 | AOV-GS | [2610.29374](https://arxiv.org/abs/2610.29374) | **已开源** | [paper-aov-gs](../../wiki/entities/paper-aov-gs.md) |
| 25 | 人形运动与奖励设计 | LocoManiLab | [2610.30876](https://arxiv.org/abs/2610.30876) | **已开源** | [paper-locomanilab](../../wiki/entities/paper-locomanilab.md)（**复用**） |
| 26 | 人形运动与奖励设计 | MPBRS | [2610.30334](https://arxiv.org/abs/2610.30334) | **已开源** | [paper-mpbrs-humanoid-reward-shaping](../../wiki/entities/paper-mpbrs-humanoid-reward-shaping.md)（**复用**） |
| 27 | 示范数据与数据引擎 | DexPort | [2610.32969](https://arxiv.org/abs/2610.32969) | **已开源** | [paper-dexport](../../wiki/entities/paper-dexport.md) |
| 28 | VLA、智能体与策略后训练 | Execution Horizon vs Sampling Depth | [2610.32102](https://arxiv.org/abs/2610.32102) | **项目页未列 GitHub** | [paper-execution-horizon-vs-sampling-depth](../../wiki/entities/paper-execution-horizon-vs-sampling-depth.md) |
| 29 | VLA、智能体与策略后训练 | FORCA | [2610.31841](https://arxiv.org/abs/2610.31841) | **项目页未列 GitHub** | [paper-forca](../../wiki/entities/paper-forca.md) |
| 30 | 评测、安全与不确定性 | SEVI | [2610.31829](https://arxiv.org/abs/2610.31829) | **待发布** | [paper-sevi](../../wiki/entities/paper-sevi.md) |
| 31 | 示范数据与数据引擎 | Reconstruction-Augmented Bimanual Pretraining | [2610.29716](https://arxiv.org/abs/2610.29716) | **已开源** | [paper-reconstruction-augmented-bimanual-pretraining-rgb-human-video](../../wiki/entities/paper-reconstruction-augmented-bimanual-pretraining-rgb-human-video.md)（**复用**） |
| 32 | VLA、智能体与策略后训练 | Plan First, Act Later | [2610.29329](https://arxiv.org/abs/2610.29329) | **部分开源** | [paper-plan-first-act-later](../../wiki/entities/paper-plan-first-act-later.md) |
| 33 | VLA、智能体与策略后训练 | Training in Deployment | [2610.29264](https://arxiv.org/abs/2610.29264) | **已开源** | [paper-training-in-deployment](../../wiki/entities/paper-training-in-deployment.md) |
| 34 | 触觉与灵巧操作 | T-WLA | [2610.28814](https://arxiv.org/abs/2610.28814) | **已开源** | [paper-t-wla-slow-vision-fast-touch](../../wiki/entities/paper-t-wla-slow-vision-fast-touch.md)（**复用**） |
| 35 | 世界模型与表征 | Physical Canvas | [2610.28613](https://arxiv.org/abs/2610.28613) | **已开源** | [paper-physical-canvas-vision-tactile-world-model](../../wiki/entities/paper-physical-canvas-vision-tactile-world-model.md)（**复用**） |
| 36 | 人形运动与奖励设计 | R2E | [2610.32439](https://arxiv.org/abs/2610.32439) | **已开源** | [paper-r2e](../../wiki/entities/paper-r2e.md) |
| 37 | 世界模型与表征 | UniWALM | [2610.28626](https://arxiv.org/abs/2610.28626) | **已开源** | [paper-uniwalm](../../wiki/entities/paper-uniwalm.md)（**复用**） |
| 38 | 示范数据与数据引擎 | WorldMirror | [2610.31834](https://arxiv.org/abs/2610.31834) | **项目页未列 GitHub** | [paper-worldmirror](../../wiki/entities/paper-worldmirror.md) |
| 39 | 触觉与灵巧操作 | DONIC | [2610.29997](https://arxiv.org/abs/2610.29997) | **已开源** | [paper-donic](../../wiki/entities/paper-donic.md)（**复用**） |
| 40 | 人形运动与奖励设计 | SMR | [2610.29169](https://arxiv.org/abs/2610.29169) | **已开源** | [paper-smr-streaming-motion-retargeting](../../wiki/entities/paper-smr-streaming-motion-retargeting.md)（**复用**） |
| 41 | 触觉与灵巧操作 | VecTile | [2610.31472](https://arxiv.org/abs/2610.31472) | **待发布** | [paper-vectile](../../wiki/entities/paper-vectile.md) |
| 42 | 世界模型与表征 | UniFlowWAM | [2610.30991](https://arxiv.org/abs/2610.30991) | **已开源** | [paper-uniflowwam-semantics-4d-geometry-world-action-model](../../wiki/entities/paper-uniflowwam-semantics-4d-geometry-world-action-model.md)（**复用**） |
| 43 | 人形运动与奖励设计 | Test-Time Motion Steering Fall Recovery | [2610.30061](https://arxiv.org/abs/2610.30061) | **已开源** | [paper-rise-humanoid-fall-recovery](../../wiki/entities/paper-rise-humanoid-fall-recovery.md)（**复用**） |
| 44 | 示范数据与数据引擎 | Real2Sem2Real | [2610.28880](https://arxiv.org/abs/2610.28880) | **待发布** | [paper-real2sem2real](../../wiki/entities/paper-real2sem2real.md) |
| 45 | VLA、智能体与策略后训练 | State-to-Vision Distribution Matching | [2610.31010](https://arxiv.org/abs/2610.31010) | **已开源** | [paper-laparoscopic-state-to-vision-policy-distillation](../../wiki/entities/paper-laparoscopic-state-to-vision-policy-distillation.md)（**复用**） |
| 46 | 评测、安全与不确定性 | From Reasoning to Action | [2610.30051](https://arxiv.org/abs/2610.30051) | **已开源** | [paper-from-reasoning-to-action](../../wiki/entities/paper-from-reasoning-to-action.md) |
| 47 | 导航、规划与定位 | Root Locus Harmonic Path Generation | [2610.31625](https://arxiv.org/abs/2610.31625) | **已开源** | [paper-root-locus-harmonic-path-generation](../../wiki/entities/paper-root-locus-harmonic-path-generation.md) |
| 48 | 评测、安全与不确定性 | Alpha | [2610.30197](https://arxiv.org/abs/2610.30197) | **已开源** | [paper-alpha-embodied-fault-injection](../../wiki/entities/paper-alpha-embodied-fault-injection.md) |
| 49 | 导航、规划与定位 | RangeNav | [2610.29845](https://arxiv.org/abs/2610.29845) | **待发布** | [paper-rangenav](../../wiki/entities/paper-rangenav.md) |
| 50 | 示范数据与数据引擎 | KiRoSe | [2610.31801](https://arxiv.org/abs/2610.31801) | **项目页未列 GitHub** | [paper-kirose](../../wiki/entities/paper-kirose.md) |
| 51 | VLA、智能体与策略后训练 | SCOPE | [2610.31518](https://arxiv.org/abs/2610.31518) | **待发布** | [paper-scope](../../wiki/entities/paper-scope.md) |
| 52 | 世界模型与表征 | Skeleton-Grounded Latent Action Model | [2610.31474](https://arxiv.org/abs/2610.31474) | **已开源** | [paper-skeleton-grounded-latent-action-model](../../wiki/entities/paper-skeleton-grounded-latent-action-model.md)（**复用**） |
| 53 | 人形运动与奖励设计 | Motion Transition with Human Prior | [2610.30817](https://arxiv.org/abs/2610.30817) | **已开源** | [paper-stamp](../../wiki/entities/paper-stamp.md)（**复用**） |
| 54 | VLA、智能体与策略后训练 | Language-Guided Representation Editing | [2610.28893](https://arxiv.org/abs/2610.28893) | **已开源** | [paper-language-guided-representation-editing](../../wiki/entities/paper-language-guided-representation-editing.md) |
| 55 | 导航、规划与定位 | Distributed Invariant EKF SLAM | [2610.31221](https://arxiv.org/abs/2610.31221) | **项目页未列 GitHub** | [paper-distributed-invariant-ekf-slam](../../wiki/entities/paper-distributed-invariant-ekf-slam.md) |
| 56 | 世界模型与表征 | DepState-JEPA | [2610.29246](https://arxiv.org/abs/2610.29246) | **已开源** | [paper-depstate-jepa](../../wiki/entities/paper-depstate-jepa.md) |
| 57 | 人形运动与奖励设计 | One Policy, Many Styles | [2610.28925](https://arxiv.org/abs/2610.28925) | **已开源** | [paper-one-policy-many-styles-preference-aggregation](../../wiki/entities/paper-one-policy-many-styles-preference-aggregation.md)（**复用**） |
| 58 | VLA、智能体与策略后训练 | CEMIR vs. FISER | [2610.32022](https://arxiv.org/abs/2610.32022) | **已开源** | [paper-cemir-vs-fiser](../../wiki/entities/paper-cemir-vs-fiser.md) |

## 对 wiki 的映射

- **58/58 独立详情节点**；同 arXiv 不重复造页（复用节点已补本文来源与地图关联）。
- 阅读坐标：[多样成功示范、触觉工具链与预见式感知：具身智能小站 10-29 期 58 篇阅读坐标](../../wiki/overview/embodied-station-2026-10-29-58-papers-technology-map.md)
- 开源结论口径：GitHub 仓库用 `git ls-remote` 核实可达；项目页用 curl 读取扫描 GitHub 链接与 "coming soon"；「项目页未列 GitHub」表示入库日页面无代码链接且未见开源承诺字样。

## 当前提炼状态

- [x] 公众号正文抓取（2026-10-29）
- [x] 58 篇独立节点（35 新建 / 23 复用）
- [x] 项目页 / 仓库开源状态核查（步骤 2.5）
- [ ] 原文实验表格核验（节点内数字仅为公众号转述）
