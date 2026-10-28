# 人类视频怎样变成机器人能用的动作经验？

> 来源归档（blog / 微信公众号）

- **标题：** 人类视频怎样变成机器人能用的动作经验？
- **类型：** blog
- **作者：** 具身智能小站（微信公众号）
- **原始链接：** https://mp.weixin.qq.com/s/JSya9kmE3Fcon3BY_DcCVA
- **发表日期：** 2026-10-28
- **入库日期：** 2026-10-28
- **抓取方式：** `scripts/fetch_wechat_article.py`（Chromium + 微信 UA）
- **原始抓取落盘：** [`sources/raw/wechat_embodied_station_59_papers_human_video_to_robot_experience_2026-10-28.md`](../raw/wechat_embodied_station_59_papers_human_video_to_robot_experience_2026-10-28.md)
- **一句话说明：** 人类视频缺控制标签，所以本期的共同线索是把它转成机器人可用的动作经验：潜在动作、三维运动监督与示范迁移；另一半篇幅在补评测——成功率之外还要看过程、失败原因与安全语义。深读免渲染事件仿真，跟进 RynnVLA-Latent 与 AuRA。 共 59 篇；**59/59 独立详情节点**（本 ingest **新建 55**、**复用 4**）。

## 59 篇 → 本库节点

| # | 组 | 论文 | arXiv | 开源结论 | wiki |
|---|----|------|-------|----------|------|
| 01 | 导航、无人机与群体 | Rendering-Free Event Sim | [2610.37543](https://arxiv.org/abs/2610.37543) | **已开源** | [paper-rendering-free-event-sim](../../wiki/entities/paper-rendering-free-event-sim.md) |
| 02 | 潜在动作与世界模型 | RynnVLA-Latent | [2610.34356](https://arxiv.org/abs/2610.34356) | **已开源** | [paper-rynnvla-latent](../../wiki/entities/paper-rynnvla-latent.md)（**复用**） |
| 03 | 数据采集与示范迁移 | AuRA | [2610.34452](https://arxiv.org/abs/2610.34452) | **已开源** | [paper-aura](../../wiki/entities/paper-aura.md) |
| 04 | 评测与基准 | EmbodiedLab | [2610.35385](https://arxiv.org/abs/2610.35385) | **已开源** | [paper-embodiedlab](../../wiki/entities/paper-embodiedlab.md) |
| 05 | 评测与基准 | SO-101 Bench | [2610.36420](https://arxiv.org/abs/2610.36420) | **已开源** | [paper-so-101-bench](../../wiki/entities/paper-so-101-bench.md) |
| 06 | 评测与基准 | Specola | [2610.35730](https://arxiv.org/abs/2610.35730) | **已开源** | [paper-specola](../../wiki/entities/paper-specola.md) |
| 07 | 评测与基准 | Open-world Robot Autonomy | [2610.34649](https://arxiv.org/abs/2610.34649) | **已开源** | [paper-open-world-robot-autonomy](../../wiki/entities/paper-open-world-robot-autonomy.md) |
| 08 | 灵巧与接触操作 | AdaDexTrack | [2610.33852](https://arxiv.org/abs/2610.33852) | **已开源** | [paper-adadextrack](../../wiki/entities/paper-adadextrack.md) |
| 09 | 灵巧与接触操作 | EgoThumb | [2610.37582](https://arxiv.org/abs/2610.37582) | **已开源** | [paper-egothumb](../../wiki/entities/paper-egothumb.md) |
| 10 | 导航、无人机与群体 | Dynamic Narrow Gap Flight | [2610.37557](https://arxiv.org/abs/2610.37557) | **已开源** | [paper-dynamic-narrow-gap-flight](../../wiki/entities/paper-dynamic-narrow-gap-flight.md) |
| 11 | 导航、无人机与群体 | Thermal-Headroom Routing | [2610.35231](https://arxiv.org/abs/2610.35231) | **已开源** | [paper-thermal-headroom-routing](../../wiki/entities/paper-thermal-headroom-routing.md) |
| 12 | 评测与基准 | Execution Logs Sway VLM Judges | [2610.34151](https://arxiv.org/abs/2610.34151) | **已开源** | [paper-execution-logs-sway-vlm-judges](../../wiki/entities/paper-execution-logs-sway-vlm-judges.md) |
| 13 | 评测与基准 | RoboJev | [2610.33718](https://arxiv.org/abs/2610.33718) | **已开源** | [paper-robojev](../../wiki/entities/paper-robojev.md) |
| 14 | 评测与基准 | WorldFailBench | [2610.33565](https://arxiv.org/abs/2610.33565) | **已开源** | [paper-worldfailbench](../../wiki/entities/paper-worldfailbench.md) |
| 15 | VLA 与策略后训练 | AGQ | [2610.33074](https://arxiv.org/abs/2610.33074) | **已开源** | [paper-agq](../../wiki/entities/paper-agq.md) |
| 16 | VLA 与策略后训练 | RoboDoctor | [2610.35604](https://arxiv.org/abs/2610.35604) | **已开源** | [paper-robodoctor](../../wiki/entities/paper-robodoctor.md) |
| 17 | 灵巧与接触操作 | FLoRe | [2610.35862](https://arxiv.org/abs/2610.35862) | **待发布** | [paper-flore](../../wiki/entities/paper-flore.md) |
| 18 | VLA 与策略后训练 | Sim-Trained Compositional Agent | [2610.35817](https://arxiv.org/abs/2610.35817) | **项目页未列 GitHub** | [paper-sim-trained-compositional-agent](../../wiki/entities/paper-sim-trained-compositional-agent.md) |
| 19 | 潜在动作与世界模型 | RynnWorld-Latent | [2610.36791](https://arxiv.org/abs/2610.36791) | **已开源** | [paper-rynnworld-latent](../../wiki/entities/paper-rynnworld-latent.md) |
| 20 | 数据采集与示范迁移 | PrecisionReach-GS | [2610.35626](https://arxiv.org/abs/2610.35626) | **已开源** | [paper-precisionreach-gs](../../wiki/entities/paper-precisionreach-gs.md) |
| 21 | VLA 与策略后训练 | HuGe | [2610.34186](https://arxiv.org/abs/2610.34186) | **已开源** | [paper-huge-humanoid-vla](../../wiki/entities/paper-huge-humanoid-vla.md)（**复用**） |
| 22 | 控制、规划与系统 | HA-DiPO | [2610.33584](https://arxiv.org/abs/2610.33584) | **已开源** | [paper-ha-dipo](../../wiki/entities/paper-ha-dipo.md) |
| 23 | 潜在动作与世界模型 | Tapestry | [2610.36823](https://arxiv.org/abs/2610.36823) | **项目页未列 GitHub** | [paper-tapestry](../../wiki/entities/paper-tapestry.md) |
| 24 | 数据采集与示范迁移 | OpenJaws | [2610.36598](https://arxiv.org/abs/2610.36598) | **已开源** | [paper-openjaws](../../wiki/entities/paper-openjaws.md) |
| 25 | 评测与基准 | EBench Frontier Agent Profiling | [2610.35832](https://arxiv.org/abs/2610.35832) | **已开源** | [paper-ebench-frontier-agent-profiling](../../wiki/entities/paper-ebench-frontier-agent-profiling.md) |
| 26 | 评测与基准 | Obstacle-Conditioned Safety Nav | [2610.34674](https://arxiv.org/abs/2610.34674) | **项目页未列 GitHub** | [paper-obstacle-conditioned-safety-nav](../../wiki/entities/paper-obstacle-conditioned-safety-nav.md) |
| 27 | 潜在动作与世界模型 | ThinkWorld | [2610.37666](https://arxiv.org/abs/2610.37666) | **已开源** | [paper-thinkworld](../../wiki/entities/paper-thinkworld.md)（**复用**） |
| 28 | 灵巧与接触操作 | CARD | [2610.36539](https://arxiv.org/abs/2610.36539) | **已开源** | [paper-card-contact-anchored-retargeting](../../wiki/entities/paper-card-contact-anchored-retargeting.md)（**复用**） |
| 29 | 控制、规划与系统 | cuArms | [2610.36250](https://arxiv.org/abs/2610.36250) | **项目页未列 GitHub** | [paper-cuarms](../../wiki/entities/paper-cuarms.md) |
| 30 | 导航、无人机与群体 | Motion-Induced WiFi-CSI Sim | [2610.34034](https://arxiv.org/abs/2610.34034) | **已开源** | [paper-motion-induced-wifi-csi-sim](../../wiki/entities/paper-motion-induced-wifi-csi-sim.md) |
| 31 | 导航、无人机与群体 | NIT-TrajMamba | [2610.33916](https://arxiv.org/abs/2610.33916) | **已开源** | [paper-nit-trajmamba](../../wiki/entities/paper-nit-trajmamba.md) |
| 32 | VLA 与策略后训练 | Retrieve to Act | [2610.37215](https://arxiv.org/abs/2610.37215) | **待发布** | [paper-retrieve-to-act](../../wiki/entities/paper-retrieve-to-act.md) |
| 33 | VLA 与策略后训练 | RoboAuto | [2610.36533](https://arxiv.org/abs/2610.36533) | **待发布** | [paper-roboauto](../../wiki/entities/paper-roboauto.md) |
| 34 | 控制、规划与系统 | Behavioural Quality vs Learning Utility | [2610.34839](https://arxiv.org/abs/2610.34839) | **项目页未列 GitHub** | [paper-behavioural-quality-vs-learning-utility](../../wiki/entities/paper-behavioural-quality-vs-learning-utility.md) |
| 35 | 评测与基准 | RoboWorM | [2610.33807](https://arxiv.org/abs/2610.33807) | **已开源** | [paper-roboworm](../../wiki/entities/paper-roboworm.md) |
| 36 | 灵巧与接触操作 | Differentiable Contact Legged | [2610.35852](https://arxiv.org/abs/2610.35852) | **项目页未列 GitHub** | [paper-differentiable-contact-legged](../../wiki/entities/paper-differentiable-contact-legged.md) |
| 37 | 导航、无人机与群体 | Distributed Predictive Flocking | [2610.35582](https://arxiv.org/abs/2610.35582) | **已开源** | [paper-distributed-predictive-flocking](../../wiki/entities/paper-distributed-predictive-flocking.md) |
| 38 | 灵巧与接触操作 | Functional Specialisation Bimanual | [2610.33947](https://arxiv.org/abs/2610.33947) | **已开源** | [paper-functional-specialisation-bimanual](../../wiki/entities/paper-functional-specialisation-bimanual.md) |
| 39 | 导航、无人机与群体 | DSGN | [2610.34084](https://arxiv.org/abs/2610.34084) | **已开源** | [paper-dsgn](../../wiki/entities/paper-dsgn.md) |
| 40 | 数据采集与示范迁移 | SceneAgent | [2610.34745](https://arxiv.org/abs/2610.34745) | **待发布** | [paper-sceneagent](../../wiki/entities/paper-sceneagent.md) |
| 41 | 控制、规划与系统 | Expressive Motion Behavior | [2610.35615](https://arxiv.org/abs/2610.35615) | **已开源** | [paper-expressive-motion-behavior](../../wiki/entities/paper-expressive-motion-behavior.md) |
| 42 | VLA 与策略后训练 | Human-Guided Adaptive Residual Correction | [2610.37547](https://arxiv.org/abs/2610.37547) | **项目页未列 GitHub** | [paper-human-guided-adaptive-residual-correction](../../wiki/entities/paper-human-guided-adaptive-residual-correction.md) |
| 43 | VLA 与策略后训练 | Quality-Controllable Policies | [2610.37393](https://arxiv.org/abs/2610.37393) | **待发布** | [paper-quality-controllable-policies](../../wiki/entities/paper-quality-controllable-policies.md) |
| 44 | 导航、无人机与群体 | RememberWhereYouWalk | [2610.37203](https://arxiv.org/abs/2610.37203) | **已开源** | [paper-rememberwhereyouwalk](../../wiki/entities/paper-rememberwhereyouwalk.md) |
| 45 | 灵巧与接触操作 | Keeping Contact | [2610.36961](https://arxiv.org/abs/2610.36961) | **项目页未列 GitHub** | [paper-keeping-contact](../../wiki/entities/paper-keeping-contact.md) |
| 46 | 潜在动作与世界模型 | Trace-WAM | [2610.36840](https://arxiv.org/abs/2610.36840) | **项目页未列 GitHub** | [paper-trace-wam](../../wiki/entities/paper-trace-wam.md) |
| 47 | 数据采集与示范迁移 | RoboSTD | [2610.36685](https://arxiv.org/abs/2610.36685) | **已开源** | [paper-robostd](../../wiki/entities/paper-robostd.md) |
| 48 | VLA 与策略后训练 | OmniBrain-0 | [2610.36013](https://arxiv.org/abs/2610.36013) | **已开源** | [paper-omnibrain-0](../../wiki/entities/paper-omnibrain-0.md) |
| 49 | 灵巧与接触操作 | Action Representation Pretraining | [2610.34453](https://arxiv.org/abs/2610.34453) | **项目页未列 GitHub** | [paper-action-representation-pretraining](../../wiki/entities/paper-action-representation-pretraining.md) |
| 50 | VLA 与策略后训练 | AdaPolicy | [2610.34280](https://arxiv.org/abs/2610.34280) | **部分开源** | [paper-adapolicy](../../wiki/entities/paper-adapolicy.md) |
| 51 | 潜在动作与世界模型 | X-World | [2610.34105](https://arxiv.org/abs/2610.34105) | **项目页未列 GitHub** | [paper-x-world-variable-view-video-priors](../../wiki/entities/paper-x-world-variable-view-video-priors.md) |
| 52 | VLA 与策略后训练 | RoboLongR | [2610.33863](https://arxiv.org/abs/2610.33863) | **项目页未列 GitHub** | [paper-robolongr](../../wiki/entities/paper-robolongr.md) |
| 53 | 评测与基准 | RoboReason | [2610.33778](https://arxiv.org/abs/2610.33778) | **待发布** | [paper-roboreason](../../wiki/entities/paper-roboreason.md) |
| 54 | 潜在动作与世界模型 | GEMS | [2610.33158](https://arxiv.org/abs/2610.33158) | **已开源** | [paper-gems](../../wiki/entities/paper-gems.md) |
| 55 | 控制、规划与系统 | SysML v2 ROS 2 Library | [2610.33362](https://arxiv.org/abs/2610.33362) | **已开源** | [paper-sysml-v2-ros-2-library](../../wiki/entities/paper-sysml-v2-ros-2-library.md) |
| 56 | VLA 与策略后训练 | FISRF | [2610.35686](https://arxiv.org/abs/2610.35686) | **项目页未列 GitHub** | [paper-fisrf](../../wiki/entities/paper-fisrf.md) |
| 57 | VLA 与策略后训练 | From Language to Drives | [2610.35629](https://arxiv.org/abs/2610.35629) | **已开源** | [paper-from-language-to-drives](../../wiki/entities/paper-from-language-to-drives.md) |
| 58 | 控制、规划与系统 | Aerial Manipulator Nonlinear Control | [2610.34454](https://arxiv.org/abs/2610.34454) | **已开源** | [paper-aerial-manipulator-nonlinear-control](../../wiki/entities/paper-aerial-manipulator-nonlinear-control.md) |
| 59 | 控制、规划与系统 | Spiking Periodic Orbit Stabilization | [2610.37310](https://arxiv.org/abs/2610.37310) | **部分开源** | [paper-spiking-periodic-orbit-stabilization](../../wiki/entities/paper-spiking-periodic-orbit-stabilization.md) |

## 对 wiki 的映射

- **59/59 独立详情节点**；同 arXiv 不重复造页（复用节点已补本文来源与地图关联）。
- 阅读坐标：[潜在动作、免渲染事件仿真与机器人评测：具身智能小站 10-28 期 59 篇阅读坐标](../../wiki/overview/embodied-station-2026-10-28-59-papers-technology-map.md)
- 开源结论口径：GitHub 仓库用 `git ls-remote` 核实可达；项目页用 curl 读取扫描 GitHub 链接与 "coming soon"；「项目页未列 GitHub」表示入库日页面无代码链接且未见开源承诺字样。

## 当前提炼状态

- [x] 公众号正文抓取（2026-10-28）
- [x] 59 篇独立节点（55 新建 / 4 复用）
- [x] 项目页 / 仓库开源状态核查（步骤 2.5）
- [ ] 原文实验表格核验（节点内数字仅为公众号转述）
