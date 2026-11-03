# 机器人不仅要完成任务，还要知道哪里出了错

> 来源归档（blog / 微信公众号）

- **标题：** 机器人不仅要完成任务，还要知道哪里出了错
- **类型：** blog
- **作者：** 具身智能小站（微信公众号）
- **原始链接：** https://mp.weixin.qq.com/s/KBuNOCHIWGw905j4IpOhZw
- **发表日期：** 2026-11-03
- **入库日期：** 2026-11-03
- **抓取方式：** `scripts/fetch_wechat_article.py`（Chromium + 微信 UA）
- **原始抓取落盘：** [`sources/raw/wechat_embodied_station_49_papers_sacq_action_aware_quantization_2026-11-03.md`](../raw/wechat_embodied_station_49_papers_sacq_action_aware_quantization_2026-11-03.md)
- **一句话说明：** 本期主线是让机器人不只完成任务、还要知道哪里出了错：SACQ 把量化目标对齐最终动作误差，另有失败模式定位、操作质量与仿真接触伪影审计、可验证自主等工作；其余覆盖策略部署、评测数据、人形与灵巧控制、导航与优化。深读 SACQ，跟进可达性采样极限与仿真驱动膝关节外骨骼。 共 49 篇；**49/49 独立详情节点**（本 ingest **新建 44**、**复用 5**）。

## 49 篇 → 本库节点

| # | 组 | 论文 | arXiv | 开源结论 | wiki |
|---|----|------|-------|----------|------|
| 01 | 策略学习与部署 | SACQ | [2610.41408](https://arxiv.org/abs/2610.41408) | **已开源** | [paper-sacq](../../wiki/entities/paper-sacq.md) |
| 02 | 规划、优化与感知基础 | Sampling-Based Reachability Limits | [2610.40473](https://arxiv.org/abs/2610.40473) | **已开源** | [paper-sampling-based-reachability-limits](../../wiki/entities/paper-sampling-based-reachability-limits.md) |
| 03 | 外骨骼与教程 | Knee Exoskeleton Sim-Derived Torque | [2610.41981](https://arxiv.org/abs/2610.41981) | **已开源** | [paper-knee-exoskeleton-sim-derived-torque](../../wiki/entities/paper-knee-exoskeleton-sim-derived-torque.md) |
| 04 | 策略学习与部署 | Auriga Technical Report | [2611.01112](https://arxiv.org/abs/2611.01112) | **已开源** | [paper-auriga-technical-report](../../wiki/entities/paper-auriga-technical-report.md) |
| 05 | 策略学习与部署 | ResMerge | [2610.41663](https://arxiv.org/abs/2610.41663) | **已开源** | [paper-resmerge](../../wiki/entities/paper-resmerge.md) |
| 06 | 人形与灵巧控制 | ∇Mimic | [2610.41654](https://arxiv.org/abs/2610.41654) | **已开源** | [paper-nablamimic](../../wiki/entities/paper-nablamimic.md) |
| 07 | 评测、数据与基准 | MolmoSpaces-CommonSense | [2610.41377](https://arxiv.org/abs/2610.41377) | **已开源** | [paper-molmospaces-commonsense](../../wiki/entities/paper-molmospaces-commonsense.md) |
| 08 | 规划、优化与感知基础 | CuPIQP | [2610.41021](https://arxiv.org/abs/2610.41021) | **已开源** | [paper-cupiqp](../../wiki/entities/paper-cupiqp.md) |
| 09 | 策略学习与部署 | STAR | [2610.41009](https://arxiv.org/abs/2610.41009) | **已开源** | [paper-star](../../wiki/entities/paper-star.md) |
| 10 | 策略学习与部署 | CaPtain | [2610.40799](https://arxiv.org/abs/2610.40799) | **已开源** | [paper-captain](../../wiki/entities/paper-captain.md) |
| 11 | 评测、数据与基准 | Gym-Anything for Robotics | [2610.40565](https://arxiv.org/abs/2610.40565) | **已开源** | [paper-gym-anything-for-robotics](../../wiki/entities/paper-gym-anything-for-robotics.md) |
| 12 | 评测、数据与基准 | RoboAtom | [2610.40207](https://arxiv.org/abs/2610.40207) | **已开源** | [paper-roboatom-atomic-action-supervision](../../wiki/entities/paper-roboatom-atomic-action-supervision.md)（**复用**） |
| 13 | 规划、优化与感知基础 | Differentiable ILC Black-Box Manipulators | [2611.01220](https://arxiv.org/abs/2611.01220) | **项目页未列 GitHub** | [paper-differentiable-ilc-black-box-manipulators](../../wiki/entities/paper-differentiable-ilc-black-box-manipulators.md) |
| 14 | 四足与导航 | Decentralized Collision Verification Swarms | [2611.01081](https://arxiv.org/abs/2611.01081) | **已开源** | [paper-decentralized-collision-verification-swarms](../../wiki/entities/paper-decentralized-collision-verification-swarms.md) |
| 15 | 策略学习与部署 | Validation Loss vs Real-Robot Success | [2611.00283](https://arxiv.org/abs/2611.00283) | **项目页未列 GitHub** | [paper-validation-loss-vs-real-robot-success](../../wiki/entities/paper-validation-loss-vs-real-robot-success.md) |
| 16 | 人形与灵巧控制 | ROOF | [2610.42097](https://arxiv.org/abs/2610.42097) | **项目页未列 GitHub** | [paper-roof](../../wiki/entities/paper-roof.md) |
| 17 | 世界模型、记忆与交互 | WorldBridge | [2610.40748](https://arxiv.org/abs/2610.40748) | **项目页未列 GitHub** | [paper-worldbridge](../../wiki/entities/paper-worldbridge.md) |
| 18 | 人形与灵巧控制 | TouchBridge | [2610.40349](https://arxiv.org/abs/2610.40349) | **已开源** | [paper-touchbridge-contact-aware-dexterous-human-interaction](../../wiki/entities/paper-touchbridge-contact-aware-dexterous-human-interaction.md)（**复用**） |
| 19 | 评测、数据与基准 | Shades of Success | [2611.01250](https://arxiv.org/abs/2611.01250) | **项目页未列 GitHub** | [paper-shades-of-success](../../wiki/entities/paper-shades-of-success.md) |
| 20 | 世界模型、记忆与交互 | MOST | [2611.00734](https://arxiv.org/abs/2611.00734) | **已开源** | [paper-most](../../wiki/entities/paper-most.md) |
| 21 | 策略学习与部署 | Editing the Goal | [2611.00602](https://arxiv.org/abs/2611.00602) | **项目页未列 GitHub** | [paper-editing-the-goal](../../wiki/entities/paper-editing-the-goal.md) |
| 22 | 世界模型、记忆与交互 | Knowing Where to Look | [2611.00499](https://arxiv.org/abs/2611.00499) | **项目页未列 GitHub** | [paper-knowing-where-to-look](../../wiki/entities/paper-knowing-where-to-look.md) |
| 23 | 评测、数据与基准 | TacMind | [2611.00455](https://arxiv.org/abs/2611.00455) | **已开源** | [paper-tacmind](../../wiki/entities/paper-tacmind.md) |
| 24 | 评测、数据与基准 | Behavior Graphs | [2611.00405](https://arxiv.org/abs/2611.00405) | **项目页未列 GitHub** | [paper-behavior-graphs](../../wiki/entities/paper-behavior-graphs.md) |
| 25 | 世界模型、记忆与交互 | Teach, Abstract, Apply | [2611.00297](https://arxiv.org/abs/2611.00297) | **已开源** | [paper-teach-abstract-apply](../../wiki/entities/paper-teach-abstract-apply.md) |
| 26 | 人形与灵巧控制 | OmniPiano | [2611.00243](https://arxiv.org/abs/2611.00243) | **已开源** | [paper-omnipiano](../../wiki/entities/paper-omnipiano.md) |
| 27 | 人形与灵巧控制 | Anchor 1 | [2611.00193](https://arxiv.org/abs/2611.00193) | **待发布** | [paper-anchor-1](../../wiki/entities/paper-anchor-1.md) |
| 28 | 策略学习与部署 | Scaling Trusted Autonomy | [2611.00158](https://arxiv.org/abs/2611.00158) | **已开源** | [paper-scaling-trusted-autonomy](../../wiki/entities/paper-scaling-trusted-autonomy.md) |
| 29 | 策略学习与部署 | RoboArchitect | [2610.42141](https://arxiv.org/abs/2610.42141) | **已开源** | [paper-roboarchitect](../../wiki/entities/paper-roboarchitect.md) |
| 30 | 评测、数据与基准 | Sensorium | [2610.41946](https://arxiv.org/abs/2610.41946) | **已开源** | [paper-sensorium](../../wiki/entities/paper-sensorium.md) |
| 31 | 人形与灵巧控制 | Movens-Zero | [2610.41877](https://arxiv.org/abs/2610.41877) | **已开源** | [paper-movens-zero](../../wiki/entities/paper-movens-zero.md) |
| 32 | 评测、数据与基准 | Simulator Contact Artifacts Audit | [2610.41857](https://arxiv.org/abs/2610.41857) | **已开源** | [paper-simulator-contact-artifacts-audit](../../wiki/entities/paper-simulator-contact-artifacts-audit.md) |
| 33 | 规划、优化与感知基础 | OpenVIO | [2610.41671](https://arxiv.org/abs/2610.41671) | **已开源** | [paper-openvio](../../wiki/entities/paper-openvio.md) |
| 34 | 策略学习与部署 | RTPolicy | [2610.41625](https://arxiv.org/abs/2610.41625) | **已开源** | [paper-rtpolicy](../../wiki/entities/paper-rtpolicy.md) |
| 35 | 规划、优化与感知基础 | Contact-Based Million-Robot Planning | [2610.41548](https://arxiv.org/abs/2610.41548) | **已开源** | [paper-contact-based-million-robot-planning](../../wiki/entities/paper-contact-based-million-robot-planning.md) |
| 36 | 策略学习与部署 | BioAgent | [2610.41383](https://arxiv.org/abs/2610.41383) | **待发布** | [paper-bioagent](../../wiki/entities/paper-bioagent.md) |
| 37 | 人形与灵巧控制 | HandMimicX | [2610.41336](https://arxiv.org/abs/2610.41336) | **项目页未列 GitHub** | [paper-handmimicx](../../wiki/entities/paper-handmimicx.md) |
| 38 | 规划、优化与感知基础 | RapidDOC | [2610.41324](https://arxiv.org/abs/2610.41324) | **已开源** | [paper-rapiddoc](../../wiki/entities/paper-rapiddoc.md) |
| 39 | 人形与灵巧控制 | Contact Imitation Interface | [2610.41028](https://arxiv.org/abs/2610.41028) | **已开源** | [paper-contact-imitation-interface](../../wiki/entities/paper-contact-imitation-interface.md) |
| 40 | 规划、优化与感知基础 | FlapSim | [2610.40902](https://arxiv.org/abs/2610.40902) | **已开源** | [paper-flapsim](../../wiki/entities/paper-flapsim.md) |
| 41 | 人形与灵巧控制 | SwapBot | [2610.40646](https://arxiv.org/abs/2610.40646) | **项目页未列 GitHub** | [paper-swapbot](../../wiki/entities/paper-swapbot.md) |
| 42 | 外骨骼与教程 | Shape Servoing Tutorial Part I | [2610.40559](https://arxiv.org/abs/2610.40559) | **已开源** | [paper-shape-servoing-deformable-objects-tutorial-part1](../../wiki/entities/paper-shape-servoing-deformable-objects-tutorial-part1.md)（**复用**） |
| 43 | 评测、数据与基准 | BURGER | [2610.40532](https://arxiv.org/abs/2610.40532) | **已开源** | [paper-burger](../../wiki/entities/paper-burger.md) |
| 44 | 四足与导航 | UED Quadruped Benchmark | [2610.40364](https://arxiv.org/abs/2610.40364) | **已开源** | [paper-unsupervised-environment-design-quadruped-benchmark](../../wiki/entities/paper-unsupervised-environment-design-quadruped-benchmark.md)（**复用**） |
| 45 | 四足与导航 | LLM-Guided Planar Loco-Manipulation | [2610.40248](https://arxiv.org/abs/2610.40248) | **已开源** | [paper-llm-guided-planar-loco-manipulation-quadruped](../../wiki/entities/paper-llm-guided-planar-loco-manipulation-quadruped.md)（**复用**） |
| 46 | 四足与导航 | Dont Risk It | [2610.40050](https://arxiv.org/abs/2610.40050) | **已开源** | [paper-dont-risk-it-crowd-navigation](../../wiki/entities/paper-dont-risk-it-crowd-navigation.md) |
| 47 | 评测、数据与基准 | ClothDojo | [2611.01190](https://arxiv.org/abs/2611.01190) | **部分开源** | [paper-clothdojo](../../wiki/entities/paper-clothdojo.md) |
| 48 | 规划、优化与感知基础 | VI RGB-D SLAM with Encoders | [2610.41768](https://arxiv.org/abs/2610.41768) | **部分开源** | [paper-vi-rgb-d-slam-with-encoders](../../wiki/entities/paper-vi-rgb-d-slam-with-encoders.md) |
| 49 | 规划、优化与感知基础 | GPU Incremental EDT | [2610.41667](https://arxiv.org/abs/2610.41667) | **已开源** | [paper-gpu-incremental-edt](../../wiki/entities/paper-gpu-incremental-edt.md) |

## 对 wiki 的映射

- **49/49 独立详情节点**；同 arXiv 不重复造页（复用节点已补本文来源与地图关联）。
- 阅读坐标：[动作感知量化、失败定位与可验证自主：具身智能小站 11-03 期 49 篇阅读坐标](../../wiki/overview/embodied-station-2026-11-03-49-papers-technology-map.md)
- 开源结论口径：GitHub 仓库用 `git ls-remote` 核实可达；项目页用 curl 读取扫描 GitHub 链接与 "coming soon"；「项目页未列 GitHub」表示入库日页面无代码链接且未见开源承诺字样。

## 当前提炼状态

- [x] 公众号正文抓取（2026-11-03）
- [x] 49 篇独立节点（44 新建 / 5 复用）
- [x] 项目页 / 仓库开源状态核查（步骤 2.5）
- [ ] 原文实验表格核验（节点内数字仅为公众号转述）
