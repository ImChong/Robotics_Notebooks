# 模型还在思考，目标已经移动：机器人评测该把时间算进来吗？

> 来源归档（blog / 微信公众号）

- **标题：** 模型还在思考，目标已经移动：机器人评测该把时间算进来吗？
- **类型：** blog
- **作者：** 具身智能小站（微信公众号）
- **原始链接：** https://mp.weixin.qq.com/s/1XR48_smtmSoYQgYSXPKaw
- **发表日期：** 2026-11-01
- **入库日期：** 2026-11-01
- **抓取方式：** `scripts/fetch_wechat_article.py`（Chromium + 微信 UA）
- **原始抓取落盘：** [`sources/raw/wechat_embodied_station_56_papers_libero_dynamic_latency_eval_2026-11-01.md`](../raw/wechat_embodied_station_56_papers_libero_dynamic_latency_eval_2026-11-01.md)
- **一句话说明：** 本期主线是评测与适配要把真实时间与隐藏环境因素算进来：LIBERO-Dynamic 让推理延迟与目标运动同台评测，RAX 把探索经验编译成可复用程序，T2-z 在测试时适配摩擦与质量等环境潜变量；其余覆盖鲁棒与个性化基准、世界模型、VLA 后训练、导航、灵巧接触与人形控制。深读 LIBERO-Dynamic，跟进 RAX 与 T2-z。 共 56 篇；**56/56 独立详情节点**（本 ingest **新建 43**、**复用 13**）。

## 56 篇 → 本库节点

| # | 组 | 论文 | arXiv | 开源结论 | wiki |
|---|----|------|-------|----------|------|
| 01 | 评测与基准 | LIBERO-Dynamic | [2610.24579](https://arxiv.org/abs/2610.24579) | **已开源** | [paper-libero-dynamic](../../wiki/entities/paper-libero-dynamic.md) |
| 02 | 具身智能体与导航 | RAX | [2610.22496](https://arxiv.org/abs/2610.22496) | **已开源** | [rax-scaling-online-exploration](../../wiki/entities/rax-scaling-online-exploration.md)（**复用**） |
| 03 | 世界模型与在线适配 | T2-z | [2610.22364](https://arxiv.org/abs/2610.22364) | **已开源** | [paper-t2-z-environment-latents](../../wiki/entities/paper-t2-z-environment-latents.md)（**复用**） |
| 04 | 评测与基准 | LIBERO-Pro2.0 | [2610.24587](https://arxiv.org/abs/2610.24587) | **已开源** | [paper-libero-pro2-0](../../wiki/entities/paper-libero-pro2-0.md) |
| 05 | VLA 与策略学习 | Goal Images Are Control Targets | [2610.24885](https://arxiv.org/abs/2610.24885) | **已开源** | [paper-goal-images-are-control-targets](../../wiki/entities/paper-goal-images-are-control-targets.md) |
| 06 | VLA 与策略学习 | MambaAct | [2610.22951](https://arxiv.org/abs/2610.22951) | **已开源** | [paper-mambaact](../../wiki/entities/paper-mambaact.md) |
| 07 | VLA 与策略学习 | Value as Memory | [2610.22368](https://arxiv.org/abs/2610.22368) | **已开源** | [paper-vam-value-as-memory](../../wiki/entities/paper-vam-value-as-memory.md)（**复用**） |
| 08 | 具身智能体与导航 | ScaleNav | [2610.25690](https://arxiv.org/abs/2610.25690) | **已开源** | [paper-scalenav](../../wiki/entities/paper-scalenav.md) |
| 09 | 评测与基准 | BiDex-100 | [2610.25459](https://arxiv.org/abs/2610.25459) | **已开源** | [paper-bidex-100](../../wiki/entities/paper-bidex-100.md) |
| 10 | 世界模型与在线适配 | Dream What I Can | [2610.24981](https://arxiv.org/abs/2610.24981) | **已开源** | [paper-dreamcan-sparse-visual-planning](../../wiki/entities/paper-dreamcan-sparse-visual-planning.md)（**复用**） |
| 11 | 世界模型与在线适配 | Fréchet Forcing | [2610.25666](https://arxiv.org/abs/2610.25666) | **已开源** | [paper-frechet-forcing](../../wiki/entities/paper-frechet-forcing.md) |
| 12 | 灵巧、接触与精细操作 | Blind Precise Assembly | [2610.23849](https://arxiv.org/abs/2610.23849) | **已开源** | [paper-blind-proprioceptive-assembly](../../wiki/entities/paper-blind-proprioceptive-assembly.md)（**复用**） |
| 13 | VLA 与策略学习 | VLA-Fabric | [2610.23018](https://arxiv.org/abs/2610.23018) | **已开源** | [paper-vla-fabric](../../wiki/entities/paper-vla-fabric.md) |
| 14 | 感知、规划与系统 | CusADi2 | [2610.25476](https://arxiv.org/abs/2610.25476) | **已开源** | [paper-cusadi2](../../wiki/entities/paper-cusadi2.md) |
| 15 | 感知、规划与系统 | PrismPlan | [2610.23884](https://arxiv.org/abs/2610.23884) | **已开源** | [paper-prismplan](../../wiki/entities/paper-prismplan.md) |
| 16 | 评测与基准 | ContactBench 1.0 | [2610.24512](https://arxiv.org/abs/2610.24512) | **待发布** | [paper-contactbench-1-0](../../wiki/entities/paper-contactbench-1-0.md) |
| 17 | 具身智能体与导航 | Learning to Compose | [2610.24464](https://arxiv.org/abs/2610.24464) | **已开源** | [paper-learning-to-compose](../../wiki/entities/paper-learning-to-compose.md) |
| 18 | 具身智能体与导航 | HexaShell | [2610.23479](https://arxiv.org/abs/2610.23479) | **已开源** | [paper-hexashell](../../wiki/entities/paper-hexashell.md)（**复用**） |
| 19 | 世界模型与在线适配 | LAW | [2610.23334](https://arxiv.org/abs/2610.23334) | **已开源** | [paper-law](../../wiki/entities/paper-law.md)（**复用**） |
| 20 | 评测与基准 | GPT-6-Astra Embodied Intelligence | [2610.22471](https://arxiv.org/abs/2610.22471) | **已开源** | [paper-gpt-6-astra-embodied-intelligence](../../wiki/entities/paper-gpt-6-astra-embodied-intelligence.md) |
| 21 | 灵巧、接触与精细操作 | ESCAPE | [2610.23943](https://arxiv.org/abs/2610.23943) | **已开源** | [paper-escape](../../wiki/entities/paper-escape.md) |
| 22 | VLA 与策略学习 | Noise-Invariant Spatial Encoding | [2610.24674](https://arxiv.org/abs/2610.24674) | **项目页未列 GitHub** | [paper-noise-invariant-spatial-encoding](../../wiki/entities/paper-noise-invariant-spatial-encoding.md) |
| 23 | 具身智能体与导航 | REAP | [2610.23973](https://arxiv.org/abs/2610.23973) | **已开源** | [paper-reap](../../wiki/entities/paper-reap.md) |
| 24 | 灵巧、接触与精细操作 | MoDex | [2610.23888](https://arxiv.org/abs/2610.23888) | **已开源** | [paper-modex](../../wiki/entities/paper-modex.md) |
| 25 | 灵巧、接触与精细操作 | No Eyes, No Problem | [2610.23438](https://arxiv.org/abs/2610.23438) | **已开源** | [paper-no-eyes-no-problem](../../wiki/entities/paper-no-eyes-no-problem.md) |
| 26 | VLA 与策略学习 | SteerAct | [2610.25632](https://arxiv.org/abs/2610.25632) | **已开源** | [paper-steeract-few-shot-temporal-steering](../../wiki/entities/paper-steeract-few-shot-temporal-steering.md)（**复用**） |
| 27 | 人形与腿足控制 | Humanoid SRL | [2610.25628](https://arxiv.org/abs/2610.25628) | **已开源** | [paper-humanoid-srl](../../wiki/entities/paper-humanoid-srl.md)（**复用**） |
| 28 | 世界模型与在线适配 | BeJEPA | [2610.23127](https://arxiv.org/abs/2610.23127) | **已开源** | [paper-bejepa](../../wiki/entities/paper-bejepa.md)（**复用**） |
| 29 | 具身智能体与导航 | Spatial Memory by Agents | [2610.22736](https://arxiv.org/abs/2610.22736) | **待发布** | [paper-spatial-memory-by-agents](../../wiki/entities/paper-spatial-memory-by-agents.md) |
| 30 | 灵巧、接触与精细操作 | Contact-Aware Adaptive Impedance | [2610.24235](https://arxiv.org/abs/2610.24235) | **待发布** | [paper-contact-aware-adaptive-impedance](../../wiki/entities/paper-contact-aware-adaptive-impedance.md) |
| 31 | 感知、规划与系统 | Structured Perception Off-road Dynamics | [2610.22900](https://arxiv.org/abs/2610.22900) | **已开源** | [paper-structured-perception-off-road-dynamics](../../wiki/entities/paper-structured-perception-off-road-dynamics.md) |
| 32 | VLA 与策略学习 | Flow-Reversed Action Anchors | [2610.22852](https://arxiv.org/abs/2610.22852) | **待发布** | [paper-flow-reversed-action-anchors](../../wiki/entities/paper-flow-reversed-action-anchors.md) |
| 33 | 具身智能体与导航 | HAVANA | [2610.24122](https://arxiv.org/abs/2610.24122) | **已开源** | [paper-havana](../../wiki/entities/paper-havana.md) |
| 34 | 感知、规划与系统 | MicroSplat | [2610.23899](https://arxiv.org/abs/2610.23899) | **项目页未列 GitHub** | [paper-microsplat](../../wiki/entities/paper-microsplat.md) |
| 35 | VLA 与策略学习 | NestedVLA | [2610.23271](https://arxiv.org/abs/2610.23271) | **已开源** | [paper-nestedvla](../../wiki/entities/paper-nestedvla.md) |
| 36 | 感知、规划与系统 | Shape-Aware Neural Time Fields | [2610.22762](https://arxiv.org/abs/2610.22762) | **已开源** | [paper-shape-aware-neural-time-fields](../../wiki/entities/paper-shape-aware-neural-time-fields.md) |
| 37 | VLA 与策略学习 | TILDE | [2610.25376](https://arxiv.org/abs/2610.25376) | **已开源** | [paper-tilde](../../wiki/entities/paper-tilde.md) |
| 38 | 世界模型与在线适配 | UMI-WM | [2610.23913](https://arxiv.org/abs/2610.23913) | **项目页未列 GitHub** | [paper-umi-wm](../../wiki/entities/paper-umi-wm.md) |
| 39 | 世界模型与在线适配 | ORD-WM | [2610.23156](https://arxiv.org/abs/2610.23156) | **项目页未列 GitHub** | [paper-ord-wm](../../wiki/entities/paper-ord-wm.md) |
| 40 | 人形与腿足控制 | FCTS | [2610.22619](https://arxiv.org/abs/2610.22619) | **已开源** | [paper-fcts-failure-guided-curriculum](../../wiki/entities/paper-fcts-failure-guided-curriculum.md)（**复用**） |
| 41 | 评测与基准 | LIBERO-PeRM | [2610.22372](https://arxiv.org/abs/2610.22372) | **已开源** | [paper-libero-perm-personalized-manipulation](../../wiki/entities/paper-libero-perm-personalized-manipulation.md)（**复用**） |
| 42 | 评测与基准 | Interventional Memory Evaluation VLA | [2610.23372](https://arxiv.org/abs/2610.23372) | **项目页未列 GitHub** | [paper-interventional-memory-evaluation-vla](../../wiki/entities/paper-interventional-memory-evaluation-vla.md) |
| 43 | 感知、规划与系统 | SpaRV | [2610.25665](https://arxiv.org/abs/2610.25665) | **已开源** | [paper-sparv](../../wiki/entities/paper-sparv.md) |
| 44 | VLA 与策略学习 | Relational Spatial Alignment VLA | [2610.24705](https://arxiv.org/abs/2610.24705) | **已开源** | [paper-relational-spatial-alignment-vla](../../wiki/entities/paper-relational-spatial-alignment-vla.md) |
| 45 | 世界模型与在线适配 | DefFlow | [2610.22827](https://arxiv.org/abs/2610.22827) | **待发布** | [paper-defflow](../../wiki/entities/paper-defflow.md) |
| 46 | 世界模型与在线适配 | ACTS | [2610.24285](https://arxiv.org/abs/2610.24285) | **项目页未列 GitHub** | [paper-acts](../../wiki/entities/paper-acts.md) |
| 47 | 具身智能体与导航 | Structure over Reasoning | [2610.25130](https://arxiv.org/abs/2610.25130) | **项目页未列 GitHub** | [paper-structure-over-reasoning](../../wiki/entities/paper-structure-over-reasoning.md) |
| 48 | 人形与腿足控制 | Compositional Motion Styles | [2610.24008](https://arxiv.org/abs/2610.24008) | **待发布** | [paper-compositional-motion-styles](../../wiki/entities/paper-compositional-motion-styles.md) |
| 49 | 灵巧、接触与精细操作 | Embodiment-Aware Heterogeneous Bimanual | [2610.23551](https://arxiv.org/abs/2610.23551) | **项目页未列 GitHub** | [paper-embodiment-aware-heterogeneous-bimanual](../../wiki/entities/paper-embodiment-aware-heterogeneous-bimanual.md) |
| 50 | 人形与腿足控制 | Giant Rubik's Cube Quadruped | [2610.24610](https://arxiv.org/abs/2610.24610) | **已开源** | [paper-rubik-skills](../../wiki/entities/paper-rubik-skills.md)（**复用**） |
| 51 | 人形与腿足控制 | Lightweight Contact Feedback Teleop | [2610.24257](https://arxiv.org/abs/2610.24257) | **项目页未列 GitHub** | [paper-lightweight-contact-feedback-teleop](../../wiki/entities/paper-lightweight-contact-feedback-teleop.md) |
| 52 | 感知、规划与系统 | Fourier Transmission Error Compensation | [2610.23882](https://arxiv.org/abs/2610.23882) | **项目页未列 GitHub** | [paper-fourier-transmission-error-compensation](../../wiki/entities/paper-fourier-transmission-error-compensation.md) |
| 53 | 感知、规划与系统 | Gearbox Lubricant Temperature Observer | [2610.23881](https://arxiv.org/abs/2610.23881) | **项目页未列 GitHub** | [paper-gearbox-lubricant-temperature-observer](../../wiki/entities/paper-gearbox-lubricant-temperature-observer.md) |
| 54 | 世界模型与在线适配 | LY-GWM-RoboCasa-Human300 | [2610.23274](https://arxiv.org/abs/2610.23274) | **已开源** | [paper-ly-gwm-robocasa-human300](../../wiki/entities/paper-ly-gwm-robocasa-human300.md) |
| 55 | 灵巧、接触与精细操作 | Pedicel Strawberry Harvesting | [2610.22765](https://arxiv.org/abs/2610.22765) | **已开源** | [paper-pedicel-strawberry-harvesting](../../wiki/entities/paper-pedicel-strawberry-harvesting.md) |
| 56 | 感知、规划与系统 | Multi-Robot Authoring | [2610.22670](https://arxiv.org/abs/2610.22670) | **项目页未列 GitHub** | [paper-multi-robot-authoring](../../wiki/entities/paper-multi-robot-authoring.md) |

## 对 wiki 的映射

- **56/56 独立详情节点**；同 arXiv 不重复造页（复用节点已补本文来源与地图关联）。
- 阅读坐标：[动态评测、技能编译与环境潜变量适配：具身智能小站 11-01 期 56 篇阅读坐标](../../wiki/overview/embodied-station-2026-11-01-56-papers-technology-map.md)
- 开源结论口径：GitHub 仓库用 `git ls-remote` 核实可达；项目页用 curl 读取扫描 GitHub 链接与 "coming soon"；「项目页未列 GitHub」表示入库日页面无代码链接且未见开源承诺字样。

## 当前提炼状态

- [x] 公众号正文抓取（2026-11-01）
- [x] 56 篇独立节点（43 新建 / 13 复用）
- [x] 项目页 / 仓库开源状态核查（步骤 2.5）
- [ ] 原文实验表格核验（节点内数字仅为公众号转述）
