# 一张照片变成仿真场景，机器人真能用吗？｜附代码与项目入口

> 来源归档（blog / 微信公众号）

- **标题：** 一张照片变成仿真场景，机器人真能用吗？｜附代码与项目入口
- **类型：** blog
- **作者：** 具身智能小站（微信公众号）
- **原始链接：** https://mp.weixin.qq.com/s/qUVNn3FiT89CDRzqmfN-0Q
- **发表日期：** 2026-10-31
- **入库日期：** 2026-10-31
- **抓取方式：** `scripts/fetch_wechat_article.py`（Chromium + 微信 UA）
- **原始抓取落盘：** [`sources/raw/wechat_embodied_station_50_papers_scenerig_photo_to_sim_2026-10-31.md`](../raw/wechat_embodied_station_50_papers_scenerig_photo_to_sim_2026-10-31.md)
- **一句话说明：** 本期主线是让仿真与评测经得起真实机器人检验：SceneRig 把物理验证放进重建循环，VLA-Retain 把适配收益与非目标退化分开，Ropedia W0 用人类视频补跨形态世界模型；其余覆盖触觉安全、策略可靠性、人形交互与移动控制。深读 SceneRig，跟进 VLA-Retain 与 Ropedia W0。 共 50 篇；**50/50 独立详情节点**（本 ingest **新建 34**、**复用 16**）。

## 50 篇 → 本库节点

| # | 组 | 论文 | arXiv | 开源结论 | wiki |
|---|----|------|-------|----------|------|
| 01 | 世界模型与仿真重建 | SceneRig | [2610.28585](https://arxiv.org/abs/2610.28585) | **已开源** | [paper-scenerig](../../wiki/entities/paper-scenerig.md) |
| 02 | 评测、基准与可靠性 | VLA-Retain | [2610.27698](https://arxiv.org/abs/2610.27698) | **已开源** | [paper-vla-retain](../../wiki/entities/paper-vla-retain.md) |
| 03 | 世界模型与仿真重建 | Ropedia W0 | [2610.26855](https://arxiv.org/abs/2610.26855) | **已开源** | [paper-ropedia-w0](../../wiki/entities/paper-ropedia-w0.md)（**复用**） |
| 04 | 策略学习与 VLA | CEL Fleet Experience | [2610.25736](https://arxiv.org/abs/2610.25736) | **部分开源** | [paper-cel-fleet-experience](../../wiki/entities/paper-cel-fleet-experience.md) |
| 05 | 触觉与精细操作 | PhantomTac | [2610.26174](https://arxiv.org/abs/2610.26174) | **项目页未列 GitHub** | [paper-phantomtac](../../wiki/entities/paper-phantomtac.md) |
| 06 | 评测、基准与可靠性 | HandsOn | [2610.26765](https://arxiv.org/abs/2610.26765) | **已开源** | [paper-handson](../../wiki/entities/paper-handson.md)（**复用**） |
| 07 | 评测、基准与可靠性 | RoboValue | [2610.26663](https://arxiv.org/abs/2610.26663) | **已开源** | [paper-robovalue-benchmark](../../wiki/entities/paper-robovalue-benchmark.md)（**复用**） |
| 08 | 世界模型与仿真重建 | LaLa-World | [2610.27851](https://arxiv.org/abs/2610.27851) | **已开源** | [paper-lala-world](../../wiki/entities/paper-lala-world.md) |
| 09 | 策略学习与 VLA | VaDar | [2610.26629](https://arxiv.org/abs/2610.26629) | **已开源** | [paper-vadar](../../wiki/entities/paper-vadar.md)（**复用**） |
| 10 | 评测、基准与可靠性 | VLA Corruption Failure Attribution | [2610.27277](https://arxiv.org/abs/2610.27277) | **项目页未列 GitHub** | [paper-vla-corruption-failure-attribution](../../wiki/entities/paper-vla-corruption-failure-attribution.md) |
| 11 | 策略学习与 VLA | LMM Direct Robot Control | [2610.28259](https://arxiv.org/abs/2610.28259) | **已开源** | [paper-lmm-direct-robot-control](../../wiki/entities/paper-lmm-direct-robot-control.md) |
| 12 | 触觉与精细操作 | Tactile In-Context Learning Agents | [2610.27613](https://arxiv.org/abs/2610.27613) | **已开源** | [paper-tactile-in-context-learning-agents](../../wiki/entities/paper-tactile-in-context-learning-agents.md) |
| 13 | 评测、基准与可靠性 | Embodied Playground | [2610.27039](https://arxiv.org/abs/2610.27039) | **已开源** | [paper-embodied-playground](../../wiki/entities/paper-embodied-playground.md) |
| 14 | 数据采集与示范迁移 | Head-Cam UMI Glove | [2610.27622](https://arxiv.org/abs/2610.27622) | **已开源** | [paper-head-cam-umi-glove](../../wiki/entities/paper-head-cam-umi-glove.md) |
| 15 | 世界模型与仿真重建 | Plausible Future, Wrong Choice | [2610.28431](https://arxiv.org/abs/2610.28431) | **项目页未列 GitHub** | [paper-plausible-future-wrong-choice](../../wiki/entities/paper-plausible-future-wrong-choice.md) |
| 16 | 策略学习与 VLA | EventSync | [2610.27917](https://arxiv.org/abs/2610.27917) | **待发布** | [paper-eventsync](../../wiki/entities/paper-eventsync.md) |
| 17 | 策略学习与 VLA | Multi-View Latent Action Pretraining | [2610.27794](https://arxiv.org/abs/2610.27794) | **已开源** | [paper-multi-view-latent-action-pretraining-cross-embodiment](../../wiki/entities/paper-multi-view-latent-action-pretraining-cross-embodiment.md)（**复用**） |
| 18 | 数据采集与示范迁移 | DexOAK | [2610.26762](https://arxiv.org/abs/2610.26762) | **项目页未列 GitHub** | [paper-dexoak](../../wiki/entities/paper-dexoak.md) |
| 19 | 世界模型与仿真重建 | TrajWAM | [2610.25763](https://arxiv.org/abs/2610.25763) | **已开源** | [paper-trajwam](../../wiki/entities/paper-trajwam.md) |
| 20 | 策略学习与 VLA | Reinforced ILC Elastic Joint Arms | [2610.28329](https://arxiv.org/abs/2610.28329) | **已开源** | [paper-reinforced-ilc-elastic-joint-arms](../../wiki/entities/paper-reinforced-ilc-elastic-joint-arms.md) |
| 21 | 策略学习与 VLA | Learning to Be Different | [2610.27194](https://arxiv.org/abs/2610.27194) | **已开源** | [paper-learning-to-be-different](../../wiki/entities/paper-learning-to-be-different.md) |
| 22 | 导航、感知与多机器人 | CoopUQ | [2610.26852](https://arxiv.org/abs/2610.26852) | **已开源** | [paper-coopuq](../../wiki/entities/paper-coopuq.md) |
| 23 | 导航、感知与多机器人 | GPU-Resident Multi-Robot Inspection | [2610.26106](https://arxiv.org/abs/2610.26106) | **已开源** | [paper-gpu-resident-multi-robot-inspection](../../wiki/entities/paper-gpu-resident-multi-robot-inspection.md) |
| 24 | 策略学习与 VLA | VLDA | [2610.25874](https://arxiv.org/abs/2610.25874) | **已开源** | [paper-vlda](../../wiki/entities/paper-vlda.md) |
| 25 | 触觉与精细操作 | Touch-JEPA | [2610.28519](https://arxiv.org/abs/2610.28519) | **部分开源** | [paper-touch-jepa](../../wiki/entities/paper-touch-jepa.md) |
| 26 | 世界模型与仿真重建 | NeoWorld-3 | [2610.27690](https://arxiv.org/abs/2610.27690) | **部分开源** | [paper-neoworld-3](../../wiki/entities/paper-neoworld-3.md) |
| 27 | 世界模型与仿真重建 | Physical Words | [2610.26854](https://arxiv.org/abs/2610.26854) | **已开源** | [paper-physical-words-latent-reasoning-manipulation](../../wiki/entities/paper-physical-words-latent-reasoning-manipulation.md)（**复用**） |
| 28 | 评测、基准与可靠性 | Runtime Safety Check Robot Loops | [2610.27350](https://arxiv.org/abs/2610.27350) | **已开源** | [paper-runtime-safety-check-robot-loops](../../wiki/entities/paper-runtime-safety-check-robot-loops.md) |
| 29 | 触觉与精细操作 | GPT-6 Astra Visuo-Tactile Policy | [2610.26846](https://arxiv.org/abs/2610.26846) | **已开源** | [paper-tactalent-gpt-6-astra-visuo-tactile](../../wiki/entities/paper-tactalent-gpt-6-astra-visuo-tactile.md) |
| 30 | 数据采集与示范迁移 | Fast Motion Retargeting | [2610.28470](https://arxiv.org/abs/2610.28470) | **已开源** | [paper-fast-motion-retargeting-pipeline-humanoid](../../wiki/entities/paper-fast-motion-retargeting-pipeline-humanoid.md)（**复用**） |
| 31 | 人形交互与移动控制 | Wheel Motion Uncertainty LIO | [2610.26550](https://arxiv.org/abs/2610.26550) | **已开源** | [paper-wheel-motion-uncertainty-degeneracy-aware-lio](../../wiki/entities/paper-wheel-motion-uncertainty-degeneracy-aware-lio.md)（**复用**） |
| 32 | 策略学习与 VLA | ReDOA | [2610.25691](https://arxiv.org/abs/2610.25691) | **项目页未列 GitHub** | [paper-redoa](../../wiki/entities/paper-redoa.md) |
| 33 | 触觉与精细操作 | FR-VLA | [2610.25981](https://arxiv.org/abs/2610.25981) | **待发布** | [paper-fr-vla](../../wiki/entities/paper-fr-vla.md) |
| 34 | 人形交互与移动控制 | REACT | [2610.28523](https://arxiv.org/abs/2610.28523) | **已开源** | [paper-vlm-contact-retargeting-partner-modeling](../../wiki/entities/paper-vlm-contact-retargeting-partner-modeling.md)（**复用**） |
| 35 | 人形交互与移动控制 | OmniInteract | [2610.26043](https://arxiv.org/abs/2610.26043) | **已开源** | [paper-omniinteract-dual-humanoids](../../wiki/entities/paper-omniinteract-dual-humanoids.md)（**复用**） |
| 36 | 评测、基准与可靠性 | ForagerSim | [2610.26395](https://arxiv.org/abs/2610.26395) | **已开源** | [paper-foragersim](../../wiki/entities/paper-foragersim.md) |
| 37 | 人形交互与移动控制 | LocoPilot | [2610.27998](https://arxiv.org/abs/2610.27998) | **已开源** | [paper-locopilot-bilateral-teleop-loco-manipulation](../../wiki/entities/paper-locopilot-bilateral-teleop-loco-manipulation.md)（**复用**） |
| 38 | 评测、基准与可靠性 | K-AoI | [2610.26201](https://arxiv.org/abs/2610.26201) | **已开源** | [paper-k-aoi](../../wiki/entities/paper-k-aoi.md) |
| 39 | 评测、基准与可靠性 | Frontier LLMs RoboColiseum | [2610.27865](https://arxiv.org/abs/2610.27865) | **项目页未列 GitHub** | [paper-frontier-llms-robocoliseum](../../wiki/entities/paper-frontier-llms-robocoliseum.md) |
| 40 | 数据采集与示范迁移 | Novice Operators Keypoint Hand View | [2610.27907](https://arxiv.org/abs/2610.27907) | **项目页未列 GitHub** | [paper-novice-operators-keypoint-hand-view](../../wiki/entities/paper-novice-operators-keypoint-hand-view.md) |
| 41 | 策略学习与 VLA | AWLS | [2610.27736](https://arxiv.org/abs/2610.27736) | **已开源** | [paper-awls-advantage-weighted-latent-steering](../../wiki/entities/paper-awls-advantage-weighted-latent-steering.md)（**复用**） |
| 42 | 导航、感知与多机器人 | VEOcc++ | [2610.27704](https://arxiv.org/abs/2610.27704) | **待发布** | [paper-veocc-plus-plus](../../wiki/entities/paper-veocc-plus-plus.md) |
| 43 | 人形交互与移动控制 | SkateNav | [2610.25962](https://arxiv.org/abs/2610.25962) | **已开源** | [paper-skatenav-passive-wheel-quadruped-skating](../../wiki/entities/paper-skatenav-passive-wheel-quadruped-skating.md)（**复用**） |
| 44 | 人形交互与移动控制 | SWAP | [2610.25711](https://arxiv.org/abs/2610.25711) | **已开源** | [paper-swap-humanoid-collision-avoidance](../../wiki/entities/paper-swap-humanoid-collision-avoidance.md)（**复用**） |
| 45 | 人形交互与移动控制 | Spot Bricklaying | [2610.27472](https://arxiv.org/abs/2610.27472) | **已开源** | [paper-autonomous-bricklaying-boston-dynamics-spot](../../wiki/entities/paper-autonomous-bricklaying-boston-dynamics-spot.md)（**复用**） |
| 46 | 评测、基准与可靠性 | FIT-MAC | [2610.28109](https://arxiv.org/abs/2610.28109) | **部分开源** | [paper-fit-mac](../../wiki/entities/paper-fit-mac.md) |
| 47 | 触觉与精细操作 | Biomimetic Skin Sensory Arrangements | [2610.26680](https://arxiv.org/abs/2610.26680) | **项目页未列 GitHub** | [paper-biomimetic-skin-sensory-arrangements](../../wiki/entities/paper-biomimetic-skin-sensory-arrangements.md) |
| 48 | 触觉与精细操作 | GPT-6 Astra Tactile Correction | [2610.28457](https://arxiv.org/abs/2610.28457) | **待发布** | [paper-gpt-6-astra-tactile-correction](../../wiki/entities/paper-gpt-6-astra-tactile-correction.md) |
| 49 | 评测、基准与可靠性 | HATCH | [2610.27485](https://arxiv.org/abs/2610.27485) | **已开源** | [paper-hatch](../../wiki/entities/paper-hatch.md) |
| 50 | 数据采集与示范迁移 | Feature-Level LfD for Humanoids | [2610.26170](https://arxiv.org/abs/2610.26170) | **已开源** | [paper-feature-level-lfd-prompt-detection-nao](../../wiki/entities/paper-feature-level-lfd-prompt-detection-nao.md)（**复用**） |

## 对 wiki 的映射

- **50/50 独立详情节点**；同 arXiv 不重复造页（复用节点已补本文来源与地图关联）。
- 阅读坐标：[单图重建仿真、VLA 适配评测与跨形态世界模型：具身智能小站 10-31 期 50 篇阅读坐标](../../wiki/overview/embodied-station-2026-10-31-50-papers-technology-map.md)
- 开源结论口径：GitHub 仓库用 `git ls-remote` 核实可达；项目页用 curl 读取扫描 GitHub 链接与 "coming soon"；「项目页未列 GitHub」表示入库日页面无代码链接且未见开源承诺字样。

## 当前提炼状态

- [x] 公众号正文抓取（2026-10-31）
- [x] 50 篇独立节点（34 新建 / 16 复用）
- [x] 项目页 / 仓库开源状态核查（步骤 2.5）
- [ ] 原文实验表格核验（节点内数字仅为公众号转述）
