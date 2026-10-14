# 机器人动作为什么总在换块时卡顿？｜附代码与项目入口

> 来源归档（blog / 微信公众号）

- **标题：** 机器人动作为什么总在换块时卡顿？｜附代码与项目入口
- **类型：** blog
- **作者：** 具身智能小站（微信公众号）
- **原始链接：** https://mp.weixin.qq.com/s/pxBq6lFyzH05LGiBkkR9mQ
- **发表日期：** 2026-10-14
- **入库日期：** 2026-10-14
- **抓取方式：** `scripts/fetch_wechat_article.py`（Chromium + 微信 UA）
- **原始抓取落盘：** [`sources/raw/wechat_embodied_station_41_papers_streamingwam_2026-10-14.md`](../raw/wechat_embodied_station_41_papers_streamingwam_2026-10-14.md)
- **一句话说明：** 「换块卡顿」是动作块策略部署的共性问题：相邻动作块是否接得上、新观察到达后能否持续细化，比单次推理更快更关键。本期深读 StreamingWAM，跟进 FreeBar 与 RoboCombine。 共 41 篇；**41/41 独立详情节点**（本 ingest **新建 41**、**复用 0**）。

## 41 篇 → 本库节点

| # | 组 | 论文 | arXiv | 开源结论 | wiki |
|---|----|------|-------|----------|------|
| 01 | VLA 与动作生成 | StreamingWAM | [2610.15585](https://arxiv.org/abs/2610.15585) | **已开源** | [paper-streamingwam](../../wiki/entities/paper-streamingwam.md) |
| 02 | VLA 与动作生成 | FreeBar | [2610.14997](https://arxiv.org/abs/2610.14997) | **已开源** | [paper-freebar-barrier-guidance-async-vla](../../wiki/entities/paper-freebar-barrier-guidance-async-vla.md) |
| 03 | 多机器人协作与评测 | RoboCombine | [2610.15425](https://arxiv.org/abs/2610.15425) | **部分开源** | [paper-robocombine-combinatorial-generalization](../../wiki/entities/paper-robocombine-combinatorial-generalization.md) |
| 04 | 导航、足式与规划 | nanoflann | [2610.15215](https://arxiv.org/abs/2610.15215) | **已开源** | [paper-nanoflann-product-manifold-kdtree](../../wiki/entities/paper-nanoflann-product-manifold-kdtree.md) |
| 05 | VLA 与动作生成 | CS-VGB | [2610.15281](https://arxiv.org/abs/2610.15281) | **已开源** | [paper-cs-vgb-inference-time-scaling](../../wiki/entities/paper-cs-vgb-inference-time-scaling.md) |
| 06 | 世界模型与记忆 | ReCAP | [2610.14583](https://arxiv.org/abs/2610.14583) | **部分开源** | [paper-recap-long-horizon-world-model-memory](../../wiki/entities/paper-recap-long-horizon-world-model-memory.md) |
| 07 | VLA 与动作生成 | FlowVLA | [2610.16160](https://arxiv.org/abs/2610.16160) | **已开源** | [paper-flowvla-pluggable-3d-flow](../../wiki/entities/paper-flowvla-pluggable-3d-flow.md) |
| 08 | VLA 与动作生成 | Δ-Flow | [2610.15794](https://arxiv.org/abs/2610.15794) | **已开源** | [paper-delta-flow-matching-robot-foundation-models](../../wiki/entities/paper-delta-flow-matching-robot-foundation-models.md) |
| 09 | 人形与移动操作 | AIDA | [2610.15572](https://arxiv.org/abs/2610.15572) | **项目页未列 GitHub** | [paper-aida-agent-interface-aligned-mobile-manipulation](../../wiki/entities/paper-aida-agent-interface-aligned-mobile-manipulation.md) |
| 10 | VLA 与动作生成 | PRISM (Precision Forcing) | [2610.14704](https://arxiv.org/abs/2610.14704) | **已开源** | [paper-prism-precision-forcing-vla](../../wiki/entities/paper-prism-precision-forcing-vla.md) |
| 11 | 操作与柔顺控制 | Residual Compliant Policy | [2610.14035](https://arxiv.org/abs/2610.14035) | **待发布** | [paper-residual-compliant-policy](../../wiki/entities/paper-residual-compliant-policy.md) |
| 12 | 多机器人协作与评测 | AnyRank | [2610.14603](https://arxiv.org/abs/2610.14603) | **已开源** | [paper-anyrank-anytime-policy-ranking](../../wiki/entities/paper-anyrank-anytime-policy-ranking.md) |
| 13 | 多机器人协作与评测 | 多操作员多机器人遥操作界面 | [2610.15001](https://arxiv.org/abs/2610.15001) | **已开源** | [paper-collaborative-teleoperation-multi-operator-multi-robot](../../wiki/entities/paper-collaborative-teleoperation-multi-operator-multi-robot.md) |
| 14 | 多机器人协作与评测 | B-Human '26 | [2610.14733](https://arxiv.org/abs/2610.14733) | **已开源** | [paper-b-human-26-scalable-team-play](../../wiki/entities/paper-b-human-26-scalable-team-play.md) |
| 15 | 世界模型与记忆 | PhySimCode | [2610.15722](https://arxiv.org/abs/2610.15722) | **已开源** | [paper-physimcode-physics-video-to-code](../../wiki/entities/paper-physimcode-physics-video-to-code.md) |
| 16 | 世界模型与记忆 | BridgeWAM | [2610.14746](https://arxiv.org/abs/2610.14746) | **已开源** | [paper-bridgewam-latent-bridge-queries](../../wiki/entities/paper-bridgewam-latent-bridge-queries.md) |
| 17 | 导航、足式与规划 | GG-Nav | [2610.14541](https://arxiv.org/abs/2610.14541) | **已开源** | [paper-gg-nav-goal-grounding-cot](../../wiki/entities/paper-gg-nav-goal-grounding-cot.md) |
| 18 | 世界模型与记忆 | IMPRINT | [2610.14406](https://arxiv.org/abs/2610.14406) | **项目页未列 GitHub** | [paper-imprint-multi-stream-latent-memory](../../wiki/entities/paper-imprint-multi-stream-latent-memory.md) |
| 19 | 操作与柔顺控制 | BAMPC | [2610.14064](https://arxiv.org/abs/2610.14064) | **已开源** | [paper-bampc-compute-budget-sampling-mpc](../../wiki/entities/paper-bampc-compute-budget-sampling-mpc.md) |
| 20 | 操作与柔顺控制 | RAIN | [2610.14039](https://arxiv.org/abs/2610.14039) | **已开源** | [paper-rain-region-aware-interaction-networks](../../wiki/entities/paper-rain-region-aware-interaction-networks.md) |
| 21 | VLA 与动作生成 | VPA | [2610.15556](https://arxiv.org/abs/2610.15556) | **已开源** | [paper-vision-predictor-action-jepa](../../wiki/entities/paper-vision-predictor-action-jepa.md) |
| 22 | VLA 与动作生成 | CFG-Policy | [2610.14936](https://arxiv.org/abs/2610.14936) | **待发布** | [paper-cfg-policy-unified-diffusion-policy](../../wiki/entities/paper-cfg-policy-unified-diffusion-policy.md) |
| 23 | 人形与移动操作 | TTCO | [2610.14896](https://arxiv.org/abs/2610.14896) | **待发布** | [paper-ttco-test-time-compute-humanoid-bfm](../../wiki/entities/paper-ttco-test-time-compute-humanoid-bfm.md) |
| 24 | 导航、足式与规划 | TempoLoco | [2610.14753](https://arxiv.org/abs/2610.14753) | **部分开源** | [paper-tempoloco-motion-adaptive-temporal-receptive-fields](../../wiki/entities/paper-tempoloco-motion-adaptive-temporal-receptive-fields.md) |
| 25 | 导航、足式与规划 | LOADNav | [2610.14596](https://arxiv.org/abs/2610.14596) | **已开源** | [paper-loadnav-load-aware-quadruped-navigation](../../wiki/entities/paper-loadnav-load-aware-quadruped-navigation.md) |
| 26 | 多机器人协作与评测 | CARE | [2610.14551](https://arxiv.org/abs/2610.14551) | **项目页未列 GitHub** | [paper-care-team-response-beliefs-multi-robot](../../wiki/entities/paper-care-team-response-beliefs-multi-robot.md) |
| 27 | 感知与 4D 世界 | TartanMatch | [2610.14513](https://arxiv.org/abs/2610.14513) | **已开源** | [paper-tartanmatch-universal-dense-correspondence](../../wiki/entities/paper-tartanmatch-universal-dense-correspondence.md) |
| 28 | 世界模型与记忆 | PILOT (Visual Subgoals) | [2610.14335](https://arxiv.org/abs/2610.14335) | **待发布** | [paper-pilot-visual-subgoal-world-model-planning](../../wiki/entities/paper-pilot-visual-subgoal-world-model-planning.md) |
| 29 | 多机器人协作与评测 | Akule | [2610.14248](https://arxiv.org/abs/2610.14248) | **已开源** | [paper-akule-sparse-interaction-diffusion](../../wiki/entities/paper-akule-sparse-interaction-diffusion.md) |
| 30 | 策略与后训练 | DiTaR | [2610.14159](https://arxiv.org/abs/2610.14159) | **待发布** | [paper-ditar-task-relevant-latent-dynamics-reward](../../wiki/entities/paper-ditar-task-relevant-latent-dynamics-reward.md) |
| 31 | 操作与柔顺控制 | 硬件感知 LLM-VLM 零样本规划 | [2610.14106](https://arxiv.org/abs/2610.14106) | **已开源** | [paper-hardware-aware-llm-vlm-zero-shot-task-planning](../../wiki/entities/paper-hardware-aware-llm-vlm-zero-shot-task-planning.md) |
| 32 | 感知与 4D 世界 | TIDE/TIDES | [2610.16167](https://arxiv.org/abs/2610.16167) | **已开源** | [paper-tide-tides-4d-liquid-forecasting](../../wiki/entities/paper-tide-tides-4d-liquid-forecasting.md) |
| 33 | 导航、足式与规划 | LERR-Nav | [2610.15362](https://arxiv.org/abs/2610.15362) | **已开源** | [paper-lerr-nav-exploration-recovery-reminiscing](../../wiki/entities/paper-lerr-nav-exploration-recovery-reminiscing.md) |
| 34 | 多机器人协作与评测 | CoPlanVLM | [2610.14341](https://arxiv.org/abs/2610.14341) | **已开源** | [paper-coplanvlm-coordinated-path-planning](../../wiki/entities/paper-coplanvlm-coordinated-path-planning.md) |
| 35 | 感知与 4D 世界 | SurgCast | [2610.15645](https://arxiv.org/abs/2610.15645) | **项目页未列 GitHub** | [paper-surgcast-surgical-world-model-skeletons](../../wiki/entities/paper-surgcast-surgical-world-model-skeletons.md) |
| 36 | 感知与 4D 世界 | Demiurge | [2610.14685](https://arxiv.org/abs/2610.14685) | **已开源** | [paper-demiurge-navigable-4d-worlds](../../wiki/entities/paper-demiurge-navigable-4d-worlds.md) |
| 37 | 导航、足式与规划 | 四旋翼软连续体臂 MuJoCo 框架 | [2610.15308](https://arxiv.org/abs/2610.15308) | **已开源** | [paper-quadrotor-tendon-soft-continuum-arm-mujoco](../../wiki/entities/paper-quadrotor-tendon-soft-continuum-arm-mujoco.md) |
| 38 | 多机器人协作与评测 | 类条件保形 STL 推断 | [2610.14352](https://arxiv.org/abs/2610.14352) | **项目页未列 GitHub** | [paper-class-conditional-conformal-stl-inference](../../wiki/entities/paper-class-conditional-conformal-stl-inference.md) |
| 39 | 人形与移动操作 | RoboCompiler-Humanoids | [2610.14237](https://arxiv.org/abs/2610.14237) | **已开源** | [paper-robocompiler-humanoids-composable-dynamics](../../wiki/entities/paper-robocompiler-humanoids-composable-dynamics.md) |
| 40 | 多机器人协作与评测 | 海上异构任务分配 RL | [2610.15237](https://arxiv.org/abs/2610.15237) | **已开源** | [paper-rl-heterogeneous-task-allocation-spatio-temporal](../../wiki/entities/paper-rl-heterogeneous-task-allocation-spatio-temporal.md) |
| 41 | VLA 与动作生成 | π0.5 CALL Astra | [2610.14117](https://arxiv.org/abs/2610.14117) | **项目页未列 GitHub** | [paper-pi05-learn-to-call-astra](../../wiki/entities/paper-pi05-learn-to-call-astra.md) |

## 对 wiki 的映射

- **41/41 独立详情节点**；同 arXiv 不重复造页（复用节点已补本文来源与地图关联）。
- 阅读坐标：[流式动作块与世界模型记忆：具身智能小站 10-14 期 41 篇阅读坐标](../../wiki/overview/embodied-station-2026-10-14-41-papers-technology-map.md)
- 开源结论口径：GitHub 仓库用 `git ls-remote` 核实可达；项目页用 curl 读取扫描 GitHub 链接与 "coming soon"；「项目页未列 GitHub」表示入库日页面无代码链接且未见开源承诺字样。

## 当前提炼状态

- [x] 公众号正文抓取（2026-10-14）
- [x] 41 篇独立节点（41 新建 / 0 复用）
- [x] 项目页 / 仓库开源状态核查（步骤 2.5）
- [ ] 原文实验表格核验（节点内数字仅为公众号转述）
