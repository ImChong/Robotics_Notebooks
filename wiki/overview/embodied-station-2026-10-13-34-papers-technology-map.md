---
type: overview
tags: [overview, survey, embodied-ai, wam, vla, technology-map, wechat]
status: complete
updated: 2026-10-14
related:
  - ../entities/paper-mulligan-performance-guided-data-collection.md
  - ../entities/paper-wamjet-wam-acceleration-harness.md
  - ../entities/paper-affordcraft-task-ready-simulation-assets.md
  - ../entities/paper-openwam-composable-world-action-models.md
  - ../entities/paper-eigendexplore.md
  - ../methods/vla.md
  - ../concepts/world-action-models.md
  - ./embodied-station-2026-10-14-41-papers-technology-map.md
sources:
  - ../../sources/blogs/wechat_embodied_station_34_papers_mulligan_2026-10-13.md
summary: "具身智能小站 2026-10-13 盘点：Mulligan 失败导向真机采集、OpenWAM 可组合 WAM、EigenDEXplore 结构化探索，以及 RL 后训练、世界模型、灵巧操作、人形 34 篇；34/34 独立节点（新建 25 / 复用 9）。"
---

# 失败导向采集与可组合世界动作模型：具身智能小站 10-13 期 34 篇阅读坐标

> **本页定位**：为 [具身智能小站 · 机器人越熟练，采集数据越要盯住失败｜附代码与项目入口](https://mp.weixin.qq.com/s/mGe8FNbW8HXnr5TUb7PHGw)（2026-10-13）提供按研究方向组织的阅读坐标；34 篇全部有独立详情节点。

## 一句话观点

**真机策略越熟练，均匀采集越浪费预算：把采集起点、价值函数与人工介入作为可优化变量，是部署后持续学习的主线。本期深读 Mulligan，跟进 OpenWAM 与 EigenDEXplore。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| WAM | World Action Model | 世界–动作联合模型 |
| RL | Reinforcement Learning | 强化学习 |
| FM | Flow Matching | 流匹配生成 |

## 为什么单独做这张地图

- 公众号一次列出 34 篇、方向跨度大，需要横切面索引；每篇只保留一个判断，细节在各节点。
- 节点内的数字为公众号转述，未核验原文表格；开源结论为入库日核查结果，见各节点「为什么重要」。
- 与 [流式动作块与世界模型记忆：具身智能小站 10-14 期 41 篇阅读坐标](./embodied-station-2026-10-14-41-papers-technology-map.md) 为相邻两期，可对照阅读（异步动作块：StreamingWAM / FreeBar / EAPN；世界模型记忆：ReCAP / IMPRINT / TTT-RM）。

## 流程总览

```mermaid
flowchart TB
  G0["强化学习与数据采集（2）"] --> GOAL
  G1["世界模型与仿真（11）"] --> GOAL
  G2["策略与后训练（6）"] --> GOAL
  G3["灵巧操作（5）"] --> GOAL
  G4["人形与移动操作（4）"] --> GOAL
  G5["导航、足式与规划（1）"] --> GOAL
  G6["VLA 与动作生成（3）"] --> GOAL
  G7["多机器人与评测（2）"] --> GOAL
  GOAL[可部署具身系统]
```

## 分组索引

### 强化学习与数据采集（2 篇）

| 论文 | 节点 | 开源 |
|------|------|------|
| Mulligan | [paper-mulligan-performance-guided-data-collection](../entities/paper-mulligan-performance-guided-data-collection.md) | 已开源 |
| UniIntervene++ | [paper-uniintervene-plus-plus-adaptive-intervention](../entities/paper-uniintervene-plus-plus-adaptive-intervention.md) | 已开源 |

### 世界模型与仿真（11 篇）

| 论文 | 节点 | 开源 |
|------|------|------|
| WAMJET | [paper-wamjet-wam-acceleration-harness](../entities/paper-wamjet-wam-acceleration-harness.md) | 已开源 |
| AffordCraft | [paper-affordcraft-task-ready-simulation-assets](../entities/paper-affordcraft-task-ready-simulation-assets.md) | 已开源 |
| Video2World | [paper-video2world-coding-agents-benchmark](../entities/paper-video2world-coding-agents-benchmark.md) | 已开源 |
| SimForcing | [paper-simforcing-simulation-motion-priors](../entities/paper-simforcing-simulation-motion-priors.md) | 已开源 |
| SDOB | [paper-sdob-sampling-based-disturbance-observers](../entities/paper-sdob-sampling-based-disturbance-observers.md) | 待发布 |
| TAPDreamer | [paper-tapdreamer-adversarial-patches-wam](../entities/paper-tapdreamer-adversarial-patches-wam.md) | 待发布 |
| TTT-RM | [paper-ttt-rm-residual-memory](../entities/paper-ttt-rm-residual-memory.md) | 项目页未列 GitHub |
| DepthWorld | [paper-depthworld-3d-world-model](../entities/paper-depthworld-3d-world-model.md) | 已开源 |
| KineWorld | [paper-kineworld-action-induced-transport-fields](../entities/paper-kineworld-action-induced-transport-fields.md) | 已开源 |
| 技能级世界模型的流策略动作 | [paper-flow-policies-skill-level-world-models](../entities/paper-flow-policies-skill-level-world-models.md) | 项目页未列 GitHub |
| OpenWAM | [paper-openwam-composable-world-action-models](../entities/paper-openwam-composable-world-action-models.md)（复用） | 已开源 |

### 策略与后训练（6 篇）

| 论文 | 节点 | 开源 |
|------|------|------|
| RobotUse | [paper-robotuse-allocating-computation-context-decisions](../entities/paper-robotuse-allocating-computation-context-decisions.md) | 已开源 |
| VLA 后训练数据增强 | [paper-vla-augm-data-augmentation-post-training](../entities/paper-vla-augm-data-augmentation-post-training.md) | 已开源 |
| VeriFine | [paper-verifine-scaling-verification-self-improvement](../entities/paper-verifine-scaling-verification-self-improvement.md) | 待发布 |
| FreeSpeed | [paper-freespeed](../entities/paper-freespeed.md)（复用） | 项目页未列 GitHub |
| QF3 | [paper-qf3-filtered-q-gradients](../entities/paper-qf3-filtered-q-gradients.md)（复用） | 项目页未列 GitHub |
| EgoLAP | [paper-egolap-egocentric-language-action-reasoning](../entities/paper-egolap-egocentric-language-action-reasoning.md)（复用） | 待发布 |

### 灵巧操作（5 篇）

| 论文 | 节点 | 开源 |
|------|------|------|
| EyeRobot 2.0 | [paper-eyerobot-2-active-gaze-manipulation](../entities/paper-eyerobot-2-active-gaze-manipulation.md) | 已开源 |
| PEARS | [paper-pears-tactile-failure-reasoning-diffusion-steering](../entities/paper-pears-tactile-failure-reasoning-diffusion-steering.md) | 项目页未列 GitHub |
| GOTT | [paper-gott-cross-embodiment-contact-primitive](../entities/paper-gott-cross-embodiment-contact-primitive.md) | 待发布 |
| PatternDex | [paper-patterndex-interaction-patterns-bimanual-dexterous](../entities/paper-patterndex-interaction-patterns-bimanual-dexterous.md) | 待发布 |
| EigenDEXplore | [paper-eigendexplore](../entities/paper-eigendexplore.md)（复用） | 已开源 |

### 人形与移动操作（4 篇）

| 论文 | 节点 | 开源 |
|------|------|------|
| MobiAgent | [paper-mobiagent-dual-loop-self-improvement](../entities/paper-mobiagent-dual-loop-self-improvement.md) | 部分开源 |
| Humanoid Horizon | [paper-humanoid-horizon-long-horizon-loco-manipulation](../entities/paper-humanoid-horizon-long-horizon-loco-manipulation.md)（复用） | 项目页未列 GitHub |
| InterMimicGen | [paper-intermimicgen](../entities/paper-intermimicgen.md)（复用） | 项目页未列 GitHub |
| CoDance | [paper-codance](../entities/paper-codance.md)（复用） | 项目页未列 GitHub |

### 导航、足式与规划（1 篇）

| 论文 | 节点 | 开源 |
|------|------|------|
| CUSP | [paper-cusp-survival-hazard-alarms-offroad](../entities/paper-cusp-survival-hazard-alarms-offroad.md) | 待发布 |

### VLA 与动作生成（3 篇）

| 论文 | 节点 | 开源 |
|------|------|------|
| StairVLA | [paper-stairvla-stage-aware-hierarchical-action](../entities/paper-stairvla-stage-aware-hierarchical-action.md) | 已开源 |
| EAPN | [paper-eapn-execution-aligned-progressive-noise](../entities/paper-eapn-execution-aligned-progressive-noise.md) | 项目页未列 GitHub |
| CNP-Flow | [paper-cnp-flow-conditional-source-distribution](../entities/paper-cnp-flow-conditional-source-distribution.md) | 项目页未列 GitHub |

### 多机器人与评测（2 篇）

| 论文 | 节点 | 开源 |
|------|------|------|
| QUEEN | [paper-queen-chess-language-model-explanations](../entities/paper-queen-chess-language-model-explanations.md) | 已开源 |
| CoRE | [paper-core-collaboration-role-experts](../entities/paper-core-collaboration-role-experts.md)（复用） | 待发布 |

## 关联页面

- [VLA](../methods/vla.md)
- [World Action Models](../concepts/world-action-models.md)
- [Flow Matching 具身策略](../concepts/flow-matching-embodied-policy.md)
- [流式动作块与世界模型记忆：具身智能小站 10-14 期 41 篇阅读坐标](./embodied-station-2026-10-14-41-papers-technology-map.md)

## 参考来源

- [wechat_embodied_station_34_papers_mulligan_2026-10-13.md](../../sources/blogs/wechat_embodied_station_34_papers_mulligan_2026-10-13.md)

## 推荐继续阅读

- [Mulligan](../entities/paper-mulligan-performance-guided-data-collection.md)
- [WAMJET](../entities/paper-wamjet-wam-acceleration-harness.md)
