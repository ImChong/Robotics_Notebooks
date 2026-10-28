---
type: overview
tags: [overview, survey, embodied-ai, vla, world-model, benchmark, technology-map, wechat]
status: complete
updated: 2026-10-28
related:
  - ../entities/paper-rendering-free-event-sim.md
  - ../entities/paper-rynnvla-latent.md
  - ../entities/paper-aura.md
  - ../methods/vla.md
  - ../concepts/world-action-models.md
sources:
  - ../../sources/blogs/wechat_embodied_station_59_papers_human_video_to_robot_experience_2026-10-28.md
summary: "具身智能小站 2026-10-28 盘点：免渲染事件帧仿真（深读）、RynnVLA-Latent 三维潜在动作、AuRA 自主交互采集，以及评测基准、灵巧操作、导航与无人机、世界模型等 59 篇；59/59 独立节点（新建 56 / 复用 3）。"
---

# 潜在动作、免渲染事件仿真与机器人评测：具身智能小站 10-28 期 59 篇阅读坐标

> **本页定位**：为 [具身智能小站 · 人类视频怎样变成机器人能用的动作经验？](https://mp.weixin.qq.com/s/JSya9kmE3Fcon3BY_DcCVA)（2026-10-28）提供按研究方向组织的阅读坐标；59 篇全部有独立详情节点。

## 一句话观点

**人类视频缺控制标签，所以本期的共同线索是把它转成机器人可用的动作经验：潜在动作、三维运动监督与示范迁移；另一半篇幅在补评测——成功率之外还要看过程、失败原因与安全语义。深读免渲染事件仿真，跟进 RynnVLA-Latent 与 AuRA。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| WAM | World Action Model | 世界–动作联合模型 |
| VLM | Vision-Language Model | 视觉语言模型 |

## 为什么单独做这张地图

- 公众号一次列出 59 篇（1 深读 + 2 跟进 + 56 扫读）、方向跨度大，需要横切面索引；每篇只保留一个判断，细节在各节点。
- 节点内的数字为公众号转述，未核验原文表格；开源结论为入库日核查结果，见各节点「为什么重要」。

## 流程总览

```mermaid
flowchart TB
  G0["导航、无人机与群体（8）"] --> GOAL
  G1["潜在动作与世界模型（7）"] --> GOAL
  G2["数据采集与示范迁移（5）"] --> GOAL
  G3["评测与基准（11）"] --> GOAL
  G4["灵巧与接触操作（8）"] --> GOAL
  G5["VLA 与策略后训练（13）"] --> GOAL
  G6["控制、规划与系统（7）"] --> GOAL
  GOAL[可部署具身系统]
```

## 分组索引

### 导航、无人机与群体（8 篇）

| 论文 | 节点 | 开源 |
|------|------|------|
| Rendering-Free Event Sim | [paper-rendering-free-event-sim](../entities/paper-rendering-free-event-sim.md) | 已开源 |
| Dynamic Narrow Gap Flight | [paper-dynamic-narrow-gap-flight](../entities/paper-dynamic-narrow-gap-flight.md) | 已开源 |
| Thermal-Headroom Routing | [paper-thermal-headroom-routing](../entities/paper-thermal-headroom-routing.md) | 已开源 |
| Motion-Induced WiFi-CSI Sim | [paper-motion-induced-wifi-csi-sim](../entities/paper-motion-induced-wifi-csi-sim.md) | 已开源 |
| NIT-TrajMamba | [paper-nit-trajmamba](../entities/paper-nit-trajmamba.md) | 已开源 |
| Distributed Predictive Flocking | [paper-distributed-predictive-flocking](../entities/paper-distributed-predictive-flocking.md) | 已开源 |
| DSGN | [paper-dsgn](../entities/paper-dsgn.md) | 已开源 |
| RememberWhereYouWalk | [paper-rememberwhereyouwalk](../entities/paper-rememberwhereyouwalk.md) | 已开源 |

### 潜在动作与世界模型（7 篇）

| 论文 | 节点 | 开源 |
|------|------|------|
| RynnVLA-Latent | [paper-rynnvla-latent](../entities/paper-rynnvla-latent.md)（复用） | 已开源 |
| RynnWorld-Latent | [paper-rynnworld-latent](../entities/paper-rynnworld-latent.md) | 已开源 |
| Tapestry | [paper-tapestry](../entities/paper-tapestry.md) | 项目页未列 GitHub |
| ThinkWorld | [paper-thinkworld](../entities/paper-thinkworld.md) | 待发布 |
| Trace-WAM | [paper-trace-wam](../entities/paper-trace-wam.md) | 项目页未列 GitHub |
| X-World | [paper-x-world-variable-view-video-priors](../entities/paper-x-world-variable-view-video-priors.md) | 项目页未列 GitHub |
| GEMS | [paper-gems](../entities/paper-gems.md) | 已开源 |

### 数据采集与示范迁移（5 篇）

| 论文 | 节点 | 开源 |
|------|------|------|
| AuRA | [paper-aura](../entities/paper-aura.md) | 已开源 |
| PrecisionReach-GS | [paper-precisionreach-gs](../entities/paper-precisionreach-gs.md) | 已开源 |
| OpenJaws | [paper-openjaws](../entities/paper-openjaws.md) | 已开源 |
| SceneAgent | [paper-sceneagent](../entities/paper-sceneagent.md) | 待发布 |
| RoboSTD | [paper-robostd](../entities/paper-robostd.md) | 已开源 |

### 评测与基准（11 篇）

| 论文 | 节点 | 开源 |
|------|------|------|
| EmbodiedLab | [paper-embodiedlab](../entities/paper-embodiedlab.md) | 已开源 |
| SO-101 Bench | [paper-so-101-bench](../entities/paper-so-101-bench.md) | 已开源 |
| Specola | [paper-specola](../entities/paper-specola.md) | 已开源 |
| Open-world Robot Autonomy | [paper-open-world-robot-autonomy](../entities/paper-open-world-robot-autonomy.md) | 已开源 |
| Execution Logs Sway VLM Judges | [paper-execution-logs-sway-vlm-judges](../entities/paper-execution-logs-sway-vlm-judges.md) | 已开源 |
| RoboJev | [paper-robojev](../entities/paper-robojev.md) | 已开源 |
| WorldFailBench | [paper-worldfailbench](../entities/paper-worldfailbench.md) | 已开源 |
| EBench Frontier Agent Profiling | [paper-ebench-frontier-agent-profiling](../entities/paper-ebench-frontier-agent-profiling.md) | 已开源 |
| Obstacle-Conditioned Safety Nav | [paper-obstacle-conditioned-safety-nav](../entities/paper-obstacle-conditioned-safety-nav.md) | 项目页未列 GitHub |
| RoboWorM | [paper-roboworm](../entities/paper-roboworm.md) | 已开源 |
| RoboReason | [paper-roboreason](../entities/paper-roboreason.md) | 待发布 |

### 灵巧与接触操作（8 篇）

| 论文 | 节点 | 开源 |
|------|------|------|
| AdaDexTrack | [paper-adadextrack](../entities/paper-adadextrack.md) | 已开源 |
| EgoThumb | [paper-egothumb](../entities/paper-egothumb.md) | 已开源 |
| FLoRe | [paper-flore](../entities/paper-flore.md) | 待发布 |
| CARD | [paper-card-contact-anchored-retargeting](../entities/paper-card-contact-anchored-retargeting.md)（复用） | 已开源 |
| Differentiable Contact Legged | [paper-differentiable-contact-legged](../entities/paper-differentiable-contact-legged.md) | 项目页未列 GitHub |
| Functional Specialisation Bimanual | [paper-functional-specialisation-bimanual](../entities/paper-functional-specialisation-bimanual.md) | 已开源 |
| Keeping Contact | [paper-keeping-contact](../entities/paper-keeping-contact.md) | 项目页未列 GitHub |
| Action Representation Pretraining | [paper-action-representation-pretraining](../entities/paper-action-representation-pretraining.md) | 项目页未列 GitHub |

### VLA 与策略后训练（13 篇）

| 论文 | 节点 | 开源 |
|------|------|------|
| AGQ | [paper-agq](../entities/paper-agq.md) | 已开源 |
| RoboDoctor | [paper-robodoctor](../entities/paper-robodoctor.md) | 已开源 |
| Sim-Trained Compositional Agent | [paper-sim-trained-compositional-agent](../entities/paper-sim-trained-compositional-agent.md) | 项目页未列 GitHub |
| HuGe | [paper-huge-humanoid-vla](../entities/paper-huge-humanoid-vla.md)（复用） | 已开源 |
| Retrieve to Act | [paper-retrieve-to-act](../entities/paper-retrieve-to-act.md) | 待发布 |
| RoboAuto | [paper-roboauto](../entities/paper-roboauto.md) | 待发布 |
| Human-Guided Adaptive Residual Correction | [paper-human-guided-adaptive-residual-correction](../entities/paper-human-guided-adaptive-residual-correction.md) | 项目页未列 GitHub |
| Quality-Controllable Policies | [paper-quality-controllable-policies](../entities/paper-quality-controllable-policies.md) | 待发布 |
| OmniBrain-0 | [paper-omnibrain-0](../entities/paper-omnibrain-0.md) | 已开源 |
| AdaPolicy | [paper-adapolicy](../entities/paper-adapolicy.md) | 部分开源 |
| RoboLongR | [paper-robolongr](../entities/paper-robolongr.md) | 项目页未列 GitHub |
| FISRF | [paper-fisrf](../entities/paper-fisrf.md) | 项目页未列 GitHub |
| From Language to Drives | [paper-from-language-to-drives](../entities/paper-from-language-to-drives.md) | 已开源 |

### 控制、规划与系统（7 篇）

| 论文 | 节点 | 开源 |
|------|------|------|
| HA-DiPO | [paper-ha-dipo](../entities/paper-ha-dipo.md) | 已开源 |
| cuArms | [paper-cuarms](../entities/paper-cuarms.md) | 项目页未列 GitHub |
| Behavioural Quality vs Learning Utility | [paper-behavioural-quality-vs-learning-utility](../entities/paper-behavioural-quality-vs-learning-utility.md) | 项目页未列 GitHub |
| Expressive Motion Behavior | [paper-expressive-motion-behavior](../entities/paper-expressive-motion-behavior.md) | 已开源 |
| SysML v2 ROS 2 Library | [paper-sysml-v2-ros-2-library](../entities/paper-sysml-v2-ros-2-library.md) | 已开源 |
| Aerial Manipulator Nonlinear Control | [paper-aerial-manipulator-nonlinear-control](../entities/paper-aerial-manipulator-nonlinear-control.md) | 已开源 |
| Spiking Periodic Orbit Stabilization | [paper-spiking-periodic-orbit-stabilization](../entities/paper-spiking-periodic-orbit-stabilization.md) | 部分开源 |

## 关联页面

- [VLA](../methods/vla.md)
- [World Action Models](../concepts/world-action-models.md)

## 参考来源

- [wechat_embodied_station_59_papers_human_video_to_robot_experience_2026-10-28.md](../../sources/blogs/wechat_embodied_station_59_papers_human_video_to_robot_experience_2026-10-28.md)

## 推荐继续阅读

- [Rendering-Free Event Sim](../entities/paper-rendering-free-event-sim.md)
- [RynnVLA-Latent](../entities/paper-rynnvla-latent.md)
