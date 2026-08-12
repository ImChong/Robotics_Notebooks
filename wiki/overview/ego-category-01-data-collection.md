---
type: overview
tags: [egocentric, ego-survey, category-hub, data-collection, dataset]
status: complete
updated: 2026-08-12
summary: "Ego 9 篇专题 · 01 数据采集（2 篇）— 机器人数据贵，Ego 让人类成为分布式采集者；核心是把「日常第一视角」做成可过滤、可规模化的训练素材。旁路对照：Ego4D、Ego-OSCAR/Stereo-550、EgoVerse、EgoWorld-100W、RekaDaily-10k、RekaCS2-10k 与 Macrodata 度量手轨迹配方。"
related:
  - ./ego-9-papers-technology-map.md
  - ./ego-category-02-human-to-robot.md
  - ../entities/paper-ego-01-aoe.md
  - ../entities/paper-ego-02-egolive.md
  - ../entities/paper-vidihand.md
  - ../entities/egoworld-100w.md
  - ../entities/rekadaily-10k-dataset.md
  - ../entities/rekacs2-10k-dataset.md
  - ../entities/paper-egoverse.md
  - ../entities/paper-ego4d.md
  - ../entities/paper-ego-oscar.md
  - ../methods/macrodata-egocentric-hand-action.md
sources:
  - ../../sources/blogs/wechat_embodied_ai_lab_ego_9_papers_survey.md
  - ../../sources/papers/ego_9_papers_catalog.md
  - ../../sources/blogs/stellarnex_egoworld_100w.md
  - ../../sources/papers/egoverse_arxiv_2604_07607.md
  - ../../sources/papers/ego4d_arxiv_2110_07058.md
  - ../../sources/sites/ego4d-data-org.md
  - ../../sources/sites/rekadaily-10k.md
  - ../../sources/sites/rekacs2-10k.md
  - ../../sources/blogs/macrodata_egocentric_video_3d_hand_actions.md
  - ../../sources/papers/ego_oscar_arxiv_2608_08285.md
---

# Ego 分类 01：数据采集

> **图谱分类节点**：对应 [具身智能研究室 · Ego 9 篇专题](https://mp.weixin.qq.com/s/4JQ1xa-cJ7J1ep_e4txNnA) 的 **01 数据采集** 分组；总地图见 [Ego 9 篇技术地图](./ego-9-papers-technology-map.md)。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Ego | Egocentric Vision | 第一人称视角感知与控制 |

## 核心问题

**Ego 数据怎么大规模、低成本采？** 真机遥操作与多场景部署贵；让人类戴设备完成真实任务，可把未被机器人系统记录的经验变成可整理数据。

## 本组论文（2 篇）

| # | 工作 | Wiki 实体 | Source |
|---|------|-----------|--------|
| 01 | AoE | [paper-ego-01-aoe.md](../entities/paper-ego-01-aoe.md) | [source](../../sources/papers/ego_survey_01_aoe.md) |
| 02 | EgoLive | [paper-ego-02-egolive.md](../entities/paper-ego-02-egolive.md) | [source](../../sources/papers/ego_survey_02_egolive.md) |

## 关联页面

- [Ego 9 篇技术地图](./ego-9-papers-technology-map.md)
- [人→机器人](./ego-category-02-human-to-robot.md)
- [ViDiHand](../entities/paper-vidihand.md) — 采集后的 **双手 4D 标注** 可用 video diffusion 先验 **无 detector** 规模化重建，支撑模仿/策略监督
- [Ego4D](../entities/paper-ego4d.md) — FAIR 联盟 **~3,670 h** 全球日常 egocentric 视频 + 五大 benchmark（CVPR 2022）；后续多数 Ego 语料的规模/任务锚点（HumanNet 表标 **Indirect**）
- [Ego-OSCAR / Stereo-550](../entities/paper-ego-oscar.md) — 第一人称视觉实验室（FPV Labs）**~USD 200** 开源硬件硬同步立体+IMU 头戴；**~550 h/相机** gated 语料验证众包续采（观测-only，非 teleop）
- [EgoVerse](../entities/paper-egoverse.md) — 联盟式 egocentric 活数据集（Aria / 产业 / 手机采集）与 EgoDB 接入；与本组「人类作分布式采集者」同动机、更偏操纵向 Direct 标注
- [EgoWorld-100W](../entities/egoworld-100w.md) — StellarNex **百万级** 第一人称操作语料（**申请制**；四维 Scene×Object×Action×Handedness）；与 ICLR [EgoWorld 视图翻译](../entities/paper-egoworld.md) **同名异物**
- [RekaDaily-10k](../entities/rekadaily-10k-dataset.md) — Reka/Claru **10k+ 小时** 无剧本家务 ego 视频（Apache 2.0；raw HF 增量 + processed/captioned）
- [RekaCS2-10k](../entities/rekacs2-10k-dataset.md) — 职业 CS2 demo 渲染的 **10k+ 小时** ego 视频 + 逐帧键鼠/轨迹（世界模型沙盒；非真实家务）
- [Macrodata Egocentric Hand-Action](../methods/macrodata-egocentric-hand-action.md) — 把已采 egocentric RGB **重建为度量 21 关节手轨迹** 的开源配方与 HOT3D Action MPJPE 标尺（采集后的几何标注层）

## 参考来源

- [wechat_embodied_ai_lab_ego_9_papers_survey.md](../../sources/blogs/wechat_embodied_ai_lab_ego_9_papers_survey.md)
- [ego_9_papers_catalog.md](../../sources/papers/ego_9_papers_catalog.md)
- [EgoVerse 论文摘录](../../sources/papers/egoverse_arxiv_2604_07607.md)
- [Ego4D 论文摘录](../../sources/papers/ego4d_arxiv_2110_07058.md)
- [Ego4D 项目页归档](../../sources/sites/ego4d-data-org.md)
- [stellarnex_egoworld_100w.md](../../sources/blogs/stellarnex_egoworld_100w.md)
- [RekaDaily-10k 研究页归档](../../sources/sites/rekadaily-10k.md)
- [RekaCS2-10k 新闻页归档](../../sources/sites/rekacs2-10k.md)
- [macrodata_egocentric_video_3d_hand_actions.md](../../sources/blogs/macrodata_egocentric_video_3d_hand_actions.md)

## 推荐继续阅读

- [机器人论文阅读笔记：EmbodMocap](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/14_Human_Motion/EmbodMocap__In-the-Wild_4D_Human-Scene_Reconstruction_for_Embodied_Agents/EmbodMocap__In-the-Wild_4D_Human-Scene_Reconstruction_for_Embodied_Agents.html)
- [Ego4D 项目页](https://ego4d-data.org/)
- [Stereo-550（Ego-OSCAR）](https://huggingface.co/datasets/fpvlabs/stereo-550)
- [EgoVerse 项目页](https://egoverse.ai/)
- [EgoWorld-100W 官方介绍](https://stellarnexrobotics.com/blog)
- [RekaDaily-10k 研究页](https://reka.ai/labs/research/rekadaily-10k-egocentric-household-manipulation-data)
- [RekaCS2-10k 新闻页](https://reka.ai/news/cs2-10k-a-large-scale-egocentric-counter-strike-2-dataset)
