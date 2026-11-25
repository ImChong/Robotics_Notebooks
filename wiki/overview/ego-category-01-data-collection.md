---
type: overview
tags: [egocentric, ego-survey, category-hub, data-collection, dataset]
status: complete
updated: 2026-11-22
summary: "Ego 9 篇专题 · 01 数据采集（2 篇）— 机器人数据贵，Ego 让人类成为分布式采集者；核心是把「日常第一视角」做成可过滤、可规模化的训练素材。旁路对照：Ego4D、Ego-OSCAR/Stereo-550、EgoVerse、EgoWorld-100W、RekaDaily-10k、RekaCS2-10k、Unidata Pico 4 Ultra 量产采集与 Macrodata 度量手轨迹配方。"
related:
  - ./ego-9-papers-technology-map.md
  - ../entities/maxinsights-physical-ai-data.md
  - ./ego-category-02-human-to-robot.md
  - ../entities/paper-ego-01-aoe.md
  - ../entities/paper-ego-02-egolive.md
  - ../entities/paper-vidihand.md
  - ../entities/egoworld-100w.md
  - ../entities/rekadaily-10k-dataset.md
  - ../entities/rekacs2-10k-dataset.md
  - ../entities/humanplus-1000-dataset.md
  - ../entities/gen-human-ego-dataset.md
  - ../entities/paper-egoverse.md
  - ../entities/paper-ego4d.md
  - ../entities/paper-ego-oscar.md
  - ../methods/macrodata-egocentric-hand-action.md
  - ../entities/pico-4-ultra-egocentric-capture.md
  - ../entities/tsing-ego-pro.md
  - ../entities/paper-vbot-exo2ego-policy.md
  - ../entities/paper-humancoin-human-intent-egocentric-manipulation.md
sources:
  - ../../sources/papers/humancoin_human_intent_egocentric_manipulation_arxiv_2611_26434.md
  - ../../sources/sites/maxinsights-official.md
  - ../../sources/blogs/maxinsights-beyond-1m-hours-physical-ai-data.md
  - ../../sources/blogs/wechat_embodied_ai_lab_ego_9_papers_survey.md
  - ../../sources/papers/ego_9_papers_catalog.md
  - ../../sources/blogs/stellarnex_egoworld_100w.md
  - ../../sources/papers/egoverse_arxiv_2604_07607.md
  - ../../sources/papers/ego4d_arxiv_2110_07058.md
  - ../../sources/sites/ego4d-data-org.md
  - ../../sources/sites/rekadaily-10k.md
  - ../../sources/sites/rekacs2-10k.md
  - ../../sources/sites/humanplus-1000.md
  - ../../sources/datasets/gen-human-ego-genrobot.md
  - ../../sources/sites/genrobot-open-dataset.md
  - ../../sources/blogs/macrodata_egocentric_video_3d_hand_actions.md
  - ../../sources/papers/ego_oscar_arxiv_2608_08285.md
  - ../../sources/blogs/unidata_pico_4_ultra_egocentric_data_collection.md
original:
  - ../../sources/sites/maxinsights-official.md
---

# Ego 分类 01：数据采集

## 一句话定义

> **分类页**：对应 [具身智能研究室 · Ego 9 篇专题](https://mp.weixin.qq.com/s/4JQ1xa-cJ7J1ep_e4txNnA) 的 **01 数据采集** 分组；总地图见 [Ego 9 篇技术地图](./ego-9-papers-technology-map.md)。

本页归纳第一视角（Egocentric）人视频与多模态采集在具身智能中的核心问题与代表工作。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Ego | Egocentric Vision | 第一人称视角感知与控制 |

## 为什么重要

第一视角人视频能把人类日常活动转化为低成本、高多样性的分布式采集源，是突破真机遥操作吞吐瓶颈的重要路径。

## 核心原理

- **视点与手部表征对齐**：通过头戴/胸挂相机与手部 3D 关键点追踪，将人类日常操作视频转化为可与机器人腕部视角及末端动作对齐的监督信号。

## 核心问题

**Ego 数据怎么大规模、低成本采？** 真机遥操作与多场景部署贵；让人类戴设备完成真实任务，可把未被机器人系统记录的经验变成可整理数据。

## 本组论文（2 篇）

| # | 工作 | Wiki 实体 | Source |
|---|------|-----------|--------|
| 01 | AoE | [paper-ego-01-aoe.md](../entities/paper-ego-01-aoe.md) | [source](../../sources/papers/ego_survey_01_aoe.md) |
| 02 | EgoLive | [paper-ego-02-egolive.md](../entities/paper-ego-02-egolive.md) | [source](../../sources/papers/ego_survey_02_egolive.md) |

## 工程实践

- **优先记录视点与手部尺度元数据**：采集第一视角人视频时同步保留相机内参、头戴高度与手部关键点置信度，便于下游做视点对齐与手–爪映射。

## 局限

- **手–爪形态差异与无动作真值**：人手自由度远高于二指夹爪，且纯视频缺乏关节力矩与本体状态，通常需配合少量真机遥操作做 mid-training 对齐。

## 架构与路线图

```mermaid
flowchart LR
    Cap["分布式人类第一视角采集 (头戴/胸挂相机 + 传感)"] --> Papers["代表工作与采集方案: AoE / EgoLive / Ego-OSCAR / Pico 4 Ultra"]
    Papers --> Meta["保留视点内参、头戴高度与手部 3D 关键点"]
    Meta --> Anno["后处理几何与 4D 标注 (ViDiHand / Macrodata / Ego-Exo4D-HM)"]
    Anno --> Align["对齐机器人腕部视角 + 少量真机遥操作 mid-training"]
```

## 关联页面

- [HumanCOIN：注视/肌电/脑电的分层第一视角数据](../entities/paper-humancoin-human-intent-egocentric-manipulation.md) — L1 估计注视、L2 实测肌电、L3 实测脑电，不能把 12,500 h 全部当实测三模态示教。

- [Ego 9 篇技术地图](./ego-9-papers-technology-map.md)
- [人→机器人](./ego-category-02-human-to-robot.md)
- [ViDiHand](../entities/paper-vidihand.md) — 采集后的 **双手 4D 标注** 可用 video diffusion 先验 **无 detector** 规模化重建，支撑模仿/策略监督
- [Ego4D](../entities/paper-ego4d.md) — FAIR 联盟 **~3,670 h** 全球日常 egocentric 视频 + 五大 benchmark（CVPR 2022）；后续多数 Ego 语料的规模/任务锚点（HumanNet 表标 **Indirect**）
- [Ego-Exo4D-HM](../entities/paper-egoexo4d-hm.md) — 在 [Ego-Exo4D](https://ego-exo4d-data.org/) 同步多视角上发布 **2,649 takes / 104.59 h** SMPL-H 4D 重建（HF npz + MIT 管线）；ego-exo 视频→**稠密 3D 说明书** 层，机器人策略收益仍待下游验证
- [Ego-OSCAR / Stereo-550](../entities/paper-ego-oscar.md) — 第一人称视觉实验室（FPV Labs）**~USD 200**（v3 Table 1 BoM INR 19,600 / ~USD 205）开源硬件硬同步立体+IMU 头戴；`fpv-labs/ego-oscar` 已以 Apache-2.0 开源固件与 Radxa 录制器，**~550 h/相机** Stereo-550 语料仍须申请 gated 定制许可；v3 与录制器默认 IMU **180 Hz**（官网博客留存 120 Hz 口径差异）
- [EgoVerse](../entities/paper-egoverse.md) — 联盟式 egocentric 活数据集（Aria / 产业 / 手机采集）与 EgoDB 接入；与本组「人类作分布式采集者」同动机、更偏操纵向 Direct 标注
- [VBot Exo2Ego Policy](../entities/paper-vbot-exo2ego-policy.md) — 双 iPhone 真实 EGO/EXO 配对与第三人称→第一人称视频生成的数据扩增；生成影像并非具动作真值的示教轨迹
- [EgoWorld-100W](../entities/egoworld-100w.md) — StellarNex **百万级** 第一人称操作语料（**申请制**；四维 Scene×Object×Action×Handedness）；与 ICLR [EgoWorld 视图翻译](../entities/paper-egoworld.md) **同名异物**
- [RekaDaily-10k](../entities/rekadaily-10k-dataset.md) — Reka/Claru **10k+ 小时** 无剧本家务 ego 视频（Apache 2.0；raw HF 增量 + processed/captioned）
- [RekaCS2-10k](../entities/rekacs2-10k-dataset.md) — 职业 CS2 demo 渲染的 **10k+ 小时** ego 视频 + 逐帧键鼠/轨迹（世界模型沙盒；非真实家务）
- [HumanPlus-1000](../entities/humanplus-1000-dataset.md) — HumanPlus **1000+ h** 同步立体 ego + SMPL-H/MANO/SLAM/IMU（HF 预览 100 session；全量申请）
- [Gen-HumanEgo](../entities/gen-human-ego-dataset.md) — GenRobot **1,848 h** DAS-Ego 六相机 MCAP + 官方 DFM 手/深度/三级语义（HF ungated CC BY-SA 4.0；das-datakit 消费）
- [Macrodata Egocentric Hand-Action](../methods/macrodata-egocentric-hand-action.md) — 把已采 egocentric RGB **重建为度量 21 关节手轨迹** 的开源配方与 HOT3D Action MPJPE 标尺（采集后的几何标注层）
- [Pico 4 Ultra（Egocentric 采集）](../entities/pico-4-ultra-egocentric-capture.md) — Unidata **4,050 h** 量产工作流：头显-only vs ZED+Orin 双 rig、episode QA 与 Quest 3 / Aria 选型对照
- [Maxinsights](../entities/maxinsights-physical-ai-data.md) — 商业化第一人称多模态数据服务；官网自报规模与月产能需和可用数据、客户交付口径分开核验

- [TSING-EGO PRO](../entities/tsing-ego-pro.md) — 清研精准面向工业操作的第一视角采集方案；详细参数仍需核实

## 参考来源

- [HumanCOIN 论文原文归档](../../sources/papers/humancoin_human_intent_egocentric_manipulation_arxiv_2611_26434.md)

- [wechat_embodied_ai_lab_ego_9_papers_survey.md](../../sources/blogs/wechat_embodied_ai_lab_ego_9_papers_survey.md)
- [ego_9_papers_catalog.md](../../sources/papers/ego_9_papers_catalog.md)
- [EgoVerse 论文摘录](../../sources/papers/egoverse_arxiv_2604_07607.md)
- [Ego4D 论文摘录](../../sources/papers/ego4d_arxiv_2110_07058.md)
- [VBot Exo2Ego Policy（arXiv:2611.16417v1）](../../sources/papers/vbot_exo2ego_arxiv_2611_16417.md)
- [Ego4D 项目页归档](../../sources/sites/ego4d-data-org.md)
- [stellarnex_egoworld_100w.md](../../sources/blogs/stellarnex_egoworld_100w.md)
- [RekaDaily-10k 研究页归档](../../sources/sites/rekadaily-10k.md)
- [RekaCS2-10k 新闻页归档](../../sources/sites/rekacs2-10k.md)
- [HumanPlus-1000 项目页归档](../../sources/sites/humanplus-1000.md)
- [macrodata_egocentric_video_3d_hand_actions.md](../../sources/blogs/macrodata_egocentric_video_3d_hand_actions.md)
- [unidata_pico_4_ultra_egocentric_data_collection.md](../../sources/blogs/unidata_pico_4_ultra_egocentric_data_collection.md)
- [Maxinsights 官方网站归档](../../sources/sites/maxinsights-official.md)
- [Maxinsights 官方规模/密度观点](../../sources/blogs/maxinsights-beyond-1m-hours-physical-ai-data.md)

## 推荐继续阅读

- [机器人论文阅读笔记：EmbodMocap](https://imchong.github.io/Robot_Learning_Paper_Notebooks/papers/14_Human_Motion/EmbodMocap__In-the-Wild_4D_Human-Scene_Reconstruction_for_Embodied_Agents/EmbodMocap__In-the-Wild_4D_Human-Scene_Reconstruction_for_Embodied_Agents.html)
- [Ego4D 项目页](https://ego4d-data.org/)
- [Stereo-550（Ego-OSCAR）](https://huggingface.co/datasets/fpvlabs/stereo-550)
- [EgoVerse 项目页](https://egoverse.ai/)
- [EgoWorld-100W 官方介绍](https://stellarnexrobotics.com/blog)
- [RekaDaily-10k 研究页](https://reka.ai/labs/research/rekadaily-10k-egocentric-household-manipulation-data)
- [RekaCS2-10k 新闻页](https://reka.ai/news/cs2-10k-a-large-scale-egocentric-counter-strike-2-dataset)
- [Unidata：Pico 4 Ultra egocentric 采集工作流](https://unidata.pro/blog/pico-4-ultra-for-egocentric-data-collection/)
