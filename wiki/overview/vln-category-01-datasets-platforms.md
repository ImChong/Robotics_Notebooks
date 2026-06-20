---
type: overview
tags: [vln, navigation, datasets, simulation, category-hub, survey]
status: complete
updated: 2026-06-20
summary: "VLN 10 篇盘点 · 01 数据集与仿真平台（3 篇）— 任务如何定义、评测，并从离散导航图走向连续环境与高层目标定位？"
related:
  - ./vln-10-papers-technology-map.md
  - ./vln-category-02-algorithm-frameworks.md
  - ../entities/paper-vln-01-r2r.md
  - ../entities/paper-vln-02-vln-ce.md
  - ../entities/paper-vln-03-reverie.md
  - ../tasks/vision-language-navigation.md
sources:
  - ../../sources/blogs/wechat_shenlan_vln_10_papers_survey.md
  - ../../sources/papers/vln_10_papers_catalog.md
---

# VLN 分类 01：数据集与仿真平台

> **图谱分类节点**：对应 [深蓝具身智能 · VLN 10 项代表性研究](https://mp.weixin.qq.com/s/2_dYaN6IeWn_vvS_jmGqRQ) 的 **01 数据集与仿真平台** 分组；总地图见 [VLN 10 篇技术地图](./vln-10-papers-technology-map.md)。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLN | Vision-and-Language Navigation | 依据自然语言在环境中导航的具身任务 |
| R2R | Room-to-Room | Matterport3D 经典逐步导航指令数据集 |
| VLN-CE | VLN in Continuous Environments | 连续动作空间下的 VLN 设定与 benchmark 族 |

## 核心问题

**VLN 任务的基础设施如何演进？** 从离散导航图 + 逐步指令（R2R），到连续底层动作（VLN-CE），再到高层目标描述 + 物体定位（REVERIE）——每一步都在拉近仿真设定与真实机器人部署的距离。

## 本组论文（3 篇）

| # | 工作 | Wiki 实体 | Source |
|---|------|-----------|--------|
| 01 | R2R | [paper-vln-01-r2r.md](../entities/paper-vln-01-r2r.md) | [source](../../sources/papers/vln_survey_01_r2r.md) |
| 02 | VLN-CE | [paper-vln-02-vln-ce.md](../entities/paper-vln-02-vln-ce.md) | [source](../../sources/papers/vln_survey_02_vln_ce.md) |
| 03 | REVERIE | [paper-vln-03-reverie.md](../entities/paper-vln-03-reverie.md) | [source](../../sources/papers/vln_survey_03_reverie.md) |

## 与算法框架组的分工

| 维度 | 本组（01 数据/平台） | [02 算法框架](./vln-category-02-algorithm-frameworks.md) |
|------|----------------------|----------------------------------------------------------|
| 回答 | **测什么、在哪测、动作空间如何定义** | **模型如何对齐、记忆、规划、扩数据、接 VLM** |
| 代表 | Matterport3D / Habitat / 高层 referring | PREVALENT → NaVid |

## 关联页面

- [VLN 10 篇技术地图](./vln-10-papers-technology-map.md)
- [算法框架](./vln-category-02-algorithm-frameworks.md)
- [视觉–语言导航（VLN）](../tasks/vision-language-navigation.md)
- [SceneVerse++](../entities/sceneverse-pp.md) — 互联网视频→R2R 风格数据（后续数据扩展路线对照）

## 参考来源

- [wechat_shenlan_vln_10_papers_survey.md](../../sources/blogs/wechat_shenlan_vln_10_papers_survey.md)
- [vln_10_papers_catalog.md](../../sources/papers/vln_10_papers_catalog.md)

## 推荐继续阅读

- [VLN 四范式开源复现路径](./vln-open-source-repro-paradigms.md)
