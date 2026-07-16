---
type: entity
tags: [paper, loco-manipulation, loco-manip-161-survey, humanoid]
status: complete
updated: 2026-07-16
venue: curated
summary: "Gallant 先从相机图像/多视角观测、本体状态与关节序列、深度/点云/高度图恢复场景、目标或运动表征，再用策略网络和控制模块生成地形/场景表征、导航/到达目标。关键点是把策略网络和控制模块放在同一条训练/部署链路里，减少高层目标到低层动作之间的断点。"
related:
  - ../overview/humanoid-loco-manip-161-papers-technology-map.md
  - ../overview/loco-manip-161-category-08-hardware-deployment.md
  - ../tasks/loco-manipulation.md
sources:
  - ../../sources/papers/loco_manip_161_survey_137_gallant.md
  - ../../sources/blogs/wechat_embodied_ai_lab_humanoid_loco_manip_161_survey.md
  - ../../sources/papers/humanoid_loco_manip_161_catalog.md
---

# Gallant

**Gallant** 收录于 [具身智能研究室 · 人形 Loco-Manip 161 篇长文](https://mp.weixin.qq.com/s/pACh9EhsISiyPGdiiR0C3A) **第 137/161** 篇，归类为 **08 硬件平台、感知配置与部署扩展**。

## 一句话定义

Gallant 先从相机图像/多视角观测、本体状态与关节序列、深度/点云/高度图恢复场景、目标或运动表征，再用策略网络和控制模块生成地形/场景表征、导航/到达目标。关键点是把策略网络和控制模块放在同一条训练/部署链路里，减少高层目标到低层动作之间的断点。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Loco-Manip | Loco-Manipulation | 行走与操作动力学耦合的全身任务 |
| WBC | Whole-Body Control | 协调全身关节满足多任务/约束的控制层 |
| VLA | Vision-Language-Action | 视觉-语言-动作多模态策略 |

## 为什么重要

- Gallant 先从相机图像/多视角观测、本体状态与关节序列、深度/点云/高度图恢复场景、目标或运动表征，再用策略网络和控制模块生成地形/场景表征、导航/到达目标。关键点是把策略网络和控制模块放在同一条训练/部署链路里，减少高层目标到低层动作之间的断点。
- 人形 Loco-Manip 161 篇 **#137/161** · 硬件平台、感知配置与部署扩展。

## 核心信息（索引级）

| 字段 | 内容 |
|------|------|
| 编号 | 137/161 |
| 分组 | 08 硬件平台、感知配置与部署扩展 |
| 原文题目 | Gallant: Voxel Grid-based Humanoid Locomotion and Local-navigation across 3D Constrained Terrains |
| 机构 | Shanghai Artificial Intelligence Laboratory、The Chinese University of Hong Kong、University of Science and Technology of China、University of Tokyo |
| 发表日期 | 2025年11月18日 |
| 论文/项目 | https://gallantloco.github.io/ |

## 核心机制（归纳）

### 策展导读要点

Gallant 先从相机图像/多视角观测、本体状态与关节序列、深度/点云/高度图恢复场景、目标或运动表征，再用策略网络和控制模块生成地形/场景表征、导航/到达目标。关键点是把策略网络和控制模块放在同一条训练/部署链路里，减少高层目标到低层动作之间的断点。

## 评测与指标（索引级）

- 本条目为 161 篇策展索引级摘录，**未搬运原文量化 benchmark 与实机指标**；评测口径与具体数值以原文 PDF / 项目页为准。
- 评测原始出处：[原文 / 项目页](https://gallantloco.github.io/)（见上方「核心信息」表「论文/项目」一行）。
- 横向评测对照请回到 [分类 hub](../overview/loco-manip-161-category-08-hardware-deployment.md) 与 [技术地图](../overview/humanoid-loco-manip-161-papers-technology-map.md)。

## 常见误区

1. 161 篇策展条目提供 **地图坐标**；量化 benchmark 与实机指标以原文 PDF / 项目页为准。
2. Loco-manip 单篇工作不自动解决 **底层 WBC 鲁棒性**；须与运控/接触控制对照。

## 与其他页面的关系

- 技术地图：[humanoid-loco-manip-161-papers-technology-map.md](../overview/humanoid-loco-manip-161-papers-technology-map.md)
- 分类 hub：[loco-manip-161-category-08-hardware-deployment.md](../overview/loco-manip-161-category-08-hardware-deployment.md)
- 原始 source：[loco_manip_161_survey_137_gallant.md](../../sources/papers/loco_manip_161_survey_137_gallant.md)

## 参考来源

- [loco_manip_161_survey_137_gallant.md](../../sources/papers/loco_manip_161_survey_137_gallant.md) — 161 篇策展摘录
- [humanoid_loco_manip_161_catalog.md](../../sources/papers/humanoid_loco_manip_161_catalog.md)
- [wechat_embodied_ai_lab_humanoid_loco_manip_161_survey.md](../../sources/blogs/wechat_embodied_ai_lab_humanoid_loco_manip_161_survey.md)

## 推荐继续阅读

- [机器人论文阅读笔记：Gallant](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/08_Navigation/Gallant__Voxel_Grid-based_Humanoid_Locomotion_and_Local-navigation_across_3D_Constrained_Terrains/Gallant__Voxel_Grid-based_Humanoid_Locomotion_and_Local-navigation_across_3D_Constrained_Terrains.html)
- [Loco-Manipulation 任务页](../tasks/loco-manipulation.md)
- 同题深读/既有实体：[paper-loco-manip-161-137-gallant](../entities/paper-loco-manip-161-137-gallant.md)
