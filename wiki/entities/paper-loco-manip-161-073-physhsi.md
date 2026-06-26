---
type: entity
tags: [paper, loco-manipulation, loco-manip-161-survey, humanoid]
status: complete
updated: 2026-06-26
venue: curated
summary: "PhysHSI 先从相机图像/多视角观测、本体状态与关节序列、深度/点云/高度图恢复场景、目标或运动表征，再用AMP/运动先验、DINO/视觉特征抽取生成地形/场景表征。关键点是把AMP/运动先验、DINO/视觉特征抽取放在同一条训练/部署链路里，减少高层目标到低层动作之间的断点。"
related:
  - ../overview/humanoid-loco-manip-161-papers-technology-map.md
  - ../overview/loco-manip-161-category-03-visuomotor.md
  - ../tasks/loco-manipulation.md
  - ../entities/paper-amp-survey-15-physhsi.md
sources:
  - ../../sources/papers/loco_manip_161_survey_073_physhsi.md
  - ../../sources/blogs/wechat_embodied_ai_lab_humanoid_loco_manip_161_survey.md
  - ../../sources/papers/humanoid_loco_manip_161_catalog.md
---

# PhysHSI

**PhysHSI** 收录于 [具身智能研究室 · 人形 Loco-Manip 161 篇长文](https://mp.weixin.qq.com/s/pACh9EhsISiyPGdiiR0C3A) **第 073/161** 篇，归类为 **03 视觉感知驱动的人形移动操作**。

## 一句话定义

PhysHSI 先从相机图像/多视角观测、本体状态与关节序列、深度/点云/高度图恢复场景、目标或运动表征，再用AMP/运动先验、DINO/视觉特征抽取生成地形/场景表征。关键点是把AMP/运动先验、DINO/视觉特征抽取放在同一条训练/部署链路里，减少高层目标到低层动作之间的断点。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Loco-Manip | Loco-Manipulation | 行走与操作动力学耦合的全身任务 |
| WBC | Whole-Body Control | 协调全身关节满足多任务/约束的控制层 |
| VLA | Vision-Language-Action | 视觉-语言-动作多模态策略 |

## 为什么重要

- PhysHSI 先从相机图像/多视角观测、本体状态与关节序列、深度/点云/高度图恢复场景、目标或运动表征，再用AMP/运动先验、DINO/视觉特征抽取生成地形/场景表征。关键点是把AMP/运动先验、DINO/视觉特征抽取放在同一条训练/部署链路里，减少高层目标到低层动作之间的断点。
- 人形 Loco-Manip 161 篇 **#073/161** · 视觉感知驱动的人形移动操作。

## 核心信息（索引级）

| 字段 | 内容 |
|------|------|
| 编号 | 073/161 |
| 分组 | 03 视觉感知驱动的人形移动操作 |
| 原文题目 | PhysHSI: Towards a Real-World Generalizable and Natural Humanoid-Scene Interaction System |
| 机构 | （见原文） |
| 发表日期 | 2025年10月13日 |
| 论文/项目 | https://why618188.github.io/physhsi |

## 核心机制（归纳）

### 策展导读要点

PhysHSI 先从相机图像/多视角观测、本体状态与关节序列、深度/点云/高度图恢复场景、目标或运动表征，再用AMP/运动先验、DINO/视觉特征抽取生成地形/场景表征。关键点是把AMP/运动先验、DINO/视觉特征抽取放在同一条训练/部署链路里，减少高层目标到低层动作之间的断点。

## 常见误区

1. 161 篇策展条目提供 **地图坐标**；量化 benchmark 与实机指标以原文 PDF / 项目页为准。
2. Loco-manip 单篇工作不自动解决 **底层 WBC 鲁棒性**；须与运控/接触控制对照。

## 与其他页面的关系

- 技术地图：[humanoid-loco-manip-161-papers-technology-map.md](../overview/humanoid-loco-manip-161-papers-technology-map.md)
- 分类 hub：[loco-manip-161-category-03-visuomotor.md](../overview/loco-manip-161-category-03-visuomotor.md)
- 原始 source：[loco_manip_161_survey_073_physhsi.md](../../sources/papers/loco_manip_161_survey_073_physhsi.md)

## 参考来源

- [loco_manip_161_survey_073_physhsi.md](../../sources/papers/loco_manip_161_survey_073_physhsi.md) — 161 篇策展摘录
- [humanoid_loco_manip_161_catalog.md](../../sources/papers/humanoid_loco_manip_161_catalog.md)
- [wechat_embodied_ai_lab_humanoid_loco_manip_161_survey.md](../../sources/blogs/wechat_embodied_ai_lab_humanoid_loco_manip_161_survey.md)

## 推荐继续阅读

- [Loco-Manipulation 任务页](../tasks/loco-manipulation.md)
- 同题深读/既有实体：[paper-amp-survey-15-physhsi](../entities/paper-amp-survey-15-physhsi.md)
