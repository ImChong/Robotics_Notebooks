---
type: entity
tags: [paper, loco-manipulation, loco-manip-161-survey, humanoid]
status: complete
updated: 2026-06-30
venue: curated
summary: "SIMPLE 的实现路径是先把相机图像/多视角观测、本体状态与关节序列、遥操作/外骨骼数据编码成多模态表征，再用VLA 多模态动作模型、世界模型/视频预测预测全身轨迹/动作序列。关键点是让视频/世界模型提供可预测的物理先验，再由动作头把语义目标变成连续控制。"
related:
  - ../overview/humanoid-loco-manip-161-papers-technology-map.md
  - ../overview/loco-manip-161-category-03-visuomotor.md
  - ../tasks/loco-manipulation.md
sources:
  - ../../sources/papers/loco_manip_161_survey_075_simple.md
  - ../../sources/blogs/wechat_embodied_ai_lab_humanoid_loco_manip_161_survey.md
  - ../../sources/papers/humanoid_loco_manip_161_catalog.md
---

# SIMPLE

**SIMPLE** 收录于 [具身智能研究室 · 人形 Loco-Manip 161 篇长文](https://mp.weixin.qq.com/s/pACh9EhsISiyPGdiiR0C3A) **第 075/161** 篇，归类为 **03 视觉感知驱动的人形移动操作**。

## 一句话定义

SIMPLE 的实现路径是先把相机图像/多视角观测、本体状态与关节序列、遥操作/外骨骼数据编码成多模态表征，再用VLA 多模态动作模型、世界模型/视频预测预测全身轨迹/动作序列。关键点是让视频/世界模型提供可预测的物理先验，再由动作头把语义目标变成连续控制。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Loco-Manip | Loco-Manipulation | 行走与操作动力学耦合的全身任务 |
| WBC | Whole-Body Control | 协调全身关节满足多任务/约束的控制层 |
| VLA | Vision-Language-Action | 视觉-语言-动作多模态策略 |

## 为什么重要

- SIMPLE 的实现路径是先把相机图像/多视角观测、本体状态与关节序列、遥操作/外骨骼数据编码成多模态表征，再用VLA 多模态动作模型、世界模型/视频预测预测全身轨迹/动作序列。关键点是让视频/世界模型提供可预测的物理先验，再由动作头把语义目标变成连续控制。
- 人形 Loco-Manip 161 篇 **#075/161** · 视觉感知驱动的人形移动操作。

## 核心信息（索引级）

| 字段 | 内容 |
|------|------|
| 编号 | 075/161 |
| 分组 | 03 视觉感知驱动的人形移动操作 |
| 原文题目 | SIMPLE: Simulation-Based Policy Learning and Evaluation for Humanoid Loco-manipulation |
| 机构 | （见原文） |
| 发表日期 | 2026年6月6日 |
| 论文/项目 | https://psi-lab.ai/SIMPLE |

## 核心机制（归纳）

### 策展导读要点

SIMPLE 的实现路径是先把相机图像/多视角观测、本体状态与关节序列、遥操作/外骨骼数据编码成多模态表征，再用VLA 多模态动作模型、世界模型/视频预测预测全身轨迹/动作序列。关键点是让视频/世界模型提供可预测的物理先验，再由动作头把语义目标变成连续控制。

## 评测与指标（索引级）

- 本条目为 161 篇策展索引级摘录，**未搬运原文量化 benchmark 与实机指标**；评测口径与具体数值以原文 PDF / 项目页为准。
- 评测原始出处：[原文 / 项目页](https://psi-lab.ai/SIMPLE)（见上方「核心信息」表「论文/项目」一行）。
- 横向评测对照请回到 [分类 hub](../overview/loco-manip-161-category-03-visuomotor.md) 与 [技术地图](../overview/humanoid-loco-manip-161-papers-technology-map.md)。

## 常见误区

1. 161 篇策展条目提供 **地图坐标**；量化 benchmark 与实机指标以原文 PDF / 项目页为准。
2. Loco-manip 单篇工作不自动解决 **底层 WBC 鲁棒性**；须与运控/接触控制对照。

## 与其他页面的关系

- 技术地图：[humanoid-loco-manip-161-papers-technology-map.md](../overview/humanoid-loco-manip-161-papers-technology-map.md)
- 分类 hub：[loco-manip-161-category-03-visuomotor.md](../overview/loco-manip-161-category-03-visuomotor.md)
- 原始 source：[loco_manip_161_survey_075_simple.md](../../sources/papers/loco_manip_161_survey_075_simple.md)

## 参考来源

- [loco_manip_161_survey_075_simple.md](../../sources/papers/loco_manip_161_survey_075_simple.md) — 161 篇策展摘录
- [humanoid_loco_manip_161_catalog.md](../../sources/papers/humanoid_loco_manip_161_catalog.md)
- [wechat_embodied_ai_lab_humanoid_loco_manip_161_survey.md](../../sources/blogs/wechat_embodied_ai_lab_humanoid_loco_manip_161_survey.md)

## 推荐继续阅读

- [Loco-Manipulation 任务页](../tasks/loco-manipulation.md)
