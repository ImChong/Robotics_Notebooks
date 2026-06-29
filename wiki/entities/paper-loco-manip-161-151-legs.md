---
type: entity
tags: [paper, loco-manipulation, loco-manip-161-survey, humanoid]
status: complete
updated: 2026-06-29
venue: curated
summary: "LEGS 的实现路径是先把语言指令、相机图像/多视角观测、本体状态与关节序列编码成多模态表征，再用ACT/行为克隆模仿学习、VLA 多模态动作模型、世界模型/视频预测预测低层控制器目标、地形/场景表征。关键点是让视频/世界模型提供可预测的物理先验，再由动作头把语义目标变成连续控制。"
related:
  - ../overview/humanoid-loco-manip-161-papers-technology-map.md
  - ../overview/loco-manip-161-category-09-vla-world-models.md
  - ../tasks/loco-manipulation.md
  - ../entities/paper-legs-embodied-gaussian-splatting-vla.md
sources:
  - ../../sources/papers/loco_manip_161_survey_151_legs.md
  - ../../sources/blogs/wechat_embodied_ai_lab_humanoid_loco_manip_161_survey.md
  - ../../sources/papers/humanoid_loco_manip_161_catalog.md
---

# LEGS

**LEGS** 收录于 [具身智能研究室 · 人形 Loco-Manip 161 篇长文](https://mp.weixin.qq.com/s/pACh9EhsISiyPGdiiR0C3A) **第 151/161** 篇，归类为 **09 人形 VLA、世界模型与通用操作**。

## 一句话定义

LEGS 的实现路径是先把语言指令、相机图像/多视角观测、本体状态与关节序列编码成多模态表征，再用ACT/行为克隆模仿学习、VLA 多模态动作模型、世界模型/视频预测预测低层控制器目标、地形/场景表征。关键点是让视频/世界模型提供可预测的物理先验，再由动作头把语义目标变成连续控制。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Loco-Manip | Loco-Manipulation | 行走与操作动力学耦合的全身任务 |
| WBC | Whole-Body Control | 协调全身关节满足多任务/约束的控制层 |
| VLA | Vision-Language-Action | 视觉-语言-动作多模态策略 |

## 为什么重要

- LEGS 的实现路径是先把语言指令、相机图像/多视角观测、本体状态与关节序列编码成多模态表征，再用ACT/行为克隆模仿学习、VLA 多模态动作模型、世界模型/视频预测预测低层控制器目标、地形/场景表征。关键点是让视频/世界模型提供可预测的物理先验，再由动作头把语义目标变成连续控制。
- 人形 Loco-Manip 161 篇 **#151/161** · 人形 VLA、世界模型与通用操作。

## 核心信息（索引级）

| 字段 | 内容 |
|------|------|
| 编号 | 151/161 |
| 分组 | 09 人形 VLA、世界模型与通用操作 |
| 原文题目 | LEGS: Fine-Tuning Teleop-Free VLAs for Humanoid Loco-manipulation in an Embodied Gaussian Splatting World |
| 机构 | Stanford University |
| 发表日期 | 2026年5月31日 |
| 论文/项目 | https://legsvla.github.io/ |

## 核心机制（归纳）

### 策展导读要点

LEGS 的实现路径是先把语言指令、相机图像/多视角观测、本体状态与关节序列编码成多模态表征，再用ACT/行为克隆模仿学习、VLA 多模态动作模型、世界模型/视频预测预测低层控制器目标、地形/场景表征。关键点是让视频/世界模型提供可预测的物理先验，再由动作头把语义目标变成连续控制。

## 评测与指标（索引级）

- 本条目为 161 篇策展索引级摘录，**未搬运原文量化 benchmark 与实机指标**；评测口径与具体数值以原文 PDF / 项目页为准。
- 评测原始出处：[原文 / 项目页](https://legsvla.github.io/)（见上方「核心信息」表「论文/项目」一行）。
- 横向评测对照请回到 [分类 hub](../overview/loco-manip-161-category-09-vla-world-models.md) 与 [技术地图](../overview/humanoid-loco-manip-161-papers-technology-map.md)。

## 常见误区

1. 161 篇策展条目提供 **地图坐标**；量化 benchmark 与实机指标以原文 PDF / 项目页为准。
2. Loco-manip 单篇工作不自动解决 **底层 WBC 鲁棒性**；须与运控/接触控制对照。

## 与其他页面的关系

- 技术地图：[humanoid-loco-manip-161-papers-technology-map.md](../overview/humanoid-loco-manip-161-papers-technology-map.md)
- 分类 hub：[loco-manip-161-category-09-vla-world-models.md](../overview/loco-manip-161-category-09-vla-world-models.md)
- 原始 source：[loco_manip_161_survey_151_legs.md](../../sources/papers/loco_manip_161_survey_151_legs.md)

## 参考来源

- [loco_manip_161_survey_151_legs.md](../../sources/papers/loco_manip_161_survey_151_legs.md) — 161 篇策展摘录
- [humanoid_loco_manip_161_catalog.md](../../sources/papers/humanoid_loco_manip_161_catalog.md)
- [wechat_embodied_ai_lab_humanoid_loco_manip_161_survey.md](../../sources/blogs/wechat_embodied_ai_lab_humanoid_loco_manip_161_survey.md)

## 推荐继续阅读

- [Loco-Manipulation 任务页](../tasks/loco-manipulation.md)
- 同题深读/既有实体：[paper-legs-embodied-gaussian-splatting-vla](../entities/paper-legs-embodied-gaussian-splatting-vla.md)
