---
type: entity
tags: [paper, loco-manipulation, loco-manip-161-survey, humanoid]
status: complete
updated: 2026-07-06
venue: curated
summary: "这篇工作先从相机图像/多视角观测、本体状态与关节序列、深度/点云/高度图恢复场景、目标或运动表征，再用下视深度相机和 U-Net 高度图重建、步态相位/频率调节、教师-学生知识迁移生成关节位置/力矩命令、地形/场景表征。关键点是把地形重建、步态相位和全身姿态放进同一个控制回路，而不是把感知和运控拆成松散串联。"
related:
  - ../overview/humanoid-loco-manip-161-papers-technology-map.md
  - ../overview/loco-manip-161-category-08-hardware-deployment.md
  - ../tasks/loco-manipulation.md
sources:
  - ../../sources/papers/loco_manip_161_survey_142_n142.md
  - ../../sources/blogs/wechat_embodied_ai_lab_humanoid_loco_manip_161_survey.md
  - ../../sources/papers/humanoid_loco_manip_161_catalog.md
---

# 具有实时基础地形重建的步态自适应感知人形运动

**具有实时基础地形重建的步态自适应感知人形运动** 收录于 [具身智能研究室 · 人形 Loco-Manip 161 篇长文](https://mp.weixin.qq.com/s/pACh9EhsISiyPGdiiR0C3A) **第 142/161** 篇，归类为 **08 硬件平台、感知配置与部署扩展**。

## 一句话定义

这篇工作先从相机图像/多视角观测、本体状态与关节序列、深度/点云/高度图恢复场景、目标或运动表征，再用下视深度相机和 U-Net 高度图重建、步态相位/频率调节、教师-学生知识迁移生成关节位置/力矩命令、地形/场景表征。关键点是把地形重建、步态相位和全身姿态放进同一个控制回路，而不是把感知和运控拆成松散串联。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Loco-Manip | Loco-Manipulation | 行走与操作动力学耦合的全身任务 |
| WBC | Whole-Body Control | 协调全身关节满足多任务/约束的控制层 |
| VLA | Vision-Language-Action | 视觉-语言-动作多模态策略 |

## 为什么重要

- 这篇工作先从相机图像/多视角观测、本体状态与关节序列、深度/点云/高度图恢复场景、目标或运动表征，再用下视深度相机和 U-Net 高度图重建、步态相位/频率调节、教师-学生知识迁移生成关节位置/力矩命令、地形/场景表征。关键点是把地形重建、步态相位和全身姿态放进同一个控制回路，而不是把感知和运控拆成松散串联。
- 人形 Loco-Manip 161 篇 **#142/161** · 硬件平台、感知配置与部署扩展。

## 核心信息（索引级）

| 字段 | 内容 |
|------|------|
| 编号 | 142/161 |
| 分组 | 08 硬件平台、感知配置与部署扩展 |
| 原文题目 | Gait-Adaptive Perceptive Humanoid Locomotion with Real-Time Under-Base Terrain Reconstruction |
| 机构 | （见原文） |
| 发表日期 | 2025年12月8日 |
| 论文/项目 | https://ga-phl.github.io/ |

## 核心机制（归纳）

### 策展导读要点

这篇工作先从相机图像/多视角观测、本体状态与关节序列、深度/点云/高度图恢复场景、目标或运动表征，再用下视深度相机和 U-Net 高度图重建、步态相位/频率调节、教师-学生知识迁移生成关节位置/力矩命令、地形/场景表征。关键点是把地形重建、步态相位和全身姿态放进同一个控制回路，而不是把感知和运控拆成松散串联。

## 评测与指标（索引级）

- 本条目为 161 篇策展索引级摘录，**未搬运原文量化 benchmark 与实机指标**；评测口径与具体数值以原文 PDF / 项目页为准。
- 评测原始出处：[原文 / 项目页](https://ga-phl.github.io/)（见上方「核心信息」表「论文/项目」一行）。
- 横向评测对照请回到 [分类 hub](../overview/loco-manip-161-category-08-hardware-deployment.md) 与 [技术地图](../overview/humanoid-loco-manip-161-papers-technology-map.md)。

## 常见误区

1. 161 篇策展条目提供 **地图坐标**；量化 benchmark 与实机指标以原文 PDF / 项目页为准。
2. Loco-manip 单篇工作不自动解决 **底层 WBC 鲁棒性**；须与运控/接触控制对照。

## 与其他页面的关系

- 技术地图：[humanoid-loco-manip-161-papers-technology-map.md](../overview/humanoid-loco-manip-161-papers-technology-map.md)
- 分类 hub：[loco-manip-161-category-08-hardware-deployment.md](../overview/loco-manip-161-category-08-hardware-deployment.md)
- 原始 source：[loco_manip_161_survey_142_n142.md](../../sources/papers/loco_manip_161_survey_142_n142.md)

## 参考来源

- [loco_manip_161_survey_142_n142.md](../../sources/papers/loco_manip_161_survey_142_n142.md) — 161 篇策展摘录
- [humanoid_loco_manip_161_catalog.md](../../sources/papers/humanoid_loco_manip_161_catalog.md)
- [wechat_embodied_ai_lab_humanoid_loco_manip_161_survey.md](../../sources/blogs/wechat_embodied_ai_lab_humanoid_loco_manip_161_survey.md)

## 推荐继续阅读

- [Loco-Manipulation 任务页](../tasks/loco-manipulation.md)
