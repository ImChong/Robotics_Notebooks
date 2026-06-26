---
type: entity
tags: [paper, loco-manipulation, loco-manip-161-survey, humanoid]
status: complete
updated: 2026-06-26
venue: curated
summary: "RPL 先从相机图像/多视角观测、本体状态与关节序列、深度/点云/高度图恢复场景、目标或运动表征，再用下视深度相机和 U-Net 高度图重建、教师-学生知识迁移、世界模型/视频预测生成全身轨迹/动作序列、低层控制器目标、地形/场景表征。关键点是把地形重建、步态相位和全身姿态放进同一个控制回路，而不是把感知和运控拆成松散串联。"
related:
  - ../overview/humanoid-loco-manip-161-papers-technology-map.md
  - ../overview/loco-manip-161-category-02-upper-body-interface.md
  - ../tasks/loco-manipulation.md
  - ../entities/paper-rpl-robust-humanoid-perceptive-locomotion.md
sources:
  - ../../sources/papers/loco_manip_161_survey_044_rpl.md
  - ../../sources/blogs/wechat_embodied_ai_lab_humanoid_loco_manip_161_survey.md
  - ../../sources/papers/humanoid_loco_manip_161_catalog.md
---

# RPL

**RPL** 收录于 [具身智能研究室 · 人形 Loco-Manip 161 篇长文](https://mp.weixin.qq.com/s/pACh9EhsISiyPGdiiR0C3A) **第 044/161** 篇，归类为 **02 上半身中心控制与移动操作接口**。

## 一句话定义

RPL 先从相机图像/多视角观测、本体状态与关节序列、深度/点云/高度图恢复场景、目标或运动表征，再用下视深度相机和 U-Net 高度图重建、教师-学生知识迁移、世界模型/视频预测生成全身轨迹/动作序列、低层控制器目标、地形/场景表征。关键点是把地形重建、步态相位和全身姿态放进同一个控制回路，而不是把感知和运控拆成松散串联。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Loco-Manip | Loco-Manipulation | 行走与操作动力学耦合的全身任务 |
| WBC | Whole-Body Control | 协调全身关节满足多任务/约束的控制层 |
| VLA | Vision-Language-Action | 视觉-语言-动作多模态策略 |

## 为什么重要

- RPL 先从相机图像/多视角观测、本体状态与关节序列、深度/点云/高度图恢复场景、目标或运动表征，再用下视深度相机和 U-Net 高度图重建、教师-学生知识迁移、世界模型/视频预测生成全身轨迹/动作序列、低层控制器目标、地形/场景表征。关键点是把地形重建、步态相位和全身姿态放进同一个控制回路，而不是把感知和运控拆成松散串联。
- 人形 Loco-Manip 161 篇 **#044/161** · 上半身中心控制与移动操作接口。

## 核心信息（索引级）

| 字段 | 内容 |
|------|------|
| 编号 | 044/161 |
| 分组 | 02 上半身中心控制与移动操作接口 |
| 原文题目 | RPL: Learning Robust Humanoid Perceptive Locomotion on Challenging Terrains |
| 机构 | Carnegie Mellon University、Stanford University、UC Berkeley |
| 发表日期 | 2026年2月3日 |
| 论文/项目 | https://rpl-humanoid.github.io/ |

## 核心机制（归纳）

### 策展导读要点

RPL 先从相机图像/多视角观测、本体状态与关节序列、深度/点云/高度图恢复场景、目标或运动表征，再用下视深度相机和 U-Net 高度图重建、教师-学生知识迁移、世界模型/视频预测生成全身轨迹/动作序列、低层控制器目标、地形/场景表征。关键点是把地形重建、步态相位和全身姿态放进同一个控制回路，而不是把感知和运控拆成松散串联。

## 常见误区

1. 161 篇策展条目提供 **地图坐标**；量化 benchmark 与实机指标以原文 PDF / 项目页为准。
2. Loco-manip 单篇工作不自动解决 **底层 WBC 鲁棒性**；须与运控/接触控制对照。

## 与其他页面的关系

- 技术地图：[humanoid-loco-manip-161-papers-technology-map.md](../overview/humanoid-loco-manip-161-papers-technology-map.md)
- 分类 hub：[loco-manip-161-category-02-upper-body-interface.md](../overview/loco-manip-161-category-02-upper-body-interface.md)
- 原始 source：[loco_manip_161_survey_044_rpl.md](../../sources/papers/loco_manip_161_survey_044_rpl.md)

## 参考来源

- [loco_manip_161_survey_044_rpl.md](../../sources/papers/loco_manip_161_survey_044_rpl.md) — 161 篇策展摘录
- [humanoid_loco_manip_161_catalog.md](../../sources/papers/humanoid_loco_manip_161_catalog.md)
- [wechat_embodied_ai_lab_humanoid_loco_manip_161_survey.md](../../sources/blogs/wechat_embodied_ai_lab_humanoid_loco_manip_161_survey.md)

## 推荐继续阅读

- [Loco-Manipulation 任务页](../tasks/loco-manipulation.md)
- 同题深读/既有实体：[paper-rpl-robust-humanoid-perceptive-locomotion](../entities/paper-rpl-robust-humanoid-perceptive-locomotion.md)
