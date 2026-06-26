---
type: entity
tags: [paper, loco-manipulation, loco-manip-161-survey, humanoid]
status: complete
updated: 2026-06-26
venue: curated
summary: "TWIST2 的实现路径是先把相机图像/多视角观测、人类视频/动捕轨迹、遥操作/外骨骼数据编码成多模态表征，再用ACT/行为克隆模仿学习、VLA 多模态动作模型、全身控制器/WBC/MPC预测可执行动作命令。关键点是保留 VLM 的语义理解，同时增加机器人状态和动作头，避免只停留在语言规划。"
related:
  - ../overview/humanoid-loco-manip-161-papers-technology-map.md
  - ../overview/loco-manip-161-category-07-data-teleop.md
  - ../tasks/loco-manipulation.md
  - ../entities/paper-twist2.md
sources:
  - ../../sources/papers/loco_manip_161_survey_132_twist2.md
  - ../../sources/blogs/wechat_embodied_ai_lab_humanoid_loco_manip_161_survey.md
  - ../../sources/papers/humanoid_loco_manip_161_catalog.md
---

# TWIST2

**TWIST2** 收录于 [具身智能研究室 · 人形 Loco-Manip 161 篇长文](https://mp.weixin.qq.com/s/pACh9EhsISiyPGdiiR0C3A) **第 132/161** 篇，归类为 **07 数据采集与遥操作系统**。

## 一句话定义

TWIST2 的实现路径是先把相机图像/多视角观测、人类视频/动捕轨迹、遥操作/外骨骼数据编码成多模态表征，再用ACT/行为克隆模仿学习、VLA 多模态动作模型、全身控制器/WBC/MPC预测可执行动作命令。关键点是保留 VLM 的语义理解，同时增加机器人状态和动作头，避免只停留在语言规划。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Loco-Manip | Loco-Manipulation | 行走与操作动力学耦合的全身任务 |
| WBC | Whole-Body Control | 协调全身关节满足多任务/约束的控制层 |
| VLA | Vision-Language-Action | 视觉-语言-动作多模态策略 |

## 为什么重要

- TWIST2 的实现路径是先把相机图像/多视角观测、人类视频/动捕轨迹、遥操作/外骨骼数据编码成多模态表征，再用ACT/行为克隆模仿学习、VLA 多模态动作模型、全身控制器/WBC/MPC预测可执行动作命令。关键点是保留 VLM 的语义理解，同时增加机器人状态和动作头，避免只停留在语言规划。
- 人形 Loco-Manip 161 篇 **#132/161** · 数据采集与遥操作系统。

## 核心信息（索引级）

| 字段 | 内容 |
|------|------|
| 编号 | 132/161 |
| 分组 | 07 数据采集与遥操作系统 |
| 原文题目 | TWIST2: Scalable, Portable, and Holistic Humanoid Data Collection System |
| 机构 | Stanford University、UC Berkeley、CMU |
| 发表日期 | 2025年11月4日 |
| 论文/项目 | https://twist-data.github.io |

## 核心机制（归纳）

### 策展导读要点

TWIST2 的实现路径是先把相机图像/多视角观测、人类视频/动捕轨迹、遥操作/外骨骼数据编码成多模态表征，再用ACT/行为克隆模仿学习、VLA 多模态动作模型、全身控制器/WBC/MPC预测可执行动作命令。关键点是保留 VLM 的语义理解，同时增加机器人状态和动作头，避免只停留在语言规划。

## 常见误区

1. 161 篇策展条目提供 **地图坐标**；量化 benchmark 与实机指标以原文 PDF / 项目页为准。
2. Loco-manip 单篇工作不自动解决 **底层 WBC 鲁棒性**；须与运控/接触控制对照。

## 与其他页面的关系

- 技术地图：[humanoid-loco-manip-161-papers-technology-map.md](../overview/humanoid-loco-manip-161-papers-technology-map.md)
- 分类 hub：[loco-manip-161-category-07-data-teleop.md](../overview/loco-manip-161-category-07-data-teleop.md)
- 原始 source：[loco_manip_161_survey_132_twist2.md](../../sources/papers/loco_manip_161_survey_132_twist2.md)

## 参考来源

- [loco_manip_161_survey_132_twist2.md](../../sources/papers/loco_manip_161_survey_132_twist2.md) — 161 篇策展摘录
- [humanoid_loco_manip_161_catalog.md](../../sources/papers/humanoid_loco_manip_161_catalog.md)
- [wechat_embodied_ai_lab_humanoid_loco_manip_161_survey.md](../../sources/blogs/wechat_embodied_ai_lab_humanoid_loco_manip_161_survey.md)

## 推荐继续阅读

- [Loco-Manipulation 任务页](../tasks/loco-manipulation.md)
- 同题深读/既有实体：[paper-twist2](../entities/paper-twist2.md)
