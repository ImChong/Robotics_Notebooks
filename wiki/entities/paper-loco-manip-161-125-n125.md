---
type: entity
tags: [paper, loco-manipulation, loco-manip-161-survey, humanoid]
status: complete
updated: 2026-06-26
venue: curated
summary: "这篇工作先从相机图像/多视角观测、本体状态与关节序列、仿真交互数据恢复场景、目标或运动表征，再用教师-学生知识迁移、PPO/RL 策略训练、全身控制器/WBC/MPC生成可执行动作命令。关键点是用特权信息训练教师策略，再把能力蒸馏到只能使用部署观测的学生策略。"
related:
  - ../overview/humanoid-loco-manip-161-papers-technology-map.md
  - ../overview/loco-manip-161-category-06-contact-tasks.md
  - ../tasks/loco-manipulation.md
sources:
  - ../../sources/papers/loco_manip_161_survey_125_n125.md
  - ../../sources/blogs/wechat_embodied_ai_lab_humanoid_loco_manip_161_survey.md
  - ../../sources/papers/humanoid_loco_manip_161_catalog.md
---

# 学习人形机器人视觉驱动的反应式足球技能

**学习人形机器人视觉驱动的反应式足球技能** 收录于 [具身智能研究室 · 人形 Loco-Manip 161 篇长文](https://mp.weixin.qq.com/s/pACh9EhsISiyPGdiiR0C3A) **第 125/161** 篇，归类为 **06 特殊任务、接触规划与视觉闭环**。

## 一句话定义

这篇工作先从相机图像/多视角观测、本体状态与关节序列、仿真交互数据恢复场景、目标或运动表征，再用教师-学生知识迁移、PPO/RL 策略训练、全身控制器/WBC/MPC生成可执行动作命令。关键点是用特权信息训练教师策略，再把能力蒸馏到只能使用部署观测的学生策略。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Loco-Manip | Loco-Manipulation | 行走与操作动力学耦合的全身任务 |
| WBC | Whole-Body Control | 协调全身关节满足多任务/约束的控制层 |
| VLA | Vision-Language-Action | 视觉-语言-动作多模态策略 |

## 为什么重要

- 这篇工作先从相机图像/多视角观测、本体状态与关节序列、仿真交互数据恢复场景、目标或运动表征，再用教师-学生知识迁移、PPO/RL 策略训练、全身控制器/WBC/MPC生成可执行动作命令。关键点是用特权信息训练教师策略，再把能力蒸馏到只能使用部署观测的学生策略。
- 人形 Loco-Manip 161 篇 **#125/161** · 特殊任务、接触规划与视觉闭环。

## 核心信息（索引级）

| 字段 | 内容 |
|------|------|
| 编号 | 125/161 |
| 分组 | 06 特殊任务、接触规划与视觉闭环 |
| 原文题目 | Learning Agile Striker Skills for Humanoid Soccer Robots from Noisy Sensory Input |
| 机构 | Department of Computer Science, The University of Texas at Austin、Sony AI |
| 发表日期 | 2026年5月10日 |
| 论文/项目 | https://humanoidsoccer.github.io |

## 核心机制（归纳）

### 策展导读要点

这篇工作先从相机图像/多视角观测、本体状态与关节序列、仿真交互数据恢复场景、目标或运动表征，再用教师-学生知识迁移、PPO/RL 策略训练、全身控制器/WBC/MPC生成可执行动作命令。关键点是用特权信息训练教师策略，再把能力蒸馏到只能使用部署观测的学生策略。

## 常见误区

1. 161 篇策展条目提供 **地图坐标**；量化 benchmark 与实机指标以原文 PDF / 项目页为准。
2. Loco-manip 单篇工作不自动解决 **底层 WBC 鲁棒性**；须与运控/接触控制对照。

## 与其他页面的关系

- 技术地图：[humanoid-loco-manip-161-papers-technology-map.md](../overview/humanoid-loco-manip-161-papers-technology-map.md)
- 分类 hub：[loco-manip-161-category-06-contact-tasks.md](../overview/loco-manip-161-category-06-contact-tasks.md)
- 原始 source：[loco_manip_161_survey_125_n125.md](../../sources/papers/loco_manip_161_survey_125_n125.md)

## 参考来源

- [loco_manip_161_survey_125_n125.md](../../sources/papers/loco_manip_161_survey_125_n125.md) — 161 篇策展摘录
- [humanoid_loco_manip_161_catalog.md](../../sources/papers/humanoid_loco_manip_161_catalog.md)
- [wechat_embodied_ai_lab_humanoid_loco_manip_161_survey.md](../../sources/blogs/wechat_embodied_ai_lab_humanoid_loco_manip_161_survey.md)

## 推荐继续阅读

- [Loco-Manipulation 任务页](../tasks/loco-manipulation.md)
