---
type: entity
tags: [paper, loco-manipulation, loco-manip-161-survey, humanoid]
status: complete
updated: 2026-06-26
venue: curated
summary: "CLONE 主要解决数据闭环：用遥操作/外骨骼数据采集人类操作和机器人状态，再通过分层技能/专家策略、闭环纠错/人类干预转成可训练、可复用的全身轨迹/动作序列、低层控制器目标。关键点是把任务拆成可路由的技能或专家策略，再用高层模块在执行中选择和组合。"
related:
  - ../overview/humanoid-loco-manip-161-papers-technology-map.md
  - ../overview/loco-manip-161-category-01-motion-base-wbt.md
  - ../tasks/loco-manipulation.md
  - ../entities/paper-bfm-12-clone.md
sources:
  - ../../sources/papers/loco_manip_161_survey_006_clone.md
  - ../../sources/blogs/wechat_embodied_ai_lab_humanoid_loco_manip_161_survey.md
  - ../../sources/papers/humanoid_loco_manip_161_catalog.md
---

# CLONE

**CLONE** 收录于 [具身智能研究室 · 人形 Loco-Manip 161 篇长文](https://mp.weixin.qq.com/s/pACh9EhsISiyPGdiiR0C3A) **第 006/161** 篇，归类为 **01 运控基座与通用全身跟踪**。

## 一句话定义

CLONE 主要解决数据闭环：用遥操作/外骨骼数据采集人类操作和机器人状态，再通过分层技能/专家策略、闭环纠错/人类干预转成可训练、可复用的全身轨迹/动作序列、低层控制器目标。关键点是把任务拆成可路由的技能或专家策略，再用高层模块在执行中选择和组合。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Loco-Manip | Loco-Manipulation | 行走与操作动力学耦合的全身任务 |
| WBC | Whole-Body Control | 协调全身关节满足多任务/约束的控制层 |
| VLA | Vision-Language-Action | 视觉-语言-动作多模态策略 |

## 为什么重要

- CLONE 主要解决数据闭环：用遥操作/外骨骼数据采集人类操作和机器人状态，再通过分层技能/专家策略、闭环纠错/人类干预转成可训练、可复用的全身轨迹/动作序列、低层控制器目标。关键点是把任务拆成可路由的技能或专家策略，再用高层模块在执行中选择和组合。
- 人形 Loco-Manip 161 篇 **#006/161** · 运控基座与通用全身跟踪。

## 核心信息（索引级）

| 字段 | 内容 |
|------|------|
| 编号 | 006/161 |
| 分组 | 01 运控基座与通用全身跟踪 |
| 原文题目 | CLONE: Closed-Loop Whole-Body Humanoid Teleoperation for Long-Horizon Tasks |
| 机构 | School of Computer Science and Technology, Beijing Institute of Technology、State Key Laboratory of General Artificial Intelligence, BIGAI、School of Psychological and Cognitive Sciences, Peking University、Institute for Artificial Intelligence, Peking University |
| 发表日期 | 2025年8月30日 |
| 论文/项目 | https://humanoid-clone.github.io/ |

## 核心机制（归纳）

### 策展导读要点

CLONE 主要解决数据闭环：用遥操作/外骨骼数据采集人类操作和机器人状态，再通过分层技能/专家策略、闭环纠错/人类干预转成可训练、可复用的全身轨迹/动作序列、低层控制器目标。关键点是把任务拆成可路由的技能或专家策略，再用高层模块在执行中选择和组合。

## 常见误区

1. 161 篇策展条目提供 **地图坐标**；量化 benchmark 与实机指标以原文 PDF / 项目页为准。
2. Loco-manip 单篇工作不自动解决 **底层 WBC 鲁棒性**；须与运控/接触控制对照。

## 与其他页面的关系

- 技术地图：[humanoid-loco-manip-161-papers-technology-map.md](../overview/humanoid-loco-manip-161-papers-technology-map.md)
- 分类 hub：[loco-manip-161-category-01-motion-base-wbt.md](../overview/loco-manip-161-category-01-motion-base-wbt.md)
- 原始 source：[loco_manip_161_survey_006_clone.md](../../sources/papers/loco_manip_161_survey_006_clone.md)

## 参考来源

- [loco_manip_161_survey_006_clone.md](../../sources/papers/loco_manip_161_survey_006_clone.md) — 161 篇策展摘录
- [humanoid_loco_manip_161_catalog.md](../../sources/papers/humanoid_loco_manip_161_catalog.md)
- [wechat_embodied_ai_lab_humanoid_loco_manip_161_survey.md](../../sources/blogs/wechat_embodied_ai_lab_humanoid_loco_manip_161_survey.md)

## 推荐继续阅读

- [Loco-Manipulation 任务页](../tasks/loco-manipulation.md)
- 同题深读/既有实体：[paper-bfm-12-clone](../entities/paper-bfm-12-clone.md)
