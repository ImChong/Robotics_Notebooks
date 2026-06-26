---
type: entity
tags: [paper, loco-manipulation, loco-manip-161-survey, humanoid]
status: complete
updated: 2026-06-26
venue: curated
summary: "Agility Meets Stability 把本体状态与关节序列、人类视频/动捕轨迹、仿真交互数据转成可跟踪的身体目标，并通过异构动捕与合成平衡数据、混合奖励与动态采样、分层技能/专家策略训练或组合全身策略，最终输出全身轨迹/动作序列、低层控制器目标。关键点是把敏捷动捕和合成平衡样本放进同一策略，但用混合奖励分别约束动作跟踪与极端平衡。"
related:
  - ../overview/humanoid-loco-manip-161-papers-technology-map.md
  - ../overview/loco-manip-161-category-01-motion-base-wbt.md
  - ../tasks/loco-manipulation.md
  - ../methods/ams.md
sources:
  - ../../sources/papers/loco_manip_161_survey_001_agility-meets-stability.md
  - ../../sources/blogs/wechat_embodied_ai_lab_humanoid_loco_manip_161_survey.md
  - ../../sources/papers/humanoid_loco_manip_161_catalog.md
---

# Agility Meets Stability

**Agility Meets Stability** 收录于 [具身智能研究室 · 人形 Loco-Manip 161 篇长文](https://mp.weixin.qq.com/s/pACh9EhsISiyPGdiiR0C3A) **第 001/161** 篇，归类为 **01 运控基座与通用全身跟踪**。

## 一句话定义

Agility Meets Stability 把本体状态与关节序列、人类视频/动捕轨迹、仿真交互数据转成可跟踪的身体目标，并通过异构动捕与合成平衡数据、混合奖励与动态采样、分层技能/专家策略训练或组合全身策略，最终输出全身轨迹/动作序列、低层控制器目标。关键点是把敏捷动捕和合成平衡样本放进同一策略，但用混合奖励分别约束动作跟踪与极端平衡。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Loco-Manip | Loco-Manipulation | 行走与操作动力学耦合的全身任务 |
| WBC | Whole-Body Control | 协调全身关节满足多任务/约束的控制层 |
| VLA | Vision-Language-Action | 视觉-语言-动作多模态策略 |

## 为什么重要

- Agility Meets Stability 把本体状态与关节序列、人类视频/动捕轨迹、仿真交互数据转成可跟踪的身体目标，并通过异构动捕与合成平衡数据、混合奖励与动态采样、分层技能/专家策略训练或组合全身策略，最终输出全身轨迹/动作序列、低层控制器目标。关键点是把敏捷动捕和合成平衡样本放进同一策略，但用混合奖励分别约束动作跟踪与极端平衡。
- 人形 Loco-Manip 161 篇 **#001/161** · 运控基座与通用全身跟踪。

## 核心信息（索引级）

| 字段 | 内容 |
|------|------|
| 编号 | 001/161 |
| 分组 | 01 运控基座与通用全身跟踪 |
| 原文题目 | Agility Meets Stability: Versatile Humanoid Control with Heterogeneous Data |
| 机构 | The University of Hong Kong、NVIDIA、Tsinghua University |
| 发表日期 | 2026年3月3日 |
| 论文/项目 | https://opendrivelab.com/AMS/ |

## 核心机制（归纳）

### 策展导读要点

Agility Meets Stability 把本体状态与关节序列、人类视频/动捕轨迹、仿真交互数据转成可跟踪的身体目标，并通过异构动捕与合成平衡数据、混合奖励与动态采样、分层技能/专家策略训练或组合全身策略，最终输出全身轨迹/动作序列、低层控制器目标。关键点是把敏捷动捕和合成平衡样本放进同一策略，但用混合奖励分别约束动作跟踪与极端平衡。

## 常见误区

1. 161 篇策展条目提供 **地图坐标**；量化 benchmark 与实机指标以原文 PDF / 项目页为准。
2. Loco-manip 单篇工作不自动解决 **底层 WBC 鲁棒性**；须与运控/接触控制对照。

## 与其他页面的关系

- 技术地图：[humanoid-loco-manip-161-papers-technology-map.md](../overview/humanoid-loco-manip-161-papers-technology-map.md)
- 分类 hub：[loco-manip-161-category-01-motion-base-wbt.md](../overview/loco-manip-161-category-01-motion-base-wbt.md)
- 原始 source：[loco_manip_161_survey_001_agility-meets-stability.md](../../sources/papers/loco_manip_161_survey_001_agility-meets-stability.md)

## 参考来源

- [loco_manip_161_survey_001_agility-meets-stability.md](../../sources/papers/loco_manip_161_survey_001_agility-meets-stability.md) — 161 篇策展摘录
- [humanoid_loco_manip_161_catalog.md](../../sources/papers/humanoid_loco_manip_161_catalog.md)
- [wechat_embodied_ai_lab_humanoid_loco_manip_161_survey.md](../../sources/blogs/wechat_embodied_ai_lab_humanoid_loco_manip_161_survey.md)

## 推荐继续阅读

- [Loco-Manipulation 任务页](../tasks/loco-manipulation.md)
- 同题深读/既有实体：[ams](../methods/ams.md)
