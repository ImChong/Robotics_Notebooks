---
type: entity
tags: [paper, loco-manipulation, loco-manip-161-survey, humanoid]
status: complete
updated: 2026-06-26
venue: curated
summary: "SplitAdapter 把仿真交互数据转成可跟踪的身体目标，并通过PPO/RL 策略训练、扩散策略/流匹配、全身控制器/WBC/MPC训练或组合全身策略，最终输出全身轨迹/动作序列、低层控制器目标。关键点是把动作生成看成条件生成问题，用扩散或流匹配在多模态动作分布里采样可执行轨迹。"
related:
  - ../overview/humanoid-loco-manip-161-papers-technology-map.md
  - ../overview/loco-manip-161-category-02-upper-body-interface.md
  - ../tasks/loco-manipulation.md
  - ../entities/paper-splitadapter-load-aware-loco-manipulation.md
sources:
  - ../../sources/papers/loco_manip_161_survey_045_splitadapter.md
  - ../../sources/blogs/wechat_embodied_ai_lab_humanoid_loco_manip_161_survey.md
  - ../../sources/papers/humanoid_loco_manip_161_catalog.md
---

# SplitAdapter

**SplitAdapter** 收录于 [具身智能研究室 · 人形 Loco-Manip 161 篇长文](https://mp.weixin.qq.com/s/pACh9EhsISiyPGdiiR0C3A) **第 045/161** 篇，归类为 **02 上半身中心控制与移动操作接口**。

## 一句话定义

SplitAdapter 把仿真交互数据转成可跟踪的身体目标，并通过PPO/RL 策略训练、扩散策略/流匹配、全身控制器/WBC/MPC训练或组合全身策略，最终输出全身轨迹/动作序列、低层控制器目标。关键点是把动作生成看成条件生成问题，用扩散或流匹配在多模态动作分布里采样可执行轨迹。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Loco-Manip | Loco-Manipulation | 行走与操作动力学耦合的全身任务 |
| WBC | Whole-Body Control | 协调全身关节满足多任务/约束的控制层 |
| VLA | Vision-Language-Action | 视觉-语言-动作多模态策略 |

## 为什么重要

- SplitAdapter 把仿真交互数据转成可跟踪的身体目标，并通过PPO/RL 策略训练、扩散策略/流匹配、全身控制器/WBC/MPC训练或组合全身策略，最终输出全身轨迹/动作序列、低层控制器目标。关键点是把动作生成看成条件生成问题，用扩散或流匹配在多模态动作分布里采样可执行轨迹。
- 人形 Loco-Manip 161 篇 **#045/161** · 上半身中心控制与移动操作接口。

## 核心信息（索引级）

| 字段 | 内容 |
|------|------|
| 编号 | 045/161 |
| 分组 | 02 上半身中心控制与移动操作接口 |
| 原文题目 | SplitAdapter: Load-Aware Humanoid Loco-Manipulation via Factorized Adaptation |
| 机构 | （见原文） |
| 发表日期 | 2026年6月2日 |
| 论文/项目 | https://splitadapter.github.io/ |

## 核心机制（归纳）

### 策展导读要点

SplitAdapter 把仿真交互数据转成可跟踪的身体目标，并通过PPO/RL 策略训练、扩散策略/流匹配、全身控制器/WBC/MPC训练或组合全身策略，最终输出全身轨迹/动作序列、低层控制器目标。关键点是把动作生成看成条件生成问题，用扩散或流匹配在多模态动作分布里采样可执行轨迹。

## 常见误区

1. 161 篇策展条目提供 **地图坐标**；量化 benchmark 与实机指标以原文 PDF / 项目页为准。
2. Loco-manip 单篇工作不自动解决 **底层 WBC 鲁棒性**；须与运控/接触控制对照。

## 与其他页面的关系

- 技术地图：[humanoid-loco-manip-161-papers-technology-map.md](../overview/humanoid-loco-manip-161-papers-technology-map.md)
- 分类 hub：[loco-manip-161-category-02-upper-body-interface.md](../overview/loco-manip-161-category-02-upper-body-interface.md)
- 原始 source：[loco_manip_161_survey_045_splitadapter.md](../../sources/papers/loco_manip_161_survey_045_splitadapter.md)

## 参考来源

- [loco_manip_161_survey_045_splitadapter.md](../../sources/papers/loco_manip_161_survey_045_splitadapter.md) — 161 篇策展摘录
- [humanoid_loco_manip_161_catalog.md](../../sources/papers/humanoid_loco_manip_161_catalog.md)
- [wechat_embodied_ai_lab_humanoid_loco_manip_161_survey.md](../../sources/blogs/wechat_embodied_ai_lab_humanoid_loco_manip_161_survey.md)

## 推荐继续阅读

- [Loco-Manipulation 任务页](../tasks/loco-manipulation.md)
- 同题深读/既有实体：[paper-splitadapter-load-aware-loco-manipulation](../entities/paper-splitadapter-load-aware-loco-manipulation.md)
