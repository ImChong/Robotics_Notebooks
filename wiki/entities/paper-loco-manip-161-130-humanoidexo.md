---
type: entity
tags: [paper, loco-manipulation, loco-manip-161-survey, humanoid]
status: complete
updated: 2026-06-26
venue: curated
summary: "HumanoidExo 主要解决数据闭环：用人类视频/动捕轨迹、遥操作/外骨骼数据采集人类操作和机器人状态，再通过扩散策略/流匹配、全身控制器/WBC/MPC、分层技能/专家策略转成可训练、可复用的全身轨迹/动作序列。关键点是把动作生成看成条件生成问题，用扩散或流匹配在多模态动作分布里采样可执行轨迹。"
related:
  - ../overview/humanoid-loco-manip-161-papers-technology-map.md
  - ../overview/loco-manip-161-category-07-data-teleop.md
  - ../tasks/loco-manipulation.md
  - ../entities/paper-loco-manip-161-067-humanoidexo.md
sources:
  - ../../sources/papers/loco_manip_161_survey_130_humanoidexo.md
  - ../../sources/blogs/wechat_embodied_ai_lab_humanoid_loco_manip_161_survey.md
  - ../../sources/papers/humanoid_loco_manip_161_catalog.md
---

# HumanoidExo

**HumanoidExo** 收录于 [具身智能研究室 · 人形 Loco-Manip 161 篇长文](https://mp.weixin.qq.com/s/pACh9EhsISiyPGdiiR0C3A) **第 130/161** 篇，归类为 **07 数据采集与遥操作系统**。

## 一句话定义

HumanoidExo 主要解决数据闭环：用人类视频/动捕轨迹、遥操作/外骨骼数据采集人类操作和机器人状态，再通过扩散策略/流匹配、全身控制器/WBC/MPC、分层技能/专家策略转成可训练、可复用的全身轨迹/动作序列。关键点是把动作生成看成条件生成问题，用扩散或流匹配在多模态动作分布里采样可执行轨迹。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Loco-Manip | Loco-Manipulation | 行走与操作动力学耦合的全身任务 |
| WBC | Whole-Body Control | 协调全身关节满足多任务/约束的控制层 |
| VLA | Vision-Language-Action | 视觉-语言-动作多模态策略 |

## 为什么重要

- HumanoidExo 主要解决数据闭环：用人类视频/动捕轨迹、遥操作/外骨骼数据采集人类操作和机器人状态，再通过扩散策略/流匹配、全身控制器/WBC/MPC、分层技能/专家策略转成可训练、可复用的全身轨迹/动作序列。关键点是把动作生成看成条件生成问题，用扩散或流匹配在多模态动作分布里采样可执行轨迹。
- 人形 Loco-Manip 161 篇 **#130/161** · 数据采集与遥操作系统。

## 核心信息（索引级）

| 字段 | 内容 |
|------|------|
| 编号 | 130/161 |
| 分组 | 07 数据采集与遥操作系统 |
| 原文题目 | HumanoidExo: Scalable Whole-Body Humanoid Manipulation via Wearable Exoskeleton |
| 机构 | National University of Defense Technology、tasks from limited data, and acquire new skills like walking |
| 发表日期 | 2025年10月3日 |
| 论文/项目 | https://humanoid-exo.github.io/ |

## 核心机制（归纳）

### 策展导读要点

HumanoidExo 主要解决数据闭环：用人类视频/动捕轨迹、遥操作/外骨骼数据采集人类操作和机器人状态，再通过扩散策略/流匹配、全身控制器/WBC/MPC、分层技能/专家策略转成可训练、可复用的全身轨迹/动作序列。关键点是把动作生成看成条件生成问题，用扩散或流匹配在多模态动作分布里采样可执行轨迹。

## 常见误区

1. 161 篇策展条目提供 **地图坐标**；量化 benchmark 与实机指标以原文 PDF / 项目页为准。
2. Loco-manip 单篇工作不自动解决 **底层 WBC 鲁棒性**；须与运控/接触控制对照。

## 与其他页面的关系

- 技术地图：[humanoid-loco-manip-161-papers-technology-map.md](../overview/humanoid-loco-manip-161-papers-technology-map.md)
- 分类 hub：[loco-manip-161-category-07-data-teleop.md](../overview/loco-manip-161-category-07-data-teleop.md)
- 原始 source：[loco_manip_161_survey_130_humanoidexo.md](../../sources/papers/loco_manip_161_survey_130_humanoidexo.md)

## 参考来源

- [loco_manip_161_survey_130_humanoidexo.md](../../sources/papers/loco_manip_161_survey_130_humanoidexo.md) — 161 篇策展摘录
- [humanoid_loco_manip_161_catalog.md](../../sources/papers/humanoid_loco_manip_161_catalog.md)
- [wechat_embodied_ai_lab_humanoid_loco_manip_161_survey.md](../../sources/blogs/wechat_embodied_ai_lab_humanoid_loco_manip_161_survey.md)

## 推荐继续阅读

- [Loco-Manipulation 任务页](../tasks/loco-manipulation.md)
- 同题深读/既有实体：[paper-loco-manip-161-067-humanoidexo](../entities/paper-loco-manip-161-067-humanoidexo.md)
