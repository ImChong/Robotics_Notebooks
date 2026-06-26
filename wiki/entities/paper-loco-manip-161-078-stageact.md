---
type: entity
tags: [paper, loco-manipulation, loco-manip-161-survey, humanoid]
status: complete
updated: 2026-06-26
venue: curated
summary: "StageACT 主要解决数据闭环：用相机图像/多视角观测、本体状态与关节序列、遥操作/外骨骼数据采集人类操作和机器人状态，再通过ACT/行为克隆模仿学习、扩散策略/流匹配、VLM 语义规划/路由转成可训练、可复用的可执行动作命令。关键点是把动作生成看成条件生成问题，用扩散或流匹配在多模态动作分布里采样可执行轨迹。"
related:
  - ../overview/humanoid-loco-manip-161-papers-technology-map.md
  - ../overview/loco-manip-161-category-03-visuomotor.md
  - ../tasks/loco-manipulation.md
  - ../entities/paper-notebook-stageact.md
sources:
  - ../../sources/papers/loco_manip_161_survey_078_stageact.md
  - ../../sources/blogs/wechat_embodied_ai_lab_humanoid_loco_manip_161_survey.md
  - ../../sources/papers/humanoid_loco_manip_161_catalog.md
---

# StageACT

**StageACT** 收录于 [具身智能研究室 · 人形 Loco-Manip 161 篇长文](https://mp.weixin.qq.com/s/pACh9EhsISiyPGdiiR0C3A) **第 078/161** 篇，归类为 **03 视觉感知驱动的人形移动操作**。

## 一句话定义

StageACT 主要解决数据闭环：用相机图像/多视角观测、本体状态与关节序列、遥操作/外骨骼数据采集人类操作和机器人状态，再通过ACT/行为克隆模仿学习、扩散策略/流匹配、VLM 语义规划/路由转成可训练、可复用的可执行动作命令。关键点是把动作生成看成条件生成问题，用扩散或流匹配在多模态动作分布里采样可执行轨迹。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Loco-Manip | Loco-Manipulation | 行走与操作动力学耦合的全身任务 |
| WBC | Whole-Body Control | 协调全身关节满足多任务/约束的控制层 |
| VLA | Vision-Language-Action | 视觉-语言-动作多模态策略 |

## 为什么重要

- StageACT 主要解决数据闭环：用相机图像/多视角观测、本体状态与关节序列、遥操作/外骨骼数据采集人类操作和机器人状态，再通过ACT/行为克隆模仿学习、扩散策略/流匹配、VLM 语义规划/路由转成可训练、可复用的可执行动作命令。关键点是把动作生成看成条件生成问题，用扩散或流匹配在多模态动作分布里采样可执行轨迹。
- 人形 Loco-Manip 161 篇 **#078/161** · 视觉感知驱动的人形移动操作。

## 核心信息（索引级）

| 字段 | 内容 |
|------|------|
| 编号 | 078/161 |
| 分组 | 03 视觉感知驱动的人形移动操作 |
| 原文题目 | StageACT: Stage-Conditioned Imitation for Robust Humanoid Door Opening |
| 机构 | StageACT: Stage-Conditioned Imitation for Robust |
| 发表日期 | 2025年9月18日 |
| 论文/项目 | https://icradooropen.github.io/icradooropen/ |

## 核心机制（归纳）

### 策展导读要点

StageACT 主要解决数据闭环：用相机图像/多视角观测、本体状态与关节序列、遥操作/外骨骼数据采集人类操作和机器人状态，再通过ACT/行为克隆模仿学习、扩散策略/流匹配、VLM 语义规划/路由转成可训练、可复用的可执行动作命令。关键点是把动作生成看成条件生成问题，用扩散或流匹配在多模态动作分布里采样可执行轨迹。

## 常见误区

1. 161 篇策展条目提供 **地图坐标**；量化 benchmark 与实机指标以原文 PDF / 项目页为准。
2. Loco-manip 单篇工作不自动解决 **底层 WBC 鲁棒性**；须与运控/接触控制对照。

## 与其他页面的关系

- 技术地图：[humanoid-loco-manip-161-papers-technology-map.md](../overview/humanoid-loco-manip-161-papers-technology-map.md)
- 分类 hub：[loco-manip-161-category-03-visuomotor.md](../overview/loco-manip-161-category-03-visuomotor.md)
- 原始 source：[loco_manip_161_survey_078_stageact.md](../../sources/papers/loco_manip_161_survey_078_stageact.md)

## 参考来源

- [loco_manip_161_survey_078_stageact.md](../../sources/papers/loco_manip_161_survey_078_stageact.md) — 161 篇策展摘录
- [humanoid_loco_manip_161_catalog.md](../../sources/papers/humanoid_loco_manip_161_catalog.md)
- [wechat_embodied_ai_lab_humanoid_loco_manip_161_survey.md](../../sources/blogs/wechat_embodied_ai_lab_humanoid_loco_manip_161_survey.md)

## 推荐继续阅读

- [Loco-Manipulation 任务页](../tasks/loco-manipulation.md)
- 同题深读/既有实体：[paper-notebook-stageact](../entities/paper-notebook-stageact.md)
