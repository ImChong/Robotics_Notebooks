---
type: entity
tags: [paper, loco-manipulation, loco-manip-161-survey, humanoid]
status: complete
updated: 2026-06-26
venue: curated
summary: "AgiBot World Colosseo 的实现路径是先把语言指令、相机图像/多视角观测、本体状态与关节序列编码成多模态表征，再用VLM 语义规划/路由、潜变量/动作 token、MM-DiT/Transformer 动作头预测全身轨迹/动作序列、末端执行器/腕手目标、动作 chunk/token。关键点是把任务拆成可路由的技能或专家策略，再用高层模块在执行中选择和组合。"
related:
  - ../overview/humanoid-loco-manip-161-papers-technology-map.md
  - ../overview/loco-manip-161-category-09-vla-world-models.md
  - ../tasks/loco-manipulation.md
  - ../entities/agibot-world-2026.md
sources:
  - ../../sources/papers/loco_manip_161_survey_146_agibot-world-colosseo.md
  - ../../sources/blogs/wechat_embodied_ai_lab_humanoid_loco_manip_161_survey.md
  - ../../sources/papers/humanoid_loco_manip_161_catalog.md
---

# AgiBot World Colosseo

**AgiBot World Colosseo** 收录于 [具身智能研究室 · 人形 Loco-Manip 161 篇长文](https://mp.weixin.qq.com/s/pACh9EhsISiyPGdiiR0C3A) **第 146/161** 篇，归类为 **09 人形 VLA、世界模型与通用操作**。

## 一句话定义

AgiBot World Colosseo 的实现路径是先把语言指令、相机图像/多视角观测、本体状态与关节序列编码成多模态表征，再用VLM 语义规划/路由、潜变量/动作 token、MM-DiT/Transformer 动作头预测全身轨迹/动作序列、末端执行器/腕手目标、动作 chunk/token。关键点是把任务拆成可路由的技能或专家策略，再用高层模块在执行中选择和组合。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Loco-Manip | Loco-Manipulation | 行走与操作动力学耦合的全身任务 |
| WBC | Whole-Body Control | 协调全身关节满足多任务/约束的控制层 |
| VLA | Vision-Language-Action | 视觉-语言-动作多模态策略 |

## 为什么重要

- AgiBot World Colosseo 的实现路径是先把语言指令、相机图像/多视角观测、本体状态与关节序列编码成多模态表征，再用VLM 语义规划/路由、潜变量/动作 token、MM-DiT/Transformer 动作头预测全身轨迹/动作序列、末端执行器/腕手目标、动作 chunk/token。关键点是把任务拆成可路由的技能或专家策略，再用高层模块在执行中选择和组合。
- 人形 Loco-Manip 161 篇 **#146/161** · 人形 VLA、世界模型与通用操作。

## 核心信息（索引级）

| 字段 | 内容 |
|------|------|
| 编号 | 146/161 |
| 分组 | 09 人形 VLA、世界模型与通用操作 |
| 原文题目 | AgiBot World Colosseo: A Large-scale Manipulation Platform for Scalable and Intelligent Embodied Systems |
| 机构 | （见原文） |
| 发表日期 | 2025年8月4日 |
| 论文/项目 | https://github.com/OpenDriveLab/AgiBot-World |

## 核心机制（归纳）

### 策展导读要点

AgiBot World Colosseo 的实现路径是先把语言指令、相机图像/多视角观测、本体状态与关节序列编码成多模态表征，再用VLM 语义规划/路由、潜变量/动作 token、MM-DiT/Transformer 动作头预测全身轨迹/动作序列、末端执行器/腕手目标、动作 chunk/token。关键点是把任务拆成可路由的技能或专家策略，再用高层模块在执行中选择和组合。

## 评测与指标（索引级）

- 本条目为 161 篇策展索引级摘录，**未搬运原文量化 benchmark 与实机指标**；评测口径与具体数值以原文 PDF / 项目页为准。
- 评测原始出处：[原文 / 项目页](https://github.com/OpenDriveLab/AgiBot-World)（见上方「核心信息」表「论文/项目」一行）。
- 横向评测对照请回到 [分类 hub](../overview/loco-manip-161-category-09-vla-world-models.md) 与 [技术地图](../overview/humanoid-loco-manip-161-papers-technology-map.md)。

## 常见误区

1. 161 篇策展条目提供 **地图坐标**；量化 benchmark 与实机指标以原文 PDF / 项目页为准。
2. Loco-manip 单篇工作不自动解决 **底层 WBC 鲁棒性**；须与运控/接触控制对照。

## 与其他页面的关系

- 技术地图：[humanoid-loco-manip-161-papers-technology-map.md](../overview/humanoid-loco-manip-161-papers-technology-map.md)
- 分类 hub：[loco-manip-161-category-09-vla-world-models.md](../overview/loco-manip-161-category-09-vla-world-models.md)
- 原始 source：[loco_manip_161_survey_146_agibot-world-colosseo.md](../../sources/papers/loco_manip_161_survey_146_agibot-world-colosseo.md)

## 参考来源

- [loco_manip_161_survey_146_agibot-world-colosseo.md](../../sources/papers/loco_manip_161_survey_146_agibot-world-colosseo.md) — 161 篇策展摘录
- [humanoid_loco_manip_161_catalog.md](../../sources/papers/humanoid_loco_manip_161_catalog.md)
- [wechat_embodied_ai_lab_humanoid_loco_manip_161_survey.md](../../sources/blogs/wechat_embodied_ai_lab_humanoid_loco_manip_161_survey.md)

## 推荐继续阅读

- [Loco-Manipulation 任务页](../tasks/loco-manipulation.md)
- 同题深读/既有实体：[agibot-world-2026](../entities/agibot-world-2026.md)
