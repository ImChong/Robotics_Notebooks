---
type: entity
tags: [paper, loco-manipulation, loco-manip-161-survey, humanoid]
status: complete
updated: 2026-08-09
venue: curated
summary: "HumanPlus 主要解决数据闭环：用相机图像/多视角观测、本体状态与关节序列、人类视频/动捕轨迹采集人类操作和机器人状态，再通过PPO/RL 策略训练、ACT/行为克隆模仿学习、分层技能/专家策略转成可训练、可复用的全身轨迹/动作序列、低层控制器目标。关键点是把任务拆成可路由的技能或专家策略，再用高层模块在执行中选择和组合。"
related:
  - ../overview/humanoid-loco-manip-161-papers-technology-map.md
  - ../overview/loco-manip-161-category-01-motion-base-wbt.md
  - ../tasks/loco-manipulation.md
sources:
  - ../../sources/papers/loco_manip_161_survey_012_humanplus.md
  - ../../sources/blogs/wechat_embodied_ai_lab_humanoid_loco_manip_161_survey.md
  - ../../sources/papers/humanoid_loco_manip_161_catalog.md
---

# HumanPlus

**HumanPlus** 收录于 [具身智能研究室 · 人形 Loco-Manip 161 篇长文](https://mp.weixin.qq.com/s/pACh9EhsISiyPGdiiR0C3A) **第 012/161** 篇，归类为 **01 运控基座与通用全身跟踪**。

## 一句话定义

HumanPlus 主要解决数据闭环：用相机图像/多视角观测、本体状态与关节序列、人类视频/动捕轨迹采集人类操作和机器人状态，再通过PPO/RL 策略训练、ACT/行为克隆模仿学习、分层技能/专家策略转成可训练、可复用的全身轨迹/动作序列、低层控制器目标。关键点是把任务拆成可路由的技能或专家策略，再用高层模块在执行中选择和组合。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Loco-Manip | Loco-Manipulation | 行走与操作动力学耦合的全身任务 |
| WBC | Whole-Body Control | 协调全身关节满足多任务/约束的控制层 |
| VLA | Vision-Language-Action | 视觉-语言-动作多模态策略 |

## 为什么重要

- HumanPlus 主要解决数据闭环：用相机图像/多视角观测、本体状态与关节序列、人类视频/动捕轨迹采集人类操作和机器人状态，再通过PPO/RL 策略训练、ACT/行为克隆模仿学习、分层技能/专家策略转成可训练、可复用的全身轨迹/动作序列、低层控制器目标。关键点是把任务拆成可路由的技能或专家策略，再用高层模块在执行中选择和组合。
- 人形 Loco-Manip 161 篇 **#012/161** · 运控基座与通用全身跟踪。

## 核心信息（索引级）

| 字段 | 内容 |
|------|------|
| 编号 | 012/161 |
| 分组 | 01 运控基座与通用全身跟踪 |
| 原文题目 | HumanPlus: Humanoid Shadowing and Imitation from Humans |
| 机构 | HumanPlus: Humanoid Shadowing and Imitation、project co-leads、Stanford University |
| 发表日期 | 2024年6月15日 |
| 论文/项目 | https://humanoid-ai.github.io |

## 核心机制（归纳）

### 策展导读要点

HumanPlus 主要解决数据闭环：用相机图像/多视角观测、本体状态与关节序列、人类视频/动捕轨迹采集人类操作和机器人状态，再通过PPO/RL 策略训练、ACT/行为克隆模仿学习、分层技能/专家策略转成可训练、可复用的全身轨迹/动作序列、低层控制器目标。关键点是把任务拆成可路由的技能或专家策略，再用高层模块在执行中选择和组合。

## 评测与指标（索引级）

- 本条目为 161 篇策展索引级摘录，**未搬运原文量化 benchmark 与实机指标**；评测口径与具体数值以原文 PDF / 项目页为准。
- 评测原始出处：[原文 / 项目页](https://humanoid-ai.github.io)（见上方「核心信息」表「论文/项目」一行）。
- 横向评测对照请回到 [分类 hub](../overview/loco-manip-161-category-01-motion-base-wbt.md) 与 [技术地图](../overview/humanoid-loco-manip-161-papers-technology-map.md)。

## 结论

**HumanPlus 的落点是数据闭环而不是单一算法：把人类视频/动捕与机器人本体状态接到同一条 shadowing → imitation 管线上，用 RL 撑运控基座、用行为克隆把演示转成可复用技能。**

- 真正起作用的是 **两段分工**：PPO/RL 训练全身跟踪与低层控制器目标，ACT/行为克隆把人类操作转成可训练、可复用的动作序列。
- 结构上把任务拆成 **可路由的技能或专家策略**，由高层模块在执行中选择与组合，因此能力扩展靠加专家而非重训整体。
- 分类坐标是 **01 运控基座与通用全身跟踪**（Stanford，2024-06），读它是为了拿基座层的组织方式，而非任务级方案。
- 本页为 161 篇索引级条目，未搬运原文量化 benchmark 与实机指标；数值以 [原文 / 项目页](https://humanoid-ai.github.io) 为准。

## 常见误区

1. 161 篇策展条目提供 **地图坐标**；量化 benchmark 与实机指标以原文 PDF / 项目页为准。
2. Loco-manip 单篇工作不自动解决 **底层 WBC 鲁棒性**；须与运控/接触控制对照。

## 与其他页面的关系

- 技术地图：[humanoid-loco-manip-161-papers-technology-map.md](../overview/humanoid-loco-manip-161-papers-technology-map.md)
- 分类 hub：[loco-manip-161-category-01-motion-base-wbt.md](../overview/loco-manip-161-category-01-motion-base-wbt.md)
- 低负担语音腿控对照（非全身 shadowing）：[Immersive Social VR+LLM](./paper-immersive-social-vr-llm-humanoids.md)
- 原始 source：[loco_manip_161_survey_012_humanplus.md](../../sources/papers/loco_manip_161_survey_012_humanplus.md)

## 参考来源

- [loco_manip_161_survey_012_humanplus.md](../../sources/papers/loco_manip_161_survey_012_humanplus.md) — 161 篇策展摘录
- [humanoid_loco_manip_161_catalog.md](../../sources/papers/humanoid_loco_manip_161_catalog.md)
- [wechat_embodied_ai_lab_humanoid_loco_manip_161_survey.md](../../sources/blogs/wechat_embodied_ai_lab_humanoid_loco_manip_161_survey.md)

## 推荐继续阅读

- [Loco-Manipulation 任务页](../tasks/loco-manipulation.md)
