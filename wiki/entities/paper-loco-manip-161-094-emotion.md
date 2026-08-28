---
type: entity
tags: [paper, loco-manipulation, loco-manip-161-survey, humanoid]
status: complete
updated: 2026-08-28
venue: curated
summary: "EMOTION 先从视觉、状态和动作数据恢复场景、目标或运动表征，再用策略网络和控制模块生成全身轨迹/动作序列。关键点是把策略网络和控制模块放在同一条训练/部署链路里，减少高层目标到低层动作之间的断点。"
related:
  - ../overview/humanoid-loco-manip-161-papers-technology-map.md
  - ../overview/loco-manip-161-category-04-generative-language-trajectory.md
  - ../tasks/loco-manipulation.md
sources:
  - ../../sources/papers/loco_manip_161_survey_094_emotion.md
  - ../../sources/blogs/wechat_embodied_ai_lab_humanoid_loco_manip_161_survey.md
  - ../../sources/papers/humanoid_loco_manip_161_catalog.md
---

# EMOTION

**EMOTION** 收录于 [具身智能研究室 · 人形 Loco-Manip 161 篇长文](https://mp.weixin.qq.com/s/pACh9EhsISiyPGdiiR0C3A) **第 094/161** 篇，归类为 **04 生成式运动、语言控制与轨迹规划**。

## 一句话定义

EMOTION 先从视觉、状态和动作数据恢复场景、目标或运动表征，再用策略网络和控制模块生成全身轨迹/动作序列。关键点是把策略网络和控制模块放在同一条训练/部署链路里，减少高层目标到低层动作之间的断点。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Loco-Manip | Loco-Manipulation | 行走与操作动力学耦合的全身任务 |
| WBC | Whole-Body Control | 协调全身关节满足多任务/约束的控制层 |
| VLA | Vision-Language-Action | 视觉-语言-动作多模态策略 |

## 为什么重要

- EMOTION 先从视觉、状态和动作数据恢复场景、目标或运动表征，再用策略网络和控制模块生成全身轨迹/动作序列。关键点是把策略网络和控制模块放在同一条训练/部署链路里，减少高层目标到低层动作之间的断点。
- 人形 Loco-Manip 161 篇 **#094/161** · 生成式运动、语言控制与轨迹规划。

## 核心信息（索引级）

| 字段 | 内容 |
|------|------|
| 编号 | 094/161 |
| 分组 | 04 生成式运动、语言控制与轨迹规划 |
| 原文题目 | EMOTION: Expressive Motion Sequence Generation for Humanoid Robots with In-Context Learning |
| 机构 | （见原文） |
| 发表日期 | 2024年10月30日 |
| 论文/项目 | （见原文） |

## 核心机制（归纳）

### 策展导读要点

EMOTION 先从视觉、状态和动作数据恢复场景、目标或运动表征，再用策略网络和控制模块生成全身轨迹/动作序列。关键点是把策略网络和控制模块放在同一条训练/部署链路里，减少高层目标到低层动作之间的断点。

## 评测与指标（索引级）

- 本条目为 161 篇策展索引级摘录，**未搬运原文量化 benchmark 与实机指标**；评测口径与具体数值以原文 PDF / 项目页为准。
- 评测原始出处：[原文 / 项目页](（见原文）)（见上方「核心信息」表「论文/项目」一行）。
- 横向评测对照请回到 [分类 hub](../overview/loco-manip-161-category-04-generative-language-trajectory.md) 与 [技术地图](../overview/humanoid-loco-manip-161-papers-technology-map.md)。

## 结论

**EMOTION 的定位是「表达性动作序列生成」：用 in-context learning 产出人形可执行的动作序列，并把生成与控制放在同一条链路上，而不是追求 loco-manip 的负载或接触性能。**

- 真正的机制点是链路不断裂：策略网络与控制模块同处一条训练/部署链，减少高层目标到低层动作之间的断点。
- 归入 04 类（[生成式运动、语言控制与轨迹规划](../overview/loco-manip-161-category-04-generative-language-trajectory.md)），评价重心在动作的可表达性与可执行性。
- 边界最需注意：本页为索引级摘录，机构与项目页字段均为「见原文」，量化评测缺失，任何引用都必须回到原文 PDF。

## 常见误区

1. 161 篇策展条目提供 **地图坐标**；量化 benchmark 与实机指标以原文 PDF / 项目页为准。
2. Loco-manip 单篇工作不自动解决 **底层 WBC 鲁棒性**；须与运控/接触控制对照。

## 与其他页面的关系

- 技术地图：[humanoid-loco-manip-161-papers-technology-map.md](../overview/humanoid-loco-manip-161-papers-technology-map.md)
- 分类 hub：[loco-manip-161-category-04-generative-language-trajectory.md](../overview/loco-manip-161-category-04-generative-language-trajectory.md)
- 原始 source：[loco_manip_161_survey_094_emotion.md](../../sources/papers/loco_manip_161_survey_094_emotion.md)

## 参考来源

- [loco_manip_161_survey_094_emotion.md](../../sources/papers/loco_manip_161_survey_094_emotion.md) — 161 篇策展摘录
- [humanoid_loco_manip_161_catalog.md](../../sources/papers/humanoid_loco_manip_161_catalog.md)
- [wechat_embodied_ai_lab_humanoid_loco_manip_161_survey.md](../../sources/blogs/wechat_embodied_ai_lab_humanoid_loco_manip_161_survey.md)

## 推荐继续阅读

- [Loco-Manipulation 任务页](../tasks/loco-manipulation.md)
- 同题深读/既有实体：[paper-loco-manip-161-094-emotion](../entities/paper-loco-manip-161-094-emotion.md)
