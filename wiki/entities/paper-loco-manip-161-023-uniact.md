---
type: entity
tags: [paper, loco-manipulation, loco-manip-161-survey, humanoid]
status: complete
updated: 2026-06-30
venue: curated
summary: "UniAct 把语言指令转成可跟踪的身体目标，并通过策略网络和控制模块训练或组合全身策略，最终输出全身轨迹/动作序列、低层控制器目标。关键点是把策略网络和控制模块放在同一条训练/部署链路里，减少高层目标到低层动作之间的断点。"
related:
  - ../overview/humanoid-loco-manip-161-papers-technology-map.md
  - ../overview/loco-manip-161-category-01-motion-base-wbt.md
  - ../tasks/loco-manipulation.md
sources:
  - ../../sources/papers/loco_manip_161_survey_023_uniact.md
  - ../../sources/blogs/wechat_embodied_ai_lab_humanoid_loco_manip_161_survey.md
  - ../../sources/papers/humanoid_loco_manip_161_catalog.md
---

# UniAct

**UniAct** 收录于 [具身智能研究室 · 人形 Loco-Manip 161 篇长文](https://mp.weixin.qq.com/s/pACh9EhsISiyPGdiiR0C3A) **第 023/161** 篇，归类为 **01 运控基座与通用全身跟踪**。

## 一句话定义

UniAct 把语言指令转成可跟踪的身体目标，并通过策略网络和控制模块训练或组合全身策略，最终输出全身轨迹/动作序列、低层控制器目标。关键点是把策略网络和控制模块放在同一条训练/部署链路里，减少高层目标到低层动作之间的断点。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Loco-Manip | Loco-Manipulation | 行走与操作动力学耦合的全身任务 |
| WBC | Whole-Body Control | 协调全身关节满足多任务/约束的控制层 |
| VLA | Vision-Language-Action | 视觉-语言-动作多模态策略 |

## 为什么重要

- UniAct 把语言指令转成可跟踪的身体目标，并通过策略网络和控制模块训练或组合全身策略，最终输出全身轨迹/动作序列、低层控制器目标。关键点是把策略网络和控制模块放在同一条训练/部署链路里，减少高层目标到低层动作之间的断点。
- 人形 Loco-Manip 161 篇 **#023/161** · 运控基座与通用全身跟踪。

## 核心信息（索引级）

| 字段 | 内容 |
|------|------|
| 编号 | 023/161 |
| 分组 | 01 运控基座与通用全身跟踪 |
| 原文题目 | UniAct: Unified Motion Generation and Action Streaming for Humanoid Robots |
| 机构 | Institute for AI, Peking University、Beijing Institute for General Artificial Intelligence (BIGAI)、School of Psychological and Cognitive Sciences, Peking University、School of Computer Science, Peking University |
| 发表日期 | 2025年12月30日 |
| 论文/项目 | https://jnnan.github.io/uniact/ |

## 核心机制（归纳）

### 策展导读要点

UniAct 把语言指令转成可跟踪的身体目标，并通过策略网络和控制模块训练或组合全身策略，最终输出全身轨迹/动作序列、低层控制器目标。关键点是把策略网络和控制模块放在同一条训练/部署链路里，减少高层目标到低层动作之间的断点。

## 评测与指标（索引级）

- 本条目为 161 篇策展索引级摘录，**未搬运原文量化 benchmark 与实机指标**；评测口径与具体数值以原文 PDF / 项目页为准。
- 评测原始出处：[原文 / 项目页](https://jnnan.github.io/uniact/)（见上方「核心信息」表「论文/项目」一行）。
- 横向评测对照请回到 [分类 hub](../overview/loco-manip-161-category-01-motion-base-wbt.md) 与 [技术地图](../overview/humanoid-loco-manip-161-papers-technology-map.md)。

## 常见误区

1. 161 篇策展条目提供 **地图坐标**；量化 benchmark 与实机指标以原文 PDF / 项目页为准。
2. Loco-manip 单篇工作不自动解决 **底层 WBC 鲁棒性**；须与运控/接触控制对照。

## 与其他页面的关系

- 技术地图：[humanoid-loco-manip-161-papers-technology-map.md](../overview/humanoid-loco-manip-161-papers-technology-map.md)
- 分类 hub：[loco-manip-161-category-01-motion-base-wbt.md](../overview/loco-manip-161-category-01-motion-base-wbt.md)
- 原始 source：[loco_manip_161_survey_023_uniact.md](../../sources/papers/loco_manip_161_survey_023_uniact.md)

## 参考来源

- [loco_manip_161_survey_023_uniact.md](../../sources/papers/loco_manip_161_survey_023_uniact.md) — 161 篇策展摘录
- [humanoid_loco_manip_161_catalog.md](../../sources/papers/humanoid_loco_manip_161_catalog.md)
- [wechat_embodied_ai_lab_humanoid_loco_manip_161_survey.md](../../sources/blogs/wechat_embodied_ai_lab_humanoid_loco_manip_161_survey.md)

## 推荐继续阅读

- [Loco-Manipulation 任务页](../tasks/loco-manipulation.md)
- 同题深读/既有实体：[paper-loco-manip-161-023-uniact](../entities/paper-loco-manip-161-023-uniact.md)
