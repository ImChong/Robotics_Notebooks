---
type: entity
tags: [paper, loco-manipulation, loco-manip-161-survey, humanoid]
status: complete
updated: 2026-06-29
venue: curated
summary: "Hold 先从相机图像/多视角观测恢复场景、目标或运动表征，再用ACT/行为克隆模仿学习生成全身轨迹/动作序列、末端执行器/腕手目标、低层控制器目标。关键点是把示范轨迹压成可监督的动作预测问题，再通过动作 chunk 或闭环执行降低时序抖动。"
related:
  - ../overview/humanoid-loco-manip-161-papers-technology-map.md
  - ../overview/loco-manip-161-category-02-upper-body-interface.md
  - ../tasks/loco-manipulation.md
sources:
  - ../../sources/papers/loco_manip_161_survey_042_hold.md
  - ../../sources/blogs/wechat_embodied_ai_lab_humanoid_loco_manip_161_survey.md
  - ../../sources/papers/humanoid_loco_manip_161_catalog.md
---

# Hold

**Hold** 收录于 [具身智能研究室 · 人形 Loco-Manip 161 篇长文](https://mp.weixin.qq.com/s/pACh9EhsISiyPGdiiR0C3A) **第 042/161** 篇，归类为 **02 上半身中心控制与移动操作接口**。

## 一句话定义

Hold 先从相机图像/多视角观测恢复场景、目标或运动表征，再用ACT/行为克隆模仿学习生成全身轨迹/动作序列、末端执行器/腕手目标、低层控制器目标。关键点是把示范轨迹压成可监督的动作预测问题，再通过动作 chunk 或闭环执行降低时序抖动。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Loco-Manip | Loco-Manipulation | 行走与操作动力学耦合的全身任务 |
| WBC | Whole-Body Control | 协调全身关节满足多任务/约束的控制层 |
| VLA | Vision-Language-Action | 视觉-语言-动作多模态策略 |

## 为什么重要

- Hold 先从相机图像/多视角观测恢复场景、目标或运动表征，再用ACT/行为克隆模仿学习生成全身轨迹/动作序列、末端执行器/腕手目标、低层控制器目标。关键点是把示范轨迹压成可监督的动作预测问题，再通过动作 chunk 或闭环执行降低时序抖动。
- 人形 Loco-Manip 161 篇 **#042/161** · 上半身中心控制与移动操作接口。

## 核心信息（索引级）

| 字段 | 内容 |
|------|------|
| 编号 | 042/161 |
| 分组 | 02 上半身中心控制与移动操作接口 |
| 原文题目 | Hold My Beer: Learning Gentle Humanoid Locomotion and End-Effector Stabilization Control |
| 机构 | Carnegie Mellon University |
| 发表日期 | 2025年6月3日 |
| 论文/项目 | https://lecar-lab.github.io/SoFTA/ |

## 核心机制（归纳）

### 策展导读要点

Hold 先从相机图像/多视角观测恢复场景、目标或运动表征，再用ACT/行为克隆模仿学习生成全身轨迹/动作序列、末端执行器/腕手目标、低层控制器目标。关键点是把示范轨迹压成可监督的动作预测问题，再通过动作 chunk 或闭环执行降低时序抖动。

## 评测与指标（索引级）

- 本条目为 161 篇策展索引级摘录，**未搬运原文量化 benchmark 与实机指标**；评测口径与具体数值以原文 PDF / 项目页为准。
- 评测原始出处：[原文 / 项目页](https://lecar-lab.github.io/SoFTA/)（见上方「核心信息」表「论文/项目」一行）。
- 横向评测对照请回到 [分类 hub](../overview/loco-manip-161-category-02-upper-body-interface.md) 与 [技术地图](../overview/humanoid-loco-manip-161-papers-technology-map.md)。

## 常见误区

1. 161 篇策展条目提供 **地图坐标**；量化 benchmark 与实机指标以原文 PDF / 项目页为准。
2. Loco-manip 单篇工作不自动解决 **底层 WBC 鲁棒性**；须与运控/接触控制对照。

## 与其他页面的关系

- 技术地图：[humanoid-loco-manip-161-papers-technology-map.md](../overview/humanoid-loco-manip-161-papers-technology-map.md)
- 分类 hub：[loco-manip-161-category-02-upper-body-interface.md](../overview/loco-manip-161-category-02-upper-body-interface.md)
- 原始 source：[loco_manip_161_survey_042_hold.md](../../sources/papers/loco_manip_161_survey_042_hold.md)

## 参考来源

- [loco_manip_161_survey_042_hold.md](../../sources/papers/loco_manip_161_survey_042_hold.md) — 161 篇策展摘录
- [humanoid_loco_manip_161_catalog.md](../../sources/papers/humanoid_loco_manip_161_catalog.md)
- [wechat_embodied_ai_lab_humanoid_loco_manip_161_survey.md](../../sources/blogs/wechat_embodied_ai_lab_humanoid_loco_manip_161_survey.md)

## 推荐继续阅读

- [Loco-Manipulation 任务页](../tasks/loco-manipulation.md)
