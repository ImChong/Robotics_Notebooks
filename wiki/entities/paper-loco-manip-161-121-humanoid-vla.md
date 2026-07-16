---
type: entity
tags: [paper, loco-manipulation, loco-manip-161-survey, humanoid]
status: complete
updated: 2026-07-16
venue: curated
summary: "Humanoid-VLA 的实现路径是先把相机图像/多视角观测、人类视频/动捕轨迹编码成多模态表征，再用扩散策略/流匹配、VLA 多模态动作模型、全身控制器/WBC/MPC预测全身轨迹/动作序列。关键点是保留 VLM 的语义理解，同时增加机器人状态和动作头，避免只停留在语言规划。"
related:
  - ../overview/humanoid-loco-manip-161-papers-technology-map.md
  - ../overview/loco-manip-161-category-06-contact-tasks.md
  - ../tasks/loco-manipulation.md
sources:
  - ../../sources/papers/loco_manip_161_survey_121_humanoid-vla.md
  - ../../sources/blogs/wechat_embodied_ai_lab_humanoid_loco_manip_161_survey.md
  - ../../sources/papers/humanoid_loco_manip_161_catalog.md
---

# Humanoid-VLA

**Humanoid-VLA** 收录于 [具身智能研究室 · 人形 Loco-Manip 161 篇长文](https://mp.weixin.qq.com/s/pACh9EhsISiyPGdiiR0C3A) **第 121/161** 篇，归类为 **06 特殊任务、接触规划与视觉闭环**。

## 一句话定义

Humanoid-VLA 的实现路径是先把相机图像/多视角观测、人类视频/动捕轨迹编码成多模态表征，再用扩散策略/流匹配、VLA 多模态动作模型、全身控制器/WBC/MPC预测全身轨迹/动作序列。关键点是保留 VLM 的语义理解，同时增加机器人状态和动作头，避免只停留在语言规划。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Loco-Manip | Loco-Manipulation | 行走与操作动力学耦合的全身任务 |
| WBC | Whole-Body Control | 协调全身关节满足多任务/约束的控制层 |
| VLA | Vision-Language-Action | 视觉-语言-动作多模态策略 |

## 为什么重要

- Humanoid-VLA 的实现路径是先把相机图像/多视角观测、人类视频/动捕轨迹编码成多模态表征，再用扩散策略/流匹配、VLA 多模态动作模型、全身控制器/WBC/MPC预测全身轨迹/动作序列。关键点是保留 VLM 的语义理解，同时增加机器人状态和动作头，避免只停留在语言规划。
- 人形 Loco-Manip 161 篇 **#121/161** · 特殊任务、接触规划与视觉闭环。

## 核心信息（索引级）

| 字段 | 内容 |
|------|------|
| 编号 | 121/161 |
| 分组 | 06 特殊任务、接触规划与视觉闭环 |
| 原文题目 | Humanoid-VLA: Towards Universal Humanoid Control with Visual Integration |
| 机构 | （见原文） |
| 发表日期 | 2025年2月21日 |
| 论文/项目 | （见原文） |

## 核心机制（归纳）

### 策展导读要点

Humanoid-VLA 的实现路径是先把相机图像/多视角观测、人类视频/动捕轨迹编码成多模态表征，再用扩散策略/流匹配、VLA 多模态动作模型、全身控制器/WBC/MPC预测全身轨迹/动作序列。关键点是保留 VLM 的语义理解，同时增加机器人状态和动作头，避免只停留在语言规划。

## 评测与指标（索引级）

- 本条目为 161 篇策展索引级摘录，**未搬运原文量化 benchmark 与实机指标**；评测口径与具体数值以原文 PDF / 项目页为准。
- 评测原始出处：[原文 / 项目页](（见原文）)（见上方「核心信息」表「论文/项目」一行）。
- 横向评测对照请回到 [分类 hub](../overview/loco-manip-161-category-06-contact-tasks.md) 与 [技术地图](../overview/humanoid-loco-manip-161-papers-technology-map.md)。

## 常见误区

1. 161 篇策展条目提供 **地图坐标**；量化 benchmark 与实机指标以原文 PDF / 项目页为准。
2. Loco-manip 单篇工作不自动解决 **底层 WBC 鲁棒性**；须与运控/接触控制对照。

## 与其他页面的关系

- 技术地图：[humanoid-loco-manip-161-papers-technology-map.md](../overview/humanoid-loco-manip-161-papers-technology-map.md)
- 分类 hub：[loco-manip-161-category-06-contact-tasks.md](../overview/loco-manip-161-category-06-contact-tasks.md)
- 原始 source：[loco_manip_161_survey_121_humanoid-vla.md](../../sources/papers/loco_manip_161_survey_121_humanoid-vla.md)

## 参考来源

- [loco_manip_161_survey_121_humanoid-vla.md](../../sources/papers/loco_manip_161_survey_121_humanoid-vla.md) — 161 篇策展摘录
- [humanoid_loco_manip_161_catalog.md](../../sources/papers/humanoid_loco_manip_161_catalog.md)
- [wechat_embodied_ai_lab_humanoid_loco_manip_161_survey.md](../../sources/blogs/wechat_embodied_ai_lab_humanoid_loco_manip_161_survey.md)

## 推荐继续阅读

- [Loco-Manipulation 任务页](../tasks/loco-manipulation.md)
