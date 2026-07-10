---
type: entity
tags: [paper, loco-manipulation, loco-manip-161-survey, humanoid]
status: complete
updated: 2026-07-08
venue: curated
summary: "EgoVLA 的实现路径是先把语言指令、相机图像/多视角观测、本体状态与关节序列编码成多模态表征，再用ACT/行为克隆模仿学习、VLA 多模态动作模型、潜变量/动作 token预测全身轨迹/动作序列、末端执行器/腕手目标。关键点是保留 VLM 的语义理解，同时增加机器人状态和动作头，避免只停留在语言规划。"
related:
  - ../overview/humanoid-loco-manip-161-papers-technology-map.md
  - ../overview/loco-manip-161-category-10-ego-video.md
  - ../tasks/loco-manipulation.md
sources:
  - ../../sources/papers/loco_manip_161_survey_161_egovla.md
  - ../../sources/blogs/wechat_embodied_ai_lab_humanoid_loco_manip_161_survey.md
  - ../../sources/papers/humanoid_loco_manip_161_catalog.md
---

# EgoVLA

**EgoVLA** 收录于 [具身智能研究室 · 人形 Loco-Manip 161 篇长文](https://mp.weixin.qq.com/s/pACh9EhsISiyPGdiiR0C3A) **第 161/161** 篇，归类为 **10 从人类第一视角视频学习**。

## 一句话定义

EgoVLA 的实现路径是先把语言指令、相机图像/多视角观测、本体状态与关节序列编码成多模态表征，再用ACT/行为克隆模仿学习、VLA 多模态动作模型、潜变量/动作 token预测全身轨迹/动作序列、末端执行器/腕手目标。关键点是保留 VLM 的语义理解，同时增加机器人状态和动作头，避免只停留在语言规划。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Loco-Manip | Loco-Manipulation | 行走与操作动力学耦合的全身任务 |
| WBC | Whole-Body Control | 协调全身关节满足多任务/约束的控制层 |
| VLA | Vision-Language-Action | 视觉-语言-动作多模态策略 |

## 为什么重要

- EgoVLA 的实现路径是先把语言指令、相机图像/多视角观测、本体状态与关节序列编码成多模态表征，再用ACT/行为克隆模仿学习、VLA 多模态动作模型、潜变量/动作 token预测全身轨迹/动作序列、末端执行器/腕手目标。关键点是保留 VLM 的语义理解，同时增加机器人状态和动作头，避免只停留在语言规划。
- 人形 Loco-Manip 161 篇 **#161/161** · 从人类第一视角视频学习。

## 核心信息（索引级）

| 字段 | 内容 |
|------|------|
| 编号 | 161/161 |
| 分组 | 10 从人类第一视角视频学习 |
| 原文题目 | EgoVLA: Learning Vision-Language-Action Models from Egocentric Human Videos |
| 机构 | UC San Diego、MIT、NVIDIA |
| 发表日期 | 2025年7月18日 |
| 论文/项目 | https://rchalyang.github.io/EgoVLA/ |

## 核心机制（归纳）

### 策展导读要点

EgoVLA 的实现路径是先把语言指令、相机图像/多视角观测、本体状态与关节序列编码成多模态表征，再用ACT/行为克隆模仿学习、VLA 多模态动作模型、潜变量/动作 token预测全身轨迹/动作序列、末端执行器/腕手目标。关键点是保留 VLM 的语义理解，同时增加机器人状态和动作头，避免只停留在语言规划。

## 评测与指标（索引级）

- 本条目为 161 篇策展索引级摘录，**未搬运原文量化 benchmark 与实机指标**；评测口径与具体数值以原文 PDF / 项目页为准。
- 评测原始出处：[原文 / 项目页](https://rchalyang.github.io/EgoVLA/)（见上方「核心信息」表「论文/项目」一行）。
- 横向评测对照请回到 [分类 hub](../overview/loco-manip-161-category-10-ego-video.md) 与 [技术地图](../overview/humanoid-loco-manip-161-papers-technology-map.md)。

## 常见误区

1. 161 篇策展条目提供 **地图坐标**；量化 benchmark 与实机指标以原文 PDF / 项目页为准。
2. Loco-manip 单篇工作不自动解决 **底层 WBC 鲁棒性**；须与运控/接触控制对照。

## 与其他页面的关系

- 技术地图：[humanoid-loco-manip-161-papers-technology-map.md](../overview/humanoid-loco-manip-161-papers-technology-map.md)
- 分类 hub：[loco-manip-161-category-10-ego-video.md](../overview/loco-manip-161-category-10-ego-video.md)
- 原始 source：[loco_manip_161_survey_161_egovla.md](../../sources/papers/loco_manip_161_survey_161_egovla.md)

## 参考来源

- [loco_manip_161_survey_161_egovla.md](../../sources/papers/loco_manip_161_survey_161_egovla.md) — 161 篇策展摘录
- [humanoid_loco_manip_161_catalog.md](../../sources/papers/humanoid_loco_manip_161_catalog.md)
- [wechat_embodied_ai_lab_humanoid_loco_manip_161_survey.md](../../sources/blogs/wechat_embodied_ai_lab_humanoid_loco_manip_161_survey.md)

## 推荐继续阅读

- [机器人论文阅读笔记：EgoVLA](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/06_Manipulation/EgoVLA__Learning_Vision-Language-Action_Models_from_Egocentric_Human_Videos/EgoVLA__Learning_Vision-Language-Action_Models_from_Egocentric_Human_Videos.html)
- [Loco-Manipulation 任务页](../tasks/loco-manipulation.md)
- 同题深读/既有实体：[paper-loco-manip-161-161-egovla](../entities/paper-loco-manip-161-161-egovla.md)
