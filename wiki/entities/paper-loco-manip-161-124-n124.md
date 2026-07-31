---
type: entity
tags: [paper, loco-manipulation, loco-manip-161-survey, humanoid]
status: complete
updated: 2026-07-16
venue: curated
summary: "这篇工作先从相机图像/多视角观测、人类视频/动捕轨迹、仿真交互数据恢复场景、目标或运动表征，再用扩散策略/流匹配、IK/动作重定向、分层技能/专家策略生成可执行动作命令。关键点是把动作生成看成条件生成问题，用扩散或流匹配在多模态动作分布里采样可执行轨迹。"
related:
  - ../overview/humanoid-loco-manip-161-papers-technology-map.md
  - ../overview/loco-manip-161-category-06-contact-tasks.md
  - ../tasks/loco-manipulation.md
sources:
  - ../../sources/papers/loco_manip_161_survey_124_n124.md
  - ../../sources/blogs/wechat_embodied_ai_lab_humanoid_loco_manip_161_survey.md
  - ../../sources/papers/humanoid_loco_manip_161_catalog.md
---

# 学习人形机器人的足球技能：渐进式感知-行动框架

**学习人形机器人的足球技能：渐进式感知-行动框架** 收录于 [具身智能研究室 · 人形 Loco-Manip 161 篇长文](https://mp.weixin.qq.com/s/pACh9EhsISiyPGdiiR0C3A) **第 124/161** 篇，归类为 **06 特殊任务、接触规划与视觉闭环**。

## 一句话定义

这篇工作先从相机图像/多视角观测、人类视频/动捕轨迹、仿真交互数据恢复场景、目标或运动表征，再用扩散策略/流匹配、IK/动作重定向、分层技能/专家策略生成可执行动作命令。关键点是把动作生成看成条件生成问题，用扩散或流匹配在多模态动作分布里采样可执行轨迹。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Loco-Manip | Loco-Manipulation | 行走与操作动力学耦合的全身任务 |
| WBC | Whole-Body Control | 协调全身关节满足多任务/约束的控制层 |
| VLA | Vision-Language-Action | 视觉-语言-动作多模态策略 |

## 为什么重要

- 这篇工作先从相机图像/多视角观测、人类视频/动捕轨迹、仿真交互数据恢复场景、目标或运动表征，再用扩散策略/流匹配、IK/动作重定向、分层技能/专家策略生成可执行动作命令。关键点是把动作生成看成条件生成问题，用扩散或流匹配在多模态动作分布里采样可执行轨迹。
- 人形 Loco-Manip 161 篇 **#124/161** · 特殊任务、接触规划与视觉闭环。

## 核心信息（索引级）

| 字段 | 内容 |
|------|------|
| 编号 | 124/161 |
| 分组 | 06 特殊任务、接触规划与视觉闭环 |
| 原文题目 | Learning Soccer Skills for Humanoid Robots: A Progressive Perception-Action Framework |
| 机构 | Institute of Artificial Intelligence (TeleAI), China Telecom、ShanghaiTech University、Zhejiang University、Shanghai Jiao Tong University |
| 发表日期 | 2026年2月5日 |
| 论文/项目 | https://soccer-humanoid.github.io/ |

## 核心机制（归纳）

### 策展导读要点

这篇工作先从相机图像/多视角观测、人类视频/动捕轨迹、仿真交互数据恢复场景、目标或运动表征，再用扩散策略/流匹配、IK/动作重定向、分层技能/专家策略生成可执行动作命令。关键点是把动作生成看成条件生成问题，用扩散或流匹配在多模态动作分布里采样可执行轨迹。

## 评测与指标（索引级）

- 本条目为 161 篇策展索引级摘录，**未搬运原文量化 benchmark 与实机指标**；评测口径与具体数值以原文 PDF / 项目页为准。
- 评测原始出处：[原文 / 项目页](https://soccer-humanoid.github.io/)（见上方「核心信息」表「论文/项目」一行）。
- 横向评测对照请回到 [分类 hub](../overview/loco-manip-161-category-06-contact-tasks.md) 与 [技术地图](../overview/humanoid-loco-manip-161-papers-technology-map.md)。

## 结论

**足球是「视觉闭环加接触」的压力测试：这篇不硬训一个端到端策略，而是用渐进式的感知-行动框架，把技能分层与条件生成动作组合起来。**

- 起作用的是三段拼装：多视角视觉与人类视频/仿真交互数据 → 扩散策略/流匹配的条件生成 → IK/动作重定向落到可执行动作命令，分层技能或专家策略负责执行中的路由。
- 归入 06 类（[特殊任务、接触规划与视觉闭环](../overview/loco-manip-161-category-06-contact-tasks.md)）意味着评价重心在特定任务的接触与闭环质量，而不是通用操作泛化。
- 边界：索引级摘录未搬运射门成功率等量化指标，评测口径以 <https://soccer-humanoid.github.io/> 为准。

## 常见误区

1. 161 篇策展条目提供 **地图坐标**；量化 benchmark 与实机指标以原文 PDF / 项目页为准。
2. Loco-manip 单篇工作不自动解决 **底层 WBC 鲁棒性**；须与运控/接触控制对照。

## 与其他页面的关系

- 技术地图：[humanoid-loco-manip-161-papers-technology-map.md](../overview/humanoid-loco-manip-161-papers-technology-map.md)
- 分类 hub：[loco-manip-161-category-06-contact-tasks.md](../overview/loco-manip-161-category-06-contact-tasks.md)
- 原始 source：[loco_manip_161_survey_124_n124.md](../../sources/papers/loco_manip_161_survey_124_n124.md)

## 参考来源

- [loco_manip_161_survey_124_n124.md](../../sources/papers/loco_manip_161_survey_124_n124.md) — 161 篇策展摘录
- [humanoid_loco_manip_161_catalog.md](../../sources/papers/humanoid_loco_manip_161_catalog.md)
- [wechat_embodied_ai_lab_humanoid_loco_manip_161_survey.md](../../sources/blogs/wechat_embodied_ai_lab_humanoid_loco_manip_161_survey.md)

## 推荐继续阅读

- [Loco-Manipulation 任务页](../tasks/loco-manipulation.md)
