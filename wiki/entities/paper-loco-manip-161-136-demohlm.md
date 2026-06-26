---
type: entity
tags: [paper, loco-manipulation, loco-manip-161-survey, humanoid]
status: complete
updated: 2026-06-26
venue: curated
summary: "DemoHLM 主要解决数据闭环：用相机图像/多视角观测、本体状态与关节序列、遥操作/外骨骼数据采集人类操作和机器人状态，再通过ACT/行为克隆模仿学习、全身控制器/WBC/MPC、闭环纠错/人类干预转成可训练、可复用的关节位置/力矩命令、全身轨迹/动作序列、低层控制器目标。关键点是把示范轨迹压成可监督的动作预测问题，再通过动作 chunk 或闭环执行降低时序抖动。"
related:
  - ../overview/humanoid-loco-manip-161-papers-technology-map.md
  - ../overview/loco-manip-161-category-08-hardware-deployment.md
  - ../tasks/loco-manipulation.md
sources:
  - ../../sources/papers/loco_manip_161_survey_136_demohlm.md
  - ../../sources/blogs/wechat_embodied_ai_lab_humanoid_loco_manip_161_survey.md
  - ../../sources/papers/humanoid_loco_manip_161_catalog.md
---

# DemoHLM

**DemoHLM** 收录于 [具身智能研究室 · 人形 Loco-Manip 161 篇长文](https://mp.weixin.qq.com/s/pACh9EhsISiyPGdiiR0C3A) **第 136/161** 篇，归类为 **08 硬件平台、感知配置与部署扩展**。

## 一句话定义

DemoHLM 主要解决数据闭环：用相机图像/多视角观测、本体状态与关节序列、遥操作/外骨骼数据采集人类操作和机器人状态，再通过ACT/行为克隆模仿学习、全身控制器/WBC/MPC、闭环纠错/人类干预转成可训练、可复用的关节位置/力矩命令、全身轨迹/动作序列、低层控制器目标。关键点是把示范轨迹压成可监督的动作预测问题，再通过动作 chunk 或闭环执行降低时序抖动。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Loco-Manip | Loco-Manipulation | 行走与操作动力学耦合的全身任务 |
| WBC | Whole-Body Control | 协调全身关节满足多任务/约束的控制层 |
| VLA | Vision-Language-Action | 视觉-语言-动作多模态策略 |

## 为什么重要

- DemoHLM 主要解决数据闭环：用相机图像/多视角观测、本体状态与关节序列、遥操作/外骨骼数据采集人类操作和机器人状态，再通过ACT/行为克隆模仿学习、全身控制器/WBC/MPC、闭环纠错/人类干预转成可训练、可复用的关节位置/力矩命令、全身轨迹/动作序列、低层控制器目标。关键点是把示范轨迹压成可监督的动作预测问题，再通过动作 chunk 或闭环执行降低时序抖动。
- 人形 Loco-Manip 161 篇 **#136/161** · 硬件平台、感知配置与部署扩展。

## 核心信息（索引级）

| 字段 | 内容 |
|------|------|
| 编号 | 136/161 |
| 分组 | 08 硬件平台、感知配置与部署扩展 |
| 原文题目 | DemoHLM: From One Demonstration to Generalizable Humanoid Loco-Manipulation |
| 机构 | Peking University |
| 发表日期 | 2025年10月13日 |
| 论文/项目 | https://beingbeyond.github.io/DemoHLM/ |

## 核心机制（归纳）

### 策展导读要点

DemoHLM 主要解决数据闭环：用相机图像/多视角观测、本体状态与关节序列、遥操作/外骨骼数据采集人类操作和机器人状态，再通过ACT/行为克隆模仿学习、全身控制器/WBC/MPC、闭环纠错/人类干预转成可训练、可复用的关节位置/力矩命令、全身轨迹/动作序列、低层控制器目标。关键点是把示范轨迹压成可监督的动作预测问题，再通过动作 chunk 或闭环执行降低时序抖动。

## 常见误区

1. 161 篇策展条目提供 **地图坐标**；量化 benchmark 与实机指标以原文 PDF / 项目页为准。
2. Loco-manip 单篇工作不自动解决 **底层 WBC 鲁棒性**；须与运控/接触控制对照。

## 与其他页面的关系

- 技术地图：[humanoid-loco-manip-161-papers-technology-map.md](../overview/humanoid-loco-manip-161-papers-technology-map.md)
- 分类 hub：[loco-manip-161-category-08-hardware-deployment.md](../overview/loco-manip-161-category-08-hardware-deployment.md)
- 原始 source：[loco_manip_161_survey_136_demohlm.md](../../sources/papers/loco_manip_161_survey_136_demohlm.md)

## 参考来源

- [loco_manip_161_survey_136_demohlm.md](../../sources/papers/loco_manip_161_survey_136_demohlm.md) — 161 篇策展摘录
- [humanoid_loco_manip_161_catalog.md](../../sources/papers/humanoid_loco_manip_161_catalog.md)
- [wechat_embodied_ai_lab_humanoid_loco_manip_161_survey.md](../../sources/blogs/wechat_embodied_ai_lab_humanoid_loco_manip_161_survey.md)

## 推荐继续阅读

- [Loco-Manipulation 任务页](../tasks/loco-manipulation.md)
- 同题深读/既有实体：[paper-loco-manip-161-136-demohlm](../entities/paper-loco-manip-161-136-demohlm.md)
