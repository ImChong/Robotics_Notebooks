---
type: entity
tags: [paper, loco-manipulation, loco-manip-161-survey, humanoid]
status: complete
updated: 2026-07-16
venue: curated
summary: "这篇工作先从相机图像/多视角观测、本体状态与关节序列、人类视频/动捕轨迹恢复场景、目标或运动表征，再用PPO/RL 策略训练、AMP/运动先验、全身控制器/WBC/MPC生成全身轨迹/动作序列、低层控制器目标。关键点是把PPO/RL 策略训练、AMP/运动先验放在同一条训练/部署链路里，减少高层目标到低层动作之间的断点。"
related:
  - ../overview/humanoid-loco-manip-161-papers-technology-map.md
  - ../overview/loco-manip-161-category-05-mocap-human-video.md
  - ../tasks/loco-manipulation.md
sources:
  - ../../sources/papers/loco_manip_161_survey_117_n117.md
  - ../../sources/blogs/wechat_embodied_ai_lab_humanoid_loco_manip_161_survey.md
  - ../../sources/papers/humanoid_loco_manip_161_catalog.md
---

# 多阶段强化学习的人形全身羽毛球

**多阶段强化学习的人形全身羽毛球** 收录于 [具身智能研究室 · 人形 Loco-Manip 161 篇长文](https://mp.weixin.qq.com/s/pACh9EhsISiyPGdiiR0C3A) **第 117/161** 篇，归类为 **05 动捕、人类视频与交互动作规划**。

## 一句话定义

这篇工作先从相机图像/多视角观测、本体状态与关节序列、人类视频/动捕轨迹恢复场景、目标或运动表征，再用PPO/RL 策略训练、AMP/运动先验、全身控制器/WBC/MPC生成全身轨迹/动作序列、低层控制器目标。关键点是把PPO/RL 策略训练、AMP/运动先验放在同一条训练/部署链路里，减少高层目标到低层动作之间的断点。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Loco-Manip | Loco-Manipulation | 行走与操作动力学耦合的全身任务 |
| WBC | Whole-Body Control | 协调全身关节满足多任务/约束的控制层 |
| VLA | Vision-Language-Action | 视觉-语言-动作多模态策略 |

## 为什么重要

- 这篇工作先从相机图像/多视角观测、本体状态与关节序列、人类视频/动捕轨迹恢复场景、目标或运动表征，再用PPO/RL 策略训练、AMP/运动先验、全身控制器/WBC/MPC生成全身轨迹/动作序列、低层控制器目标。关键点是把PPO/RL 策略训练、AMP/运动先验放在同一条训练/部署链路里，减少高层目标到低层动作之间的断点。
- 人形 Loco-Manip 161 篇 **#117/161** · 动捕、人类视频与交互动作规划。

## 核心信息（索引级）

| 字段 | 内容 |
|------|------|
| 编号 | 117/161 |
| 分组 | 05 动捕、人类视频与交互动作规划 |
| 原文题目 | Humanoid Whole-Body Badminton via Multi-Stage Reinforcement Learning |
| 机构 | （见原文） |
| 发表日期 | 2026年4月27日 |
| 论文/项目 | https://humanoid-badminton.github.io/Humanoid-Whole-Body-Badminton-via-Multi-Stage-Reinforcement-Learning/ |

## 核心机制（归纳）

### 策展导读要点

这篇工作先从相机图像/多视角观测、本体状态与关节序列、人类视频/动捕轨迹恢复场景、目标或运动表征，再用PPO/RL 策略训练、AMP/运动先验、全身控制器/WBC/MPC生成全身轨迹/动作序列、低层控制器目标。关键点是把PPO/RL 策略训练、AMP/运动先验放在同一条训练/部署链路里，减少高层目标到低层动作之间的断点。

## 评测与指标（索引级）

- 本条目为 161 篇策展索引级摘录，**未搬运原文量化 benchmark 与实机指标**；评测口径与具体数值以原文 PDF / 项目页为准。
- 评测原始出处：[原文 / 项目页](https://humanoid-badminton.github.io/Humanoid-Whole-Body-Badminton-via-Multi-Stage-Reinforcement-Learning/)（见上方「核心信息」表「论文/项目」一行）。
- 横向评测对照请回到 [分类 hub](../overview/loco-manip-161-category-05-mocap-human-video.md) 与 [技术地图](../overview/humanoid-loco-manip-161-papers-technology-map.md)。

## 结论

**羽毛球逼出的是「高层意图与低层全身动作必须同拍」这个问题；这篇的答案是多阶段 RL，把运动先验和全身控制压进同一条训练/部署链路，专治高层到低层的断点。**

- 真正起作用的不是某个单一算法，而是 PPO/RL 与 AMP 运动先验共处一条链路，减少高层目标传到低层动作时的信息丢失与阶段失配。
- 感知侧同时用相机图像/多视角观测与本体状态，说明它是有外部目标驱动的闭环任务，而非只跟踪一条给定参考轨迹。
- 输出既有全身轨迹/动作序列也有低层控制器目标，中间仍靠 WBC/MPC 承接，底层控制鲁棒性不由本工作保证。
- 适用边界是 **05 动捕、人类视频与交互动作规划** 里的高动态交互任务，结论不宜直接外推到准静态操作场景。
- 「多阶段」具体分几阶段、各阶段指标如何，本页未搬运，以 [原文 / 项目页](https://humanoid-badminton.github.io/Humanoid-Whole-Body-Badminton-via-Multi-Stage-Reinforcement-Learning/) 为准。

## 常见误区

1. 161 篇策展条目提供 **地图坐标**；量化 benchmark 与实机指标以原文 PDF / 项目页为准。
2. Loco-manip 单篇工作不自动解决 **底层 WBC 鲁棒性**；须与运控/接触控制对照。

## 与其他页面的关系

- 技术地图：[humanoid-loco-manip-161-papers-technology-map.md](../overview/humanoid-loco-manip-161-papers-technology-map.md)
- 分类 hub：[loco-manip-161-category-05-mocap-human-video.md](../overview/loco-manip-161-category-05-mocap-human-video.md)
- 原始 source：[loco_manip_161_survey_117_n117.md](../../sources/papers/loco_manip_161_survey_117_n117.md)

## 参考来源

- [loco_manip_161_survey_117_n117.md](../../sources/papers/loco_manip_161_survey_117_n117.md) — 161 篇策展摘录
- [humanoid_loco_manip_161_catalog.md](../../sources/papers/humanoid_loco_manip_161_catalog.md)
- [wechat_embodied_ai_lab_humanoid_loco_manip_161_survey.md](../../sources/blogs/wechat_embodied_ai_lab_humanoid_loco_manip_161_survey.md)

## 推荐继续阅读

- [Loco-Manipulation 任务页](../tasks/loco-manipulation.md)
