---
type: entity
tags: [paper, loco-manipulation, loco-manip-161-survey, humanoid]
status: complete
updated: 2026-07-16
venue: curated
summary: "这篇工作先从相机图像/多视角观测、本体状态与关节序列、人类视频/动捕轨迹恢复场景、目标或运动表征，再用PPO/RL 策略训练生成关节位置/力矩命令、地形/场景表征。关键点是把PPO/RL 策略训练放在同一条训练/部署链路里，减少高层目标到低层动作之间的断点。"
related:
  - ../overview/humanoid-loco-manip-161-papers-technology-map.md
  - ../overview/loco-manip-161-category-05-mocap-human-video.md
  - ../tasks/loco-manipulation.md
sources:
  - ../../sources/papers/loco_manip_161_survey_119_n119.md
  - ../../sources/blogs/wechat_embodied_ai_lab_humanoid_loco_manip_161_survey.md
  - ../../sources/papers/humanoid_loco_manip_161_catalog.md
---

# 迈向多样化人形乒乓球：具有预测增强的统一强化学习

**迈向多样化人形乒乓球：具有预测增强的统一强化学习** 收录于 [具身智能研究室 · 人形 Loco-Manip 161 篇长文](https://mp.weixin.qq.com/s/pACh9EhsISiyPGdiiR0C3A) **第 119/161** 篇，归类为 **05 动捕、人类视频与交互动作规划**。

## 一句话定义

这篇工作先从相机图像/多视角观测、本体状态与关节序列、人类视频/动捕轨迹恢复场景、目标或运动表征，再用PPO/RL 策略训练生成关节位置/力矩命令、地形/场景表征。关键点是把PPO/RL 策略训练放在同一条训练/部署链路里，减少高层目标到低层动作之间的断点。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Loco-Manip | Loco-Manipulation | 行走与操作动力学耦合的全身任务 |
| WBC | Whole-Body Control | 协调全身关节满足多任务/约束的控制层 |
| VLA | Vision-Language-Action | 视觉-语言-动作多模态策略 |

## 为什么重要

- 这篇工作先从相机图像/多视角观测、本体状态与关节序列、人类视频/动捕轨迹恢复场景、目标或运动表征，再用PPO/RL 策略训练生成关节位置/力矩命令、地形/场景表征。关键点是把PPO/RL 策略训练放在同一条训练/部署链路里，减少高层目标到低层动作之间的断点。
- 人形 Loco-Manip 161 篇 **#119/161** · 动捕、人类视频与交互动作规划。

## 核心信息（索引级）

| 字段 | 内容 |
|------|------|
| 编号 | 119/161 |
| 分组 | 05 动捕、人类视频与交互动作规划 |
| 原文题目 | PACE: Physics Augmentation for Coordinated End-to-end Reinforcement Learning toward Versatile Humanoid Table Tennis |
| 机构 | （见原文） |
| 发表日期 | 2026年3月21日 |
| 论文/项目 | https://github.com/purdue-tracelab/TTRL- |

## 核心机制（归纳）

### 策展导读要点

这篇工作先从相机图像/多视角观测、本体状态与关节序列、人类视频/动捕轨迹恢复场景、目标或运动表征，再用PPO/RL 策略训练生成关节位置/力矩命令、地形/场景表征。关键点是把PPO/RL 策略训练放在同一条训练/部署链路里，减少高层目标到低层动作之间的断点。

## 评测与指标（索引级）

- 本条目为 161 篇策展索引级摘录，**未搬运原文量化 benchmark 与实机指标**；评测口径与具体数值以原文 PDF / 项目页为准。
- 评测原始出处：[原文 / 项目页](https://github.com/purdue-tracelab/TTRL-)（见上方「核心信息」表「论文/项目」一行）。
- 横向评测对照请回到 [分类 hub](../overview/loco-manip-161-category-05-mocap-human-video.md) 与 [技术地图](../overview/humanoid-loco-manip-161-papers-technology-map.md)。

## 结论

**PACE 走的是与分层规划相反的路线：用统一的端到端 RL 加物理增强，把高层目标到低层动作之间的断点直接消掉，而不是用接口把它缝起来。**

- 真正的机制是单条 RL 链路直接输出关节位置/力矩命令，并顺带产出地形/场景表征，训练与部署共用同一条路径。
- 条件输入横跨图像/多视角观测、本体状态与关节序列、人类视频/动捕轨迹，协调性由训练而非手工调度获得。
- 代价是模块性：端到端换来的协调，意味着无法像分层方案那样单独替换或调试某一层。
- 本页「论文/项目」给出的 GitHub 链接末尾带 `-`，看似截断，取用前需核对实际可访问性；量化指标以原文为准，横向对照回 [分类 hub](../overview/loco-manip-161-category-05-mocap-human-video.md)。

## 常见误区

1. 161 篇策展条目提供 **地图坐标**；量化 benchmark 与实机指标以原文 PDF / 项目页为准。
2. Loco-manip 单篇工作不自动解决 **底层 WBC 鲁棒性**；须与运控/接触控制对照。

## 与其他页面的关系

- 技术地图：[humanoid-loco-manip-161-papers-technology-map.md](../overview/humanoid-loco-manip-161-papers-technology-map.md)
- 分类 hub：[loco-manip-161-category-05-mocap-human-video.md](../overview/loco-manip-161-category-05-mocap-human-video.md)
- 原始 source：[loco_manip_161_survey_119_n119.md](../../sources/papers/loco_manip_161_survey_119_n119.md)

## 参考来源

- [loco_manip_161_survey_119_n119.md](../../sources/papers/loco_manip_161_survey_119_n119.md) — 161 篇策展摘录
- [humanoid_loco_manip_161_catalog.md](../../sources/papers/humanoid_loco_manip_161_catalog.md)
- [wechat_embodied_ai_lab_humanoid_loco_manip_161_survey.md](../../sources/blogs/wechat_embodied_ai_lab_humanoid_loco_manip_161_survey.md)

## 推荐继续阅读

- [Loco-Manipulation 任务页](../tasks/loco-manipulation.md)
