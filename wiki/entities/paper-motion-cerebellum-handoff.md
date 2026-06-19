---
type: entity
tags: [paper, motion-cerebellum-survey, humanoid, motion-control]
status: complete
updated: 2026-06-18
arxiv: "2606.06493"
summary: "接口：任务空间命令 + 多教师蒸馏。输入是任务空间命令、本体状态和不同教师策略输出；实现上把移动、恢复、全身跟踪等互补教师蒸馏到一个学生控制器，并用门控/条件机制融合专家能力；上层只需给速度、手部目标、身体高度等紧凑命令。"
related:
  - ../overview/humanoid-motion-cerebellum-technology-map.md
  - ../overview/motion-cerebellum-category-07-loco-manip-interface.md
sources:
  - ../../sources/papers/motion_cerebellum_survey_48_handoff.md
  - ../../sources/blogs/wechat_embodied_ai_lab_humanoid_motion_cerebellum_survey.md
  - ../../sources/papers/motion_cerebellum_64_catalog.md
---

# HANDOFF

**HANDOFF** 收录于 [具身智能研究室 · 运动小脑 64 篇长文](https://mp.weixin.qq.com/s/Kx9myecE1Z0eGqOapoqQnA) **第 48/64** 篇，归类为 **G Loco-Manip 接口**。本页为知识库 **策展摘要**；方法细节以论文 PDF 与项目页为准。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WBC | Whole-Body Control | 协调全身关节满足多任务/约束的控制层 |
| RL | Reinforcement Learning | 通过与环境交互学习策略的范式 |
| Loco-Manip | Loco-Manipulation | 行走与操作动力学耦合的全身任务 |

## 为什么重要

- 接口：任务空间命令 + 多教师蒸馏。输入是任务空间命令、本体状态和不同教师策略输出；实现上把移动、恢复、全身跟踪等互补教师蒸馏到一个学生控制器，并用门控/条件机制融合专家能力；上层只需给速度、手部目标、身体高度等紧凑命令。
- 在 [运动小脑技术地图](../overview/humanoid-motion-cerebellum-technology-map.md) 中属于 **[Loco-Manip 接口](../overview/motion-cerebellum-category-07-loco-manip-interface.md)**。

## 核心信息（索引级）

| 字段 | 内容 |
|------|------|
| 编号 | 48/64 |
| 分组 | G Loco-Manip 接口 |
| 机构 | 加州理工学院、人类与机器认知研究所 |
| 论文/项目 | https://arxiv.org/abs/2606.06493v1 |

## 与其他页面的关系

- 技术地图：[humanoid-motion-cerebellum-technology-map.md](../overview/humanoid-motion-cerebellum-technology-map.md)
- 分类 hub：[motion-cerebellum-category-07-loco-manip-interface.md](../overview/motion-cerebellum-category-07-loco-manip-interface.md)

## 实验与评测

- 本页为 **策展索引级** 摘要；量化 benchmark、消融与实机指标以 **原文 PDF / 项目页** 为准（链接见 [参考来源](#参考来源) 与上文 **核心信息（索引级）** 表）。
- 如需与运动小脑同组篇目对照实验，请回到 [运动小脑技术地图](../overview/humanoid-motion-cerebellum-technology-map.md) 对应分类 hub 的评测段落。

## 参考来源

- [motion_cerebellum_survey_48_handoff.md](../../sources/papers/motion_cerebellum_survey_48_handoff.md)
- [motion_cerebellum_64_catalog.md](../../sources/papers/motion_cerebellum_64_catalog.md)
- [wechat_embodied_ai_lab_humanoid_motion_cerebellum_survey.md](../../sources/blogs/wechat_embodied_ai_lab_humanoid_motion_cerebellum_survey.md)

## 推荐继续阅读

- [运动小脑技术地图](../overview/humanoid-motion-cerebellum-technology-map.md)
- [人形 RL 身体系统栈](../overview/humanoid-rl-motion-control-body-system-stack.md)
