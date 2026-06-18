---
type: entity
tags: [paper, motion-cerebellum-survey, humanoid, motion-control]
status: complete
updated: 2026-06-18
summary: "任务数据：3D 资产和视频先验生成 Loco-Manip 数据。输入是移动操作任务、场景几何、物体状态和机器人模型；实现上用规划、仿真、生成式数据或自主探索产生手脚协同轨迹，再筛选成可训练示范；目标是补足人形 loco-manip 最缺的长"
related:
  - ../overview/humanoid-motion-cerebellum-technology-map.md
  - ../overview/motion-cerebellum-category-08-real-tasks.md
sources:
  - ../../sources/papers/motion_cerebellum_survey_57_grail.md
  - ../../sources/blogs/wechat_embodied_ai_lab_humanoid_motion_cerebellum_survey.md
  - ../../sources/papers/motion_cerebellum_64_catalog.md
---

# GRAIL

**GRAIL** 收录于 [具身智能研究室 · 运动小脑 64 篇长文](https://mp.weixin.qq.com/s/Kx9myecE1Z0eGqOapoqQnA) **第 57/64** 篇，归类为 **H 真实任务**。本页为知识库 **策展摘要**；方法细节以论文 PDF 与项目页为准。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WBC | Whole-Body Control | 协调全身关节满足多任务/约束的控制层 |
| RL | Reinforcement Learning | 通过与环境交互学习策略的范式 |
| Loco-Manip | Loco-Manipulation | 行走与操作动力学耦合的全身任务 |

## 为什么重要

- 任务数据：3D 资产和视频先验生成 Loco-Manip 数据。输入是移动操作任务、场景几何、物体状态和机器人模型；实现上用规划、仿真、生成式数据或自主探索产生手脚协同轨迹，再筛选成可训练示范；目标是补足人形 loco-manip 最缺的长程、接触丰富数据。
- 在 [运动小脑技术地图](../overview/humanoid-motion-cerebellum-technology-map.md) 中属于 **[真实任务](../overview/motion-cerebellum-category-08-real-tasks.md)**。

## 核心信息（索引级）

| 字段 | 内容 |
|------|------|
| 编号 | 57/64 |
| 分组 | H 真实任务 |
| 机构 | 英伟达 |
| 论文/项目 | https://arxiv.org/abs/2606.05160v1 |

## 与其他页面的关系

- 技术地图：[humanoid-motion-cerebellum-technology-map.md](../overview/humanoid-motion-cerebellum-technology-map.md)
- 分类 hub：[motion-cerebellum-category-08-real-tasks.md](../overview/motion-cerebellum-category-08-real-tasks.md)

## 参考来源

- [motion_cerebellum_survey_57_grail.md](../../sources/papers/motion_cerebellum_survey_57_grail.md)
- [motion_cerebellum_64_catalog.md](../../sources/papers/motion_cerebellum_64_catalog.md)
- [wechat_embodied_ai_lab_humanoid_motion_cerebellum_survey.md](../../sources/blogs/wechat_embodied_ai_lab_humanoid_motion_cerebellum_survey.md)

## 推荐继续阅读

- [运动小脑技术地图](../overview/humanoid-motion-cerebellum-technology-map.md)
- [人形 RL 身体系统栈](../overview/humanoid-rl-motion-control-body-system-stack.md)
