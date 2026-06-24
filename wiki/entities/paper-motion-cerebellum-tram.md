---

type: entity
tags: [paper, motion-cerebellum-survey, humanoid, motion-control, upenn]
status: complete
updated: 2026-06-18
venue: curated
summary: "数据入口：野外视频到全局人体轨迹。输入是野外单目视频；实现上恢复全局人体轨迹、相机运动和三维人体姿态，为后续机器人重定向提供带世界坐标的动作源；价值在于把互联网视频从局部姿态提升到可用的全局运动数据。"
related:
  - ../overview/humanoid-motion-cerebellum-technology-map.md
  - ../overview/motion-cerebellum-category-03-data-pipeline.md
sources:
  - ../../sources/papers/motion_cerebellum_survey_17_tram.md
  - ../../sources/blogs/wechat_embodied_ai_lab_humanoid_motion_cerebellum_survey.md
  - ../../sources/papers/motion_cerebellum_64_catalog.md
---

# TRAM

**TRAM** 收录于 [具身智能研究室 · 运动小脑 64 篇长文](https://mp.weixin.qq.com/s/Kx9myecE1Z0eGqOapoqQnA) **第 17/64** 篇，归类为 **C 数据入口**。本页为知识库 **策展摘要**；方法细节以论文 PDF 与项目页为准。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WBC | Whole-Body Control | 协调全身关节满足多任务/约束的控制层 |
| RL | Reinforcement Learning | 通过与环境交互学习策略的范式 |
| Loco-Manip | Loco-Manipulation | 行走与操作动力学耦合的全身任务 |

## 为什么重要

- 数据入口：野外视频到全局人体轨迹。输入是野外单目视频；实现上恢复全局人体轨迹、相机运动和三维人体姿态，为后续机器人重定向提供带世界坐标的动作源；价值在于把互联网视频从局部姿态提升到可用的全局运动数据。
- 在 [运动小脑技术地图](../overview/humanoid-motion-cerebellum-technology-map.md) 中属于 **[数据入口](../overview/motion-cerebellum-category-03-data-pipeline.md)**。

## 核心信息（索引级）

| 字段 | 内容 |
|------|------|
| 编号 | 17/64 |
| 分组 | C 数据入口 |
| 机构 | 宾夕法尼亚大学 |
| 论文/项目 | https://yufu-wang.github.io/tram4d/ |

## 与其他页面的关系

- 技术地图：[humanoid-motion-cerebellum-technology-map.md](../overview/humanoid-motion-cerebellum-technology-map.md)
- 分类 hub：[motion-cerebellum-category-03-data-pipeline.md](../overview/motion-cerebellum-category-03-data-pipeline.md)

## 实验与评测

- 本页为 **策展索引级** 摘要；量化 benchmark、消融与实机指标以 **原文 PDF / 项目页** 为准（链接见 [参考来源](#参考来源) 与上文 **核心信息（索引级）** 表）。
- 如需与运动小脑同组篇目对照实验，请回到 [运动小脑技术地图](../overview/humanoid-motion-cerebellum-technology-map.md) 对应分类 hub 的评测段落。

## 参考来源

- [motion_cerebellum_survey_17_tram.md](../../sources/papers/motion_cerebellum_survey_17_tram.md)
- [motion_cerebellum_64_catalog.md](../../sources/papers/motion_cerebellum_64_catalog.md)
- [wechat_embodied_ai_lab_humanoid_motion_cerebellum_survey.md](../../sources/blogs/wechat_embodied_ai_lab_humanoid_motion_cerebellum_survey.md)

## 推荐继续阅读

- [运动小脑技术地图](../overview/humanoid-motion-cerebellum-technology-map.md)
- [人形 RL 身体系统栈](../overview/humanoid-rl-motion-control-body-system-stack.md)
