---

type: entity
tags: [paper, motion-cerebellum-survey, humanoid, motion-control, upenn]
status: complete
updated: 2026-07-16
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

**TRAM** 收录于 [具身智能研究室 · 运动小脑 64 篇长文](https://mp.weixin.qq.com/s/Kx9myecE1Z0eGqOapoqQnA) **第 17/64** 篇，归类为 **C 数据入口**。

## 一句话定义

数据入口：野外视频到全局人体轨迹。输入是野外单目视频；实现上恢复全局人体轨迹、相机运动和三维人体姿态，为后续机器人重定向提供带世界坐标的动作源；价值在于把互联网视频从局部姿态提升到可用的全局运动数据。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WBC | Whole-Body Control | 协调全身关节满足多任务/约束的控制层 |
| RL | Reinforcement Learning | 通过与环境交互学习策略的范式 |
| Loco-Manip | Loco-Manipulation | 行走与操作动力学耦合的全身任务 |

## 为什么重要

- 数据入口：野外视频到全局人体轨迹。输入是野外单目视频；实现上恢复全局人体轨迹、相机运动和三维人体姿态，为后续机器人重定向提供带世界坐标的动作源；价值在于把互联网视频从局部姿态提升到可用的全局运动数据。
- 运动小脑 64 篇 **#17/64** · 数据入口：野外视频到全局人体轨迹。

## 核心信息（索引级）

| 字段 | 内容 |
|------|------|
| 编号 | 17/64 |
| 分组 | C 数据入口 |
| 机构 | 宾夕法尼亚大学 |
| 论文/项目 | https://yufu-wang.github.io/tram4d/ |

## 核心机制（归纳）

### 1）策展导读要点

数据入口：野外视频到全局人体轨迹。输入是野外单目视频；实现上恢复全局人体轨迹、相机运动和三维人体姿态，为后续机器人重定向提供带世界坐标的动作源；价值在于把互联网视频从局部姿态提升到可用的全局运动数据。

### 2）策展导读要点

机构：宾夕法尼亚大学

## 结论

**TRAM 的价值不在「又一个三维人体姿态估计」，而在于把相机运动一并解出来——只有拿到世界坐标，互联网视频才算得上可重定向的动作源。**

- 真正起作用的是 **全局人体轨迹 + 相机运动 + 三维人体姿态** 的联合恢复：局部姿态本就不稀缺，稀缺的是带世界坐标的连续轨迹。
- 它在栈中的位置是 **C 数据入口**：产出是喂给下游重定向与全身跟踪的动作源，本身不产出任何控制策略。
- 适用边界与风险：输入设定是野外单目视频，精度、失败模式与量化 benchmark 本页未给，需回到项目页与原文，不宜据本页断言可用性。
- 与运动小脑其余条目一致，它解决的是 **身体层** 的数据供给问题，不替代 VLA/世界模型的任务规划。

## 常见误区

1. 运动小脑条目解决 **身体层** 问题，不替代 VLA/世界模型的任务规划。

## 实验与评测

- 本页在公众号/survey **策展编译**基础上补充机制归纳；**量化 benchmark、消融与实机指标以原文 PDF / 项目页为准**（链接见 [参考来源](#参考来源)）。
- 与同栈姊妹篇对照时，请回到对应 **技术地图 / 42 篇栈 / BFM 地图 / VLN 地图** 总览中的实验段落。

## 与其他页面的关系

- 技术地图：[humanoid-motion-cerebellum-technology-map.md](../overview/humanoid-motion-cerebellum-technology-map.md)
- 分类 hub：[motion-cerebellum-category-03-data-pipeline.md](../overview/motion-cerebellum-category-03-data-pipeline.md)

## 参考来源

- [motion_cerebellum_survey_17_tram.md](../../sources/papers/motion_cerebellum_survey_17_tram.md)
- [motion_cerebellum_64_catalog.md](../../sources/papers/motion_cerebellum_64_catalog.md)
- [wechat_embodied_ai_lab_humanoid_motion_cerebellum_survey.md](../../sources/blogs/wechat_embodied_ai_lab_humanoid_motion_cerebellum_survey.md)

## 推荐继续阅读

- [运动小脑技术地图](../overview/humanoid-motion-cerebellum-technology-map.md)
- [人形 RL 身体系统栈](../overview/humanoid-rl-motion-control-body-system-stack.md)
