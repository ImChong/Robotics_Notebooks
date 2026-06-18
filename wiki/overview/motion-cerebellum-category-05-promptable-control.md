---
type: overview
tags: [motion-cerebellum, humanoid, category-hub, survey, promptable-control]
status: complete
updated: 2026-06-18
summary: "运动小脑 64 篇长文 · E 可提示控制（4 篇）— 可提示小脑等站位。"
related:
  - ./humanoid-motion-cerebellum-technology-map.md
  - ../entities/paper-bfm-zero.md
  - ../entities/paper-motion-cerebellum-mugen.md
  - ../entities/paper-omg-omni-modal-humanoid-control.md
  - ../entities/paper-motionwam-humanoid-loco-manipulation-wam.md
sources:
  - ../../sources/blogs/wechat_embodied_ai_lab_humanoid_motion_cerebellum_survey.md
  - ../../sources/papers/motion_cerebellum_64_catalog.md
---

# 运动小脑分类 E：可提示控制

> **图谱分类节点**：对应 [具身智能研究室 · 运动小脑 64 篇长文](https://mp.weixin.qq.com/s/Kx9myecE1Z0eGqOapoqQnA) 的 **E 可提示控制** 分组；总地图见 [运动小脑技术地图](./humanoid-motion-cerebellum-technology-map.md)。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WBC | Whole-Body Control | 协调全身关节满足多任务/约束的控制层 |
| RL | Reinforcement Learning | 通过与环境交互学习策略的范式 |
| VLA | Vision-Language-Action | 上层策略调用身体 API 的典型形态 |

## 核心问题

可提示小脑：目标、奖励、轨迹 prompt 调用身体。输入是机器人状态、目标状态/奖励函数/动作片段对应的 latent prompt；实现上用 Forward-Backward 表征学习把未来状态访问分布压到潜空间，再用 actor 在给定状态和 prompt 下最大化对应可达性；推理时可用目标姿态、奖励或轨迹编码出 prompt，从而调用不同全身行为。

## 本组论文（4 篇）

| # | 工作 | Wiki 实体 | Source |
|---|------|-----------|--------|
| 38 | BFM-Zero | [paper-bfm-zero.md](../entities/paper-bfm-zero.md) | [catalog](../../sources/papers/motion_cerebellum_64_catalog.md) |
| 39 | MuGen | [paper-motion-cerebellum-mugen.md](../entities/paper-motion-cerebellum-mugen.md) | [catalog](../../sources/papers/motion_cerebellum_survey_39_mugen.md) |
| 40 | OMG | [paper-omg-omni-modal-humanoid-control.md](../entities/paper-omg-omni-modal-humanoid-control.md) | [catalog](../../sources/papers/motion_cerebellum_64_catalog.md) |
| 41 | MotionWAM | [paper-motionwam-humanoid-loco-manipulation-wam.md](../entities/paper-motionwam-humanoid-loco-manipulation-wam.md) | [catalog](../../sources/papers/motion_cerebellum_64_catalog.md) |

## 关联页面

- [运动小脑技术地图](./humanoid-motion-cerebellum-technology-map.md)
- [人形 RL 身体系统栈](./humanoid-rl-motion-control-body-system-stack.md)

## 参考来源

- [wechat_embodied_ai_lab_humanoid_motion_cerebellum_survey.md](../../sources/blogs/wechat_embodied_ai_lab_humanoid_motion_cerebellum_survey.md)
- [motion_cerebellum_64_catalog.md](../../sources/papers/motion_cerebellum_64_catalog.md)

## 推荐继续阅读

- [Loco-Manip 8 篇技术地图](./loco-manip-8-papers-technology-map.md)
