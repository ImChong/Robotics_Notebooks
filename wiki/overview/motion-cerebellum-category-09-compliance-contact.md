---
type: overview
tags: [motion-cerebellum, humanoid, category-hub, survey, compliance-contact]
status: complete
updated: 2026-06-18
summary: "运动小脑 64 篇长文 · I 柔顺与接触（5 篇）— 接触等站位。"
related:
  - ./humanoid-motion-cerebellum-technology-map.md
  - ../entities/paper-notebook-softmimic-learning-compliant-whole-body-control.md
  - ../entities/paper-hrl-stack-36-chip.md
  - ../entities/paper-hrl-stack-37-gentlehumanoid.md
  - ../entities/paper-hrl-stack-42-thor.md
sources:
  - ../../sources/blogs/wechat_embodied_ai_lab_humanoid_motion_cerebellum_survey.md
  - ../../sources/papers/motion_cerebellum_64_catalog.md
---

# 运动小脑分类 I：柔顺与接触

> **图谱分类节点**：对应 [具身智能研究室 · 运动小脑 64 篇长文](https://mp.weixin.qq.com/s/Kx9myecE1Z0eGqOapoqQnA) 的 **I 柔顺与接触** 分组；总地图见 [运动小脑技术地图](./humanoid-motion-cerebellum-technology-map.md)。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WBC | Whole-Body Control | 协调全身关节满足多任务/约束的控制层 |
| RL | Reinforcement Learning | 通过与环境交互学习策略的范式 |
| VLA | Vision-Language-Action | 上层策略调用身体 API 的典型形态 |

## 核心问题

接触：柔顺全身控制也在成为 tracking 条件。输入是示例动作、本体状态和期望柔顺程度；实现上先用离线优化/扰动生成带外力、刚度条件的柔顺参考，再训练带刚度条件的全身策略去跟踪参考动作；部署时通过调节刚度命令，让同一个 G1 策略在接触场景里表现出更软或更硬的身体响应。

## 本组论文（5 篇）

| # | 工作 | Wiki 实体 | Source |
|---|------|-----------|--------|
| 60 | SoftMimic | [paper-notebook-softmimic-learning-compliant-whole-body-control.md](../entities/paper-notebook-softmimic-learning-compliant-whole-body-control.md) | [catalog](../../sources/papers/motion_cerebellum_64_catalog.md) |
| 61 | CHIP | [paper-hrl-stack-36-chip.md](../entities/paper-hrl-stack-36-chip.md) | [catalog](../../sources/papers/motion_cerebellum_64_catalog.md) |
| 62 | GentleHumanoid | [paper-hrl-stack-37-gentlehumanoid.md](../entities/paper-hrl-stack-37-gentlehumanoid.md) | [catalog](../../sources/papers/motion_cerebellum_64_catalog.md) |
| 63 | Thor | [paper-hrl-stack-42-thor.md](../entities/paper-hrl-stack-42-thor.md) | [catalog](../../sources/papers/motion_cerebellum_64_catalog.md) |
| 64 | WT-UMI | [paper-loco-manip-07-wt-umi.md](../entities/paper-loco-manip-07-wt-umi.md) | [catalog](../../sources/papers/motion_cerebellum_64_catalog.md) |

## 关联页面

- [运动小脑技术地图](./humanoid-motion-cerebellum-technology-map.md)
- [人形 RL 身体系统栈](./humanoid-rl-motion-control-body-system-stack.md)

## 参考来源

- [wechat_embodied_ai_lab_humanoid_motion_cerebellum_survey.md](../../sources/blogs/wechat_embodied_ai_lab_humanoid_motion_cerebellum_survey.md)
- [motion_cerebellum_64_catalog.md](../../sources/papers/motion_cerebellum_64_catalog.md)

## 推荐继续阅读

- [Loco-Manip 8 篇技术地图](./loco-manip-8-papers-technology-map.md)
