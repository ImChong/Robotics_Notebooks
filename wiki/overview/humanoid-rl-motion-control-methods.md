---
type: overview
tags: [humanoid, reinforcement-learning, imitation-learning, bfm, perception]
status: complete
updated: 2026-07-14
related:
  - ./humanoid-motion-control-know-how-technology-map.md
  - ../../roadmap/depth-rl-locomotion.md
  - ../methods/reinforcement-learning.md
  - ../concepts/behavior-foundation-model.md
sources:
  - ../../sources/papers/humanoid_motion_control_know_how.md
summary: "飞书「深度强化学习运动控制方法（Learning Base）」父节点：RL 理论、TS+DAgger、感知盲走/单阶段、重定向、DeepMimic/AMP 与 BFM 三线的方法族索引。"
---

# 深度强化学习运动控制方法（Learning-based）

飞书 Know-How **「深度强化学习运动控制方法（Learning Base）」** 的图谱父节点：覆盖 **RL 基础 → 特权/模仿训练 → 感知 loco → 重定向与跟踪 → BFM 三线** 的方法族，每主题有独立 wiki 节点。

## 一句话定义

用仿真与数据学策略，从盲走、感知行走、动作模仿到运控大模型，与传统 Model-based 栈并行互补。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RL | Reinforcement Learning | 基础范式 |
| IL | Imitation Learning | 重定向与跟踪上游 |
| AMP | Adversarial Motion Prior | 仿人行走先验 |
| BFM | Behavior Foundation Model | 多行为身体基座 |
| TS | Teacher-Student | 特权蒸馏 |
| Sim2Real | Simulation to Real | 训练主场景 |

## 方法族索引

| 飞书主题 | Wiki |
|----------|------|
| RL 理论基础 | [Reinforcement Learning](../methods/reinforcement-learning.md) |
| Teacher-Student + DAgger | [Teacher-Student + DAgger](../methods/teacher-student-dagger-training.md) |
| DreamWaq 盲走 | [DreamWaQ](../methods/dreamwaq.md) |
| PIE 感知一阶段 | [PIE](../methods/pie-perceptive-locomotion.md) |
| Attention 落足 | [Attention 落足点](../methods/attention-foot-placement.md) |
| Retarget | [Motion Retargeting](../concepts/motion-retargeting.md)、[GMR](../methods/motion-retargeting-gmr.md) |
| DeepMimic 跳舞 | [DeepMimic](../methods/deepmimic.md) |
| AMP 仿人行走 | [AMP](../methods/amp-reward.md) |
| BFM 总览 | [BFM](../concepts/behavior-foundation-model.md) |
| BFM-Zero | [BFM-Zero](../entities/paper-bfm-zero.md) |
| SONIC | [SONIC](../methods/sonic-motion-tracking.md) |
| TS 多动作 BFM | [TS 多技能 BFM](../methods/teacher-student-multi-skill-bfm.md) |

## 学习路线

- [depth-rl-locomotion](../../roadmap/depth-rl-locomotion.md)
- 四足入门视频（飞书外链）：B 站「四足运控从入门到精通」

## 局限与风险

- **仿真依赖**：奖励与域随机设计决定上限。
- **与 Model-based 分工**：高保证任务（力控、安全）常需 WBC/MPC 托底。

## 关联页面

- [Know-How 技术地图](./humanoid-motion-control-know-how-technology-map.md)
- [八层身体系统栈](./humanoid-rl-motion-control-body-system-stack.md)
- [depth-bfm](../../roadmap/depth-bfm.md)

## 参考来源

- [humanoid_motion_control_know_how.md](../../sources/papers/humanoid_motion_control_know_how.md)

## 推荐继续阅读

- [depth-rl-locomotion](../../roadmap/depth-rl-locomotion.md)
