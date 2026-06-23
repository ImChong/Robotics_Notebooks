---

type: entity
tags: [repo, motion-retargeting, humanoid, video-imitation, reinforcement-learning, nvidia]
status: complete
updated: 2026-06-19
summary: "VideoMimic 将单目人体视频转为可仿真跟踪的人形参考运动并训练 RL 策略；与 CRISP、OmniRetarget 常在论文/项目页作基线对比。"
related:
  - ../concepts/motion-retargeting.md
  - ../methods/crisp-real2sim.md
  - ./paper-hrl-stack-03-omniretarget.md
  - ../tasks/humanoid-locomotion.md
sources:
  - ../../sources/repos/videomimic.md
  - ../../sources/papers/omniretarget_arxiv_2509_26633.md
---

# VideoMimic

**VideoMimic**（<https://github.com/hongsukchoi/VideoMimic>，<https://videomimic.github.io/>）实现 **视频驱动的人形运动模仿**：从单目人体视频估计运动，生成仿真中可跟踪的参考轨迹，并训练物理策略使人形复现技能。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RL | Reinforcement Learning | 用参考奖励训练跟踪策略 |
| HMR | Human Mesh Recovery | 视频侧人体姿态估计 |
| Retargeting | Motion Retargeting | 人体参考映射到机器人骨架 |
| Sim2Real | Simulation to Real | 仿真策略迁移真机 |

## 为什么重要

- **端到端视频→机器人**：把重定向、地形/接触建模与模仿学习放在同一研究叙事里，代表「不先手工 MoCap」的一条路线。
- **对比基准**：[OmniRetarget](./paper-hrl-stack-03-omniretarget.md) 论文将 VideoMimic 列为基线（软惩罚、偏地形交互，缺 interaction-preserving 硬约束与数据增广）。

## 流程概念

```mermaid
flowchart LR
  video["单目视频"] --> est["人体运动估计"]
  est --> ret["重定向 + 场景/接触建模"]
  ret --> sim["仿真 RL 跟踪"]
  sim --> real["真机部署（论文设定）"]
```

## 关联页面

- [Motion Retargeting](../concepts/motion-retargeting.md)
- [CRISP](../methods/crisp-real2sim.md)
- [OmniRetarget](./paper-hrl-stack-03-omniretarget.md)
- [holosoma](./holosoma.md)

## 参考来源

- [VideoMimic 仓库归档](../../sources/repos/videomimic.md)
- [OmniRetarget 论文基线表](../../sources/papers/omniretarget_arxiv_2509_26633.md)

## 推荐继续阅读

- 项目页：<https://videomimic.github.io/>
- GitHub：<https://github.com/hongsukchoi/VideoMimic>
