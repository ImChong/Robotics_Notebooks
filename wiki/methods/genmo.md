---
type: method
tags: [human-motion, diffusion, smpl, perception, video-to-control]
status: complete
updated: 2026-05-07
related:
  - ./exoactor.md
  - ./diffusion-motion-generation.md
  - ./sonic-motion-tracking.md
  - ./wilor.md
sources:
  - ../../sources/repos/genmo.md
summary: "GENMO（官方代码常以 GEM 名义发布）把人体运动估计表述为带多模态条件的约束式扩散生成，统一视频、2D 关键点、文本、音频与 SMPL 关键帧等输入下的轨迹恢复与合成。"
---

# GENMO（统一人体运动估计与生成）

**GENMO**（*A GENeralist Model for Human MOtion*，NVIDIA，ICCV 2025 Highlight）将 **运动估计** 与 **运动生成** 放进同一套扩散式框架：在给定观测约束（如 2D 关键点轨迹）的前提下生成完整 SMPL 参数序列，从而利用生成先验修补遮挡、截断等脆弱帧。它与 [Video-as-Simulation（视频即仿真）](../concepts/video-as-simulation.md) 所讨论的「像素域动力学接口」相衔接：先把视频变成可跟踪的人体运动，再交给下游执行器。

## 为什么重要？

- **视频→机器人链路的中枢**：从单目或生成视频恢复时间一致的全身参数，是「像素 → 跟踪控制器」管道的标准中间表示之一（参见 [ExoActor](./exoactor.md)）。
- **估计与生成的协同**：论文强调生成模型提供的运动先验可提升困难条件下的估计质量，而大规模野外视频弱标注又可增强生成分布——区别于「只做回归」或「无条件生成」两条孤立路线。
- **多条件混合**：同一模型可在时间轴上混合多种条件（关键点片段、文本描述、音频节拍等），适合与上层语言或视听模块拼接。

## 主要技术路线

| 组件 | 要点 |
|------|------|
| **问题形式** | 将估计视为 **约束运动生成**：采样轨迹必须贴近观测（如重投影一致的 2D）。 |
| **训练** | 结合动捕/标注数据与野外视频上的 2D、文本等弱监督，做估计引导的生成训练。 |
| **接口** | 以 SMPL 族参数为主干输出之一，便于下游直接使用或与手部估计（如 [WiLoR](./wilor.md)）拼接。 |

## 常见误区或局限

- **不等于实时控制器**：GENMO 侧重点是人体运动推理；要在实体人形上闭环还需动力学可行的跟踪策略（如 [SONIC](./sonic-motion-tracking.md)）或传统 WBC/MPC。
- **命名迁移**：仓库与权重发布可能使用 **GEM** 品牌，检索代码时需两者兼顾。
- **域差异**：从「生成的人体视频」估计运动时，误差会传递到下游；需与视频生成模块 jointly 评估（ExoActor 论文讨论了若干失败模式）。

## 与其他页面的关系

- [ExoActor](./exoactor.md)：典型系统集成位——GENMO 承担全身 SMPL 序列恢复。
- [Diffusion-based Motion Generation](./diffusion-motion-generation.md)：同类生成式运动建模范式。
- [SONIC](./sonic-motion-tracking.md)：将估计轨迹映射到机器人可行空间。

## 推荐继续阅读

- 论文：<https://arxiv.org/abs/2505.01425>
- 代码与说明：<https://github.com/NVlabs/GENMO>、<https://research.nvidia.com/labs/dair/publication/genmo2025/>

## 参考来源

- [GENMO / GEM（统一人体运动估计与生成）](../../sources/repos/genmo.md)

## 关联页面

- [ExoActor (视频生成驱动的交互式人形控制)](./exoactor.md)
- [WiLoR（野外手部 3D 重建）](./wilor.md)
- [SONIC（规模化运动跟踪人形控制）](./sonic-motion-tracking.md)
