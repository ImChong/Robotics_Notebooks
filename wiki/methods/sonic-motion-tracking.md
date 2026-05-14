---
type: method
tags: [humanoid, imitation-learning, motion-tracking, foundation-model, nvidia]
status: complete
updated: 2026-05-07
related:
  - ./beyondmimic.md
  - ./exoactor.md
  - ./imitation-learning.md
  - ./genmo.md
sources:
  - ../../sources/repos/sonic-humanoid-motion-tracking.md
summary: "SONIC 通过规模化运动跟踪监督训练通用人形策略，把海量 MoCap 帧上的轨迹跟踪当作预训练任务，支持 VR、人体视频、文本与音乐等多模态指令，并作为真实机器人的通用低层执行器。"
---

# SONIC（规模化运动跟踪人形控制）

**SONIC**（*Supersizing Motion Tracking for Natural Humanoid Whole-Body Control*）论证：在人形控制上 **大规模拟合多样参考运动**（motion tracking）可获得稳健的全身体现与少手工奖励设计，并随 **模型容量、数据量与算力** 同步扩展性能。项目由 NVIDIA 等与 CMU 等合作者推动（详见论文作者列表）。实现层面与 [Whole-Body Control (WBC)](../concepts/whole-body-control.md) 所讨论的「高自由度协调」问题同一战场：SONIC 用学习策略把参考运动映射为全身扭矩/位置指令。

## 为什么重要？

- **执行层「基础模型」叙事**：把跟踪当作统一预训练目标，再用统一 token 空间接入 VR、视频、VLA、文本等不同上游——降低「每换一个接口就重写 reward」的成本。
- **与 BeyondMimic  lineage**：同属高质量仿真里的模仿 / 跟踪路线；SONIC 强调 **scaling law** 式的数据与网络扩展（参见 [BeyondMimic](./beyondmimic.md) 中的物理建模与采样细节对照阅读）。
- **视频驱动现实的落脚点**：人体运动估计（如 [GENMO](./genmo.md)、[WiLoR](./wilor.md)）给出参考轨迹后，需要动力学可行的跟踪策略；SONIC 在 [ExoActor](./exoactor.md) 中被用作「物理过滤器」，直接把人体运动喂入策略而省略部分经典重定向步骤（该结论具有任务与平台依赖性）。

## 主要技术路线

| 维度 | 要点 |
|------|------|
| **监督** | 密集跟踪损失：策略输出逼近参考全身运动（具体观测与动作空间以论文为准）。 |
| **规模** | 论文公开量级包含 **上亿帧**、**数百小时** MoCap 与大规模 GPU 训练时间；性能随规模单调改善的报道支撑「跟踪可扩展」论点。 |
| **接口** | 多模态条件映射到共享表示，支持交互式部署与实时规划桥接。 |

## 常见误区或局限

- **不是万能仿真替身**：跟踪器只能在其训练分布与机器人动力学对齐的范围内泛化；极端杂技或强接触任务仍可能失败。
- **跳过重定向的前提**：ExoActor 显示「人体轨迹 → SONIC」可优于某些 SMPL→机器人重定向流水线，但不等于所有平台都应丢弃重定向（参见 [GMR](./motion-retargeting-gmr.md) 讨论）。
- **硬件差异**：同一策略在不同人形硬件上仍需适配观测与动作映射。

## 与其他页面的关系

- [BeyondMimic](./beyondmimic.md)：相近生态里的高性能模仿框架，可作为物理建模与采样策略的对照。
- [ExoActor](./exoactor.md)：SONIC 作为生成视频流水线的机器人侧执行模块的案例。
- [Imitation Learning](./imitation-learning.md)：大规模跟踪可视为广义的演示驱动学习。
- [Zhengyi Luo（罗正宜）](../entities/zhengyi-luo.md)：论文共同一作与项目核心贡献者之一，主页汇总 SONIC 与相邻人形工作入口。

## 推荐继续阅读

- 论文：<https://arxiv.org/abs/2511.07820>
- 项目页：<https://nvlabs.github.io/SONIC/>

## 参考来源

- [SONIC（规模化人体运动跟踪驱动的人形全身控制）](../../sources/repos/sonic-humanoid-motion-tracking.md)

## 关联页面

- [BeyondMimic](./beyondmimic.md)
- [ExoActor (视频生成驱动的交互式人形控制)](./exoactor.md)
- [GENMO（统一人体运动估计与生成）](./genmo.md)
