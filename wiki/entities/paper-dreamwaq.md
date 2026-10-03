---
type: entity
tags: [paper, quadruped, locomotion, proprioception, implicit-terrain, reinforcement-learning, kaist]
status: complete
updated: 2026-10-03
arxiv: "2301.10602"
related:
  - ../methods/dreamwaq.md
  - ../concepts/privileged-training.md
  - ./dreamwaq-plus.md
  - ./paper-robust-perceptive-locomotion-wild.md
sources:
  - ../../sources/papers/dreamwaq_arxiv_2301_10602.md
  - ../../sources/blogs/humanoid_motion_intelligence_day2_locomotion_motion_priors_2026_10_03.md
summary: "DreamWaQ（ICRA 2023）：由本体历史估计机身速度和隐式地形上下文，与策略联合训练，部署时不依赖外部地形感知。"
---

# DreamWaQ：从本体历史学习鲁棒四足行走

**DreamWaQ**（*Learning Robust Quadrupedal Locomotion With Implicit Terrain Imagination via Deep Reinforcement Learning*，[arXiv:2301.10602](https://arxiv.org/abs/2301.10602)，ICRA 2023）研究四足机器人如何只靠身体反馈适应地形和动力学变化。它从短时本体观测历史估计机身速度与隐式环境表示，并将估计结果用于行走策略。

## 一句话理解

机器人不能直接读到地面摩擦等属性时，利用近期身体响应形成控制所需的环境线索，再据此调整步态。

## 方法要点

- **历史编码：** Context Estimation Network（CENet）处理本体历史，预测机身线速度并学习隐式环境表示。
- **联合优化：** 状态估计目标与策略训练共同更新；价值网络可用训练期特权信息，部署策略依靠本体输入。
- **作用边界：** 身体历史有助于接触后的适应，但不能替代对沟隙、台阶等前方障碍的预先观察。

## 与相邻工作

- [DreamWaQ 方法页](../methods/dreamwaq.md)归纳算法机制；本页是论文的独立详情节点。
- [DreamWaQ++](./dreamwaq-plus.md)在后续工作中加入点云等外部感知。
- [Robust Perceptive Locomotion](./paper-robust-perceptive-locomotion-wild.md)提供带噪高程图与本体历史融合的另一条路线。

## 来源

- [arXiv:2301.10602](https://arxiv.org/abs/2301.10602)
- [项目页](https://sites.google.com/view/dreamwaq)
- [Day 2 文章来源索引](../../sources/blogs/humanoid_motion_intelligence_day2_locomotion_motion_priors_2026_10_03.md)
