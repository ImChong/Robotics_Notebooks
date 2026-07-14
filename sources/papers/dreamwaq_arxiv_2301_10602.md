# DreamWaQ: Learning Robust Quadrupedal Locomotion With Implicit Terrain Imagination（arXiv:2301.10602）

- **标题：** DreamWaQ: Learning Robust Quadrupedal Locomotion With Implicit Terrain Imagination
- **会议：** ICRA 2023
- **链接：** <https://arxiv.org/abs/2301.10602>
- **扩展：** [DreamWaQ++（arXiv:2409.19709）](dreamwaq_plus_arxiv_2409_19709.md)（T-RO 2026，多模态点云）
- **一句话说明：** 四足**盲走**单阶段 RL：CENet 从本体历史估计隐式地形上下文与体速，非对称 Actor–Critic 在仅本体部署下实现鲁棒 locomotion，为后续障碍感知扩展奠基。

## 核心机制

- **CENet（Context Estimation Network）：** 编码本体历史 → 隐式地形想象 + 速度估计。
- **训练：** 特权 critic + 部分观测 actor；无需显式高度图或外感知传感器。
- **局限：** 楼梯/缺口等需**接触后**才能调整步态；前瞻障碍见 DreamWaQ++ 与 PIE 路线。

## 对 wiki 的映射

- [DreamWaQ 方法页](../../wiki/methods/dreamwaq.md)
- [DreamWaQ++ 实体页](../../wiki/entities/dreamwaq-plus.md)
- [Privileged Training](../../wiki/concepts/privileged-training.md)
- 飞书 Know-How「DreamWaq盲走一阶段鲁棒行走」：[humanoid-motion-control-know-how-technology-map](../../wiki/overview/humanoid-motion-control-know-how-technology-map.md)
