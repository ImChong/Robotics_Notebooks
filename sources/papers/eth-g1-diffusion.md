# Learning Whole-Body Humanoid Locomotion via Motion Generation and Motion Tracking

- **URL**: https://arxiv.org/html/2604.17335v1
- **Authors**: Zewei Zhang, Kehan Wen, Michael Xu, Junzhe He, Chenhao Li, Takahiro Miki, Clemens Schwarke, Chong Zhang, Xue Bin Peng, and Marco Hutter.
- **Date**: 2026-04 (ArXiv)
- **Tags**: #humanoid #diffusion-model #whole-body-control #RL #locomotion #Unitree-G1

## 核心摘要

本文提出了一种分层的全身人形机器人移动框架，通过结合**扩散模型（Diffusion Models）**的高层规划能力和**强化学习（RL）**的底层控制鲁棒性，解决了类人机器人在复杂地形下的全身协调运动问题。

### 技术路线
1. **Motion Generation (Diffusion)**: 训练一个条件扩散运动生成器，根据目标指令、历史状态和局部地形（Elevation Map），生成未来 0.5s 的全身参考运动。
2. **Motion Tracking (RL)**: 使用 PPO 训练一个全身运动跟踪器，使其能够实时模仿扩散模型生成的参考序列。
3. **Closed-loop Fine-tuning**: 在仿真中进行闭环微调，使跟踪器能够处理生成器的预测误差，提高鲁棒性。

### 关键成果
- 在 **Unitree G1** 硬件上实现了实时、全 onboard 的感知运动控制。
- 机器人展现了利用手、膝等部位进行攀爬（爬上 75cm 高箱）和翻越复杂障碍的能力。
- 采用 **TensorRT** 加速扩散模型推理，延迟仅为 20ms。

## 关键术语
- **Diffusion-based Motion Generator**: 基于扩散模型的运动生成器。
- **Closed-loop Fine-tuning**: 闭环微调，解决分布偏移（Distribution Mismatch）。
- **Whole-body Coordination**: 利用全身各部位（非仅足部）进行移动。

## 相关引用
- [diffusion-policy](../../wiki/methods/diffusion-policy.md)
- [ppo](../../wiki/methods/policy-optimization.md)
- [humanoid-locomotion](../../wiki/tasks/humanoid-locomotion.md)
