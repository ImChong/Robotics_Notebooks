---
type: method
tags: [world-models, generative-ai, simulation, video-generation, driving]
status: complete
updated: 2026-05-14
related:
  - ../concepts/latent-imagination.md
  - ../concepts/world-action-models.md
  - ../methods/model-based-rl.md
  - ../methods/being-h07.md
  - ../entities/nvidia-omniverse.md
sources:
  - ../../sources/papers/diffusion_and_gen.md
  - ../../sources/papers/exoactor.md
  - ../../sources/papers/being_h07.md
  - ../../sources/papers/world_action_models_survey_2605.md
summary: "生成式世界模型（Generative World Models）利用扩散模型或视频生成技术来模拟物理世界的动态，为机器人提供高保真的视频级仿真和无限的反事实推演能力。"
---

# Generative World Models (生成式世界模型)

**生成式世界模型** 是具身智能（Embodied AI）领域的下一代物理引擎替代者。不同于 Drake 或 MuJoCo 等基于严谨几何和力学方程的解析引擎，生成式世界模型直接利用**生成式 AI (Generative AI)** 的能力，通过海量视频数据学习世界的运动规律。

## 核心理念：以生成代替计算

在传统仿真中，我们需要手动编写复杂的接触力方程；而在生成式世界模型中，模型学会了“如果机器人向左打方向盘，画面应该如何平滑变化”。

### 主要架构
1. **视频生成器 (Video Diffusion/Autoregressive)**：如 GAIA-1 或 UniSim。给定当前画面和动作序列，生成一段长达数秒甚至数分钟的未来预测视频。
2. **反事实推演 (Counterfactual Reasoning)**：允许用户输入“如果没有躲避障碍物会怎样？”，模型会生成相应的碰撞视频，作为强化学习的负样本。

## 典型代表作

### 1. GAIA-1 (Wayve)
针对自动驾驶设计的世界模型。它不仅能生成真实的驾驶场景，还能根据文本描述（如“突然下起大雨”）动态改变天气和光影。

### 2. UniSim (Google DeepMind)
一个通用的具身智能世界模型。它将现实世界的视频数据和仿真数据结合，允许机器人在“视频”中练习开橱柜、拿杯子等精细操作，并将学到的技能无缝迁移到真实物理世界。

## 优势与挑战

### 优势
- **视觉真实度极致**：彻底解决了 Sim2Real 在感知层面的 Gap。
- **无需手动建模**：对于复杂的流体、软体（如折衣服、揉面团），生成式模型比物理引擎更容易捕捉其动态特性。

### 挑战
- **物理一致性缺失**：模型有时会产生违反物理常识的幻觉（如物体凭空消失）。
- **推理开销大**：目前生成一帧高质量视频的速度远低于物理引擎的 1000Hz 要求。
- **交互精度低**：很难通过生成的视频反推精确到毫米级的接触力。

### 工程折中：潜空间世界–动作（示例：Being-H0.7）

若目标是**在线操作控制**而非高保真视频预览，可把「未来结构」压进**紧凑潜变量工作空间**，训练时用未来观测分支对齐、测试时只跑先验动作头，从而保留世界建模的部分收益、避免每步显式像素 rollout。详见 [Being-H0.7](./being-h07.md)。

当讨论把「预测未来」与「输出动作」在**同一策略对象**里联合建模（综述中的 **World Action Models**）时，重点会从**像素逼真度**转向**耦合结构、动作可推断性与闭环延迟**；仓库内总览见 [World Action Models（WAM）](../concepts/world-action-models.md)。

## 主要技术路线
- **视频即仿真 (Video-as-Simulation)**：利用交互式视频预测器代替解析引擎，详见 [Video-as-Simulation](../concepts/video-as-simulation.md)。
- **扩散模型 (Diffusion-based)**：利用 DDPM 逐步去噪生成未来帧，代表：UniSim。
- **离散 Token 流 (Discrete Token flow)**：将图像量化为 Token，利用 Transformer 预测序列，代表：π₀ 的动作建模部分。
- **生成视频作为人形控制 demo 源**：把第三人称视频生成当成"想象出来的示教"，再用动作估计 + 通用动作跟踪把视频翻译为机器人动作，代表：[ExoActor](./exoactor.md)。

## 关联页面
- [Latent Imagination (潜空间想象)](../concepts/latent-imagination.md)
- [Model-Based RL](../methods/model-based-rl.md)
- [Being-H0.7](./being-h07.md) — 潜空间世界–动作模型，测试时不滚未来像素。
- [World Action Models（WAM）](../concepts/world-action-models.md) — 世界预测与动作生成的联合范式与文献taxonomy
- [NVIDIA Omniverse](../entities/nvidia-omniverse.md)
- [ExoActor](./exoactor.md) — 视频生成驱动的交互式人形控制。

## 参考来源
- Hu, A., et al. (2023). *GAIA-1: A Generative AI for Embodied AI*.
- Yang, S., et al. (2023). *Learning Interactive Real-World Simulators (UniSim)*.
- Zhou Y., et al. (2026). *ExoActor: Exocentric Video Generation as Generalizable Interactive Humanoid Control* — 见 [sources/papers/exoactor.md](../../sources/papers/exoactor.md)。
- Luo, H., et al. (2026). *Being-H0.7: A Latent World-Action Model from Egocentric Videos* — 见 [sources/papers/being_h07.md](../../sources/papers/being_h07.md)。
- Wang, S., et al. (2026). *World Action Models: The Next Frontier in Embodied AI* — 见 [sources/papers/world_action_models_survey_2605.md](../../sources/papers/world_action_models_survey_2605.md)。
