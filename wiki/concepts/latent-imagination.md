---
type: concept
tags: [world-models, reinforcement-learning, machine-learning, model-based-rl]
status: complete
updated: 2026-09-02
related:
  - ../entities/paper-odeworld.md
  - ./rl-runner.md
  - ../methods/model-based-rl.md
  - ../methods/generative-world-models.md
  - ../methods/being-h07.md
  - ../formalizations/variational-objective.md
  - ../entities/paper-shenlan-wm-13-dreamerv3.md
  - ../entities/open-dreamer.md
  - ../overview/world-models-route-03-virtual-sandbox.md
  - ../entities/paper-online-mbrl-robot-control.md
  - ../entities/paper-lucid.md
  - ../entities/paper-lawa.md
  - ../entities/paper-rise-adaptive-imagination-wam.md
sources:
  - ../../sources/personal/rl_runner_types.md
  - ../../sources/papers/rl_foundation_models.md
  - ../../sources/papers/being_h07.md
  - ../../sources/repos/open-dreamer.md
  - ../../sources/papers/online_mbrl_robot_control_arxiv_2510_18518.md
  - ../../sources/papers/lucid_arxiv_2608_07746.md
  - ../../sources/papers/odeworld_arxiv_2607_27924.md
  - ../../sources/papers/rise_adaptive_imagination_arxiv_2608_20430.md
summary: "潜空间想象（Latent Imagination）是 Model-Based RL 的核心技术，通过在紧凑的隐变量空间中预测未来状态，使智能体能够在无需真实环境交互的情况下进行无限次自我博弈与策略优化。"
---

# Latent Imagination (潜空间想象)

**潜空间想象 (Latent Imagination)** 是现代 Model-Based 强化学习（尤其是 **Dreamer** 系列）的灵魂。它彻底改变了机器人学习的范式：不再是在真实世界或沉重的物理仿真器中反复试错，而是在一个完全由数据学出来的“脑内模型”中进行极其高速的并行进化。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略的范式 |
| Dreamer | Dreamer (World Models) | 在潜空间想象中训练的 MBRL 代表 |
| Reward | Reward Function | 塑造强化学习策略行为的标量反馈 |
| DreamerV3 | Dreamer version 3 | 在潜空间想象中训练、单一超参跨 150+ 任务通用的世界模型智能体 |

## 核心工作原理

潜空间想象通常建立在 **RSSM (Recurrent State Space Model)** 之上。其流程分为“梦境构建”和“梦中训练”两个阶段：

### 1. 构建世界模型（学习“梦境”的法则）
智能体通过真实的交互数据训练一个世界模型，包含：
- **Transition Model**：预测下一步的潜状态 $z_{t+1}$。
- **Observation Model**：从潜状态重建图像或感知。
- **Reward Model**：预测每一步的即时奖励。

### 2. 潜空间展开（在“梦境”中航行）
一旦模型训练完成，智能体就可以从任意起始状态出发，完全脱离外部环境输入，利用 Transition Model 在潜空间中向未来展开 $H$ 步（Horizon）：
$$ \hat{z}_{t+1}, \hat{z}_{t+2}, \dots, \hat{z}_{t+H} $$

### 3. 策略优化（在“梦境”中进化）
Actor-Critic 策略直接在这条“想象轨迹”上运行：
- **Actor** 输出动作，使模型预测的奖励最大化。
- **Critic** 学习评估想象状态的长期价值。
- 由于一切都在向量化的潜空间进行，其速度比物理仿真快 100-1000 倍。

## 为什么它对机器人至关重要

1. **样本效率 (Sample Efficiency)**：真实机器人交互极其昂贵（硬件损耗、时间）。潜空间想象将 1 小时的真实数据“压榨”出相当于数千小时的虚拟训练经验。
2. **处理高维观测**：直接在像素级预测未来极其困难且不平滑。在紧凑的潜空间（Latent Space）中想象，可以自动过滤掉背景噪声，只保留对任务关键的物理特征。
3. **安全避障**：智能体可以在脑海中预演“如果我这样跨步会跌倒”，从而在真实动作执行前规避高风险行为。

## 代表算法
- **Dreamer V1-V3**：将潜空间想象推向通用人工智能（Atari 到机器人控制）的巅峰；策展页见 [DreamerV3](../entities/paper-shenlan-wm-13-dreamerv3.md)。
- **Dreamer 4 / Open Dreamer**：把可扩展交互式视频世界模型推到 Minecraft 级；开源 JAX 复现见 [Open Dreamer](../entities/open-dreamer.md)（完整「模型内训 agent」环仍在 roadmap）。
- **DayDreamer**：证明了该技术可以直接在真实机械臂上几小时内从零学出抓取，无需任何仿真。
- **Being-H0.7**：面向语言–视觉–操作策略的 **latent world–action** 路线——用 egocentric 人视频与机演示，在训练期用未来观测分支对齐潜空间，部署时不依赖像素 rollout；见 [Being-H0.7](../methods/being-h07.md)。
- **LUCID**：在 **技能级宏状态**（非像素 RSSM）上想象；冻结结构化 latent LLC，用 macro-dynamics WM 训 HLC 做长时程重排；见 [LUCID](../entities/paper-lucid.md)。
- **ODEWorld**：把离散 RSSM 步换成 **物理时间 ODE**——在解耦后的单 token 动力学 latent 上积分 \(v_\theta\)，任意 \(\tau\) / 反向预测，再把 \(z_\tau\) 当策略子目标；见 [ODEWorld](../entities/paper-odeworld.md)。不是 Dreamer 式「梦中 actor-critic」，而是连续时间潜空间展开。
- **RISE（酷哇，驾驶 WAM）**：测试时在 V-JEPA latent 前缀上逐步 Roll/Stop，用规划增益对代价，而不是固定 horizon 或 Dreamer 式梦中 actor-critic；见 [RISE 自适应想象](../entities/paper-rise-adaptive-imagination-wam.md)。勿与 OpenDriveLab 同名操作 RISE 混淆。

## 关联页面
- [Model-Based RL](../methods/model-based-rl.md)
- [RL Runner（训练循环编排）](./rl-runner.md) — Model-based Runner 把想象 rollout 嵌进训练循环
- [Generative World Models](../methods/generative-world-models.md)
- [Being-H0.7](../methods/being-h07.md)
- [LAWA](../entities/paper-lawa.md) — 测试时去噪紧凑 latent 意图而非像素（代码待发布）
- [DreamerV3](../entities/paper-shenlan-wm-13-dreamerv3.md)
- [Open Dreamer](../entities/open-dreamer.md)
- [LUCID](../entities/paper-lucid.md) — 人形技能级宏动力学想象
- [ODEWorld](../entities/paper-odeworld.md) — 物理时间 latent ODE 展开 + 子目标引导
- [RISE（酷哇 · 驾驶 WAM 自适应想象）](../entities/paper-rise-adaptive-imagination-wam.md) — 测试时 Roll/Stop，非梦中 RL
- [虚拟沙盒路线](../overview/world-models-route-03-virtual-sandbox.md)
- [变分目标函数 (ELBO)](../formalizations/variational-objective.md)
- [具身大模型分类学选型闭环（知识链枢纽）](../overview/hub-embodied-foundation-model.md) — 潜空间想象属五层闭环的世界模型推演层
- [Online MBRL via Online Optimization](../entities/paper-online-mbrl-robot-control.md) — 对照：策略在**真实轨迹**上更新，模型只供 Jacobian，而非想象 actor-critic

## 参考来源
- Hafner, D., et al. (2019). *Dream to Control: Learning Behaviors by Latent Imagination*.
- Hafner, D., et al. (2023). *Mastering Diverse Domains through World Models (DreamerV3)*.
- Hafner, D., Yan, W., & Lillicrap, T. (2025). *Training Agents Inside of Scalable World Models (Dreamer 4)* — <https://arxiv.org/abs/2509.24527>；开源复现归档 [sources/repos/open-dreamer.md](../../sources/repos/open-dreamer.md)。
- Luo, H., et al. (2026). *Being-H0.7: A Latent World-Action Model from Egocentric Videos* — 项目页 <https://research.beingbeyond.com/being-h07>；归档见 [sources/papers/being_h07.md](../../sources/papers/being_h07.md)。
- [Online MBRL 论文归档（真机一阶对照）](../../sources/papers/online_mbrl_robot_control_arxiv_2510_18518.md)
- [RISE 自适应想象（驾驶 WAM Roll/Stop）](../../sources/papers/rise_adaptive_imagination_arxiv_2608_20430.md)
- [lucid_arxiv_2608_07746.md](../../sources/papers/lucid_arxiv_2608_07746.md) — LUCID 技能级想象控制
- [odeworld_arxiv_2607_27924.md](../../sources/papers/odeworld_arxiv_2607_27924.md) — PT-Flow 连续时间潜空间预测
