---
type: method
tags: [locomotion, diffusion, generative-model, humanoid]
status: complete
related:
  - ../entities/kimodo.md
summary: "利用扩散模型生成机器人全身运动序列，通过闭环微调解决分布偏移，实现复杂地形下的实时运动规划。"
---

# Diffusion-based Motion Generation (基于扩散模型的运动生成)

**Diffusion-based Motion Generation**：利用扩散概率模型（Diffusion Probabilistic Models）生成机器人关节空间或笛卡尔空间的连续运动序列，通常作为分层控制架构中的高层参考规划器。

## 一句话定义

利用扩散模型的去噪过程预测未来一段时间内的全身运动轨迹，为底层控制器提供具备地形感知能力的高质量参考动作。

## 核心原理

基于扩散的运动生成将运动预测视为一个去噪过程。给定当前状态 $s_t$ 和条件项 $c$（如地形信息、导航目标），模型学习从随机噪声中恢复出一段未来运动轨迹 $\tau = \{q_{t+1}, \dots, q_{t+H}\}$。

### 主要特点
1. **多峰分布建模**：能够捕获人类运动的多样性，解决传统确定性模型在复杂决策点的“均值平滑”问题。
2. **长程一致性**：相比于逐帧预测，轨迹生成能够保证运动在时间窗口内的物理连贯性。
3. **条件约束**：可以轻松整合地形图（Elevation Maps）、任务目标或文本指令作为生成条件。

## 主要技术路线 (以 ETH G1 为例)

1. **输入表示**：
   - 历史本体感受状态（Proprioception History）。
   - 目标指令（Yaw velocity, Heading）。
   - 局部地形扫描（Local Elevation Map）。
2. **扩散生成架构**：
   - 采用 1D CNN 或 Transformer 结构的骨干网。
   - 预测未来约 0.5s - 1.0s 的全身参考姿态。
3. **闭环微调（Closed-loop Fine-tuning）**：
   - 在仿真环境中，使底层 RL 策略在扩散生成器的“实时指导”下进行演练，学习适应生成器的噪声，解决分布偏移（Distribution Mismatch）问题。

## 关键挑战与解决方案

### 1. 分布偏移
- **问题**：扩散模型在离线数据上训练，但底层跟踪器（Tracking Controller）在执行时产生的细微偏差会导致生成器进入未见过的状态空间。
- **方案**：闭环微调，将生成器集成进仿真训练循环。

### 2. 推理延迟
- **问题**：扩散模型迭代次数多，计算量大。
- **方案**：
  - **收缩时界（Receding-horizon）更新**：异步触发推理。
  - **加速库**：使用 NVIDIA TensorRT 进行模型量化与推理加速。

## 参考来源
- [sources/papers/eth-g1-diffusion.md](../../sources/papers/eth-g1-diffusion.md) — ETH Zurich 2026 G1 扩散运动生成工作，结合扩散模型与 RL 跟踪器实现全身移动。
- [sources/repos/kimodo.md](../../sources/repos/kimodo.md) — Kimodo：大规模动捕上训练的运动扩散模型与约束式生成工具链（SOMA / G1 / SMPL-X）。
- [sources/papers/genmo.md](../../sources/papers/genmo.md) — GENMO（ICCV 2025 Highlight，NVIDIA）：把人体运动估计形式化为带观测约束的扩散生成，dual-mode 训练统一估计 + 生成。
- [sources/repos/zilize-awesome-text-to-motion.md](../../sources/repos/zilize-awesome-text-to-motion.md) — Zilize 维护的文本驱动人体运动生成综述/数据集/模型精选与交互式项目页索引。
- [Diffusion Policy](./diffusion-policy.md) — 扩散策略在操作任务中的应用。
- [GENMO（统一人体运动估计与生成）](./genmo.md) — 人体运动域的扩散生成代表实现，与机器人控制域的扩散运动生成相互参照（估计 ↔ 生成的双向收益）。

## 关联页面
- [Kimodo（实体页）](../entities/kimodo.md) — 文本 + 运动学约束的人形/人体运动扩散官方实现
- [Awesome Text-to-Motion（Zilize）](../entities/awesome-text-to-motion-zilize.md) — 人体文本–运动文献与数据集的 curated 入口（单人、无 HOI）
- [Humanoid Locomotion](../tasks/humanoid-locomotion.md)
- [Motion Retargeting](../methods/motion-retargeting-gmr.md)
- [PPO](./policy-optimization.md)
- [Probability Flow](../formalizations/probability-flow.md) — 扩散模型的数学基础
- [Contact Dynamics](../concepts/contact-dynamics.md)
