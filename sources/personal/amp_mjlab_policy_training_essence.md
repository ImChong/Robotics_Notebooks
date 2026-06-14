# AMP_mjlab Policy 训练本质 FAQ（维护者整理）

- **类型**：`personal`（对话/答疑整理，非正式出版物）
- **日期**：2026-06-14
- **原始对话**（仅作溯源，不在 wiki 建独立节点）：
  - AMP_mjlab Policy 训练解析：<https://chatgpt.com/share/6a2ed607-5584-83ea-907f-55750a58d997>
- **用途**：为 [神经反馈控制器（RL Policy）](../../wiki/concepts/neural-feedback-controller.md)、[AMP_mjlab](../../wiki/entities/amp-mjlab.md)、[PPO](../../wiki/methods/ppo.md) 提供可追溯编译来源；正文以 wiki 页为准。

## 对话要点：Policy 训练产物与更新本质

### 训练产物是什么

- 最终得到 `model.pt` 或 `policy.onnx`：本质是 **MLP 神经网络**，实现 $a_t = \pi(o_t)$。
- 输入：机身状态、关节角/速、速度指令、历史堆叠等；输出：关节位置目标增量（如 G1 的 29 维）。
- **不是** 学会一条固定动作轨迹，而是学会 **状态 → 动作** 的映射规律：同一网络在不同速度指令下输出走/跑/起身等不同行为。

### AMP 在学什么

- **老师 1（任务奖励）**：速度跟踪、存活、能耗、足端打滑惩罚等。
- **老师 2（AMP 判别器）**：policy 状态转移是否像 mocap 参考动作（对抗训练，类似 GAN）。
- 数百万步仿真经验 + 参考风格 + 奖励要求，被 **压缩进几十 MB 的权重**。

### 控制论视角

- 传统：状态 → MPC/QP → 关节力矩；AMP：**状态 → 神经网络 → 关节目标**。
- 与 $u = Kx$（PID/LQR）类比：AMP policy 是 **高维非线性状态反馈控制器**（Neural Feedback Controller），参数由仿真自动学习而非人工设计。

### PPO 参数更新在干什么

- Policy 输出的是 **动作分布** $\pi(a|s)$，不是确定性动作。
- 优势 $A_t > 0$ 的动作 → 提高该动作概率；$A_t < 0$ → 降低。
- 参数更新本质：**强化导致高奖励的状态→动作神经连接，削弱导致失败/低奖励的连接**。
- 梯度 $\nabla_\theta J(\theta)$ 指向 reward 增长最快的方向；每一步是在浮点权重上做极小扰动（如 $0.513241 \to 0.513268$）。

### 推理算力（参数量 ≠ FLOPs）

- 典型 AMP/locomotion MLP：`obs(705) → 1024 → 512 → 256 → action(23~29)`，单次前向约 **1–5M FLOPs**（经验：MLP 前向 FLOPs ≈ $2 \times$ 参数量）。
- 50MB ONNX（FP32）≈ 13M 参数 ≈ 26M FLOPs/次；@100Hz ≈ **2.6 GFLOPS/s**。
- 对比：RK3588 轻松；Orin 上 policy 推理通常 **< 整机算力 1%**；瓶颈多在视觉/VLA/点云，而非低层 policy。

### 对 wiki 的映射

| 要点 | 目标页 |
|------|--------|
| Policy = 神经反馈控制器、经验压缩、AMP 双老师 | `wiki/concepts/neural-feedback-controller.md` |
| TensorBoard 曲线、训练命令、部署细节 | `wiki/entities/amp-mjlab.md`（已有，补交叉引用） |
| PPO clip/GAE 形式化 | `wiki/methods/ppo.md` |
| 网络层宽与架构代际 | `wiki/concepts/humanoid-policy-network-architecture.md` |
