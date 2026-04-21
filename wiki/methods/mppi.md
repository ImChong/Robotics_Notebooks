---
type: method
tags: [control, optimization, reinforcement-learning, model-based-rl, mpc]
status: complete
updated: 2026-04-21
related:
  - ./model-based-rl.md
  - ./model-predictive-control.md
  - ../formalizations/variational-objective.md
sources:
  - ../../sources/papers/optimal_control.md
summary: "模型预测路径积分（MPPI）是一种基于样本的概率模型预测控制方法，通过海量并行轨迹采样与加权平均，实现了对非凸、非平滑动力学系统的高效控制。"
---

# Model Predictive Path Integral (MPPI)

**MPPI** 是一种基于采样（Sampling-based）的随机最优控制算法。它不依赖于对动力学方程进行求导（与 DDP/iLQR 不同），而是通过在 GPU 上并行生成成千上万条随机轨迹，并根据每条轨迹的代价（Cost）进行加权聚合来计算最优动作。

## 主要技术路线

MPPI 的核心是基于信息论的路径积分控制理论：

1. **随机采样**：在当前动作序列上叠加高斯噪声，生成大量备选动作序列。
2. **前向仿真 (Rollout)**：在模型（解析模型或神经网络）中并行推演这些序列，计算每条路径的累计代价 $S(\tau)$。
3. **指数加权平均**：使用 Softmax 形式的权重聚合动作：
   $$ u_t^* = \frac{\sum \exp(-\frac{1}{\lambda} S(\tau)) u_t(\tau)}{\sum \exp(-\frac{1}{\lambda} S(\tau))} $$
   其中 $\lambda$ 是温度参数，控制对低代价轨迹的聚焦程度。
4. **滚动优化**：仅执行第一个动作，然后进入下一循环。

## 为什么在机器人中流行

- **非微分友好**：可以处理带有硬接触、摩擦切换或逻辑判断的不可导模型。
- **天然并行**：极其适配 NVIDIA GPU 加速，可以在几毫秒内完成数万次 Rollout。
- **鲁棒性**：作为一种随机优化方法，它比基于梯度的法方更容易跳出局部最优。

## 关联页面
- [Model-Based RL](./model-based-rl.md)
- [Model Predictive Control (MPC)](./model-predictive-control.md)
- [变分目标函数](../formalizations/variational-objective.md)

## 参考来源
- Williams, G., et al. (2017). *Information-theoretic model predictive control: Theory and applications to autonomous driving*.
