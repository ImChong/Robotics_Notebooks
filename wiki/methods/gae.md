---
type: method
tags: [rl, policy-optimization, math, optimization]
status: complete
updated: 2026-11-07
related:
  - ./policy-optimization.md
  - ./reinforcement-learning.md
  - ./ppo.md
  - ../formalizations/gae.md
  - ../formalizations/bellman-equation.md
  - ../concepts/rl-runner.md
  - ./intentional-updates-streaming-rl.md
  - ../queries/rl-hyperparameter-guide.md
original:
  - ../../sources/papers/gae_reading_arxiv_1506_02438.md
sources:
  - ../../sources/papers/gae_reading_arxiv_1506_02438.md
  - ../../sources/papers/ppo_reading_arxiv_1707_06347.md
  - ../../sources/papers/policy_optimization.md
  - ../../sources/papers/intentional_streaming_rl.md
  - ../../sources/personal/rl_runner_types.md
  - ../../sources/blogs/wechat_robotshub_ppo_locomotion_fundamentals.md
summary: "广义优势估计（GAE）通过引入衰减因子 λ 在偏差与方差之间进行权衡，是目前 PPO 等主流 Policy Gradient 算法中计算优势函数的标准方法。"
---

# Generalized Advantage Estimation (GAE)

**GAE** 解决了强化学习中一个核心痛点：如何准确估计一个动作比平均水平“好多少”（即优势函数 $A(s, a)$），同时保持低方差。

## 一句话定义

GAE 把所有 n-step 优势做指数加权平均，用一根旋钮 $\lambda$ 在「一步 TD（稳、偏）」和「蒙特卡洛（准、吵）」之间滑动。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| GAE | Generalized Advantage Estimation | 多步 TD 误差的指数加权优势估计 |
| TD | Temporal Difference | 用 $r+\gamma V(s')$ 自举当前价值 |
| MC | Monte Carlo | 用整条真实回报估价值，无偏高方差 |
| PPO | Proximal Policy Optimization | 运控里最常用、默认配 GAE 的 on-policy 算法 |

## 为什么重要

在 [PPO](./ppo.md) 中使用 GAE 可以显著稳定训练过程。其优势估计的准确性直接决定了 [Bellman 方程](../formalizations/bellman-equation.md) 迭代中的梯度平滑度，使模型在面对长时域任务时更容易收敛。形式化展开见 [GAE 形式化页](../formalizations/gae.md)。

**与流式 RL 的交叉：** [Intentional Updates（流式 RL）](./intentional-updates-streaming-rl.md) 在 **TD($\lambda$) + eligibility traces** 设定下，把「意图更新」写成对 **近期多状态预测折扣 RMS 变化** 与 $|\delta_t|$ 成比例——trace 几何必须与 GAE 的多步信用分配一致，否则 naive 用 $\mathbf{z}_t$ 范数归一化会导致 trace 变长时更新反而缩小。读 GAE 时若关心 **batch=1、无 replay** 的在线设定，应连同 intentional TD($\lambda$) 一并理解。

## 核心原理

GAE 由 Schulman 等人在 *High-Dimensional Continuous Control Using Generalized Advantage Estimation*（ICLR 2016，arXiv:1506.02438）提出。论文把问题写成**无折扣**的总回报最大化，$\gamma$ 在这里不是问题定义，而是降低方差的算法参数；策略梯度写成 $g=\mathbb{E}\big[\sum_t \Psi_t \nabla_\theta\log\pi_\theta(a_t|s_t)\big]$，$\Psi_t$ 取优势函数 $A^\pi$ 时方差近似最低。

## 主要技术路线

GAE 通过对不同时间跨度的 TD 误差进行加权平均来计算优势：

$$ \hat{A}_t^{GAE(\gamma, \lambda)} = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}^V $$

其中 $\delta_t^V = r_t + \gamma V(s_{t+1}) - V(s_t)$ 是单步 TD 残差。把连续若干个 $\delta$ 相加会望远镜抵消中间的 $V$，得到 n-step advantage；再对所有 $n$ 做 $\lambda$ 的指数加权，塌缩成上式。实现从轨迹末端往前递推：

$$\hat{A}_t = \delta_t + \gamma\lambda\,\hat{A}_{t+1}$$

- **$\lambda = 0$**：退化为单步 TD，低方差但高偏差（几乎全靠 critic 概括未来）。
- **$\lambda = 1$**：退化为「折扣回报减 $V(s_t)$」，即蒙特卡洛（MC）采样加基线（中间 $V$ 全部抵消）；论文指出它对 $V$ 的精度不敏感（对折扣梯度无偏），但方差高。

**$\gamma$ 和 $\lambda$ 不要混：**

| 旋钮 | 管什么 | 运控常用 |
|------|--------|----------|
| $\gamma$ | 看多远的未来（有效视野约 $1/(1-\gamma)$ **步**；$\gamma<1$ 一律引入偏差，与 $V$ 准不准无关） | $0.99$ |
| $\lambda$ | 多大程度信任 critic 来概括未来（把噪声再截断到约 $1/(1-\gamma\lambda)$ 步；$\lambda<1$ 只在 $V$ 不准时才引入偏差） | $0.95$ |

$\lambda$ 略小于 1，是用一点点偏差换明显更低的方差。论文实验中最佳取值落在 $\lambda\in[0.9,0.99]$ 的中间区间（见下文），$\lambda$ 的最优值通常比 $\gamma$ 的小，原文推测是因为 $\lambda$ 比 $\gamma$ 引入的偏差少。人形/足式高维、长 horizon、奖励噪声大，尤其吃这个红利（这一句是工程经验，非论文结论）。$\gamma$ 本身还受控制频率影响：同样 $0.99$，50 Hz 大约看 2 s，200 Hz 只剩约 0.5 s——见 [PPO](./ppo.md) 与 [MDP](../formalizations/mdp.md)。

### 论文里的另外三个结论

1. **奖励塑形解读：** 令势函数 $\Phi=V$ 做塑形 $\tilde r=r+\gamma\Phi(s')-\Phi(s)$，塑形后的奖励恰是 TD 残差 $\delta^V$，于是 $\sum_l(\gamma\lambda)^l\tilde r_{t+l}$ 正是 GAE：用 $V$ 把"延迟的因果"压缩到近乎即时，再用更陡的折扣 $\gamma\lambda$ 截断远处噪声。
2. **价值函数也用信赖域：** 论文把价值回归写成带 KL 型约束的问题（约束用 Gauss-Newton / Fisher 矩阵，共轭梯度求解），避免对最近一批数据过拟合；目标是蒙特卡洛回报 $\hat V_t=\sum_l\gamma^l r_{t+l}$。
3. **更新顺序：** 算法先用旧的 $V_{\phi_i}$ 算优势并更新策略，**再**更新价值函数；否则 $V$ 过拟合会令残差趋零、梯度估计也趋零。

### 论文实验与原文数值

| 任务 | 设置 | 结论 |
|------|------|------|
| cart-pole | 线性策略，每批 20 条轨迹、最长 1000 步，21 个种子 | 最佳 $\gamma\in[0.96,0.99]$、$\lambda\in[0.92,0.99]$ |
| 3D 双足行走 | 33 维状态、10 个执行器，每批 50000 步，9 个种子 | 最佳 $\gamma\in[0.99,0.995]$、$\lambda\in[0.96,0.99]$；1000 次迭代折合约 5.8 天实时 |
| 四足 / 双足起身 | 29 维状态、8 个执行器；批 200000 步 | 四足最好用价值函数配 $\lambda=0.96$；起身任务价值函数总有帮助 |

网络均为 100-50-25 的 tanh MLP，TRPO 更新策略；原文中的 $\lambda=0$（一步回报）"偏差过大、效果差"。

## 工程实践

- **截断 rollout + bootstrap：** On-policy Runner 通常只采 24–64 步，不是等整局结束。未终止片段用末状态 $V$ 补窗口外的未来；真正 `done` 则不 bootstrap。
- **优势归一化：** 每个 batch 内减均值除标准差，不改变「哪些动作更好」的排序，只稳住梯度尺度。不要去标准化 value 回归目标。
- **value loss 只诊断 critic。** critic 的 MSE 下降只说明「当前策略的价值估得更准」，不代表回报在涨。奖励黑客、actor 停滞、坏采样分布时，value loss 仍可收敛。真正该盯 episode return 与任务指标。
- 调参清单见 [RL 超参数指南](../queries/rl-hyperparameter-guide.md)。
- **PPO 里的取值：** PPO 论文 MuJoCo 实验取 $\gamma=0.99$、$\lambda=0.95$，并使用截断 GAE（只向前看 $T$ 步，窗口外用 $V(s_{t+T})$ 补）；见 [PPO 论文归档](../../sources/papers/ppo_reading_arxiv_1707_06347.md)。

## 局限与风险

- $\lambda$ 过小且 critic 还没学好时，偏差会顺着 bootstrap 传下去，早期训练会慢。
- $\lambda$ 过大则把后续几百步的摔倒运气算进当前动作的功劳，梯度噪声让 PPO clip 也救不回来。
- GAE 默认在**当前策略**的采样分布下定义，是 on-policy 的；把过期 rollout 上的 GAE 硬套到新策略上，优势估计就不再对应新策略（PPO 对同一批数据做多轮更新并用 clip 限制偏离，是针对同一类问题的约束，但 clip 的直接动机是限制策略步长）。
- 论文自己承认的开放问题：$\gamma,\lambda$ 需要手动选且没有自适应方案；价值函数误差与策略梯度误差之间的关系没有理论刻画；策略与价值共享网络时的收敛性未解决。
- 论文的实验是仿真器内的小规模/中规模任务（MuJoCo 双足 / 四足），不含真机，也不含 4096 并行环境那样的现代设置。

## 关联页面

- [PPO](./ppo.md) — GAE 是 clip 目标里 $\hat{A}_t$ 的标准来源
- [GAE 形式化](../formalizations/gae.md) — TD 残差、递推与代码伪代码
- [Reinforcement Learning](./reinforcement-learning.md)
- [Policy Optimization](./policy-optimization.md)
- [RL Runner（训练循环编排）](../concepts/rl-runner.md) — On-policy 循环在丢掉 rollout 前用 GAE 算优势
- [Bellman 方程](../formalizations/bellman-equation.md)

## 参考来源

- [RobotsHub：万字解析运控 PPO](../../sources/blogs/wechat_robotshub_ppo_locomotion_fundamentals.md) — $\gamma$ vs $\lambda$、望远镜抵消、value loss 陷阱
- [GAE 论文归档（原文通读）](../../sources/papers/gae_reading_arxiv_1506_02438.md)
- Schulman, J., Moritz, P., Levine, S., Jordan, M. I., Abbeel, P. (2016). *High-Dimensional Continuous Control Using Generalized Advantage Estimation*. ICLR.
- [sources/papers/intentional_streaming_rl.md](../../sources/papers/intentional_streaming_rl.md) — intentional TD($\lambda$) 与 GAE trace 几何

## 推荐继续阅读

- 原文：<https://mp.weixin.qq.com/s/MJQYYyOBSLirVr0vH1-AZg>
- Schulman et al., *GAE*（2015）— <https://arxiv.org/abs/1506.02438>
- [rsl_rl `compute_returns`](https://github.com/leggedrobotics/rsl_rl) — 反向递推 GAE 的工程实现
