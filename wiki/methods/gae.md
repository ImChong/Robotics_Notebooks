---
type: method
tags: [rl, policy-optimization, math, optimization]
status: complete
updated: 2026-08-17
related:
  - ./policy-optimization.md
  - ./reinforcement-learning.md
  - ./ppo.md
  - ../formalizations/gae.md
  - ../formalizations/bellman-equation.md
  - ../concepts/rl-runner.md
  - ./intentional-updates-streaming-rl.md
  - ../queries/rl-hyperparameter-guide.md
sources:
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

## 主要技术路线

GAE 通过对不同时间跨度的 TD 误差进行加权平均来计算优势：

$$ \hat{A}_t^{GAE(\gamma, \lambda)} = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}^V $$

其中 $\delta_t^V = r_t + \gamma V(s_{t+1}) - V(s_t)$ 是单步 TD 残差。把连续若干个 $\delta$ 相加会望远镜抵消中间的 $V$，得到 n-step advantage；再对所有 $n$ 做 $\lambda$ 的指数加权，塌缩成上式。实现从轨迹末端往前递推：

$$\hat{A}_t = \delta_t + \gamma\lambda\,\hat{A}_{t+1}$$

- **$\lambda = 0$**：退化为单步 TD，低方差但高偏差（几乎全靠 critic 概括未来）。
- **$\lambda = 1$**：退化为蒙特卡洛（MC）采样，无偏但极高方差（中间 $V$ 全部抵消）。

**$\gamma$ 和 $\lambda$ 不要混：**

| 旋钮 | 管什么 | 运控常用 |
|------|--------|----------|
| $\gamma$ | 看多远的未来（有效视野约 $1/(1-\gamma)$ **步**） | $0.99$ |
| $\lambda$ | 多大程度信任 critic 来概括未来 | $0.95$ |

$\lambda$ 略小于 1，是用一点点偏差换明显更低的方差。人形/足式高维、长 horizon、奖励噪声大，尤其吃这个红利。$\gamma$ 本身还受控制频率影响：同样 $0.99$，50 Hz 大约看 2 s，200 Hz 只剩约 0.5 s——见 [PPO](./ppo.md) 与 [MDP](../formalizations/mdp.md)。

## 工程实践

- **截断 rollout + bootstrap：** On-policy Runner 通常只采 24–64 步，不是等整局结束。未终止片段用末状态 $V$ 补窗口外的未来；真正 `done` 则不 bootstrap。
- **优势归一化：** 每个 batch 内减均值除标准差，不改变「哪些动作更好」的排序，只稳住梯度尺度。不要去标准化 value 回归目标。
- **value loss 只诊断 critic。** critic 的 MSE 下降只说明「当前策略的价值估得更准」，不代表回报在涨。奖励黑客、actor 停滞、坏采样分布时，value loss 仍可收敛。真正该盯 episode return 与任务指标。
- 调参清单见 [RL 超参数指南](../queries/rl-hyperparameter-guide.md)。

## 局限与风险

- $\lambda$ 过小且 critic 还没学好时，偏差会顺着 bootstrap 传下去，早期训练会慢。
- $\lambda$ 过大则把后续几百步的摔倒运气算进当前动作的功劳，梯度噪声让 PPO clip 也救不回来。
- GAE 默认在**当前策略**的采样分布下定义，是 on-policy 的；把过期 rollout 上的 GAE 硬套到新策略上，正是 PPO 必须 clip 的原因。

## 关联页面

- [PPO](./ppo.md) — GAE 是 clip 目标里 $\hat{A}_t$ 的标准来源
- [GAE 形式化](../formalizations/gae.md) — TD 残差、递推与代码伪代码
- [Reinforcement Learning](./reinforcement-learning.md)
- [Policy Optimization](./policy-optimization.md)
- [RL Runner（训练循环编排）](../concepts/rl-runner.md) — On-policy 循环在丢掉 rollout 前用 GAE 算优势
- [Bellman 方程](../formalizations/bellman-equation.md)

## 参考来源

- [RobotsHub：万字解析运控 PPO](../../sources/blogs/wechat_robotshub_ppo_locomotion_fundamentals.md) — $\gamma$ vs $\lambda$、望远镜抵消、value loss 陷阱
- Schulman, J., et al. (2015). *High-Dimensional Continuous Control Using Generalized Advantage Estimation*.
- [sources/papers/intentional_streaming_rl.md](../../sources/papers/intentional_streaming_rl.md) — intentional TD($\lambda$) 与 GAE trace 几何

## 推荐继续阅读

- 原文：<https://mp.weixin.qq.com/s/MJQYYyOBSLirVr0vH1-AZg>
- Schulman et al., *GAE*（2015）— <https://arxiv.org/abs/1506.02438>
- [rsl_rl `compute_returns`](https://github.com/leggedrobotics/rsl_rl) — 反向递推 GAE 的工程实现
