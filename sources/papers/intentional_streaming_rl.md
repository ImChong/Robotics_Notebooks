# intentional_streaming_rl

> 来源归档（ingest）

- **标题：** Intentional Updates for Streaming Reinforcement Learning
- **类型：** paper
- **来源：** arXiv:2604.19033v1；代码 <https://github.com/sharifnassab/Intentional_RL>
- **入库日期：** 2026-05-10
- **一句话说明：** 在 batch=1、无 replay 的流式深度 RL 中，用「先指定输出空间上的意图变化、再反解步长」稳定每步更新；给出 Intentional TD / Q / Policy Gradient 及 eligibility trace + 对角预条件的实现，实证上常接近批式与 replay baseline。

## 核心论文摘录（MVP）

### 1) 问题：参数步长 ≠ 可预测的函数输出变化（流式脆弱性）

- **链接：** <https://arxiv.org/abs/2604.19033>（PDF：<https://arxiv.org/pdf/2604.19033v1>）
- **核心贡献：** 固定学习率在参数空间缩放梯度，并不能控制价值或策略在**函数输出**上的每步变化幅度；在流式设定（batch size 1、无 minibatch 平均）下，梯度方向与范数剧烈波动，易出现过大/过小更新。这与「stream barrier」现象一致：全流式深度 RL 往往比带 replay 的设定脆得多。
- **对 wiki 的映射：**
  - [Intentional Updates（流式 RL）](../../wiki/methods/intentional-updates-streaming-rl.md)
  - [Reinforcement Learning](../../wiki/methods/reinforcement-learning.md)
  - [Online vs Offline RL](../../wiki/comparisons/online-vs-offline-rl.md)

### 2) 原则：意图更新（intentional updates）与 NLMS 类比

- **链接：** 同上
- **核心贡献：** 先指定标量目标量 $y_t(\mathbf{w})$ 上希望达到的增量 $\Delta_t$（在意图单位而非参数单位），再用一阶近似 $\alpha_t = \Delta_t / (\nabla y_t^\top \mathbf{d}_t)$ 解出步长。监督线性回归中的 **NLMS（Normalized LMS）** 是先例：用步长把输出误差按固定比例压下去。本文将该思想推广到流式深度 RL 的价值学习与策略学习。
- **对 wiki 的映射：**
  - [Intentional Updates（流式 RL）](../../wiki/methods/intentional-updates-streaming-rl.md)
  - [Policy Optimization](../../wiki/methods/policy-optimization.md)

### 3) Intentional TD / Intentional Q-learning：控制 TD 误差的按比例收缩

- **链接：** 同上
- **核心贡献：** 对半梯度 TD(0)，令当前状态价值沿更新后近似满足 $V_{\mathbf{w}_{t+1}}(s_t) \approx V_{\mathbf{w}_t}(s_t) + \eta\,\delta_t$（$\eta\in(0,1]$），在一阶近似下得到与梯度范数平方成反比的步长；Q 学习对 $(s_t,a_t)$ 做同样构造。对 **TD($\lambda$) + eligibility traces**，意图必须写成**对近期多状态预测折扣 RMS 变化**与 $|\delta_t|$ 成比例，从而得到与 trace 几何一致的保守步长（避免 naive 用 $\mathbf{z}_t$ 范数归一化导致 trace 变长时更新反而缩小的问题）。
- **对 wiki 的映射：**
  - [Intentional Updates（流式 RL）](../../wiki/methods/intentional-updates-streaming-rl.md)
  - [GAE](../../wiki/methods/gae.md)（优势与多步信用分配语境）

### 4) Intentional Policy Gradient：用采样动作 log-prob 变化代理局部 KL

- **链接：** 同上
- **核心贡献：** 策略侧没有与 TD 同构的标量「误差」可压；改为控制每步在采样动作上 $\Delta\log\pi$ 的大小，使其与 advantage 成比例且由运行尺度 $\bar{A}_t$ 归一，小步下与状态条件 KL 的二阶展开相关，作为 **PPO/TRPO 式信任域约束** 的廉价流式代理。实现上与 **RMSProp 式对角缩放**、**eligibility traces** 组合，并讨论样本依赖步长对状态/动作重加权的影响。
- **对 wiki 的映射：**
  - [Policy Optimization](../../wiki/methods/policy-optimization.md)
  - [PPO vs SAC](../../wiki/comparisons/ppo-vs-sac.md)

### 5) 工程实现与实证范围

- **链接：** <https://github.com/sharifnassab/Intentional_RL>
- **核心贡献：** 官方代码实现 Algorithm 1–3（Intentional TD($\lambda$)、Intentional Q($\lambda$)、Intentional Policy Gradient）；实验报告在价值预测、连续控制与离散动作域上，**全流式**性能常可与批式 / replay 方法比肩。
- **对 wiki 的映射：**
  - [Intentional_RL 仓库](../repos/intentional_rl.md)
  - [Intentional Updates（流式 RL）](../../wiki/methods/intentional-updates-streaming-rl.md)

## 当前提炼状态

- [x] 摘要与算法主线已映射到 wiki 方法页
- [~] 若后续仓库 README 或实验配置有重大更新，可回链补充「复现实验」要点
