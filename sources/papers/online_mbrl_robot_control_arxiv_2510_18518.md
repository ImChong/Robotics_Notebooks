# Efficient Model-Based Reinforcement Learning for Robot Control via Online Optimization（arXiv:2510.18518）

> 来源归档（ingest）

- **标题：** Efficient Model-Based Reinforcement Learning for Robot Control via Online Optimization
- **类型：** paper / model-based RL / online optimization / on-robot learning / hydraulic / soft robot
- **arXiv：** <https://arxiv.org/abs/2510.18518>（PDF：<https://arxiv.org/pdf/2510.18518.pdf>；v2，2026-05-06）
- **作者：** Fang Nan、Hao Ma、Qinghua Guan、Josie Hughes、Michael Muehlebach、Marco Hutter
- **机构：** 苏黎世联邦理工（ETH Zürich）Robotic Systems Lab；马克斯·普朗克智能系统研究所（MPI-IS）Learning and Dynamical Systems；ETH 动态系统与控制研究所；洛桑联邦理工学院（EPFL）CREATE Lab
- **项目页：** 无（截至 2026-08-11 公开检索未见独立项目页）
- **代码：** 无公开仓库（确认未开源）
- **入库日期：** 2026-08-11
- **最后更新：** 2026-08-11
- **一句话说明：** 用真机交互在线学动力学模型，再以模型 Jacobian 在**真实轨迹**上算近似策略梯度（预条件在线下降），把 MBRL 推到液压挖掘机臂与缆驱软臂的小时级真机训练。

## 开源状态（步骤 2.5，2026-08-11）

- **确认未开源：** arXiv abs/HTML 无 Code / Project 链接；GitHub / Papers with Code 检索无官方实现；补充材料入口亦未给出可运行仓。
- **复现边界：** 论文给出算法伪代码、超参（\(\alpha=0.01,\ \epsilon=0.05,\ \eta=0.5\)）与平台设定，但无可运行训练脚本或权重。

## 摘要级要点

- **动机：** sim-to-real + model-free（PPO 等）对液压/软体等难仿真系统成本高；既有 MBRL（Dreamer / TD-MPC）常依赖大量想象 rollout，计算成为真机在线瓶颈。
- **方法：** 每 episode 真机 rollout → 缓冲扩充 → MSE 更新动力学网 \(f_\theta\) → 用 \(f_\theta\) 的 Jacobian 在**真实轨迹**上构造闭环策略梯度（式 5）→ 预条件更新 \(\pi_\phi\)（式 6–7），避免在错误模型上长 horizon 想象。
- **理论：** 将模型学习与策略学习拆成两个随机在线优化问题，给出与梯度估计误差 / 分布漂移相关的 regret 界；强调策略正则化带来的时间尺度分离。
- **实验：** HEAP 液压挖掘机臂真机约 **2.5 h / 180 episode** 达均值跟踪误差 **2.7 cm**；同超参迁移缆驱软臂约 **30 episode**、均值误差 **2.95 cm**；相对 Egli & Hutter 2022 / Nan & Hutter 2024 在更高速度下 \(\rho\) 更优；负载切换可数 episode 内恢复。

## 核心论文摘录（MVP）

### 1) 真机轨迹上的模型引导一阶策略更新

- **链接：** §3.2 Algorithm 1；式 (3)(5)(6)(7)
- **摘录要点：** 模型只用于提供沿真实轨迹的局部 Jacobian，而非生成合成轨迹做 zeroth-order 优化；预条件 \(\Lambda_t\) 限制目标、动作与参数步长。
- **对 wiki 的映射：**
  - [Online MBRL via Online Optimization 实体页](../../wiki/entities/paper-online-mbrl-robot-control.md)
  - [Model-Based RL](../../wiki/methods/model-based-rl.md)

### 2) 对 Dreamer / TD-MPC2 的样本–计算对照

- **链接：** §4.1.1 Simulation；§5 Discussion
- **摘录要点：** HEAP 仿真中同交互预算下优于 TD-MPC2 / DreamerV3；作者归因于「只在真数据上做一阶更新、避免想象域差与采样规划方差」。
- **对 wiki 的映射：**
  - [TD-MPC2](../../wiki/entities/paper-td-mpc2.md)
  - [DreamerV3](../../wiki/entities/paper-shenlan-wm-13-dreamerv3.md)
  - [Latent Imagination](../../wiki/concepts/latent-imagination.md)

### 3) 难建模平台上的真机结果与局限

- **链接：** §4.1.2 / §4.2；§5.3 Limitations
- **摘录要点：** 液压臂与软臂同一超参套件；负载扰动可在线适应；局限为连续跟踪代价、接触/稀疏奖励任务需 latent 扩展，且理论未完全闭合模型–策略耦合。
- **对 wiki 的映射：**
  - [Sim2Real](../../wiki/concepts/sim2real.md) — 作为「绕开仿真」对照路线
  - [Robotic World Model（ETH RSL）](../../wiki/entities/robotic-world-model-eth-rsl.md) — 同实验室「学模型再想象训策略」对照

## BibTeX

```bibtex
@article{nan2025online_mbrl,
  title   = {Efficient Model-Based Reinforcement Learning for Robot Control via Online Optimization},
  author  = {Nan, Fang and Ma, Hao and Guan, Qinghua and Hughes, Josie and Muehlebach, Michael and Hutter, Marco},
  journal = {arXiv preprint arXiv:2510.18518},
  year    = {2025}
}
```

## 对 wiki 的映射

- 主实体页：[`wiki/entities/paper-online-mbrl-robot-control.md`](../../wiki/entities/paper-online-mbrl-robot-control.md)
- 方法页：[`wiki/methods/model-based-rl.md`](../../wiki/methods/model-based-rl.md)
- 对照：[`wiki/entities/paper-td-mpc2.md`](../../wiki/entities/paper-td-mpc2.md)、[`wiki/entities/paper-shenlan-wm-13-dreamerv3.md`](../../wiki/entities/paper-shenlan-wm-13-dreamerv3.md)、[`wiki/entities/robotic-world-model-eth-rsl.md`](../../wiki/entities/robotic-world-model-eth-rsl.md)
- 迁移语境：[`wiki/concepts/sim2real.md`](../../wiki/concepts/sim2real.md)
