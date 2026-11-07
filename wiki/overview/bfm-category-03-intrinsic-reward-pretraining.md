---
type: overview
tags: [bfm, behavior-foundation-model, category-hub, awesome-bfm-papers, intrinsic-reward]
status: complete
updated: 2026-11-07
summary: "具身智能研究室 BFM 41 篇专题 · 03 Intrinsic reward 预训练（5 篇）— 在尚无明确下游任务时，身体应先通过内在奖励积累何种可迁移探索经验？"
related:
  - ./bfm-41-papers-technology-map.md
  - ../concepts/behavior-foundation-model.md
  - ../../sources/blogs/wechat_embodied_ai_lab_bfm_41_papers_survey.md
  - ../../sources/papers/bfm_awesome_41_catalog.md
  - ./bfm-category-01-forward-backward-representation.md
  - ./bfm-category-02-goal-conditioned-learning.md
  - ./bfm-category-04-adaptation.md
  - ./bfm-category-05-hierarchical-control.md
  - ../entities/paper-bfm-26-aps.md
  - ../entities/paper-bfm-27-proto-rl.md
  - ../entities/paper-bfm-28-re3.md
  - ../entities/paper-bfm-29-rnd.md
  - ../entities/paper-bfm-30-diayn.md
original:
  - ../../sources/papers/bfm_awesome_aps_icml_2021.md
  - ../../sources/papers/bfm_awesome_proto_rl_icml_2021.md
  - ../../sources/papers/bfm_awesome_re3_icml_2020.md
  - ../../sources/papers/bfm_awesome_rnd_iclr_2019.md
  - ../../sources/papers/bfm_awesome_diayn_iclr_2018.md
sources:
  - ../../sources/blogs/wechat_embodied_ai_lab_bfm_41_papers_survey.md
  - ../../sources/papers/bfm_awesome_41_catalog.md
  - ../../sources/repos/awesome_bfm_papers.md
---

# BFM 分类 03：Intrinsic reward 预训练

> **分类页**：对应 [具身智能研究室 · BFM 41 篇专题](https://mp.weixin.qq.com/s/Ei32la_vo0UW9Y_QCAqB2g) 与 [awesome-bfm-papers](https://github.com/friedrichyuan/awesome-bfm-papers) 的 **03 Intrinsic reward 预训练** 分组；本页汇集该组 **5 篇** 论文的站内实体与 source 索引。总地图见 [BFM 技术地图](./bfm-41-papers-technology-map.md)。

## 一句话定义

**在没有下游任务奖励时，用"内在奖励"（新奇度、状态熵、技能可区分度）驱动探索，先把身体/智能体预训练成可迁移的探索经验，再用少量任务奖励微调。**本组 5 篇覆盖三类内在奖励：预测误差/新奇度（RND）、状态熵（RE3、Proto-RL）、技能互信息（DIAYN、APS）。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| BFM | Behavior Foundation Model | 大规模行为数据预训练的可复用全身行为先验 |
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略的范式 |
| RND | Random Network Distillation | 用预测误差作内在探索奖励的无监督手段 |
| DIAYN | Diversity Is All You Need | 以内在奖励发现多样技能的无监督预训练代表 |
| AMP | Adversarial Motion Prior | 用对抗判别约束状态转移接近专家运动分布的先验 |
| RE3 | Random Encoders for Efficient Exploration | 在固定随机编码器表示空间用 k 近邻估计状态熵作内在奖励 |
| Proto-RL | Reinforcement Learning with Prototypical Representations | 原型（聚类中心）表征 + 熵探索，任务无关预训练 |
| APS | Active Pretraining with Successor Features | 后继特征 + 非参数状态熵，最大化 $I(s;z)$ |
| SAC | Soft Actor-Critic | DIAYN 与多数内在奖励基线用的最大熵 off-policy 算法 |
| k-NN | k-Nearest Neighbors | RE3 / Proto-RL / APS 估计状态熵的非参数手段 |

## 核心问题（公众号分类）

在**尚无明确下游任务**时，身体应先通过内在奖励积累何种**可迁移探索经验**？

**代表工作（策展）：** APS、Proto-RL、RE3、RND、DIAYN

## 为什么重要

- 它们回答 [BFM](../concepts/behavior-foundation-model.md) 的"预训练阶段没有任务奖励怎么办"：RND 证明简单的预测误差奖励能在稀疏奖励游戏里持续探索；DIAYN / APS 把技能变成可复用的先验；RE3 / Proto-RL 说明在像素观测下"怎么度量新奇"可以靠固定随机编码器或原型来稳定。
- 与 [人形 RL 身体系统栈](./humanoid-rl-motion-control-body-system-stack.md) 的关系是**思想层面的前身**：后面的 BFM 工作常把"无监督探索 + 技能/表征预训练"当作前置模块。

## 核心原理（一手论文对照）

| 工作 | 内在奖励 / 目标（论文原式） | 关键设计 | 论文自报结果（节选） |
|------|----------------------------|----------|----------------------|
| **RND**（Burda 等，2018） | $i_t=\|\hat f(x;\theta)-f(x)\|^2$：预测固定随机目标网络的输出 | 目标网络固定且确定，避开"noisy-TV"（动力学预测会被随机转移吸引）；内在回报非 episodic、外在回报 episodic，用**双价值头** $V=V_E+V_I$；观测白化并裁剪到 [−5,5]，奖励按内在回报标准差归一 | Montezuma's Revenge：RND 8,152 vs PPO 2,497 vs 动力学基线 400（人类均值 4,753）；最佳 agent 常探索 24 个房间中的 22 个；Pitfall! 两者都拿不到正奖励 |
| **DIAYN**（Eysenbach 等，2018） | $\mathcal F=I(S;Z)+H[A\mid S]-I(A;Z\mid S)=H[Z]-H[Z\mid S]+H[A\mid S,Z]$；伪奖励 $r_z=\log q_\phi(z\mid s)-\log p(z)$ | 技能由**状态**而非动作区分；先验 $p(z)$ 固定为均匀（避免 VIC 的"马太效应"）；用 SAC（熵系数 $\alpha=0.1$）；合作博弈而非对抗 | 无奖励即学会 half-cheetah / hopper / ant（111 维）的奔跑、跳跃、翻转等；技能可作初始化、分层 RL 的元控制器（ant 导航每 100 步、cheetah 跨栏每 10 步选一个技能）、以及对专家状态轨迹做 M-投影式模仿 |
| **RE3**（Seo 等，2021） | $r^i(s_i)=\log(\|y_i-y_i^{k\text{-NN}}\|_2+1)$，$y=f_\theta(s)$ 来自**固定随机初始化编码器** | 不训练额外模型/表示；$r^{total}=r^e+\beta_t r^i$，$\beta_t=\beta_0(1-\rho)^t$ 衰减；距离到整个 replay buffer 的 k-NN（$k=3$） | DMC：RAD+RE3 在 Cheetah Run Sparse 平均回报 601.6（RAD、DrQ 失败）；MiniGrid DoorKey-8x8：A2C+RE3 回报 0.49，ICM 0.20，RND 与 A2C 失败；无监督预训练仅 500K 步（APT 报告为 5M 步） |
| **Proto-RL**（Yarats 等，2021） | 熵奖励 $\hat r_t=\|z_{t+1}-NN_{k,Q}(z_{t+1})\|$，$Q$ 为按原型聚类均匀采样的候选队列 | 编码器 + $M=512$ 个 128 维原型，用 SwAV 式 Sinkhorn-Knopp 均衡聚类 + EMA 目标网络学习；RL 梯度不回传编码器；下游冻结编码器与原型，奖励加 $\alpha\hat r_t$（$\alpha=0.2$） | 500K 任务无关预训练 + 500K 下游：在 8 个 DMC 环境中 7/8 优于 Curiosity、6/8 优于训练 1M 步的 DrQ；多任务泛化与稀疏奖励下游探索更好 |
| **APS**（Liu & Abbeel，2021） | $I(s;z)=H(s)-H(s\mid z)$；$r_{APS}=\phi(s)^\top w+\log\big(1+\tfrac1k\sum\|h-h^{(j)}\|\big)$（利用项 + 探索项） | 把 VISR 的后继特征（下游快速适配）与 APT 的非参数粒子熵（探索）结合；$z\equiv w$，用 Von Mises-Fisher 分布；测试期用线性回归从少量 rollouts 推断任务向量 $w$ | Atari-26（10 万步）中位人类归一化得分 58.8（APT 47.5、SPR 41.5、VISR 12.36），8 款游戏超人；Atari-57：15 款超人（此前最多 12） |

**读法：** RND 与 RE3 的奖励只回答"这个状态新不新"；DIAYN 与 APS 还给出"哪个技能"这一**可条件化的接口**；Proto-RL 则同时产出一个可复用的**表征空间**。三者在 BFM 语境里分别对应"探索数据""技能库""表征底座"。

## 工程实践

- **选奖励：** 观测低维、动力学确定且想要技能库 → DIAYN；像素观测且不想额外训练模型 → RE3；需要下游快速适配、可接受后继特征 → APS；要同时学表征 → Proto-RL；游戏式稀疏奖励的局部探索 → RND。
- **内外在奖励怎么混：** RND 用双价值头（允许不同折扣与 episodic/非 episodic）；RE3 用衰减系数 $\beta_t$（论文发现衰减能稳定性能）；Proto-RL 下游用固定系数 $\alpha$ 并在稀疏奖励任务上有明显收益。
- **规模与预算：** RND 大多数实验约 19.7 亿帧、128 个并行环境；Proto-RL 与 RE3 的协议是数十万步的任务无关预训练再微调；APS 的 Atari 协议是 2.5 亿步无监督 + 10 万步有监督。
- **尚需自己验证的部分：** 本组论文全部在 Atari、DeepMind Control、MiniGrid 或 MuJoCo 类仿真上评测，**没有一篇涉及人形或真机**；把这些奖励搬到人形全身控制（高维动作、接触、摔倒代价）需要另外验证（推测：分层或先验约束是必要的，不在这些论文中）。

## 局限与风险

- **论文自述的局限：** RND 只解决**局部探索**，需要长时程协同决策（如先省下钥匙再开门）的全局探索"超出方法能力"；预测误差类奖励还受随机转移（noisy-TV）与模型误设影响（RND 用确定性目标回避，但不保证所有任务）。RE3 论文强调并非否定表示学习的必要性，对视觉更复杂的域，预训练固定表示可能更有用；APS 的伪代码里 $\phi$ 与编码器共享权重是经验选择。
- **评测口径不统一：** 五篇的基准、步数与基线各不相同（Atari / DMC / MiniGrid / MuJoCo），上表数字不能跨行横向比较。
- **BFM 归类是综述视角：** 这五篇在原文中并不自称"行为基础模型"，把它们放入 BFM 分类来自具身智能研究室的策展叙事，不是原作者的表述。

## 本组论文（5 篇）

| # | 工作 | Wiki 实体 | Source |
|---|------|-----------|--------|
| 26 | Active Pretraining with Successor Features | [paper-bfm-26-aps.md](../entities/paper-bfm-26-aps.md) | [source](../../sources/papers/bfm_awesome_aps_icml_2021.md) |
| 27 | Reinforcement Learning with Prototypical Representations | [paper-bfm-27-proto-rl.md](../entities/paper-bfm-27-proto-rl.md) | [source](../../sources/papers/bfm_awesome_proto_rl_icml_2021.md) |
| 28 | State Entropy Maximization with Random Encoders for Efficient Exploration | [paper-bfm-28-re3.md](../entities/paper-bfm-28-re3.md) | [source](../../sources/papers/bfm_awesome_re3_icml_2020.md) |
| 29 | Exploration by Random Network Distillation | [paper-bfm-29-rnd.md](../entities/paper-bfm-29-rnd.md) | [source](../../sources/papers/bfm_awesome_rnd_iclr_2019.md) |
| 30 | Diversity is All You Need | [paper-bfm-30-diayn.md](../entities/paper-bfm-30-diayn.md) | [source](../../sources/papers/bfm_awesome_diayn_iclr_2018.md) |

## 在 BFM taxonomy 中的位置

| 字段 | 内容 |
|------|------|
| 分组 | 03 Intrinsic reward 预训练 |
| 篇数 | 5/41 |
| 概念对照 | [Behavior Foundation Model](../concepts/behavior-foundation-model.md) |
| 姊妹分类 | 见 [BFM 技术地图 · 五类问题](./bfm-41-papers-technology-map.md#流程总览五类问题--身体-api) |

## 关联页面

- 同一分类体系：[01 前向-后向表征](./bfm-category-01-forward-backward-representation.md)、[02 目标条件学习](./bfm-category-02-goal-conditioned-learning.md)、[04 适应](./bfm-category-04-adaptation.md)、[05 分层控制](./bfm-category-05-hierarchical-control.md)
- 本组逐篇实体：[APS](../entities/paper-bfm-26-aps.md)、[Proto-RL](../entities/paper-bfm-27-proto-rl.md)、[RE3](../entities/paper-bfm-28-re3.md)、[RND](../entities/paper-bfm-29-rnd.md)、[DIAYN](../entities/paper-bfm-30-diayn.md)
- [BFM 41 篇技术地图](./bfm-41-papers-technology-map.md)
- [Behavior Foundation Model](../concepts/behavior-foundation-model.md)
- [人形 RL 身体系统栈](./humanoid-rl-motion-control-body-system-stack.md)
- [AMP 运动先验综述](./humanoid-amp-motion-prior-survey.md)

## 参考来源

- **一手论文（已通读正文，2026-11-07）：** [APS](../../sources/papers/bfm_awesome_aps_icml_2021.md)（arXiv:2108.13956）、[Proto-RL](../../sources/papers/bfm_awesome_proto_rl_icml_2021.md)（arXiv:2102.11271）、[RE3](../../sources/papers/bfm_awesome_re3_icml_2020.md)（arXiv:2102.09430）、[RND](../../sources/papers/bfm_awesome_rnd_iclr_2019.md)（arXiv:1810.12894）、[DIAYN](../../sources/papers/bfm_awesome_diayn_iclr_2018.md)（arXiv:1802.06070）
- [wechat_embodied_ai_lab_bfm_41_papers_survey.md](../../sources/blogs/wechat_embodied_ai_lab_bfm_41_papers_survey.md) — 微信公众号编译（<https://mp.weixin.qq.com/s/Ei32la_vo0UW9Y_QCAqB2g>）
- [bfm_awesome_41_catalog.md](../../sources/papers/bfm_awesome_41_catalog.md)
- [awesome-bfm-papers](https://github.com/friedrichyuan/awesome-bfm-papers)
- [A Survey of Behavior Foundation Model](https://arxiv.org/abs/2506.20487)（TPAMI 2025）

## 推荐继续阅读

- [Exploration by Random Network Distillation（arXiv:1810.12894）](https://arxiv.org/abs/1810.12894)、[Diversity is All You Need（arXiv:1802.06070）](https://arxiv.org/abs/1802.06070) — 本组中最常被引用的两篇
- [State Entropy Maximization with Random Encoders（arXiv:2102.09430）](https://arxiv.org/abs/2102.09430)、[Reinforcement Learning with Prototypical Representations（arXiv:2102.11271）](https://arxiv.org/abs/2102.11271)、[APS（arXiv:2108.13956）](https://arxiv.org/abs/2108.13956)
