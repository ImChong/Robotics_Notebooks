# 逆强化学习（IRL）一手资料索引

> 来源归档（ingest）

- **标题：** Inverse Reinforcement Learning 经典论文与可运行实现
- **类型：** paper / repo（合集）
- **入库日期：** 2026-08-28
- **最后更新：** 2026-08-28
- **一句话说明：** 汇总 IRL 问题提出、线性规划求解、学徒学习、最大熵、深度 IOC 与对抗式 GAIL/AIRL 的一手论文，并指向可运行的现代实现。
- **沉淀到 wiki：** 是 → [inverse-reinforcement-learning](../../wiki/methods/inverse-reinforcement-learning.md)

## 为什么值得保留

- 仓库已有 [模仿学习](../../wiki/methods/imitation-learning.md)、[Behavior Cloning](../../wiki/methods/behavior-cloning.md)、[DAgger](../../wiki/methods/dagger.md) 与 [AMP](../../wiki/methods/amp-reward.md)，但 **IRL 本身没有方法页**：GAIL/AIRL 只作为 IL 的一行对照出现，读者无法从「为什么要从演示学奖励」读到 MaxEnt 与奖励可迁移性。
- IRL 是 **奖励难写、演示可得** 时连接 IL 与 RL 的形式化桥梁：先推断 $r$，再把 $r$ 交给 [强化学习](../../wiki/methods/reinforcement-learning.md)。人形 locomotion 里的 AMP 判别器奖励，是这条线的工程后代，不是另一套无关技巧。
- 一手文献把三件经常被混为一谈的事拆开：**学策略**（BC/GAIL）、**学可复用奖励**（MaxEnt / AIRL）、**只匹配占用测度**（学徒学习 / occupancy matching）。

## 开源核查（2026-08-28）

经典 2000–2008 论文 **无官方项目页、无官方训练代码**。深度 / 对抗阶段有仓，开放程度如下：

| 资料 | 开放程度 | 入口 |
|------|----------|------|
| Ng & Russell 2000 / Abbeel & Ng 2004 / Ziebart 2008 | **确认未开源**（作者 PDF 可下载，无可运行实现） | 作者页 PDF |
| Finn et al. 2016 GCL | **确认官方 GCL 专仓不存在**；相关 GPS 栈在 [cbfinn/gps](https://github.com/cbfinn/gps)，不是 GCL 论文复现入口 | 论文 arXiv |
| Ho & Ermon 2016 GAIL | **已开源** MIT，仓 **已归档**（最后推送 2018-11） | [openai/imitation](https://github.com/openai/imitation) |
| Fu, Luo & Levine 2018 AIRL | **已开源** MIT，TensorFlow 1.x 时代实现（最后推送 2018-06） | [justinjfu/inverse_rl](https://github.com/justinjfu/inverse_rl) |
| 现代可运行入口 | **已开源** MIT；PyTorch + Gymnasium；含 BC / DAgger / MCE-IRL / GAIL / AIRL | [HumanCompatibleAI/imitation](https://github.com/HumanCompatibleAI/imitation) → [repos 归档](../repos/humancompatibleai-imitation.md) |

以 **项目页 / 官方仓实际链接** 为准：GCL 论文未列出可点进的 GCL GitHub；GAIL/AIRL 官方仓可跑但依赖过时。工程复现优先 `imitation` 库，而不是 2016–2018 的 TF 仓。

## 核心摘录

### 1) Russell (1998) — 提出 IRL 问题

- **来源：** S. Russell, *Learning Agents for Uncertain Environments*, NIPS 1998 workshop invited paper. [作者 PDF](https://people.eecs.berkeley.edu/~russell/papers/nips98-agents.pdf)
- **要点：**
  - 把「给定行为，反推智能体在优化什么」写成独立问题：观测行为（及必要时感知与环境模型）→ 求奖励函数。
  - 动机有两支：用 RL 做动物/人类学习的计算模型；以及学徒学习——奖励往往比策略更紧凑、更可迁移。
- **对 wiki 的映射：** [inverse-reinforcement-learning](../../wiki/methods/inverse-reinforcement-learning.md)

### 2) Ng, Harada & Russell (1999) — 势函数奖励塑形与策略不变类

- **来源：** A. Y. Ng, D. Harada, S. Russell, *Policy Invariance Under Reward Transformations: Theory and Application to Reward Shaping*, ICML 1999. [作者 PDF](https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf)
- **要点：**
  - $\hat{r}(s,a,s') = r(s,a,s') + \gamma\Phi(s') - \Phi(s)$ 不改变最优策略；在未知动力学下，这是保持策略不变的奖励变换类。
  - IRL 的 **退化性** 由此精确化：同一专家策略对应一整族奖励，其中大量是被动力学塑过形的。AIRL 后来把「学到可迁移奖励」定义成从这族里拆出与动力学解耦的 $r(s)$。
- **对 wiki 的映射：** [inverse-reinforcement-learning](../../wiki/methods/inverse-reinforcement-learning.md)、[reward-design](../../wiki/concepts/reward-design.md)

### 3) Ng & Russell (2000) — IRL 算法与线性规划

- **来源：** A. Y. Ng, S. Russell, *Algorithms for Inverse Reinforcement Learning*, ICML 2000, pp. 663–670. [作者 PDF](https://people.eecs.berkeley.edu/~russell/papers/ml00-irl.pdf) · [Stanford 镜像](https://ai.stanford.edu/~ang/papers/icml00-irl.pdf) · [ACM](https://dl.acm.org/doi/10.5555/645529.657801)
- **要点：**
  - 形式化：已知（近似）最优策略，求使该策略最优的 $R$。给出有限状态表格 $R$、线性函数近似、以及 **只观测到有限轨迹** 三种算法。
  - 核心障碍是 **degeneracy**（$R\equiv 0$ 总是解）。启发式：选让专家相对次优动作 **margin 最大** 的 $R$，得到可解 LP。
  - 实验是离散/连续导航与 gridworld 量级，不是高维机器人控制；价值在问题定义与退化性，不在规模。
- **对 wiki 的映射：** [inverse-reinforcement-learning](../../wiki/methods/inverse-reinforcement-learning.md)、[mdp](../../wiki/formalizations/mdp.md)

### 4) Abbeel & Ng (2004) — 学徒学习：匹配特征期望即可

- **来源：** P. Abbeel, A. Y. Ng, *Apprenticeship Learning via Inverse Reinforcement Learning*, ICML 2004. [作者 PDF](https://ai.stanford.edu/~ang/papers/icml04-apprentice.pdf)
- **要点：**
  - 假设 $R^*(s)=w^*\cdot\phi(s)$。算法迭代：用当前 $w$ 解 RL → 比较策略与专家的 **feature expectations** $\mu(\pi)=\mathbb{E}[\sum_t\gamma^t\phi(s_t)]$ → 更新 $w$。
  - **关键保证：** 不必恢复真实 $w^*$。只要 $\|\mu(\tilde\pi)-\mu_E\|_2\le\varepsilon$，则对 **任意** $\|w\|_1\le 1$ 的线性奖励，回报差距 $\le\varepsilon$。学徒学习的目标是 **表现接近专家**，不是唯一还原 $R$。
  - 用公路驾驶动机：人类能开车，却写不出各项权衡的数值奖励——这是后来机器人 IRL 的标准叙事。
- **对 wiki 的映射：** [inverse-reinforcement-learning](../../wiki/methods/inverse-reinforcement-learning.md)、[imitation-learning](../../wiki/methods/imitation-learning.md)

### 5) Ziebart, Maas, Bagnell & Dey (2008) — 最大熵 IRL

- **来源：** B. D. Ziebart, A. Maas, J. A. Bagnell, A. K. Dey, *Maximum Entropy Inverse Reinforcement Learning*, AAAI 2008, pp. 1433–1438. [作者 PDF](https://ai.stanford.edu/~amaas/papers/amaas_aaai.pdf) · [AAAI](https://aaai.org/papers/01433-aaai08-227-maximum-entropy-inverse-reinforcement-learning/)
- **要点：**
  - 在特征匹配约束下，用 **最大熵** 选定轨迹分布：$P(\tau)\propto\exp(R(\tau))$，全局归一化，避免最大间隔法在噪声/次优演示上的歧义。
  - 应用是 **10 万英里出租车导航**：路网结构已知，学司机路线偏好，并支持从部分轨迹推断目的地。
  - 后续 **最大因果熵**（Ziebart 2010 博士论文）把同一思想接到随机策略与未知动力学，成为 GCL/GAIL/AIRL 的概率底座。
- **对 wiki 的映射：** [inverse-reinforcement-learning](../../wiki/methods/inverse-reinforcement-learning.md)

### 6) Wulfmeier, Ondruska & Posner (2015) — 深度最大熵 IRL

- **来源：** M. Wulfmeier, P. Ondruska, I. Posner, *Maximum Entropy Deep Inverse Reinforcement Learning*, arXiv:1507.04888. [abs](https://arxiv.org/abs/1507.04888)
- **要点：**
  - 用神经网络参数化代价/奖励，摆脱手工 $\phi$；仍在可解析或可迭代求解的占用上做 MaxEnt。
  - 评测停留在可用 value iteration 的简单域。说明「深度函数近似」本身不够：未知动力学与高维连续控制还需要采样式 IOC。
- **对 wiki 的映射：** [inverse-reinforcement-learning](../../wiki/methods/inverse-reinforcement-learning.md)

### 7) Finn, Levine & Abbeel (2016) — Guided Cost Learning

- **来源：** C. Finn, S. Levine, P. Abbeel, *Guided Cost Learning: Deep Inverse Optimal Control via Policy Optimization*, ICML 2016. [arXiv:1603.00448](https://arxiv.org/abs/1603.00448)
- **要点：**
  - 针对 **未知动力学、高维连续、真机力矩操作**：用策略优化做 MaxEnt IOC 的采样近似，神经网络代价无需精心特征。
  - 同时训练代价与策略（策略充当配分函数的 amortized sampler）。报告仿真与真机操作相对先前 IOC 的任务复杂度与样本效率提升。
  - **步骤 2.5：** 截至入库日无独立 GCL GitHub；[cbfinn/gps](https://github.com/cbfinn/gps) 是 Guided Policy Search 栈，不要当成 GCL 官方复现。
- **对 wiki 的映射：** [inverse-reinforcement-learning](../../wiki/methods/inverse-reinforcement-learning.md)、[manipulation](../../wiki/tasks/manipulation.md)

### 8) Ho & Ermon (2016) — GAIL

- **来源：** J. Ho, S. Ermon, *Generative Adversarial Imitation Learning*, NeurIPS 2016. [arXiv:1606.03476](https://arxiv.org/abs/1606.03476)
- **代码：** [openai/imitation](https://github.com/openai/imitation) — MIT，**已归档**
- **要点：**
  - 刻画「对 MaxEnt IRL 学到的代价再跑一遍 RL」所诱导的策略，证明可 **绕过显式 IRL 内环**，直接做占用测度匹配。
  - 实例化成 GAN：判别器区分专家 vs 策略 $(s,a)$，策略最大化判别器给出的信号。高维物理控制上显著优于当时的 model-free 模仿基线。
  - **不是**可迁移奖励学习：最优时判别器输出约 0.5，不能当新环境的 $r$ 用。AIRL 与 AMP 分别从「要可迁移 $r$」和「只要风格奖励」两个方向消化这一点。
- **对 wiki 的映射：** [inverse-reinforcement-learning](../../wiki/methods/inverse-reinforcement-learning.md)、[imitation-learning](../../wiki/methods/imitation-learning.md)、[amp-reward](../../wiki/methods/amp-reward.md)

### 9) Fu, Luo & Levine (2018) — AIRL

- **来源：** J. Fu, K. Luo, S. Levine, *Learning Robust Rewards with Adversarial Inverse Reinforcement Learning*, ICLR 2018. [arXiv:1710.11248](https://arxiv.org/abs/1710.11248)
- **代码：** [justinjfu/inverse_rl](https://github.com/justinjfu/inverse_rl) — MIT，TF1 时代
- **要点：**
  - 给对抗判别器加 **奖励结构** $D_\theta(s,a)=\exp(f_\theta)/(\exp(f_\theta)+\pi(a|s))$，使 $f$ 在最优时可提取为奖励，而不是 GAIL 那种不可复用的 critic。
  - 进一步把 $f$ 拆成 $r_\theta(s)+\gamma V_\phi(s')-V_\phi(s)$，用状态价值吸收塑形项，得到对 **动力学变化更稳健** 的 disentangled $r(s)$。
  - 原训练环境上与 GAIL 接近；**换动力学再优化学到的奖励** 时明显优于 GCL/GAIL。这是「IRL 相对直接模仿的独特卖点」的实验定位。
- **对 wiki 的映射：** [inverse-reinforcement-learning](../../wiki/methods/inverse-reinforcement-learning.md)、[reward-design](../../wiki/concepts/reward-design.md)

## 当前提炼状态

- [x] 九篇一手论文摘录与 wiki 映射
- [x] 开源边界（无官方仓 / 归档 TF 仓 / 现代 `imitation` 库）写入步骤 2.5
- [x] 升格 [wiki/methods/inverse-reinforcement-learning.md](../../wiki/methods/inverse-reinforcement-learning.md)
