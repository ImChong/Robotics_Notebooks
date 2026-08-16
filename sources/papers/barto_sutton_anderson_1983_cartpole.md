# Barto, Sutton, Anderson 1983 — Cart-Pole / Actor–Critic 原点

> 来源归档（ingest）

- **标题：** Neuronlike Adaptive Elements That Can Solve Difficult Learning Control Problems
- **类型：** paper
- **作者：** Andrew G. Barto, Richard S. Sutton, Charles W. Anderson
- **出处：** IEEE Transactions on Systems, Man, and Cybernetics, vol. SMC-13, no. 5, pp. 834–846, Sept.–Oct. 1983
- **DOI：** <https://doi.org/10.1109/TSMC.1983.6313077>
- **IEEE：** <https://ieeexplore.ieee.org/document/6313077>
- **公开 PDF（镜像）：** <http://www.derongliu.org/adp/adp-cdrom/refs/barto19830834.pdf>
- **复现代码：** <https://github.com/codecheckers/Barto-Sutton-Anderson-1983>（含论文附带 C 代码与 Python 翻译）
- **Sutton 教材配套 C：** <http://incompleteideas.net/sutton/book/code/pole.c>（Gymnasium CartPole 注释声明从此拷贝动力学）
- **入库日期：** 2026-08-16
- **一句话说明：** Cart-pole / 倒立摆平衡的强化学习一手论文：仅用失败信号，用 ASE+ACE（后世 Actor–Critic）学会给小车施加左右力以保持杆直立。
- **代码：** 论文附带仿真与学习器实现已开源（C 与社区 Python 翻译）；不是现代 Gym / Isaac 环境本身。
- **沉淀到 wiki：** 是 → [`wiki/concepts/cartpole.md`](../../wiki/concepts/cartpole.md)

---

## 核心摘录

### 任务定义（原文设定）

- 杆铰接在可沿轨道移动的小车上；唯一控制输入是施加在小车底座上的力。
- **不假设已知运动方程**；评价反馈只有失败信号：杆相对竖直偏角过大，或小车到达轨道端点。
- 作者论证：自适应网络中单个元件面对的学习难度，至少不亚于这一 pole-balancing 版本。

### 学习系统：ASE + ACE

- **ASE（Associative Search Element）**：在强化反馈下搜索输入–输出关联，相当于后世的 **actor**。
- **ACE（Adaptive Critic Element）**：构造比原始失败信号更信息丰富的评价函数，相当于后世的 **critic**。
- 这是 Gymnasium `CartPole` 文档明确对应的版本；Farama 源码注释写明「Classic cart-pole system implemented by Rich Sutton et al.」。

### 与更早 BOXES 工作的关系

- Michie & Chambers 的 BOXES（1968）已在离散化状态盒上做过 cart-pole。
- 1983 文把同一物理任务改写成 **稀疏失败信号 + 两个类神经元自适应元件**，并讨论与经典/工具性条件反射及神经科学的关系。
- Barto & Sutton 2021 回顾文 *Looking Back on the Actor–Critic Architecture*（IEEE TSMC）确认：该文是 Actor–Critic 架构的标志性实验，投稿前还修过仿真角度单位 bug。

## 开源与复现（步骤 2.5）

- **无独立项目页**（1983 IEEE 论文）。代码以论文附录 / Sutton `pole.c` 与社区 CODECHECK 仓为准。
- **已开源（历史实现）**：CODECHECK 仓含原 C 与可画出原文 Fig. 4/5 的 Python；许可证以该仓为准。
- **不是** 现代 `CartPole-v1` 或 `Isaac-Cartpole-v0` 的运行入口——后者分别见 [Gymnasium CartPole 文档](../sites/gymnasium-cartpole.md) 与 [Isaac Lab Cartpole](../sites/isaac-lab-cartpole.md)。

## 对 wiki 的映射

- [Cartpole 问题](../../wiki/concepts/cartpole.md) — 本资料升格的独立详情节点
- [Sutton & Barto RL 教材](../../wiki/entities/sutton-barto-rl-book.md) — 同一作者线的标准教材；`pole.c` 在教材代码目录
- [Reinforcement Learning](../../wiki/methods/reinforcement-learning.md) — Actor–Critic 谱系原点
- [MDP](../../wiki/formalizations/mdp.md) — 失败信号 + 四维状态的最小 MDP 实例
- [Gymnasium](../../wiki/entities/gymnasium.md) — `CartPole-v1` 官方声明对应本文
- [Reward Design](../../wiki/concepts/reward-design.md) — 原始设定是稀疏失败（0 / −1），不是逐步 +1

## 为什么值得保留

- Cartpole 在机器人学习课里常被当成「玩具」，但 **问题定义、失败信号、actor–critic 分工** 都从这篇一手论文来，不是从 Gym API 来。
- 后续 Gymnasium `sutton_barto_reward=True`、Isaac Lab 的 shaping 奖励，都是相对本文稀疏失败设定的工程变体；对比必须回到原文。
