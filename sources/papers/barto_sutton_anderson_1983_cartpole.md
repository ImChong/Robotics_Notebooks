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
- **入库日期：** 2026-08-16（2026-11-07 补通读核对）
- **核对范围：** 2026-11-07 取得论文全文 PDF（IEEE TSMC 14 页，SMC-13(5) pp. 834–846，镜像见上；扫描版 OCR，公式部分字符残缺）并通读：摘要、§I 引言、§II 网络内学习、§V 倒立摆任务、§VI BOXES 系统、ASE 与 ACE 的定义（含资格迹与内部强化）、§IX 仿真结果（图 4、5）、讨论（与经典条件反射、神经科学）及附录（动力学方程与参数）。ASE/ACE 更新式因 OCR 残缺只核对文字说明，没有逐符号核对公式；没有运行官方 C/Python 复现代码；没有读被引的 Michie & Chambers 原文。
- **通读：** 已完成
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

## 通读核对要点（2026-11-07，均出自原文）

- **任务设定（§V）：** 控制器只能在离散时间步对小车施加固定大小、方向向左或向右的冲击力；状态为 x、ẋ、θ、θ̇ 四维；失败信号出现于杆偏离竖直超过阈值或小车到达轨道两端，此时系统（而非控制器记忆）复位并开始下一次试验；作者强调除失败信号外没有任何评价反馈，由此带来"难的信用分配问题"，而 Widrow & Smith 的 ADALINE 做法需要每步外部给出正确的控制决定。
- **BOXES 对照（§VI）：** 沿用 Michie & Chambers，把四维状态量化为 3（x）× 3（ẋ）× 6（θ）× 3（θ̇）=162 个区域；阈值为 x：±0.8、±2.4 m；θ：0、±1、±6、±12°；ẋ：±0.5 m/s；θ̇：±50°/s；原文明说这些阈值与单位和 Michie & Chambers 不同，是为本文参数化的仿真选取的。
- **ASE/ACE（§VII–VIII）：** ASE 随机输出左/右并保持资格迹；ACE 学习状态的失败预测 p，用 r̂ = r + γp(t) − p(t−1) 形式的内部强化（原文式 7 的文字说明）让 ASE 在每个时间步都能学习，而 BOXES 与无 ACE 的 ASE 只在失败时才学习。
- **仿真（§IX，图 4、5）：** 每次试验从 x=ẋ=θ=θ̇=0 开始，θ 越出 [−12°, 12°] 或 x 越出 [−2.4, 2.4] m 即失败；一个 run 100 次试验或累计超过 500 000 时间步（约 2.8 小时仿真时间）即终止；ASE/ACE 的参数 α=1000、β=0.5、δ=0.9、γ=0.95、λ=0.8、σ=0.01（除一组极端值外只试过这一套）；结果：ASE/ACE 的表现明显优于 BOXES，且多数 ASE/ACE run 在 100 次试验之前就达到 500 000 步而被截断；作者归因于 ACE 使学习在试验期间每一步发生，而非仅在失败时。
- **附录（动力学）：** g=−9.8 m/s²，m_c=1.0 kg，m=0.1 kg，l=0.5 m（半杆长），小车与轨道摩擦系数 μ_c=0.0005，杆与铰链摩擦系数 μ_p=0.000002，F=±10 N；先用 Adams–Moulton 预测校正器，报告结果用 Euler 法，步长 0.02 s。**因此 1983 原版含摩擦项；Gymnasium 文档称"frictionless track"并省略摩擦（见 `sites/gymnasium-cartpole.md`），两者并非逐项相同。**
- **局限（原文自述）：** 状态量化由人先验给定（作者承认易选出使任务不可解的划分）；没有证明 ASE/ACE 能作为强大自适应网络的组件；仅在单一仿真任务上测试，未与自适应控制方法比较。

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
