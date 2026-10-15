# Hyperspherical Contrastive Representations Unlock Planning and Exploration（C-Slerp，arXiv:2610.20409）

> 来源归档（ingest）

- **标题：** Hyperspherical Contrastive Representations Unlock Planning and Exploration
- **方法名：** C-Slerp（Contrastive Planning via Spherical Linear Interpolation）
- **类型：** paper / reinforcement-learning / goal-conditioned / contrastive-learning / planning / exploration
- **arXiv：** <https://arxiv.org/abs/2610.20409>（PDF：<https://arxiv.org/pdf/2610.20409>；HTML：<https://arxiv.org/html/2610.20409v1>）
- **作者：** Jonghae Park, Sungil Son, Jonghun Shin, Junseo Lee, H. Jin Kim, Benjamin Eysenbach
- **机构：** 未核实（arXiv 摘要页未列作者单位）
- **代码（项目页链接）：** <https://github.com/jonghaepark/c-slerp>
- **项目页：** <https://jonghaepark.github.io/c-slerp>
- **开源核查（2026-10-15）：** **已开源**。仓库公开（MIT；第三方代码保留原许可），含官方实现，分三个包：`JaxGCRL`、`builderbench_gripper`、`builderbench_wo_gripper`，每包提供 `scripts/train_cslerp.sh`，另附两个 Conda 环境文件。README 未列单独的评测命令。
- **文章来源：** [公众号综述摘录](../raw/wechat_embodied_station_robot_31_papers_2026-10-15.md)

## 核心主张（摘要要点）

- 自监督 RL，从零学习堆叠至多 8 层立方体塔，不依赖先验数据、奖励、示范或预训练模型。
- 把对比表示当作可复用的构件，反复组合以推断通向目标的中间路标（waypoint）。
- 超球面参数化下，路标可用球面线性插值（slerp）闭式计算，无需单独的高层策略或图搜索。
- 表示几何同时指示某个目标需要多少路标（自适应 K）。
- 消融：固定路标数会明显降低性能，支持按任务难度调整计划。

## 方法要点（来自论文 HTML 正文，前 100k 字符）

- 状态编码 ψ(s) 与状态-动作编码 φ(s,a) 映射到单位超球面；可学习正交旋转 R 建模非对称动力学（Cayley 变换）。
- 两类打分：f_H(s,s⁺)=ψ(s)ᵀRψ(s⁺) 用于路标间转移；f_L(s,a,s⁺)=φ(s,a)ᵀψ(s⁺) 用于策略。
- 路标均值 μᵢ=(Rᵀ)^i·slerp(ψ(s), R^{K+1}ψ(g), i/(K+1))，方差随角度解析给出。
- 路标数 K：对 ℓ_K=ψ(s)ᵀR^{K+1}ψ(g) 做 softmax（温度 η）。
- 路标选择：取满足 ρᵢ≥ε 的最大 i（ε∈[-1,1]）；无满足者取 i=1。ε 的实验取值未核实。
- 训练：表示损失为 backward InfoNCE 加对称 InfoNCE；策略以 ψ(g) 为条件最大化 φ(s,a)ᵀψ(g)。规划器每 h 个环境步重新选择 K 与路标。

## 实验（定性，数值表未核实）

- 环境：BuilderBench（6 个操作任务，含 2/3/4/5/8 层塔与金字塔变体，带夹爪与不带夹爪两套）；JaxGCRL（6 个运动任务，Ant 迷宫与 Humanoid U 型迷宫）。
- 对比：CRL、RIS、Online HIQL、L3P、SAC+HER。
- 论文称 C-Slerp 是唯一在所有带夹爪叠方块任务上成功的方法；短任务（如 2-tower、Ant Big Maze）上与部分基线相当。
- 单目标探索（Ant Big Maze、Ant U4 Maze）：C-Slerp 优于全部基线，除 CRL 与 L3P 外其余基线为 0。
- 消融（8-tower、Ant U5 Maze）：自适应 K 优于所有固定 K；自适应路标选择优于固定 i。

## 链接

- arXiv 摘要：<https://arxiv.org/abs/2610.20409>
- 官方代码：<https://github.com/jonghaepark/c-slerp>
- wiki 节点：[paper-c-slerp-hyperspherical](../../wiki/entities/paper-c-slerp-hyperspherical.md)
