# AME-2: Agile and Generalized Legged Locomotion via Attention-Based Neural Map Encoding（arXiv:2601.08485）

> 论文来源归档（ingest · 由 progress 锚点升格为 arXiv 深读归纳）

- **标题：** AME-2: Agile and Generalized Legged Locomotion via Attention-Based Neural Map Encoding
- **类型：** paper / quadruped / biped / perceptive-locomotion / reinforcement-learning / neural-mapping / teacher-student / sim2real
- **arXiv：** <https://arxiv.org/abs/2601.08485> · PDF：<https://arxiv.org/pdf/2601.08485.pdf>
- **项目页：** <https://sites.google.com/leggedrobotics.com/ame-2>
- **机构：** ETH Zurich Robotic Systems Lab（Chong Zhang*, Victor Klemm, Fan Yang, Marco Hutter）；ETH AI Center；ETH SRI Lab
- **平台：** ANYmal-D（四足）、LimX Dynamics TRON1（双足，23-DoF 级）
- **分类（Paper Notebooks）：** 03_High_Impact_Selection
- **入库日期：** 2026-07-13（progress 锚点 2026-06-11；本次 arXiv 归纳升格）
- **一句话说明：** **AME-2** 在 [AME（arXiv:2506.09588）](ame_arxiv_2506_09588.md) 基础上加入 **全局地形特征 + 全局–本体联合 query 的 MHA**，并配套 **轻量 Bayesian 深度→局部高程+不确定性** 映射管线（Probabilistic Winner-Take-All 融合）；**Teacher（GT 高程图）→ Student（在线神经映射 + LSIO 本体 + 表征蒸馏）** 在 **Isaac Gym** 并行仿真中 **同栈训练**；ANYmal-D **零样本** 最难 parkour/rubble，TRON1 **0.48 m 上攀 / 0.88 m 下攀**，稀疏梁/踏石/组合地形与 **主动感知、膝撑恢复** 等涌现行为；相对 AME-1、MoE actor、单目 generalist 等在 benchmark 上 **敏捷与泛化兼得**。

## 核心摘录（面向 wiki 编译）

### 1) 相对 AME-1 的编码器升级

- **要点：** **AME-2 encoder**：CNN 得 **逐点 local features** + MLP positional embedding；**max-pool → global features** 捕获整体地形上下文；**global + proprio MLP → query**，MHA 对 local features 加权；**global ∥ weighted local** 为 map embedding。相对 AME-1 仅 proprio-query 点级注意力，全局特征使 **不同地形下 motion pattern 可分**（论文 Sec. VI–VII ablation）。
- **对 wiki 的映射：**
  - [`wiki/entities/paper-notebook-ame-2-agile-and-generalized-legged-locomotion-vi.md`](../../wiki/entities/paper-notebook-ame-2-agile-and-generalized-legged-locomotion-vi.md)
  - [`wiki/entities/paper-ame-attention-based-map-encoding.md`](../../wiki/entities/paper-ame-attention-based-map-encoding.md)

### 2) 统一 goal-reaching RL 与 asymmetric critic

- **要点：** **目标到达** 公式（position/heading tracking + move-to-goal + stand-at-goal）；**50 Hz** 策略、**400 Hz** PD。Teacher actor：**GT proprio + GT 3D 高程**；Student：**20 步 LSIO 时序本体（无 $v_b$）+ 4D 映射 $(x,y,z,u)$**。Critic：**MoE**（非 MHA，省算力），额外 **link contact state**；**左右对称增广** 仅用于 critic。奖励 **terrain-agnostic**，不显式惩罚/奖励 foothold 位置，允许 **膝接触、近缘落脚** 等全身接触涌现。
- **对 wiki 的映射：**
  - [`wiki/tasks/stair-obstacle-perceptive-locomotion.md`](../../wiki/tasks/stair-obstacle-perceptive-locomotion.md)
  - [`wiki/concepts/privileged-training.md`](../../wiki/concepts/privileged-training.md)

### 3) 神经映射：深度 → 局部高程 + 不确定性

- **要点：** 每帧深度点云 **投影局部 grid**（同 cell 取 max $z$）→ **浅 U-Net + gated residual** 预测 **elevation + log-variance**（$\beta$-NLL, $\beta=0.5$）；**TV 重加权** 强调崎岖 batch 样本。训练数据：**Warp 射线** 从 locomotion 地形 + 随机 box/heightfield/floating box 采样 **5400 万帧/机器人**，<1 h/模型。融合：**Probabilistic Winner-Take-All**（式 6–8）— 遮挡区不确定性 **不因重复预测而虚假下降**；Student 训练时 **部分 env 全图、部分在线部分图** 支持 **map reuse**（同楼梯往返）。
- **对 wiki 的映射：** 同上 AME-2 实体页；[`wiki/concepts/sim2real.md`](../../wiki/concepts/sim2real.md)

### 4) Teacher–Student 与训练成本

- **要点：** Teacher **80k** iter GT 地图；Student **40k** iter（前 **5k** 关 PPO surrogate、大 lr）= PPO + action distillation + **map embedding MSE**。ANYmal-D：**~60 RTX-4090-days**（8 GPU）；TRON1：**~30 days**（4 GPU）。1000 并行 ANYmal 环境映射 **<0.3 ms** 推理、**~3 GB** GPU（相对 Neural Processes 类 **>13 GB/env** 可并行训练）。机载 i7：**~5 ms/帧** 映射（**~2.5 ms** ONNX 推理），策略 **~2 ms**。
- **对 wiki 的映射：** 同上

### 5) 实机能力与相对 SOTA

- **要点：** ANYmal-D：**2 m/s** parkour 往返（controller+mapping **均未见**）；19 cm 梁、双行错落踏石、非固定梁+沟组合。**TRON1**：38 cm 台+沟+楼梯+ rough **全向**；非固定 **19 cm 浮块曲梁**；解锁轮式平台车 **攀爬/平衡**。相对 prior parkour generalist [48]：**零样本泛化** 更强；相对 biped prior **0.5 m 台**（H1, 更大扭矩/身高），TRON1 **0.48/0.88 m** 上下攀。**主动感知**：碰撞失败后地图补全，重试成功攀台。
- **对 wiki 的映射：**
  - [`wiki/entities/anymal.md`](../../wiki/entities/anymal.md)
  - [`wiki/entities/extreme-parkour.md`](../../wiki/entities/extreme-parkour.md)

## 局限（论文自述）

- 训练成本仍高；依赖 **2.5D 高程** 与机载 SLAM/深度栈。
- 未覆盖 **操作臂** 与 loco-manipulation 联合需求。

## 对 wiki 的映射

- [paper-notebook-ame-2-agile-and-generalized-legged-locomotion-vi](../../wiki/entities/paper-notebook-ame-2-agile-and-generalized-legged-locomotion-vi.md)
- [paper-ame-attention-based-map-encoding](../../wiki/entities/paper-ame-attention-based-map-encoding.md)
- 分类父节点：[paper-notebook-category-03-high-impact-selection](../../wiki/overview/paper-notebook-category-03-high-impact-selection.md)

## 参考来源（原始）

- He et al., *Attention-Based Map Encoding for Learning Generalized Legged Locomotion*, [arXiv:2506.09588](https://arxiv.org/abs/2506.09588)（AME-1 前作）
- Zhang et al., *AME-2: Agile and Generalized Legged Locomotion via Attention-Based Neural Map Encoding*, [arXiv:2601.08485](https://arxiv.org/abs/2601.08485)
- 项目页：<https://sites.google.com/leggedrobotics.com/ame-2>
