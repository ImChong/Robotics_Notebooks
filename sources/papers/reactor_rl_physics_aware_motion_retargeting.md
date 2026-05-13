# reactor_rl_physics_aware_motion_retargeting

> 来源归档（ingest）

- **标题：** ReActor: Reinforcement Learning for Physics-Aware Motion Retargeting
- **类型：** paper
- **会议标注：** SIGGRAPH 2026（arXiv comment）
- **来源：** arXiv abs / arXiv HTML / arXiv PDF
- **原始链接：**
  - <https://arxiv.org/abs/2605.06593>
  - <https://arxiv.org/html/2605.06593v1>
  - <https://arxiv.org/pdf/2605.06593>
- **作者：** David Müller, Agon Serifi, Sammy Christen, Ruben Grandia, Espen Knoop, Moritz Bächer
- **入库日期：** 2026-05-13
- **最后更新：** 2026-05-13
- **一句话说明：** 将运动重定向表述为**物理仿真内的强化学习问题**：**双层优化**同时学习**参数化参考运动**（上层，少量语义刚体对应 + 可学习偏移）与**跟踪该参考的策略**（下层 RL），用问题结构导出的**近似上层梯度**避免隐函数定理下的 Hessian 逆，在仿真与真机上验证人形与四足等强异构具身。

## 核心摘录

### 1) ReActor: Reinforcement Learning for Physics-Aware Motion Retargeting（Müller 等，2026）
- **链接：** <https://arxiv.org/abs/2605.06593>；全文 HTML：<https://arxiv.org/html/2605.06593v1>；PDF：<https://arxiv.org/pdf/2605.06593>
- **问题：** 将人体（或其它源）**运动学参考**重定向到目标机器人形态时，纯运动学优化或学习映射常产生**脚滑、自碰撞、动力学不可行**等伪影，直接损害下游模仿学习 / tracking。
- **与 motion imitation 的区分：** 类似 DeepMimic 的框架用 RL **跟踪给定运动学参考**；本文解决的是**更早一步**——在具身差异下**生成适合被跟踪的参考**，而非假设源参考已形态一致。
- **双层形式：** 上层优化重定向参数 \(\mathbf{p}\)（将源运动 \(\mathbf{m}_t\) 映射为参数化参考 \(\mathbf{g}_t\)）；下层对给定 \(\mathbf{p}\) 训练 RL 策略 \(\pi_\phi\) 最大化跟踪奖励，使 rollout 状态 \(\mathbf{s}_t\) 逼近 \(\mathbf{g}_t\)。上层损失为 rollout 与参考的误差期望。
- **可解性：** 采用**单环双层**与**双时间尺度近似（TTSA）**思路，同步更新 \(\mathbf{p}\) 与 \(\phi\)。不用隐函数定理（避免下层 Hessian 逆），而对仅依赖 \((\mathbf{g}_t-\mathbf{s}_t)\) 形式的损失施加结构假设，将总导数中的策略灵敏度项近似为 \(\alpha I\)，得到**可实现的 \(\tilde{\nabla}_{\mathbf{p}}\mathcal{L}\)**，用当前策略 batch 数据估计。
- **参数化：** 用户在标定姿态下指定**稀疏语义刚体对**与根对；由根高度比得全局尺度；在名义坐标下对每对刚体引入有界位置/朝向偏移及每条运动的竖直偏移 \(p_z\)，处理 AMASS 类数据中的漂浮/穿透残余；损失在位置、测地旋转、线速度、角速度上加权（机器人自由度不足时可对扭转轴分解 swing/twist 损失）。
- **下层 RL：** PD 关节设定点 + **根残差外力矩（RFC）**（带死区与惩罚）以跨大数据集泛化；用**重定向相位变量**做回合开始时的参考暂停、奖励混合与数据过滤；强调与纯运动学工具（PHC、ProtoMotions、GMR、OmniRetarget 等）相比**严格自碰避免、不预设接触模式、单策略跨大规模数据**等主张；与可微最优控制式 DOC 对照为 RL 下层 + 不同梯度处理。
- **实验：** 仿真与**硬件**；包含与人体形态差异显著的人形与**四足**重定向；报告运动学指标、下游 tracking 学习与泛化分析。
- **对 wiki 的映射：**
  - 新建 [ReActor（物理感知 RL 运动重定向）](../../wiki/methods/reactor-physics-aware-motion-retargeting.md)
  - 在 [Motion Retargeting](../../wiki/concepts/motion-retargeting.md)、[GMR](../../wiki/methods/motion-retargeting-gmr.md)、[NMR](../../wiki/methods/neural-motion-retargeting-nmr.md) 中补充「仿真内双层：可学习参考 + RL 跟踪」的定位与交叉引用

## 当前提炼状态

- [x] arXiv 摘要 + HTML 公开章节中的方法主线（双层、近似梯度、参数化、RL 细节）已摘录
- [x] wiki 方法页与流程图已落盘
- [x] 与 motion-retargeting / GMR / NMR 交叉引用已补
