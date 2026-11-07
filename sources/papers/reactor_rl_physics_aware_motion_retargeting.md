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
- **入库日期：** 2026-05-13；2026-11-07（全文通读更新）
- **核对范围：** 通读 arXiv:2605.06593 v1 全文 §1–§8（引言、相关工作、双层优化与近似梯度推导、重定向参数化、下层 RL 设计与奖励表 1、实现细节与超参数表 2、与 GMR / OmniRetarget 的对比表 3–4、参数化消融、泛化表 5、外力惩罚消融、四足 / 交互动画 / 真机用例、结论与局限）；参考文献仅浏览；未读论文补充材料与视频。
- **通读：** 已完成
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

## 全文核对要点（2026-11-07，均出自论文）

- **发表信息：** ACM Trans. Graph. 45(4), Article 97（SIGGRAPH 2026，July 2026）；作者 Disney Research（David Müller 一作，Agon Serifi、Sammy Christen、Ruben Grandia、Espen Knoop、Moritz Bächer）。旧 wiki 把一作写成"Müller M."有误。
- **双层问题：** $\min_{p\in P}\mathcal L(p,\phi^*(p))$，s.t. $\phi^*(p)=\arg\max_\phi\mathcal R(p,\phi)$；$P$ 为凸集（参数范数有界：$\delta_{pos}=\delta_{ori}=\delta_z=0.5$）。采用 **单环（single-loop）** 双层算法，每个 RL 迭代更新一次上层参数 $p\leftarrow P_P(p-\eta\tilde d_p\mathcal L)$。
- **近似梯度：** 不用隐函数定理（需下层 Hessian 逆）。假设①损失只依赖 $g_t-s_t^*$，故 $\partial_{s^*}\ell=-\partial_g\ell$；②最优轨迹对参考的灵敏度 $\partial_g s^*=\alpha I,\alpha\in[0,1]$（策略"部分"跟随参考的变化）。得 $\tilde d_p\ell=(1-\alpha)\,\partial_g\ell\,d_pg$，并用当前 rollout 的批数据估计。
- **参数化：** 用户在名义构型（如 T-pose）给出稀疏的语义刚体对应对和一个根对应；全局尺度 $s=h_{target}/h_{source}$（根高度比）；每个对应有位置参数 $p_{pos}^b$、朝向参数 $p_{ori}^b$（指数映射），另有每段动作的竖直偏移 $p_z$；自由度少的目标用 swing-twist 分解忽略无驱动轴的旋转误差。
- **下层 RL：** 策略输出关节位置设定值 + **根上的辅助残差力 / 力矩（RFC）**，50 Hz；RFC 带死区并被惩罚（便于单个策略学遍 AMASS，含倒立等形态上不可行的动作）；不用域随机化；retargeting 阶段变量 $\psi_t$ 先让机器人走到起始位姿；对失败多的片段自适应加大采样。PPO，MLP 3×512，Isaac Sim 4096 环境 / RTX 5090，20k 迭代约 6 小时。
- **评测：** 目标：Unitree G1（1.27 m，35 kg，29 DoF）、Lima（0.84 m，16.2 kg，20 DoF 自制小型机器人）、ANYmal D 四足。基线：GMR、OmniRetarget。PHC 过滤后的 AMASS 全集上，ReActor 的地面穿透 / 自穿透 / 脚滑 / 悬空指标整体最好（G1 下游 RL 成功率：GMR 89.93%、OmniRetarget 95.51%、ReActor 97.45%；Lima 为 91.23% / 79.85% / 95.07%，OmniRetarget 在 Lima 上因非均匀缩放与根对齐表现差）。成本：ReActor 在单 GPU 上训练 + 重定向全 AMASS 约 6.5 h，OmniRetarget 约 7 h（CPU），GMR 约 5 h（CPU）。注意 ReActor 的脚滑 / 悬空**不为零**，因为不严格强制接触匹配。
- **消融：** 双层优化持续降低跟踪损失并提高奖励；稀疏对应（只留根与末端）更偏离源动作但仍稳定；含位置目标优于仅朝向（跳跃起落时机更准）；**外力惩罚权重是最敏感超参数**——加大更物理但极端动作失败增多。泛化：100STYLE 上，训练于 AMASS 的单策略重定向误差略高于训练于同分布的策略（位置 0.19 vs 0.12 cm）。
- **用例：** 交互式动画中重定向策略以 88.3 Hz 运行；真机：用 ReActor 重定向数据训练目标条件、DeepMimic 式奖励的跟踪策略，部署到 Lima 实机（sim-to-real）；四足上外观保留但"形态差距越大越需细致调奖励"，ANYmal 会为减少外力而放弃跟踪。
- **局限（原文）：** 参数化设为时间不变；语义对应需人工选择；物理上不可能的动作本身是病态问题（走上虚拟楼梯还是投影到地面）。
