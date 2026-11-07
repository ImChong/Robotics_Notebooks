---
type: formalization
tags: [robotics, motion-retargeting, optimization, ik, math, humanoid]
status: complete
created: 2026-05-17
updated: 2026-11-07
related:
  - ../concepts/motion-retargeting.md
  - ../concepts/motion-retargeting-pipeline.md
  - ../methods/motion-retargeting-gmr.md
  - ../methods/neural-motion-retargeting-nmr.md
  - ../methods/reactor-physics-aware-motion-retargeting.md
  - ../methods/spider-physics-informed-dexterous-retargeting.md
  - ../methods/deepmimic.md
  - ./tsid-formulation.md
  - ./friction-cone.md
sources:
  - ../../sources/papers/gmr_retargeting_matters_arxiv_2510_02252.md
  - ../../sources/papers/neural_motion_retargeting_nmr.md
  - ../../sources/papers/reactor_rl_physics_aware_motion_retargeting.md
  - ../../sources/papers/spider_scalable_physics_informed_dexterous_retargeting.md
  - ../../sources/papers/toporetarget_arxiv_2606_16272.md
  - ../../sources/papers/omniretarget_arxiv_2509_26633.md
  - ../../sources/papers/motion_control_projects.md
original:
  - ../../sources/papers/gmr_retargeting_matters_arxiv_2510_02252.md
  - ../../sources/papers/neural_motion_retargeting_nmr.md
  - ../../sources/papers/reactor_rl_physics_aware_motion_retargeting.md
  - ../../sources/papers/spider_scalable_physics_informed_dexterous_retargeting.md
  - ../../sources/papers/toporetarget_arxiv_2606_16272.md
  - ../../sources/papers/omniretarget_arxiv_2509_26633.md
summary: "动作重定向目标函数形式化：把姿态相似项、末端/接触约束项、平衡项、关节限位与平滑罚项写成统一的加权和；并展示该目标在纯 IK / RL 跟踪 / 双层优化三种工程实现下的退化与扩展形态。"
---

# Motion Retargeting Objective（动作重定向目标函数形式化）

**Motion Retargeting Objective** 是 [Motion Retargeting Pipeline](../concepts/motion-retargeting-pipeline.md) 第 4–6 阶段共用的数学骨架：把「让机器人长得像人」「关键点跟得上」「不穿地、不超限位」「时序不抖」这些工程要求，统一写成**带硬约束的加权最小化问题**。同一套目标函数在不同方法（[GMR](../methods/motion-retargeting-gmr.md) 的离线 QP / [ReActor](../methods/reactor-physics-aware-motion-retargeting.md) 的双层 RL / [DeepMimic](../methods/deepmimic.md) 风格 tracking 奖励）里以不同形态出现，本页提供一个对照表式的形式化。

## 一句话定义

给定源人体序列 $\mathbf{x}^h_{1:T}$，求解机器人关节轨迹 $\mathbf{q}^r_{1:T}$，最小化「姿态相似 + 接触/末端约束 + 平衡 + 关节限位 + 平滑」的加权和，同时满足运动学（或附加动力学）硬约束。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| IK | Inverse Kinematics | 满足末端/姿态约束求解关节角的运动学逆解 |
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略的范式 |
| Retargeting | Motion Retargeting | 将人体/动物动作映射到目标机器人骨架 |
| GMR | General Motion Retargeting | 把人体/视频动作重定向为机器人可执行参考 |
| QP | Quadratic Programming | 将 WBC/控制问题写成二次规划的标准求解形式 |
| SMPL | Skinned Multi-Person Linear Model | 常见人体参数化模型与重定向源 |
| DoF | Degrees of Freedom | 自由度，人形通常 20–50+ 关节 |
| CoM | Center of Mass | 质心，平衡与 locomotion 规划的核心状态量 |
| ZMP | Zero Moment Point | 足式平衡判据，地面反力合力矩为零的点 |
| DCM | Divergent Component of Motion | 质心发散分量，用于落脚点与平衡调节 |
| LIP | Linear Inverted Pendulum | 线性倒立摆，质心动力学的常用简化模型 |
| Reward | Reward Function | 塑造强化学习策略行为的标量反馈 |
| WBC | Whole-Body Control | 协调全身关节满足多任务/约束的控制基础设施 |
| TSID | Task-Space Inverse Dynamics | 任务空间逆动力学求解关节力矩的 WBC 实现 |
| PPO | Proximal Policy Optimization | 人形/足式 locomotion 中最常用的 on-policy 策略梯度算法 |
| SAC | Soft Actor-Critic | 连续控制常用的 off-policy 最大熵算法 |
| PD | Proportional–Derivative | 关节位置/阻抗底层控制，策略输出常为其 setpoint |

## 为什么重要

- **读论文的统一坐标**：GMR、NMR、ReActor、SPIDER、OmniRetarget、TopoRetarget 都在最小化"姿态 / 末端 / 接触 / 限位 / 平滑"的某种组合，只是把哪些项放进目标、哪些项外移到约束、过滤或仿真器里不同；把它们放进同一张表，才能看清各自在哪一步修补误差。
- **重定向质量会传到下游**：GMR 的受控实验显示，地面穿透、自穿透、关节值突变这三类伪影会降低跟踪策略的鲁棒性；这些伪影都能对应到本页目标中缺失或被弱化的项（高度 / 接触项、限位 / 自碰项、平滑项）。
- **选型与调参的抓手**：知道一项在目标里的权重（或被外移到何处），才知道出现脚滑、穿模、关节跳变时该改权重、改约束还是换方法。

## 决策变量与符号

| 符号 | 含义 |
|------|------|
| $\mathbf{q}^r_t \in \mathbb{R}^{n_q}$ | 时刻 $t$ 机器人广义坐标（含浮动基座 6 维 + 关节角） |
| $\dot{\mathbf{q}}^r_t,\ \ddot{\mathbf{q}}^r_t$ | 一阶/二阶导数（用于平滑与动力学项） |
| $\mathbf{x}^h_t$ | 源人体参考（SMPL 关节、关键点位姿或骨架序列） |
| $\mathbf{p}^r_{k,t} = \mathrm{FK}_k(\mathbf{q}^r_t)$ | 第 $k$ 个机器人关键点（手、脚、肘、骨盆等）的笛卡尔位姿 |
| $\hat{\mathbf{p}}^h_{k,t}$ | 经体型缩放后映射到机器人坐标系的源关键点目标 |
| $\mathcal{C}_t$ | 时刻 $t$ 的支撑接触集合（哪些末端贴地） |
| $\boldsymbol{\theta}$ | 可选的可学习重定向参数（如 [ReActor](../methods/reactor-physics-aware-motion-retargeting.md) 中的有界刚体偏移、全局尺度等） |

## 核心原理：通用目标函数

$$
\min_{\mathbf{q}^r_{1:T},\ \boldsymbol{\theta}}\quad
\sum_{t=1}^{T}\Big[
\underbrace{\mathcal{L}^{\text{pose}}_t}_{\text{姿态相似}}
+ \underbrace{\mathcal{L}^{\text{ee}}_t}_{\text{末端/接触}}
+ \underbrace{\mathcal{L}^{\text{bal}}_t}_{\text{平衡}}
+ \underbrace{\mathcal{L}^{\text{lim}}_t}_{\text{限位}}
+ \underbrace{\mathcal{L}^{\text{smooth}}_t}_{\text{平滑}}
\Big]
$$

满足硬约束：

$$
\mathrm{FK}(\mathbf{q}^r_t)\in\mathcal{M},\quad
\mathbf{q}^{\min}\preceq\mathbf{q}^r_t\preceq\mathbf{q}^{\max},\quad
\text{支撑脚不穿地不滑移}\ \forall k\in\mathcal{C}_t.
$$

下面给出每一项常用的具体形式。

### 1. 姿态相似项（Pose Similarity）

衡量「机器人当前姿态」与「（缩放后）源人体姿态」之间的几何相似度。常见三种取法：

**1a. 关节角对应**（拓扑匹配后）：

$$
\mathcal{L}^{\text{pose,joint}}_t = \sum_{j} w^{\text{j}}_j\,\big\| \mathbf{q}^r_{t,j} - \tilde{\mathbf{q}}^h_{t,j} \big\|_2^2,
$$

其中 $\tilde{\mathbf{q}}^h$ 是经过子树/DoF 对齐与 ROM 映射后的人体关节角。

**1b. 关键点位姿**（任务空间 IK 主导，[GMR](../methods/motion-retargeting-gmr.md) 风格）：

$$
\mathcal{L}^{\text{pose,kp}}_t = \sum_{k} w^{\text{k}}_k\,\big\| \mathrm{FK}_k(\mathbf{q}^r_t) - \hat{\mathbf{p}}^h_{k,t} \big\|_2^2.
$$

**1c. 旋转一致**（避免「位置贴但姿态拧」）：用 SO(3) 上的测地距离

$$
d_{\mathrm{SO}(3)}(R^r_{k,t},\,\hat{R}^h_{k,t}) = \big\|\log(\hat{R}^{h\top}_{k,t}\,R^r_{k,t})\big\|_F.
$$

> 在 [DeepMimic](../methods/deepmimic.md) 风格的 RL tracking 中，1a/1b/1c 通常以 **指数核** 形式出现：
> $r^{\text{pose}}_t = \exp(-\alpha \,\mathcal{L}^{\text{pose}}_t)$，作为奖励而非损失参与梯度。

### 2. 末端 / 接触约束项（End-Effector & Contact）

源人体序列里"哪只脚在地、哪只手在抓"是重定向必须保留的语义；丢失就会出现脚滑或穿模。

**末端跟随**（与姿态项里 1b 部分重叠，但通常加权更高）：

$$
\mathcal{L}^{\text{ee,track}}_t = \sum_{k\in\mathcal{E}} w^{\text{ee}}_k\,\big\| \mathbf{p}^r_{k,t} - \hat{\mathbf{p}}^h_{k,t} \big\|_2^2.
$$

**接触位置锁定**（对支撑脚 $k\in\mathcal{C}_t$）：

$$
\mathbf{p}^r_{k,t}\cdot\hat{\mathbf{n}} = h_{\text{ground}},\quad
\dot{\mathbf{p}}^r_{k,t} = \mathbf{0}.
$$

**接触相位一致性罚项**（避免相位漂移）：

$$
\mathcal{L}^{\text{ee,phase}}_t = \sum_{k}\mathbb{1}\big[c^r_{k,t}\neq c^h_{k,t}\big],
$$

其中 $c_{k,t}\in\{0,1\}$ 是接触指示，由源序列或物理筛选给出。

**摩擦锥**（当下游要求动力学一致时引入）：见 [Friction Cone 形式化](./friction-cone.md)。

### 3. 平衡项（Balance）

让重定向产物在静态/准静态意义上「站得住」，是物理可行性筛选阶段最常被外移成硬阈值的项。

**质心–支撑多边形罚项**：

$$
\mathcal{L}^{\text{bal,com}}_t = \big[\,\mathrm{dist}\big(\mathrm{CoM}(\mathbf{q}^r_t),\,\mathrm{Poly}(\mathcal{C}_t)\big)\,\big]_+^2,
$$

$[\cdot]_+$ 取正部，质心进入支撑多边形时为 0。

**ZMP/DCM 漂移**（动态步态）：详见 [ZMP + LIP 形式化](./zmp-lip.md)。

**根残差力压制**（[ReActor](../methods/reactor-physics-aware-motion-retargeting.md) 等仿真内方法中常见）：

$$
\mathcal{L}^{\text{bal,rfc}}_t = \big\| \mathbf{f}^{\text{rt}}_t \big\|_1 + \big\| \boldsymbol{\tau}^{\text{rt}}_t \big\|_1 .
$$

> 根残差力（RFC）相当于在浮动基上"扶一把"。ReActor 的下层 RL 让策略输出作用在根上的辅助力 $\mathbf f^{rt}$ 与力矩 $\boldsymbol\tau^{rt}$（带死区 $d=0.1$），并在奖励里用 **L1 范数**惩罚；允许它松弛会让单个策略能学遍 AMASS（含倒立等形态上不可行的动作），但代价是结果未完全动力学一致。原文称该惩罚权重是最敏感的超参数。

### 4. 关节限位与执行器边界（Limits）

$$
\mathcal{L}^{\text{lim}}_t = \sum_j\Big(
\big[\mathbf{q}^r_{t,j}-\mathbf{q}^{\max}_j\big]_+^2 +
\big[\mathbf{q}^{\min}_j-\mathbf{q}^r_{t,j}\big]_+^2 +
\big[|\dot{\mathbf{q}}^r_{t,j}|-\dot{\mathbf{q}}^{\max}_j\big]_+^2 +
\big[|\boldsymbol{\tau}^r_{t,j}|-\boldsymbol{\tau}^{\max}_j\big]_+^2
\Big).
$$

通常作为 **硬约束**写进 QP（位置/速度），力矩边界则视是否引入动力学项决定是软是硬。

### 5. 平滑项（Smoothness）

抑制"几何最优但电机跟不上"的高频抖动：

$$
\mathcal{L}^{\text{smooth}}_t = w_v\|\dot{\mathbf{q}}^r_t\|_2^2 + w_a\|\ddot{\mathbf{q}}^r_t\|_2^2 + w_j\|\dddot{\mathbf{q}}^r_t\|_2^2.
$$

工程上常用 Savitzky–Golay 滤波或最小 jerk 重投影实现；写成罚项是把"先求解再平滑"统一进单次优化。

## 三种工程化实现下的退化形态

同一套目标函数在落地时往往只保留对应阶段最关心的项，并把其他项移到约束、过滤或下游模块里。

| 方法 | 退化形态 | 主项 | 弱化/外移项 |
|------|----------|------|-------------|
| **逐帧微分 IK**（[GMR](../methods/motion-retargeting-gmr.md)） | 两阶段：第一阶段 $\min_{\dot q}\|e(q)+J\dot q\|_W^2$ 只含全部匹配刚体的**朝向**与**末端位置**，第二阶段加入全部关键刚体的位置 + 旋转并换一组权重；Mink 求解，上一帧结果作初值 | $\mathcal{L}^{\text{pose,kp}}$（含 SO(3) 测地项 1c）+ 关节限位硬约束；非均匀局部缩放决定 $\hat p^h$（根平移统一缩放） | 无显式平滑项（靠上一帧初值与可选速度上限）；**不使用接触相位**，仅靠根统一缩放抑制脚滑、序列末做高度修正；平衡 / 动力学项外移到下游 |
| **RL Tracking Reward**（[DeepMimic](../methods/deepmimic.md) 风格） | 指数核奖励之和 | $r^{\text{pose}}+r^{\text{ee}}+r^{\text{root}}$ | 限位/平滑由仿真器与早停隐式给出；平衡由是否摔倒（终止条件）触发 |
| **双层优化**（[ReActor](../methods/reactor-physics-aware-motion-retargeting.md)） | 上层 $\min_{\boldsymbol{p}\in P}\mathcal L(\boldsymbol p,\phi^*(\boldsymbol p))$：参数化参考 $\mathbf g_t(\boldsymbol p)$ 与仿真状态 $\mathbf s_t$ 之差，$\ell=\sum_b(w_x\ell_x^b+w_R\ell_R^b+w_v\ell_v^b+w_\omega\ell_\omega^b)$（位置 / 测地旋转 / 线速度 / 角速度）；下层 RL 最大化同形的跟踪奖励 | 参数化参考（原文记号 $\boldsymbol p$，本页前文的 $\boldsymbol\theta$）+ RFC 压制 | 限位/接触/自穿透由仿真器接管；上层用 $(1-\alpha)$ 缩放的近似梯度回传 |
| **配对监督**（[NMR](../methods/neural-motion-retargeting-nmr.md) CEPR 阶段） | $L=\sum_t\|m^t_{bot}-\hat m^t_{bot}\|_1$（L1 序列回归，作用于含根速度、局部关节位置 / 速度、关节 DoF 的运动表示） | 仿真物理一致轨迹做标签 | 重定向项已"内化"到标签里；网络不显式优化几何对应 |
| **采样轨迹优化**（[SPIDER](../methods/spider-physics-informed-dexterous-retargeting.md)） | $\min_u\|x_T\ominus x^{ref}_T\|^2_{Q_T}+\sum_t(\|x_{t+1}\ominus x^{ref}_{t+1}\|^2_{Q_t}+\|u_t\|^2_{R_t})$，退火噪声无梯度采样 + 仿真 rollout 评分；可加最坏情形鲁棒项与课程式虚拟接触 | 状态跟踪（SE(3) 对数映射）+ 控制代价（对角 $Q,R$） | 物理 / 接触约束由仿真器承担；无策略网络；开环轨迹仅在对齐的初始状态附近有效 |
| **Interaction mesh Laplacian**（[OmniRetarget](../entities/paper-hrl-stack-03-omniretarget.md)、[TopoRetarget](../methods/toporetarget-interaction-preserving-dexterous-retargeting.md)） | OmniRetarget：$\min_q\sum_i\|L(p_i^{tgt}(q))-L(p_i^{src})\|^2+\|q-q_{t-1}\|_Q^2$，**均匀权重** $w_{ij}=1/|N(i)|$，硬约束 SDF 非穿透、关节 / 速度界、stance 脚不动；TopoRetarget：**距离衰减权重** $w_{ij}\propto e^{-\kappa\|v_i-v_j\|}$ + 骨方向先验 $E_{bone}$ + 穿透软 / 硬界与松弛 | $\|\Delta(V^r)-\Delta(V^s)\|^2$（人形全身 / 灵巧手各自的网格顶点） | 姿态 / 指尖项弱化为 mesh 内局部几何；强调 hand–object / body–object 相对交互；OmniRetarget 用 SQP 型求解器逐帧 warm start |

## 物理可行性筛选 = 目标函数的硬阈值化

[NMR 的 CEPR 管线](../methods/neural-motion-retargeting-nmr.md)把上述罚项 **离散为硬阈值**，作为"是/否进入下一阶段"的二值过滤（对照原文）：

- **第 1 步（沿用 PHUMA 的人体侧过滤）**：去除过大 jerk、CoM 远离支撑域、足地接触不足（悬空 / 穿地）的人体序列；
- **第 2 步（GMR 重定向后的机器人侧过滤）**：

$$
\text{accept}=
\mathbb{1}\big[\max_t\|\dot{\mathbf q}_t\|_\infty\le\dot q_{\max}\big]\cdot
\mathbb{1}\big[\text{自相交帧占比}\le 0.05\big]\cdot
\mathbb{1}\big[\text{最低脚点平均离地}\le 0.10\,\text{m}\big],
$$

其中 $\dot q_{\max}$ 为硬件速度饱和上限（自相交由 MuJoCo 碰撞检测，阈值 0.05；悬空阈值 0.10 m 均为原文取值）。旧版把"脚 z 速度 $\le\epsilon$""CoM $\in$ Poly"也写进同一个指示函数，实际它们分属 PHUMA 的人体侧过滤，不在第 2 步的三条里。

这等价于把连续罚项写成"超出某阈值即损失趋向无穷"，工程实现简单但失去梯度信息——所以下游仍需要在保留段上跑 RL（NMR 用每簇 PPO 专家 rollout 得到约 30K 对物理一致的配对）。

## 工程实践：求解器侧的常见选择

- **微分 IK / QP 风格**：GMR 用 Mink 每步求解 $\min_{\dot q}\|e(q)+J\dot q\|_W^2$；下游要"重定向轨迹 → 力矩"时，可借助 [TSID 形式化](./tsid-formulation.md) 的执行器选择矩阵接到底层 WBC。
- **SQP / 序列凸优化**：OmniRetarget 对 Laplacian 目标 + 硬约束逐帧用定制 SQP 型（序列 SOCP）求解，Drake 自动微分处理四元数；非线性优化器（IPOPT / Ceres）是同类可选项（推测，论文未用）。
- **采样优化**（CMA-ES / MPPI / 退火噪声）：用于 [SPIDER](../methods/spider-physics-informed-dexterous-retargeting.md) 式的并行仿真轨迹搜索。
- **RL 优化器**（PPO / SAC）：当目标函数以奖励形态出现且策略与轨迹同时学习时。

## 评测口径

- **几何**：MPJPE（关键点误差）、末端跟随误差、$d_{\mathrm{SO}(3)}$ 姿态误差；用户研究对"忠实度"的主观评价（GMR 论文）。
- **物理**：地面 / 自穿透（时间占比 + 深度）、脚滑速度、脚悬空高度（OmniRetarget / ReActor 的口径）；关节跳变（单步 >0.5 rad）、自碰帧占比、关节极限帧占比（NMR 的口径）；根残差力均值、仿真 PD 跟踪成功率。
- **下游**：以这批参考训练出的 [模仿学习](../methods/imitation-learning.md) / RL tracking 策略的真机/仿真成功率与样本效率（最终指标）。

## 局限与风险

- **目标函数是"拼装图"而非单一标准**：各方法的权重、阈值（0.05、0.10 m、$\delta_{pos}=0.5$ 等）是各自论文的取值，不能跨方法直接套用。
- **运动学目标放不下动力学**：GMR / OmniRetarget 这类逐帧运动学目标不含力矩与接触力，物理一致要靠下游 RL / 仿真；OmniRetarget 自述约束线性化会带来偶发小穿透，由 RL 修复。
- **学习式 / 仿真式方法的代价**：NMR 的 CEPR 是机型特定的；SPIDER 的开环轨迹只在对齐的初始状态附近有效；ReActor 的外力惩罚权重在"物理真实"与"极端动作可重定向"之间取舍，并且脚滑 / 悬空不为零。
- **源数据质量**：TopoRetarget 对"虚拟接触"（源手指本想接触却没碰到物体）效果较差；SPIDER 受感知重建误差影响；NMR 依赖上游过滤。

## 关联页面

- [Motion Retargeting（动作重定向）](../concepts/motion-retargeting.md) — 任务定义与方法分类。
- [Motion Retargeting Pipeline](../concepts/motion-retargeting-pipeline.md) — 工程化端到端流水线，本目标函数对应其第 4–6 阶段。
- [GMR（通用动作重定向）](../methods/motion-retargeting-gmr.md) — 离线 QP 退化形态的代表实现。
- [NMR（神经运动重定向）](../methods/neural-motion-retargeting-nmr.md) — CEPR 把罚项硬阈值化 + 配对监督。
- [ReActor（物理感知 RL 运动重定向）](../methods/reactor-physics-aware-motion-retargeting.md) — 双层优化下的参数化参考。
- [SPIDER（物理感知采样式灵巧重定向）](../methods/spider-physics-informed-dexterous-retargeting.md) — 采样轨迹优化形态。
- [TopoRetarget（交互保留灵巧重定向）](../methods/toporetarget-interaction-preserving-dexterous-retargeting.md) — 灵巧手 interaction mesh Laplacian 形态。
- [GMR vs NMR vs ReActor（重定向方法谱系对比）](../comparisons/gmr-vs-nmr-vs-reactor.md) — 三种工程退化形态在选型坐标里的直接对照。
- [DeepMimic](../methods/deepmimic.md) — 把目标函数写成 RL tracking 奖励的范式。
- [TSID 形式化](./tsid-formulation.md) — 下游消费重定向参考的 QP 控制层。
- [Friction Cone 形式化](./friction-cone.md) — 动力学一致化阶段的接触约束。
- [ZMP + LIP 形式化](./zmp-lip.md) — 平衡项在简化模型下的解析形式。

## 参考来源

- [GMR（Retargeting Matters）通读归档](../../sources/papers/gmr_retargeting_matters_arxiv_2510_02252.md) — arXiv:2510.02252：非均匀局部缩放 + 两阶段 Mink IK。
- [NMR 通读归档](../../sources/papers/neural_motion_retargeting_nmr.md) — arXiv:2603.22201：CEPR 硬阈值过滤 + 配对监督（L1）。
- [ReActor 通读归档](../../sources/papers/reactor_rl_physics_aware_motion_retargeting.md) — arXiv:2605.06593：双层优化与近似梯度、RFC。
- [SPIDER 通读归档](../../sources/papers/spider_scalable_physics_informed_dexterous_retargeting.md) — arXiv:2511.09484：采样轨迹优化与虚拟接触课程。
- [TopoRetarget 通读归档](../../sources/papers/toporetarget_arxiv_2606_16272.md) — arXiv:2606.16272：距离衰减 Laplacian + 骨方向先验 + 穿透松弛。
- [OmniRetarget 通读归档](../../sources/papers/omniretarget_arxiv_2509_26633.md) — arXiv:2509.26633：均匀权重 Laplacian + 硬约束 + SQP 型求解。
- [sources/papers/motion_control_projects.md](../../sources/papers/motion_control_projects.md) — 飞书《开源运动控制项目》对运动学重定向项与下游动力学一致化分层的总结（二手，未列入 original）。

## 推荐继续阅读

- Araujo et al., *Retargeting Matters*：<https://arxiv.org/abs/2510.02252>
- Müller et al., *ReActor*：<https://arxiv.org/abs/2605.06593>
- Yang et al., *OmniRetarget*：<https://arxiv.org/abs/2509.26633>
- Pan et al., *SPIDER*：<https://arxiv.org/abs/2511.09484>
