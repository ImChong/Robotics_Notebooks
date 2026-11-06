---
type: entity
tags: [paper, humanoid-paper-notebooks, sim-to-real, contrastive-learning, humanoid-locomotion]
status: complete
updated: 2026-11-06
arxiv: "2509.12858"
related:
  - ../overview/paper-notebook-category-10-sim-to-real.md
  - ../overview/humanoid-paper-notebooks-index.md
sources:
  - ../../sources/papers/humanoid_pnb_contrastive-representation-learning-for-adaptive.md
summary: "主流人形 RL 行走面临两难选择——纯本体感知策略反应快但被动（只能\"踩到了再调\"），而依赖深度图/高程图的感知驱动策略主动但脆弱（深度噪声、外参漂移、视角遮挡都会让 sim-to-real 崩掉）。本文用对比学习把仿真侧的特权环境信息（地形高度、摩擦、质量、外力等）\"蒸馏\"到 actor 的隐状态里，同时引入一个自适应步态时钟让策略根据\"已感知到但实际看不见\"的地形主动调整步频，从而在不接任何外部感知模块的前提下兼具反应与主动性，全尺寸人形零样本通过 30 cm 台阶和 26.5° 斜坡。"
---

# Contrastive Representation Learning for Robust Sim-to-Real Transfer of Adaptive Humanoid Locomotion

**Contrastive Representation Learning for Robust Sim-to-Real Transfer of Adaptive Humanoid Locomotion** 收录于 [Robot Learning Paper Notebooks](https://imchong.github.io/Robot_Learning_Paper_Notebooks/index.html)（分类：10_Sim-to-Real）。本页基于原始论文全文（arXiv:2509.12858v1，Lu et al.）整理；深读笔记与项目页仅作为辅助入口。

## 一句话定义

主流人形 RL 行走面临两难选择——纯本体感知策略反应快但被动（只能"踩到了再调"），而依赖深度图/高程图的感知驱动策略主动但脆弱（深度噪声、外参漂移、视角遮挡都会让 sim-to-real 崩掉）。本文用对比学习把仿真侧的特权环境信息（地形高度、摩擦、质量、外力等）"蒸馏"到 actor 的隐状态里，同时引入一个自适应步态时钟让策略根据"已感知到但实际看不见"的地形主动调整步频，从而在不接任何外部感知模块的前提下兼具反应与主动性，全尺寸人形零样本通过 30 cm 台阶和 26.5° 斜坡。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略；本文用 PPO 训练 |
| PPO | Proximal Policy Optimization | 带截断比率的策略梯度算法（式 (8)），约束每次更新幅度 |
| RNN | Recurrent Neural Network | Actor 与 Critic 各带一个 256 维隐状态的循环网络（式 (1)、(3)） |
| PD | Proportional-Derivative | 把目标关节角转换为力矩的比例-微分控制器，400 Hz（式 (2)） |
| InfoNCE | Noise-Contrastive Estimation 形式的对比损失 | 同一时刻的 actor/critic 嵌入为正样本对，其余时刻为负样本（式 (7)） |
| POMDP | Partially Observable Markov Decision Process | 部分可观测马尔可夫决策过程，本文将行走任务建模为 POMDP |
| DR | Domain Randomization | 仿真中随机化质量、质心、电机强度与外力扰动，减小 sim-to-real 差距 |
| HIM | Hybrid Internal Model | 本文对比学习设计的参照对象，对比历史观测与未来观测 |
| Sim2Real | Simulation to Real | 把仿真中学到的策略迁移落地真机的工程主线 |

## 为什么重要

- 行走策略的"反应 vs 主动"取舍是人形落地的核心矛盾：纯本体感知策略在台阶、斜坡上只能被动补偿，而加外感知又引入相机/LiDAR 的标定、延迟与遮挡问题。本文尝试在两者之间给出第三条路。
- 方法直接把"训练期特权信息"转化为"部署期不需要的隐式感知"，对只有关节编码器和 IMU 的真机平台有直接工程意义。
- 步态时钟的自适应调节（频率与相位偏置由策略输出）把"感知到的地形"映射到步态节奏上，是比固定周期步态更贴近人类行走的一种控制接口。
- 列入 Paper Notebooks 策展清单，便于与全库 [机器人学习论文笔记总索引](../overview/humanoid-paper-notebooks-index.md) 及分类父节点交叉检索。

## 核心原理

**输入与输出。** Actor 观测 $o_t \in \mathbb{R}^{84}$，包含控制指令 $c_t\in\mathbb{R}^3$、投影重力 $g_t\in\mathbb{R}^3$、关节角/角速度 $q_t,\dot q_t\in\mathbb{R}^{23}$、上一步动作 $a_{t-1}\in\mathbb{R}^{26}$、机体角速度 $\omega_t\in\mathbb{R}^3$、平均角速度 $\bar\omega_t\in\mathbb{R}^1$ 与步态相位 $\psi_t\in\mathbb{R}^2$。动作 $a_t\in\mathbb{R}^{26}$ 拆分为 $a^j_t\in\mathbb{R}^{23}$（相对参考姿态的关节增量，$q^*_t=\mathring q+a^j_t$）与 $a^\delta_t\in\mathbb{R}^3$（$\delta_0,\delta_1,\delta_2$，调节步态频率与相位）。关节力矩由 $\tau = k_p(q^*_t-q_t)+k_d(-\dot q_t)$ 计算（式 (2)）。

**非对称 Actor-Critic。** Actor 只用机载传感（RNN，256 维隐状态 $h^{actor}_t$）。Critic 额外拿到仿真特权信息：速度、质心位置、两只脚各 $5\times5$ 的高度图，以及全局地形图 $h_t\in\mathbb{R}^{17\times11}$（经 CNN 得到 32 维 $u_t$）。Critic 的 RNN 输出 256 维 $m^c_t$，与 $u_t$ 拼接成 288 维向量 $z_t$ 后过全连接层输出 $V$（式 (3)–(4)）。

**对比蒸馏（核心机制）。** Source Encoder 把 actor 的中间表征 $m^a_t$ 映射为 $I^S_t=\mathrm{MLP}_S(m^a_t)$；Target Encoder 以 $u_t$（停止梯度）与特权观测 $s_t$ 产生 $I^T_t=\mathrm{MLP}_T(\mathrm{sg}(u_t), s_t)$（式 (5)–(6)）。同一时刻 $(I^S_t, I^T_t)$ 为正样本对，其余时刻为负样本，损失为

$$
\mathcal{L}_{contrast} = -\frac{1}{B}\sum_{t=1}^{B}\log\frac{\exp(I^S_t\cdot I^T_t/\tau)}{\sum_{t'}\exp(I^S_t\cdot I^T_{t'}/\tau)}
$$

与 HIM 不同，本文对比的是**同一步**的 actor 与 critic 表征（而非历史窗口与下一步观测），目的是把地形相关线索直接蒸馏进策略。论文的消融把对比目标改成"下一时刻观测"（Sequential Contrast）作为对照。

**自适应步态时钟。** 维护全局相位 $\phi(t+\Delta t)=\phi(t)+(1+\alpha\,\delta_0(t))\Delta t$（式 (9)），归一化相位 $\phi(t)/T_{cycle}$（式 (10)），并构造两路正弦 $s(t)=\sin 2\pi[\varphi+\delta_1]$、$c(t)=\sin 2\pi[\varphi+\delta_2+\psi]$（式 (11)–(12)，$\psi=0.5$）。$\delta_0$ 改变全局节奏，$\delta_1,\delta_2$ 分别平移左右脚相位。接触掩码由阈值 $\gamma$（如 0 或 $-0.1$）决定（式 (13)）。

**奖励。** 奖励为各项加权和 $r_t=\sum_i w_i r_i$（Table I），包括线速度/角速度跟踪（权重 3.0）、接触模式（$\exp(-\|m^{stance}-c^{actual}\|^2)$，权重 1.0）、足部离地高度、关节力矩与加速度惩罚等。

**平台与仿真。** 全尺寸人形 Adam Lite：约 1.6 m、约 60 kg，25 DOF 中实际控制 23 个（腕部 yaw 不控制），腿部为 QDD 力控执行器（单关节峰值约 340 N·m）。训练在 Isaac Gym 中进行，4096 个并行环境，PPO 训练 10k 次迭代，单张 RTX 4090 约 9 小时。另在 MuJoCo 中做 sim-to-sim 验证。

## 工程实践

**复现要点（论文给出的参数）**

| 项目 | 论文设置 |
|------|----------|
| 仿真器 / 并行数 | Isaac Gym，4096 环境 |
| 训练算法 | PPO，clip $\varepsilon$ 约 0.2 |
| 控制频率 | PD 控制器 400 Hz |
| 域随机化 | 质量 ±5%，质心位置 ±5 cm，电机强度 0.8–1.2×，随机冲击 0–0.8 m/s，外力 ±50 N |
| 地形课程 | 台阶 5–35 cm，斜坡 2–34°，不平地形 |
| 评测规模 | 每个难度 200 个机器人；消融每个策略 25 次试验、3 个随机种子 |

**调试与评测指标**：训练侧看地形课程等级推进速度（论文称约 5k 步即可推进到最高难度）；评测侧看成功率（到达绿色目标点且不摔倒）随台阶高度、坡度与指令速度的变化。

**结果（论文原文数字）**：台阶 10–15 cm 成功率 >95%，30 cm 台阶约 85%（Fig. 11，论文未明确该组是否为仿真，未核实）；26.5° 斜坡成功率 >90%；在最高测试指令速度下成功率仍保持 >75%（Fig. 7 右，论文称其他方法随速度下降更快）。消融显示去掉对比学习、或固定周期步态、或无时钟策略均明显下降，基线 DWL 与 DreamWaQ 在高台阶场景低于 60%。

**真机**：Adam Lite 零样本完成上楼梯、下 26.5° 斜坡、不平地形（Fig. 8、Fig. 10）。论文正文真机段落写的是"上 15 cm 台阶"，而 Fig. 1 与结论写的是 30 cm 台阶，二者的真机条件差异论文未展开（未核实）。读者引用时应以具体图表为准。

**源码开源状态（步骤 2.5）**：项目页 <https://lu-yidan.github.io/cra-loco/> 的头部、资源区与页脚未列出 GitHub、Hugging Face 等代码或权重链接（仅有网站模板的 GitHub 引用）；论文正文也未声明代码发布。**截至入库日（2026-11-06）尚未发现官方代码**，故不写 `code:` 字段，也不要求源码运行时序图。若后续代码发布，需补写该节。

## 结论

**这篇的取舍很明确：不接任何外部感知模块，用特权信息蒸馏 + 自适应步态时钟把「主动性」塞进纯本体感知策略，从而绕开深度图/高程图路线的 sim-to-real 脆弱点。**

1. **对比蒸馏的关键是"同一步对齐"** — actor 隐状态与 critic 的地形嵌入在同一时刻做 InfoNCE 对齐，而不是 HIM 式的历史-未来对齐；消融中"序列对比"是对照项，读者若要复现，应保持这一设定。
2. **两个机制缺一不可** — 只有对比蒸馏而无自适应时钟，或只有时钟而无蒸馏，都会退回「反应快但被动」或「死板时钟」；论文的消融结构正是围绕这一点设计的。
3. **真实收益体现在高难度台阶与斜坡** — 论文数字在 10–15 cm 台阶上接近饱和，差距主要出现在 30 cm 台阶与 26.5° 斜坡，这是判断方法价值的主要依据。
4. **代价是部署期只能靠本体信号推断地形** — 它换掉的是深度噪声、外参漂移与遮挡，但对于视觉可直接看到的大型障碍，它不会像外感知那样提前规划。
5. **复现前提是仿真侧特权信息与域随机化** — 高度图、质心、接触信息只在仿真中可得；蒸馏质量直接依赖域随机化的覆盖范围，参数见上表。
6. **注意论文自身的数字口径** — 30 cm 台阶出现在 Fig. 1、Fig. 11 与结论中，而真机正文写的是 15 cm；引用时需区分仿真/真机与具体图表。

## 与其他页面的关系

- 分类父节点：[paper-notebook-category-10-sim-to-real](../overview/paper-notebook-category-10-sim-to-real.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)

## 参考来源

- [humanoid_pnb_contrastive-representation-learning-for-adaptive.md](../../sources/papers/humanoid_pnb_contrastive-representation-learning-for-adaptive.md)（Paper Notebooks 策展来源，含 arXiv 2509.12858 链接）
- 论文原文（已通读正文、方法、实验、结论与参考文献；论文无独立附录）：<https://arxiv.org/pdf/2509.12858>（arXiv:2509.12858v1，2025-09-16）
- 项目页：<https://lu-yidan.github.io/cra-loco/>（用于步骤 2.5 核查，未列代码）
- 深读笔记：<https://imchong.github.io/Robot_Learning_Paper_Notebooks/papers/10_Sim-to-Real/Contrastive_Representation_Learning_for_Adaptive_Humanoid_Locomotion/Contrastive_Representation_Learning_for_Adaptive_Humanoid_Locomotion.html>

## 推荐继续阅读

- [论文 PDF（arXiv:2509.12858）](https://arxiv.org/pdf/2509.12858)：原文，含 Table I 奖励全表与图 8–11 的完整数据。
- [项目页 cra-loco](https://lu-yidan.github.io/cra-loco/)：视频与论文镜像，未发布代码。
- [机器人论文阅读笔记：Contrastive Representation Learning for Robust Sim-to-Real Transfer of Adaptive Humanoid Locomotion](https://imchong.github.io/Robot_Learning_Paper_Notebooks/papers/10_Sim-to-Real/Contrastive_Representation_Learning_for_Adaptive_Humanoid_Locomotion/Contrastive_Representation_Learning_for_Adaptive_Humanoid_Locomotion.html)：深读笔记。
