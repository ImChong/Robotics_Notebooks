# Generalizing from References using a Multi-Task Reference and Goal-Driven RL Framework

> 来源归档（ingest）

- **标题：** Generalizing from References using a Multi-Task Reference and Goal-Driven RL Framework
- **工作简称：** **MTRG**（Multi-Task Reference and Goal-Driven RL；论文正文未给出官方缩写，本库用作导航标签）
- **类型：** paper / humanoid whole-body control / parkour / sim-to-real
- **arXiv：** <https://arxiv.org/abs/2602.20375>（PDF：<https://arxiv.org/pdf/2602.20375>）
- **演示视频：** <https://youtu.be/9NamvWhtFPM>
- **作者：** Jiashun Wang, M. Eva Mungai, He Li, Jean Pierre Sleiman, Jessica Hodgins, Farbod Farshidian
- **机构：** RAI Institute, Carnegie Mellon University
- **入库日期：** 2026-06-12
- **一句话说明：** 单一 **goal-conditioned** 策略在**共享观测/动作空间**上联合优化两类任务——**(i) 参考塑形模仿**（参考只进奖励、不进策略输入）与 **(ii) 纯目标泛化**（随机目标、稀疏任务奖励）——无需对抗、相位或部署时参考轨迹，在 **Unitree G1** 箱式跑酷场上实现 walk-jump / walk-climb / climb-down 的 OOD 初始条件泛化与长程技能组合。

## 核心摘录

### 1) 核心 trade-off

- **Reference tracking**（DeepMimic、GMT、ZEST）：动作自然但部署耦合参考，OOD 初始/目标易失败（如 ZEST 在 beyond-nominal 上 success 骤降）。
- **Tabula rasa RL**：可完成任务但动作质量差、需重 reward shaping。
- **AIL/HIL**：分布匹配更灵活，但动态接触下判别器难稳、与任务奖励冲突。

### 2) 统一多任务 MDP

- **观测**：本体状态 \(s_t\)（关节位姿速度、投影重力、躯干角速度，角色坐标系）+ **目标** \(g_t\)（躯干根在水平面的 2D 位置，相对角色）。
- **策略不接收**：参考轨迹、未来姿态、相位变量。
- **动作**：关节 PD **残差** \(\bm{q}^{cmd}=\bar{\bm{q}}+\bm{\Sigma}\bm{a}_t\)；**省略**参考关节前馈项，保留偏离参考的能力。

| 任务 | 目标 \(g_t\) 来源 | 奖励 |
|------|------------------|------|
| **Reference-guided imitation** | 由参考轨迹导出 | 稠密 tracking \(r_{track}\) + 正则 + survival |
| **Goal-conditioned generalization** | 与参考无关随机采样 | 稀疏 goal \(r_{goal}\)（位置/朝向进度 + reach bonus）+ 正则 + survival |

- **Critic**：非对称 actor-critic；critic 额外接收 **task indicator** \(k_t\) 与仿真特权信息（接触力、辅助扳手等）。

### 3) 课程（继承 ZEST [25]）

- 全局难度标量 \(\lambda\in[0,1]\) 在线调节：
  - **虚拟空间辅助扳手** \(\mathbf{w}_e=\beta(\lambda)[\mathbf{F}_b;\mathbf{M}_b]\)，低 \(\lambda\) 时强辅助、训练后期衰减至零；
  - **任务混合**：\(p_{imi}(\lambda)=(1-\lambda)p_0+\lambda p_{target}\)，从以模仿为主平滑过渡到模仿+泛化混合；
  - 随 \(\lambda\) 扩大初始状态与目标采样范围。
- **初始化**：模仿任务近参考小扰动；泛化任务宽分布随机状态与目标。

### 4) 实验（G1，29 DoF，Isaac Lab + PPO）

- **技能**：walk-jump、walk-climb、climb-down（每技能一条参考 + 镜像增广）。
- **对比**（Table I）：相对 **ZEST mocap**（参考作为策略输入跟踪）与 **tabula rasa RL**，在 nominal 与 beyond-nominal 初始条件下 **success rate 最高**；策略会按距离选择先走再跳或直接跳、左右腿自适应攀爬等，而非回放单条参考。
- **消融**：去掉任务课程或模仿分支 → beyond-nominal success 崩溃；去掉泛化分支 → OOD 鲁棒性显著下降。
- **长程组合**：MuJoCo 中将 walk-climb / walk-jump / climb-down 策略串联执行多箱跑酷序列。

### 5) 与相关工作的关系

- 明确将 **[HIL](../papers/hil_hybrid_imitation_learning_arxiv_2505_12619.md)** 列为「tracking + 对抗」联合训练对照，指出对抗目标难硬件调参。
- **[ZEST](zest.md)** 提供 assistive-wrench 课程与 G1 执行器建模基线；MTRG 将参考从「部署约束」降为「训练塑形」。

## 对 wiki 的映射

- 新建方法页：[`wiki/methods/mtrg-reference-goal-driven-rl.md`](../../wiki/methods/mtrg-reference-goal-driven-rl.md)
- 交叉更新：
  - [`wiki/methods/zest.md`](../../wiki/methods/zest.md) — 辅助扳手课程与 tracking 基线对照
  - [`wiki/methods/hil-hybrid-imitation-learning.md`](../../wiki/methods/hil-hybrid-imitation-learning.md) — 同主题「参考 + 任务」但对抗分支
  - [`wiki/tasks/humanoid-locomotion.md`](../../wiki/tasks/humanoid-locomotion.md) — G1 箱式跑酷泛化
  - [`wiki/concepts/curriculum-learning.md`](../../wiki/concepts/curriculum-learning.md) — \(\lambda\) 耦合扳手与任务采样

## 参考来源（原始）

- 论文：<https://arxiv.org/abs/2602.20375>
- 视频：<https://youtu.be/9NamvWhtFPM>
