# ZEST / Embodied skill transfer for locomotion control

> 来源归档（ingest）

- **期刊标题：** Embodied skill transfer for locomotion control
- **预印本标题：** ZEST: Zero-shot Embodied Skill Transfer for Athletic Robot Control
- **短名：** ZEST
- **类型：** paper / humanoid / locomotion / motion-imitation / sim2real / multi-contact
- **期刊：** *Science Robotics* 11(117)，2026-08-12
- **DOI：** [10.1126/scirobotics.aec7695](https://doi.org/10.1126/scirobotics.aec7695)
- **arXiv：** <https://arxiv.org/abs/2602.00401>
- **PDF（可直接读）：** <https://arxiv.org/pdf/2602.00401>
- **HTML：** <https://arxiv.org/html/2602.00401v1>
- **项目页：** 无独立项目页
- **代码：** 论文 *Data and materials availability* 仅写「结论所需数据见正文/附录」；检索未见官方 GitHub / HF → **确认未开源**
- **作者（核心贡献者，按贡献排序）：** Jean-Pierre Sleiman、He Li、Alphonsus Adu-Bredu、Robin Deits、Arun Kumar
- **项目负责人（按贡献倒序）：** Alfred Rizzi、Jessica Hodgins、Sylvain Bertrand、Yeuhi Abe、Scott Kuindersma、Farbod Farshidian（通讯）
- **机构：** 机器人与人工智能研究所（RAI Institute）；波士顿动力（Boston Dynamics）。Atlas / Spot 实验为全作者；Unitree G1 由 RAI 完成。
- **入库日期：** 2026-08-15
- **最后更新：** 2026-08-15
- **一句话说明：** 单阶段、极简 MDP 的运动模仿：MoCap / 单目视频 / 关键帧动画三类异构参考，不靠接触标签、历史窗、状态估计器或重奖励塑形，Isaac Lab PPO 训完后零样本上 Atlas、G1、Spot。

## 摘要级要点

- **问题：** 全尺寸人形要做多接触、高动态全身技能时，模型基 MPC/WBC 需要接触时刻表与环境建模；tabula rasa RL 又对奖励极敏感。现有模仿管线常叠接触标注、多阶段训练、观测/参考窗口或状态估计。
- **方法：** ZEST 把参考当下一步目标，策略只看本体感知 + 下一步参考 + 上一步动作，输出残差关节目标叠加到参考后再进 PD。训练侧用 **自适应 RSI**（按 bin 失败率 EMA 采样）和 **模型基辅助扳手课程**（\(\beta\) 随失败率衰减到 0）。
- **执行器：** 闭链 PLA（膝/踝/腰）用递进近似（质量可忽略连杆 → Jacobi 对角化 → 名义构型固定电枢），再按 \(K_p=I\omega_n^2\)、\(K_d=2I\omega_n\) 选增益；Spot 另加功率限制、磁饱和与正负功效率。
- **数据：** MoCap（Xsens / Vicon）→ 高保真；ViCap = MegaSaM + TRAM，手持手机「上午拍、当天训、晚上上机」；动画补非人构型（Atlas 背偏航倒立、Spot 连续后空翻）。运动学重定向做时空优化，**不标接触**。
- **开源（截至 2026-08-15）：** 无项目页、无仓库。未建 `sources/repos/` / `sources/sites/`。

## 核心摘录（面向 wiki 编译）

### 极简 MDP

- Actor：\(\mathbf{o}_t=(\mathbf{o}_{\mathrm{prop}},\mathbf{o}_{\mathrm{ref}})\)。本体 = IMU 角速度、投影重力、关节位置/速度、上一动作；**不要**全局位姿或线速度。参考 = 下一步基座高度、基座线/角速度、重力方向、参考关节位置。
- 动作：\(\bm{q}_j^{\mathrm{cmd}}=\hat{\bm{q}}_j+\bm{\Sigma}\bm{a}_t\)。髋膝尺度更大，头腕更小。
- Critic 特权：基座线速度/高度、末端位姿与接触力、课程扳手信号。
- 奖励：跟踪（指数核）+ 正则（动作变化、关节加速度、限位）+ 存活；**无任务专用项、无接触标签**。
- 算法：Isaac Lab + PPO；非对称 actor–critic；单卡 NVIDIA L4 ≈10 h / 7k iter 训一条技能策略。

### 自适应采样与辅助扳手

- 轨迹切成固定步长 bin；每 bin 维护失败水平 \(f_b\)（EMA）。Categorical 按 \(f_b\) 偏置 reset，并留地板概率防遗忘。
- 辅助扳手 = 基座位姿 PD + 躯干名义动力学前馈，再乘 \(\beta\in(0,\beta_{\max})\)、\(\beta_{\max}<1\)。难 bin 先给更大辅助，跟踪上来后自动退火到 0。
- 消融（15 条 Atlas 多技能、全 DR、1e4 rollout）：去掉课程 ≈ 基线 10 h 的 20 h 表现；去掉自适应采样则难 bin 欠采样；20 步历史或 20 步未来参考（0.4 s）反而伤收敛；绝对动作最差。

### 真机技能与跟踪误差（Table 1，成功试验均值）

| 来源 | 技能 | 平台 | 多接触 | 时长 (s) | MAE \(q\) (rad) | MAD \(R\) (rad) |
|------|------|------|--------|----------|-----------------|-----------------|
| MoCap | Walk / Jog | Atlas | 否 | 8.47 / 5.89 | 0.057 / 0.041 | 0.030 / 0.059 |
| MoCap | Army crawl / Breakdance | Atlas | 是 | 16.66 / 6.56 | 0.103 / 0.079 | 0.075 / 0.133 |
| MoCap | Cartwheel / Table-tennis | G1 | 是 / 否 | 8.0 / 29.98 | 0.078 / 0.108 | 0.331 / 0.402 |
| ViCap | Dance A / Soccer kick | Atlas | 否 | 12.70 / 6.00 | 0.051 / 0.049 | 0.039 / 0.050 |
| ViCap | Climb up / down box | G1 | 是 | 9.18 / 10.62 | 0.073 / 0.108 | 0.385 / 0.394 |
| 动画 | Handstand invert | Atlas | 是 | 5.18 | 0.074 | 0.082 |
| 动画 | Continuous backflip | Spot | 否 | 6.00 | 0.192 | 0.150 |

G1 箱体：名义高 0.75 m、训练随机 [0.70, 0.80] m；连续 5/5 成功；训练分布内 \(\pm 10\) cm \(x\)、\(\pm 0.3\) rad yaw 全成功；躯干加 2 kg 仍成功；climb-up 可到 0.55 m，climb-down 可到 0.55 / 0.95 m。策略**不看箱子位姿**，只靠本体 + 初始相对位姿随机。

### 相对 BD 全身 MPC（Table 2，仿真）

- Walk：MPC 与 RL 接近（MAE \(q\) 0.047 vs 0.055）。
- Jog / Cartwheel：RL 更好（jog MAE \(q\) 0.117 vs 0.076；cartwheel 0.237 vs 0.088）。
- MPC 失败（–）：Dance B（接触标注滑步）、Handstand invert（手臂点接触力矩饱和）、Roll on all fours（长时脚趾接触）。膝/躯干/前臂接触技能直接不在 MPC 默认能力内。

### 开源核查（步骤 2.5）

- 无 `*.github.io` / 机构项目页列出 Code。
- Science Robotics 数据可用性仅指向正文与附录；arXiv 同期亦无仓库。
- 检索「ZEST embodied skill transfer github」无官方实现。
- → **确认未开源**（截至 2026-08-15）。未建 `sources/repos/` / `sources/sites/`。

## 对 wiki 的映射

- 升格 [ZEST 论文实体](../../wiki/entities/paper-zest.md)
- 方法页 [ZEST](../../wiki/methods/zest.md)
- 交叉：[MTRG](../../wiki/methods/mtrg-reference-goal-driven-rl.md)、[HIL](../../wiki/methods/hil-hybrid-imitation-learning.md)、[HIL vs MTRG vs ZEST](../../wiki/comparisons/hil-vs-mtrg-vs-zest-parkour-imitation.md)、[Curriculum Learning](../../wiki/concepts/curriculum-learning.md)、[Sim2Real](../../wiki/concepts/sim2real.md)、[VideoMimic](../../wiki/entities/videomimic.md)、[Boston Dynamics](../../wiki/entities/boston-dynamics.md)、[Unitree G1](../../wiki/entities/unitree-g1.md)、[运动跟踪选型](../../wiki/queries/humanoid-motion-tracking-method-selection.md)

## 当前提炼状态

- [x] Science Robotics DOI / 卷期与 arXiv PDF 对齐
- [x] 开源结论：确认未开源
- [x] 升格论文实体并交叉引用
