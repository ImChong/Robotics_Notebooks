# Actuator-Constrained Reinforcement Learning for High-Speed Quadrupedal Locomotion（arXiv:2312.17507）

> 论文来源归档（ingest）

- **标题：** Actuator-Constrained Reinforcement Learning for High-Speed Quadrupedal Locomotion
- **类型：** paper / quadruped locomotion / high-speed running / actuator modeling / reinforcement-learning / sim2real
- **arXiv：** <https://arxiv.org/abs/2312.17507> · HTML：<https://arxiv.org/html/2312.17507v1> · PDF：<https://arxiv.org/pdf/2312.17507.pdf>
- **DOI：** <https://doi.org/10.48550/arxiv.2312.17507>
- **作者：** Young-Ha Shin、Tae-Gyu Song、Gwanghyeon Ji、Hae-Won Park（通讯）
- **机构：** KAIST 机械工程 · 人形机器人研究中心（Humanoid Robot Research Center）
- **资助：** 韩国国防采办项目管理局（DAPA）通过国防发展局（ADD）Challengeable Future Defense Technology（2022）
- **平台：** KAIST Hound（45 kg 自研四足，ICRA 2022 齿轮箱 MINLP 优化设计）；RaiSim 仿真；机载 ThinkStation P350 Tiny；Elmo Platinum Twitter 100 V/70 A（EtherCAT 2 kHz）
- **项目页 / 代码：** 截至 **2026-07-20** 入库日，**arXiv 页未列专用项目页或 GitHub**；论文未声明 code availability 段落。同团队后续 [APT-RL](https://skillquadsr.github.io/) 有 Zenodo 数据，**本篇训练栈视为未开源**。
- **入库日期：** 2026-07-20
- **一句话说明：** 在 RL 训练中将 **电机扭矩–转速工作区（MOR）** 作为线性不等式约束写入仿真：关节力矩经 **减速器矩阵** 映射到电机空间后按规格书 **clip**，并配合 **对称步态奖励** 与 **轻量化足端**，使 KAIST Hound 在跑步机上达到 **6.5 m/s**（当时纯电驱四足最快纪录），且无 MOR 约束的策略在 **5 m/s 实机摔倒**。

## 相关资料（策展）

| 类型 | 链接 | 说明 |
|------|------|------|
| 同平台后续 | [APT-RL（Science Robotics 2026）](../../wiki/entities/paper-apt-rl-agile-perceptive-quadruped-locomotion.md) | 同 HOUND 平台；感知多技能与野外长程 |
| HOUND 机体设计 | Shin et al., ICRA 2022 | MINLP 齿轮比优化；并行 HFE/KFE 传动 |
| 并发训练框架 | Ji et al., RA-L 2022 [17] | 策略与状态估计器并发训练（本文基线栈） |
| 速度纪录对照 | MIT Cheetah 2, IJRR 2017 [10] | 此前纯电驱腿式 **6.4 m/s** |
| 约束 RL 后续 | Kim et al., 2023 [31] | 「Not only rewards but also constraints」腿足 locomotion |

## 摘要级要点

- **问题：** 高速四足奔跑时电机输出逼近 **τ–ω 包络边界**；仿真 URDF 常把关节力矩/速度限为 **矩形常数**，与真实 **梯形 MOR** 不符，导致策略在仿真中学到 **不可实现的力矩指令**，sim2real 性能断崖。
- **MOR 建模：** 准静态 DC 电机方程 $V_{\text{bus}}$ 界定第一象限梯形工作区；考虑铁芯磁饱和 **τ_peak**（偏离线性 20%）；第二/四象限（再生制动）包络更宽。
- **关节↔电机映射：** HOUND 并行 HFE/KFE 用 **叠加原理** 得 gearbox 矩阵（式 3、10）；仿真中 PD 输出关节力矩 → 电机力矩 → **MOR clip** → 饱和关节力矩施加。
- **电流–力矩非线性：** 铁芯电机高扭矩区 $i$–$\tau$ 二次拟合补偿；推理时电流映射与仿真力矩对齐。
- **步态奖励：** 奖励 **RR+FL 触地且 RL+FR 摆动** 或对称对角步态，避免单侧电机功率偏置导致饱和瓶颈。
- **轻量化足端：** 仿真分析攻击/离地角后，足端由圆柱改为 **局部球面**；小腿质量 **−62%**、俯仰惯量 **−62.6%**（论文：质量 38.0%、惯量 37.4% 为相对原设计保留比例）。
- **训练：** 400 并行环境、RaiSim、~6 h（RTX 3080 Ti）；速度课程 **U(-0.3,1.5)→U(-1.4,7.0) m/s**；网络 [256,128,64]；控制 **100 Hz**。
- **实机：** 跑步机 **6.5 m/s** 维持 10 s+；户外塑胶跑道 **100 m / 19.87 s**（估 **5.9 m/s**）；CoT **0.29**（15 步，机械功率+焦耳热）。
- **消融（Table II）：** 无步态奖励 **6.0**、无定制足 **5.5**、无 MOR **4.5 m/s**；无 MOR 策略仿真可达 6.5 m/s 但 **5 m/s 实机摔倒**。

## 核心摘录（面向 wiki 编译）

### 1) MOR 约束与 sim2real 机制

- **要点：** 训练期 clip 违规电机力矩，使策略 **采样时主动避开不可行区**（Fig. 12：违规采样比例收敛更低）；评估时用 MOR 约束无 MOR 训练的策略，**>3.5 m/s** 起 reward gap 扩大，解释实机失败。
- **对 wiki 的映射：** [`wiki/entities/paper-actuator-constrained-rl-high-speed-quadruped-locomotion.md`](../../wiki/entities/paper-actuator-constrained-rl-high-speed-quadruped-locomotion.md)、[`wiki/concepts/sim2real.md`](../../wiki/concepts/sim2real.md)

### 2) 并行关节传动与执行器建模

- **要点：** URDF 矩形限矩 vs 电机空间梯形 MOR；减速器矩阵对并联腿的必要性。
- **对 wiki 的映射：** [`wiki/concepts/implicit-explicit-actuator-modeling.md`](../../wiki/concepts/implicit-explicit-actuator-modeling.md)、[`wiki/concepts/humanoid-parallel-joint-kinematics.md`](../../wiki/concepts/humanoid-parallel-joint-kinematics.md)

### 3) 高速四足工程（足端 + 步态奖励）

- **要点：** 高速仅需覆盖攻击/离地角的 **局部球足**；对称步态奖励均衡四腿功率。
- **对 wiki 的映射：** [`wiki/tasks/locomotion.md`](../../wiki/tasks/locomotion.md)

### 4) 开源状态

- **要点：** **无项目页、无官方代码**；复现需自建 RaiSim + 并发估计器栈 + HOUND 动力学参数。
- **对 wiki 的映射：** 实体页「局限与风险」

## 对 wiki 的映射

- 主沉淀：**[`wiki/entities/paper-actuator-constrained-rl-high-speed-quadruped-locomotion.md`](../../wiki/entities/paper-actuator-constrained-rl-high-speed-quadruped-locomotion.md)**
- 交叉：**[`wiki/concepts/sim2real.md`](../../wiki/concepts/sim2real.md)**、**[`wiki/tasks/locomotion.md`](../../wiki/tasks/locomotion.md)**、**[`wiki/entities/paper-apt-rl-agile-perceptive-quadruped-locomotion.md`](../../wiki/entities/paper-apt-rl-agile-perceptive-quadruped-locomotion.md)**、**[`roadmap/depth-sim2real.md`](../../roadmap/depth-sim2real.md)**
