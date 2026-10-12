# RACER 论文来源归档

## 元数据

- **题目：** RACER: Residual-Adaptive Closed-Loop Estimation for Sampling-Based Planning in Wheeled-Quadruped Racing
- **作者：** Yuxiang Liu, Marla Eisman, Lizhi Yang, Aaron Ames, Francesco Borrelli
- **单位：** 加州大学伯克利分校（Liu、Eisman、Borrelli）；加州理工学院（Yang、Ames）
- **预印本：** https://arxiv.org/abs/2610.07409
- **HTML：** https://arxiv.org/html/2610.07409v1
- **PDF：** https://arxiv.org/pdf/2610.07409
- **版本：** v1，2026-10-05，cs.RO
- **代码与数据：** arXiv 页面与正文均未提及项目页、仓库或数据发布；截至 2026-10-12 无公开代码。

## 论文要解决的问题

轮足四足高速过弯时，MPPI 规划器若只用简单运动学模型（独轮车），无法预测下层 RL 速度跟踪器在轮胎打滑、延迟和执行器饱和下的真实闭环行为；而在真机上采集大量数据又昂贵。论文要同时解决“规划模型失配”与“少量实机数据下的域适应”。

## 方法与训练信息

1. **分层结构：** 上层 DIAL-MPC 式采样规划器（MPPI 家族）以 10 Hz 采样速度指令并 rollout；下层为 50 Hz 前馈 PPO 速度跟踪器（用 IsaacLab 扩展 robot_lab 训练），输入关节角、关节速度、基座位姿与速度指令，输出 16 维：12 个腿关节位置目标 + 4 个轮速目标；奖励设计使其行为尽量接近独轮车模型 f_uni。
2. **名义模型 + 残差：** 规划器预测 s_{t+1} = f_uni(s_t, a_t) + g_φ(s_t, a_t)；状态 5 维 [x, y, yaw, 体坐标 v_x, 偏航角速度 ω_z]，动作 [v_x^des, ω_z^des]；残差网络 g_φ 为 5 层 MLP，输入输出空间与 f_uni 相同（各层宽度原文未给出）。
3. **LoRRA（Low-Rank Residual Adaptation）：** 损失为残差 MSE ||g_φ(s,a) − (s′ − f_uni(s,a))||²；第一阶段在约 1000 万条仿真转移上预训练；第二阶段用约 400 条实机转移在低秩约束下微调：W_l = W_l⁰ + (α/r)BA，W_l⁰ 冻结，r = 6、α = 0.35，每层统一；r 在 {2,4,6,8,10,12} 上于 sim-to-sim 摩擦差 0.55 处扫描；可训练参数占残差网络总参数的 17.7%。
4. **平台与赛道：** Unitree Go2-W（12 腿关节 + 4 轮电机），L 形赛道（尺寸原文未给）；参考轨迹由 CasADi 中动力学自行车模型的非线性规划离线预计算。
5. **域差设置：** 速度上限扫描（v_x ≤ c）；摩擦差：源域 μ_s = 0.85，目标 μ_t 取 {0.30, 0.35, …, 0.65}，每个 μ_t 每种方法 8 次独立试验；关节增益差：源 K_p,s = 70，仿真扫 K_p 差（具体数值原文未给）。
6. **基线与消融：** Naive Unicycle（去掉残差）、From-Scratch（只用目标域数据训练，沿用 Sym2Real 的思路）、Full-Tune（预训练后不加秩约束全量微调）。
7. **指标：** 圈平均速度（发生不期望接触则该圈速度记 0）、与参考轨迹的最大偏差、固定时间内无碰撞圈数。

## 论文报告的结果

**硬件，残差消融（Table I；3 次运行均值±标准差，每次 3 圈）：**

| 速度上限 | 方法 | 无碰撞圈 | 平均速度 (m/s) | 最大偏差 (m) |
|---|---|---|---|---|
| 1.5 m/s | RACER | 3.0 ± 0.0 | 1.21 ± 0.01 | 0.42 ± 0.13 |
| 1.5 m/s | Naive Unicycle | 2.3 ± 1.2 | 1.49 ± 0.04 | 0.59 ± 0.12 |
| 2.0 m/s | RACER | 3.0 ± 0.0 | 1.41 ± 0.02 | 0.45 ± 0.08 |
| 2.0 m/s | Naive Unicycle | 0.0 ± 0.0 | 0.00 | 0.83 ± 0.57 |
| 3.2 m/s | RACER | 3.0 ± 0.0 | 1.95 ± 0.04 | 0.35 ± 0.03 |
| 3.2 m/s | Naive Unicycle | 0.0 ± 0.0 | 0.00 | 0.80 ± 0.22 |

**硬件，LoRRA 消融（Table II；3 个种子，最多 3 圈无碰撞）：**

| ΔK_p | 方法 | 无碰撞圈 | 平均速度 (m/s) |
|---|---|---|---|
| 0 | LoRRA | 3.00 ± 0.00 | 1.94 ± 0.07 |
| 0 | Full-Tune | 2.67 ± 0.47 | 2.06 ± 0.05 |
| 0 | From-Scratch | 0.33 ± 0.47 | 0.59 ± 0.83 |
| 30 | LoRRA | 2.33 ± 0.94 | 1.61 ± 0.08 |
| 30 | Full-Tune | 1.00 ± 0.82 | 0.60 ± 0.44 |
| 30 | From-Scratch | 0.00 ± 0.00 | 0.00 ± 0.00 |
| 50 | LoRRA | 2.33 ± 0.94 | 1.29 ± 0.10 |
| 50 | Full-Tune | 0.00 ± 0.00 | 0.00 ± 0.00 |
| 50 | From-Scratch | 1.00 ± 1.41 | 0.80 ± 1.14 |

**仿真（仅趋势，正文无数值）：** RACER 性能随速度上限基本稳定，Naive Unicycle 在高上限处退化；Naive Unicycle 的残差预测误差在高速过弯处尖峰；LoRRA 相对 Full-Tune 的优势随摩擦差增大而增大，关节增益差结果更嘈杂；From-Scratch 平均最差且方差最大；LoRRA 适配后源域损失最低。

## 复现与解释边界

- 硬件试验每个条件仅 3 次运行/种子，方差大；ΔK_p = 0 时 Full-Tune 平均速度（2.06）高于 LoRRA（1.94）；Naive Unicycle 在 1.5 m/s 下平均速度（1.49）也高于 RACER（1.21），但无碰撞圈数较少——比较须结合无碰撞圈数。
- 作者承认关节增益差结果更嘈杂，归因于增益失配时策略立即失效（摔倒）。
- 作者列出的未来工作：硬件上在线安全地适配残差动力学；非结构化环境下的残差适配。
- 论文未提及代码或数据发布。

## Robotics_Notebooks 映射

- 详情节点：wiki/entities/paper-racer-residual-adaptive-wheeled-quadruped-racing.md
- 汇总页：wiki/overview/frontier-humanoid-quadruped-2026-10-05-10-09.md
- 概念入口：wiki/concepts/wheel-legged-quadruped.md
- 方法入口：wiki/methods/mppi.md、wiki/concepts/lora.md

## 一手来源

- arXiv 摘要：<https://arxiv.org/abs/2610.07409>
- arXiv HTML：<https://arxiv.org/html/2610.07409v1>
