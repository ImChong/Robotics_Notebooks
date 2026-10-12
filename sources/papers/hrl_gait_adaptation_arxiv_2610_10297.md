# 分层强化学习节能步态适配 论文来源归档

## 元数据

- **题目：** Energy-Efficient Gait Adaptation via Hierarchical Reinforcement Learning for Quadrupedal Locomotion Across Diverse Terrains
- **作者：** Ammar Issa, Anubhav Singh, Anton Tsaritsin, Sergey Kolyubin（前三位同等贡献）
- **单位：** arXiv 页面未列机构；周报标注为 ITMO 大学、BE2R 实验室（未在 arXiv 页面核实）
- **预印本：** https://arxiv.org/abs/2610.10297
- **HTML：** https://arxiv.org/html/2610.10297v1
- **版本：** v1，2026-10-07，cs.RO（交叉 cs.LG）；9 页，投稿 IEEE ICRA 2027
- **代码与项目页：** arXiv 页面未列，截至 2026-10-12 未发现官方代码。

## 论文要解决的问题

固定步态（如全程小跑）难以同时适配速度与地形变化；直接在端到端策略里加能耗惩罚（如 AER）又可能损害稳定性，甚至让机器人蹲伏失稳。论文把“稳定关节控制”和“能耗导向的步态参数选择”分到两个频率不同的策略。

## 方法与训练信息

1. **低层（LL）策略：** 50 Hz；58 维观测（投影重力、指令速度与步态参数、关节偏移、关节速度、上一动作、4 路相位时钟，含 5 步历史）；12 维关节位置目标，经关节 PD 跟踪；沿用 Walk These Ways（MoB）框架。
2. **高层（HL）策略：** 2 Hz（每个动作保持 H=25 个低层步）；6 维状态（窗口平均机体系 vx、vy、偏航率 + 窗口末机体系重力）；非对称 actor-critic，critic 额外看到指令步态参数；11 维动作：3 个腿间相位偏移 θ1–θ3、步频、机身高度、横滚/俯仰、步幅/步宽、摆动高度、支撑时长占比。
3. **HL 奖励：** r_HL = r_energy − 0.1·‖g_k − g_{k−1}‖² + 0.002·Σ r_LL；r_energy = exp(−Σ|P|/(σ_x|v̄x|+σ_z|ω̄z|))，P=τ·q̇（σ 值原文未给）。CoT = Σ|P_i|/(m·g·|vx|)（机器人质量未给）。
4. **三阶段训练（Isaac Gym，PPO，4096 并行环境）：** (1) 平地预训练 LL（WTW 超参不变）；(2) 冻结 LL，在复杂地形（平地、0–10 cm 粗糙地面、0.2/0.3/0.36 rad 上下坡）训练 HL，关闭速度课程；(3) 冻结 HL，在复杂地形上短时微调 LL。迭代数与墙钟时间未报告。

## 论文报告的结果

- **仿真对比（Go1，平地，Table III CoT）：** 速度 (0.5,1.0] m/s：HRL 0.421 vs AER(α=1) 0.618 vs WTW-Trotting 0.688 vs F&E 1.663；(1.0,1.5]：0.384 vs 0.681 vs 0.680 vs 1.240；(0,0.5]：HRL 0.762，AER 0.572（AER 更低）。论文给出平均 CoT 降幅：对 AER 27.7%、对 WTW 33.5%、对 F&E 70.5%。
- **跌倒率（Table II，每格 500 次）：** 平地 0%；粗糙地面仅 1.5–2.0 m/s 区间 0.6%；坡面最高 2.4%（0–0.5 m/s）。
- **AlienGo 消融（Table IV）：** 去掉 HL 能耗项后 CoT 暴增（平地 0.5–1.0 m/s 从 0.734 升到 8.539）；仅平地训练 HL 在粗糙地面最低速 CoT 5.029，复杂地形训练为 1.846。
- **LL 微调（阶段 3）：** 速度跟踪 RMSE 平地从 0.165 降到 0.039 m/s，粗糙地面从 0.182 降到 0.047 m/s。
- **步态：** 低速 pace（两拍），中速四拍行走，高速 trot，无预设切换规则；原文未给具体速度阈值。20° 上坡时机身高度与步幅下降，俯仰和步宽上升。
- **真机（Unitree AlienGo，零样本）：** 关节 PD Kp=50、Kd=2，Unitree SDK；坡度到 20°；平地、斜坡、不平粗糙地形共 10 次试验无摔倒；AER 风格 α=0.5 的端到端策略在 AlienGo 上因关节力过低出现不稳定蹲姿。

## 复现与解释边界

- 无公开代码；σ 参数、机器人质量、迭代数缺失，难以复现 CoT 数值。
- 仿真基线对比用 Go1，消融和真机用 AlienGo，两者不可直接混读。
- 仅本体感知；HL 更新率低（2 Hz）可能限制对突变地形的响应；对分频因子 H 敏感；LL 的误差会沿层级传播。
- 真机试验为 10 次定性级别，未给出真机 CoT。

## Robotics_Notebooks 映射

- 详情节点：wiki/entities/paper-hrl-energy-efficient-gait-adaptation-quadruped.md
- 任务入口：wiki/tasks/locomotion.md
- 汇总页：wiki/overview/frontier-humanoid-quadruped-2026-10-05-10-09.md

## 一手来源

- arXiv 摘要：<https://arxiv.org/abs/2610.10297>
- arXiv HTML 全文：<https://arxiv.org/html/2610.10297v1>
