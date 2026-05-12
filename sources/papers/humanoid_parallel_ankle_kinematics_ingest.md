# humanoid_parallel_ankle_kinematics_ingest

> 来源归档（ingest）

- **标题：** 人形并联踝 / 并联关节运动学与训练—部署链路文献包
- **类型：** paper
- **来源：** ResearchGate（期刊/会议预印索引页）、arXiv
- **入库日期：** 2026-05-12
- **一句话说明：** 覆盖经典机构学踝部分析、GPU 大规模 RL 中的闭链踝动力学嵌入，以及「串联主链 + 运动学驱动映射」的解析执行器模型与真机动态行为案例。

## 核心论文摘录

### 1) On the Comprehensive Kinematics Analysis of a Humanoid Parallel Ankle Mechanism（ResearchGate 索引页）

- **链接：** <https://www.researchgate.net/publication/326528067_On_the_Comprehensive_Kinematics_Analysis_of_a_Humanoid_Parallel_Ankle_Mechanism>
- **核心贡献：** 面向人形**并联踝**机构的**系统性运动学分析**议程：正/逆运动学、工作空间与奇异位形等典型并联机构问题在人形踝几何下的展开；适合作为「只做串联等效」之前的机构学对照基准。
- **对 wiki 的映射：**
  - [人形机器人并联关节解算](../../wiki/concepts/humanoid-parallel-joint-kinematics.md)

### 2) Kinematic Analysis of a Novel Parallel 2SPRR+1U Ankle Mechanism in Humanoid Robot（ResearchGate 索引页）

- **链接：** <https://www.researchgate.net/publication/325264356_Kinematic_Analysis_of_a_Novel_Parallel_2SPRR1U_Ankle_Mechanism_in_Humanoid_Robot>
- **核心贡献：** 提出 **2SPRR+1U** 拓扑的人形踝并联机构并完成**运动学分析**；强调「两路转动副 + 球副 + 万向节」一类构型在足端两自由度合成中的耦合方式，可与 RSU/四杆投影等工程抽象对照阅读。
- **对 wiki 的映射：**
  - [人形机器人并联关节解算](../../wiki/concepts/humanoid-parallel-joint-kinematics.md)

### 3) Design of a 3-DOF Hopping Robot with an Optimized Gearbox（arXiv:2505.12231）

- **链接：** <https://arxiv.org/abs/2505.12231>
- **核心贡献：** 类人布局**单腿跳跃平台**（膝 1 DoF + 踝 pitch/roll）；踝部采用**闭链并联机构**（万向节在足端、执行器输出端球铰等描述），两路执行器轴平行布置仍合成两轴足端运动；仿真侧在 **RaiSim** 用 **pin 约束**在运动树上强制闭链一致性；硬件上验证连续跳跃与抗扰。
- **对 wiki 的映射：**
  - [人形机器人并联关节解算](../../wiki/concepts/humanoid-parallel-joint-kinematics.md)

### 4) Learning Impact-Rich Rotational Maneuvers via Centroidal Velocity Rewards and Sim-to-Real Techniques（arXiv:2505.12222）

- **链接：** <https://arxiv.org/abs/2505.12222v2>
- **核心贡献：** 同一硬件平台上的**高冲击空翻类动作**与 sim-to-real；除质心角速度奖励、**电机工作区（MOR）**建模外，显式指出：**闭链踝**在落地冲击下会出现**传载路径再分配**——仅压接触冲量未必降低执行器侧峰值载荷，需在奖励/屏障中引入**传动负载正则**等项。并联运动学在此文中主要作为**真实动力学与损伤模式**的背景约束。
- **对 wiki 的映射：**
  - [人形机器人并联关节解算](../../wiki/concepts/humanoid-parallel-joint-kinematics.md)
  - [Sim2Real](../../wiki/concepts/sim2real.md)

### 5) LiPS: Large-Scale Humanoid Robot Reinforcement Learning with Parallel-Series Structures（arXiv:2503.08349）

- **链接：** <https://arxiv.org/abs/2503.08349>
- **核心贡献：** 针对 GPU 并行 RL 环境（如 Isaac Gym 系）中 URDF **开环树**与真实**串并联踝**不一致的问题，提出 **LiPS**：在仿真中嵌入**多刚体闭链踝的局部运动学/动力学**（向量环求主动臂角、再推速度/加速度与逆雅可比关系），使策略在**训练阶段**即面对与部署更接近的并联动力学，减少「训练用串联、真机再换算」的缝隙。
- **对 wiki 的映射：**
  - [人形机器人并联关节解算](../../wiki/concepts/humanoid-parallel-joint-kinematics.md)
  - [Sim2Real](../../wiki/concepts/sim2real.md)

### 6) Control of Humanoid Robots with Parallel Mechanisms using Kinematic Actuation Models（arXiv:2503.22459）

- **链接：** <https://arxiv.org/abs/2503.22459>
- **核心贡献：** 对**四杆膝**与**双侧投影四杆式 2 DoF 并联踝**给出**解析**几何/运动学映射 \(q_m = f(q_s)\) 及其一阶（**Actuation Jacobian** \(J_A\)：\(\dot q_m = J_A \dot q_s\)，\(\tau_s = J_A^\top \tau_m\)）与高阶导数，使 **DDP/MPC（Crocoddyl）** 与 **RL** 能在**低维串联主链**上优化/学习，同时仍在模型中保留**变传动比、可行域在电机空间表达**等闭链效应；与「完全忽略闭链」的简化模型对比，粗糙地/楼梯等任务上可挖掘更大可行运动包络。
- **对 wiki 的映射：**
  - [人形机器人并联关节解算](../../wiki/concepts/humanoid-parallel-joint-kinematics.md)
  - [Trajectory Optimization](../../wiki/methods/trajectory-optimization.md)

## 当前提炼状态

- [x] 论文摘要填写
- [x] wiki 页面映射确认
- [x] 关联 wiki 页面的参考来源段落已添加 ingest 链接
