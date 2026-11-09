# RoboKit: Accelerating Parallel Differentiable Optimization for Robot Learning

- **类型：** CoRL 2026 Spotlight 论文（UC San Diego）
- **作者：** Jianglong Ye*, Xuanbin Peng*, Conor Mc Gartoll*, Changwei Jing*, Lai Wei, Jiali Chen, Xueyan Zou, Xiaolong Wang（*共同贡献；Jianglong Ye 为 project lead）
- **论文原文（首选）：** https://openrobokit.com/robokit/paper.pdf （26 页，包含附录 A–D）
- **OpenReview：** https://openreview.net/forum?id=xfZJ2GKrZG （浏览器验证可能阻断，论文正文以作者提供的 PDF 为准）
- **项目主页：** https://openrobokit.com/robokit/
- **官方代码：** https://github.com/openrobokit/robokit
- **首次入库：** 2026-11-07
- **全文复核：** 2026-11-09（作者 PDF：正文 §1–6、表 1–6、附录 A–D、表 7–13、算法 1–4、图 1–4，含参考文献）
- **通读：** 已完成

## 一句话摘要

RoboKit 将 Warp 上的 GPU 多初值采样与梯度/二阶近似/MPPI 优化器组合，借助稀疏解析 Jacobian、混合 voxel SDF/mesh、分层渲染和粗细物理模型加速跨任务优化；可微残差还可直接训练/蒸馏策略。论文作者在多个不同任务和协议下报告加速，**不是同一模型、同一指标上统一的倍数**。

## 方法与实现（正文 §3；附录 C）

### 求解变量、代价与优化器

1. 支持固定/转动/移动/mimic joints、浮动基座 SE(3)/SO(3) 位姿以及重定向缩放变量；FK 将关节配置映射为 link pose 和 spatial/body Jacobian。
2. 目标为可复用加权残差的平方和：末端位姿 SE(3) log map、人体/手指对应点、接触吸引、force closure、碰撞与自碰撞、关节/速度/加速度/jerk 限位、平滑与相机 mask alignment 等（附录 C.1 表 13）。
3. 高层 particle sampler 生成多初值、按残差排名、取 top-K 并可重采样；低层使用 GD/Adam、L-BFGS、GN/LM 或 MPPI。作者给 Warp 增加 COO 稀疏 Jacobian 支持；典型曲率型迭代 16–64 步。
4. **区分梯度路径：** analytic/torch 可微残差可对 policy 输出反传；物理 rollout 上的 MPPI 为**无仿真梯度的采样优化**；online direct-gradient distillation **不反向展开优化器**。

### 四类加速结构

- **Voxel SDF + mesh：** 远离接触用常数复杂度的栅格 SDF，接近表面阈值 τ=b·h（默认 b=4 voxel）时用精确 mesh query，兼顾速度与表面法线。
- **两级碰撞球：** link 外接粗球先拒绝远处物体，近障碍再查 link 的细粒度 spheres。
- **分层可微渲染：** 先低分辨率、多初值、mask blur，再提高分辨率并筛选最佳种子。
- **粗到细物理：** MPC 前期简化场景、加大步长/减少碰撞对，后期回归完整物理。

### 学习集成与算法 1–4（附录 C.2）

- **离线蒸馏：** 批量求解 → 按优化代价或仿真可行性过滤失败解 → 用 (observation, solution) 做监督拟合。
- **在线直接梯度：** policy 预测配置后，直接对各可微 residual 的加权平方和反向传播；非对 solver 做 unroll。
- **混合蒸馏：** 接触复杂的 grasp 用过滤后的离线数据预热再在线微调；手重定向等较简单任务可直接在线优化。
- **RL：** 并行 MuJoCo Warp 仿真运行 PPO，和 mjlab 的任务/物理对齐，评估达到基线终分的训练时间。

## 定量实验：按协议拆分（正文 §4）

**表 2：单 RTX 4090，CPU Intel Xeon Gold 5120，预热后计时。通常均值 5 次；人形重定向为 3 次，手重定向遍历所有帧 1 次。**

| 任务 / 指标 | RoboKit | 对照 | 解读 |
|---|---:|---:|---|
| Franka Panda, 2000 IK，整批时间 | 7.12 ms | PyRoKi 86.20 ms | ~12.1×；成功率 99.91% vs 99.89% |
| Franka 轨迹规划，平均规划时间 | 16.3 ms | cuRobo 56.6 ms | ~3.5×；成功率 99.65% vs 99.57%，运动完成时间略长 |
| Shadow Hand 在线重定向，每帧 | 1.25 ms | PyRoKi 33.19 ms | ~26.6×，position error 6.05 vs 10.76 mm |
| Unitree G1 全身重定向，每帧 | 2.00 ms | GMR 7.91 ms | ~4.0×，MPJPE 8.08 vs 9.26 cm |
| MANO 4096 grasps 优化 | 2.97 s | Dex1B 178.45 s | ~60×；成功率 46.83% vs 23.63%；Q1 **较低** 0.077 vs 0.109 |
| 真实噪声设定的 Panda 手眼标定 | 2.0 s / 1.10 cm | EasyHEC 458.5 s / 96.43 cm | EasyHEC 在该大初始化偏差协议下未有效收敛，**不应以 229× 宣称公平同精度加速** |
| OakInk 物理重定向，每帧 | 109 ms / ADD-AUC 65.8 | SPIDER 610 ms / 51.3 | ~5.6×，对照使用作者公开 rollouts |

手眼标定**easy split**下两者均可收敛：RoboKit 1.23 s vs EasyHEC 70.4 s（~57×）；GigaHands 物理重定向则 57 vs 125 ms、ADD-AUC 63.7 vs 50.3。正文给出的“2–60×”指各个任务上的优化器对比，**不能把这些倍数直接套到全部场景**。

**表 3：蒸馏策略**，在 RTX A5000 上按各自协议重测优化器：轨迹优化 150.61 → 0.63 ms/frame，成功率 96.72% → 95.90%；人形重定向 10.86 → 0.72 ms/frame、位置误差 7.15 → 9.92 cm；grasp 蒸馏成功率 71.3%，但与表 2 的优化基线不同协议。论文概括为**9–240×**，属于策略推理与优化调用之间的比较，不是优化器本身提速。

**表 4：强化学习**。6 个 Go1/G1/Yam 任务、4096 environments、5 seeds 中位数，以达到 mjlab 终分的 95%/100% 所需分钟计；例如 Go1 Flat 95% 阈值 mjlab 7.6 min、RoboKit 3.2 min；G1 Rough 77.6 vs 46.4 min。采用相同 MuJoCo Warp physics 和对齐任务配置，**不是与 Isaac Lab/真机同口径比较**。

## 消融与附录重点

- **表 5a**：手重定向 offline / online(supervised) / online(direct grad) 位置误差 4.71 / 4.34 / **2.71 mm**；不同任务不能泛化断言 pure online 最优。
- **表 5b–c**：grasp 的 offline → offline+online，成功率 57.91% → 65.48%；不筛选示范 → 筛选，23.83% → 57.91%，表明高质量数据/接触约束关键。
- **表 5d / 表 11**：mesh BVH LM 16.61 s、成功率 48.49%、穿透 0.55 mm；voxel SDF LM 2.97 s、成功率 46.83%、穿透 0.70 mm，速度更快但质量有权衡。
- **表 6**：Franka IK full Warp+analytic Jacobian+particles 1.82 ms / 100% 成功率；不用 Warp 139.59 ms；不用解析 Jacobian 17.93 ms；不用多粒子 1.64 ms、成功率降至 72.6%。
- **附录 A**：详细列基线、成功标准、采样规模、随机种子和计时协议；物理重定向对 SPIDER 只纳入通过其数据导出检查的 218/248 trials；其它 30 条被排除。
- **附录 B.1 表 7–8**：2000 IK 对 cuRobo 50.27 ms；在线 hand 1000 批 3.02 ms、G1 1000 批 13.129 ms；后者是**整批**测量且与表 2 不同 workload。
- **附录 B.2 表 9**：蒸馏手重定向对 GeoRT 的 fingertip error 23.0 vs 37.4 mm。
- **附录 B.3 表 10**：Go2 trot/jump/crate 三场景与 DIAL-MPC 比较：step time 34.8 vs 74.4、29.1 vs 65.7、163 vs 1354 ms；任务分数按各自指标解读。
- **附录 B.5 表 12**：CPU Pinocchio/Pink IK 的默认示例任务和 joint limit 约束不同，不能只比较耗时而忽略可比性。
- **附录 C.1 表 13**：详列 cost 与 Jacobian；voxel/mesh 法线在网格/三角面切换处可能不连续。
- **附录 C.2 算法 1–4**：采样/LM → 数据过滤与 offline train → online objective gradient → hybrid schedule；附录 D 为任务可视化。

## 重要边界与复现实验检查

1. 论文 §6：目标是 GPU 并行负载，单次 CPU IK 可能因 kernel launch 开销落后于 C++ 求解器；轨迹规划没有图搜索阶段，也不提供 inverse dynamics。
2. 表 2 的 RTX 4090 与表 3 的 RTX A5000、表 4 的训练时间不是同一速度指标；不得跨表相除形成“加速”宣称。
3. 该论文报告属于**作者实验**；本次未独立运行 GPU benchmark、PPO 训练、标定或实机测试；源码现有 Wuji 示例证明仿真入口，不等同机器人硬件部署认证。
4. 注意区分“cost 可微”“solver 有解析 Jacobian”“physics-based MPPI rollout 无梯度”三个层级。

## 来源与 wiki 映射

- 官方完整 PDF：https://openrobokit.com/robokit/paper.pdf
- 作者主页（论文、项目、代码入口）：https://xuanbinpeng.com/
- 官方源码与文档：https://github.com/openrobokit/robokit
- [RoboKit 实体](../../wiki/entities/robokit-openrobokit.md)
