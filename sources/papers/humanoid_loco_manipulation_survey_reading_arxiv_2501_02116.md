# 人形机器人 Loco-Manipulation 综述通读（人形机器人总览视角）

> 来源归档（原文通读 · 支撑 `wiki/entities/humanoid-robot.md`）

- **标题：** Humanoid Locomotion and Manipulation: Current Progress and Challenges in Control, Planning, and Learning
- **作者：** Zhaoyuan Gu, Junheng Li, Wenlan Shen, Wenhao Yu, Zhaoming Xie, ... Huijun Gao, Ye Zhao（Georgia Tech 等）
- **类型：** paper / survey / humanoid
- **原文：** <https://arxiv.org/abs/2501.02116>（2026-11 通过 arXiv PDF v2，19 Apr 2025，pdftotext 获取全文）
- **一句话说明：** 以双足人形的 loco-manipulation 为主线，综述触觉、接触规划、MPC、WBC（模型方法）与 RL / IL / 基础模型（学习方法），并对两类范式做对比、讨论硬件局限。
- **核对范围：** 通读正文 §I–§X（引言、背景、触觉、接触规划、MPC、WBC、技能学习、基础模型、讨论与结论，含表 I–VI 与各节 Key Takeaways）；参考文献列表（[1]–[424]）仅作索引、未逐条核对被引原文。
- **通读：** 已完成

## 核心摘录（均出自原文）

- **定义与范围（§I）：** 人形机器人指类人形态机器人，通常有躯干、双臂、双腿，拟人程度可变（可有简单手或轮腿）；本综述聚焦双足人形的全身运动，而非多指灵巧操作。类人形态使其可利用人类数据（互联网视频、公开数据集）快速获得技能。
- **控制趋势（§I）：** 模型方法近十年趋向"预测-反应"层级：全身 MPC 或简化（质心动力学）MPC + 局部任务空间 WBC。sim-to-real RL 本身依赖仿真器的精确动力学，也可视为"模型方法"，且与模型控制互补而非对立。
- **背景（§II）：** 双足行走从被动行走、准静态到动态行走；人形 COT > 0.7，人类约 0.2。人形比四足重心高、支撑区小，早期框架把行走与操作分开控制（行走时上身固定、桌面操作时下身不动）。Table I 区分 whole-body manipulation / loco-manipulation / whole-body loco-manipulation。
- **触觉（§III）：** 触觉分手部、足底、全身三类；足底触觉可估计地面反力与地形，但人形足部冲击大、传感器耐久是难点；全身触觉（iCub、REEM-C、Punyo）支持全身柔顺与大物体操作；高维接触、触觉仿真与 sim-to-real 是挑战。
- **多接触规划（§IV）：** 分搜索、优化（接触隐式轨迹优化 CITO）、学习三类；人形上 CITO 尚未做到实时 CI-MPC；挑战为接触模式组合爆炸、点接触假设、固定接触序列假设。
- **MPC（§V）：** 模型保真度与算力折中：简化模型（SRBM、LIPM）、全身动力学（质心动力学 + 全身运动学、WBD）、混合保真度；NMPC 加速手段（结构利用 DDP/iLQR、逐次线性化 SQP/ReLU-QP——206 Hz 到 2600 Hz、热启动、MPPI 采样，Table IV）；物体可建模为外部 wrench 或并入统一机器人-物体动力学。
- **WBC（§VI）：** 每个时刻求解瞬时问题，决策变量为广义加速度、接触力、关节力矩（式 5–7）；人形欠驱动，必须靠接触维持平衡。闭式（逆动力学、OSC / stack of tasks）高效但难处理不等式；优化式（分层 QP 严格优先级 vs 加权 QP 软权重）更灵活；loco-manipulation 中交互可视为外部 wrench（全局优化 / 预优化 / 后优化三策略）或统一机器人-物体模型。数值优化仅局部最优、确定性、权重调参依赖专家。
- **技能学习（§VII）：** PPO 因仿真数据便宜被广泛用于人形 sim-to-real，SAC 等 off-policy 样本效率高但训练慢；纯 RL 在多数人形 loco-manipulation 任务上失败（引 HumanoidBench）。sim-to-real 手段：域随机化、系统辨识、域适应；模拟器接触建模（MuJoCo 软接触、PhysX 摩擦锥金字塔）仍有差距。IL 数据源四类（策略执行、遥操作、动捕、人类视频，Table V），存在形态差距、需重定向；混合方法含教师-学生、跟踪/增广参考轨迹；多技能表示含 MoE、潜空间、目标条件策略、世界模型。缺基准（HumanoidBench 等）与数据稀缺；作者主张靠含物体运动与多模态感知而非单纯数据量实现泛化。
- **基础模型（§VIII）：** 两条路：预训练 LLM/VLM 做任务规划、写代码或奖励（无需机器人数据），以及训练机器人基础模型 / VLA（分层 VL 条件策略如 GR00T N1、Helix；端到端如 RT-2、OpenVLA）；人形 loco-manipulation 的 VLA 仅见于仿真。挑战：板载算力与推理延迟（云端高层 + 板载低层分层）、训练成本、缺少缩放律、安全与伦理。
- **对比（§IX-A，Table VI）：** 学习方法在鲁棒性（RL）与多样性（IL）上占优，模型方法在运动精度上占优；VLA 泛化潜力高但对人形技能的有效性尚未被令人信服地证明；文中指出多数性能数据为个案、难以数值比较。
- **硬件局限（§IX-B）：** 高功率密度执行器、QDD（减速比 < 10:1，可反驱、力控带宽高，但大电流导致发热）、质量集中于躯干以减小腿部惯量（MIT Humanoid 约 75% 质量在躯干，代价是四杆/带传动限制关节范围或需维护）；电池是主要限制，Apollo 约 4 小时、Unitree G1 约 2 小时，实验室多仍系绳供电。
- **应用与社会（§IX-C）：** 室外救援 / 施工 / 农业、工厂仓储、家庭医院；成本高、法规缺失、安全与 HRI 是落地障碍。

## 对 wiki 的映射

- [`wiki/entities/humanoid-robot.md`](../../wiki/entities/humanoid-robot.md)
- 已有策展解读归档（非原文通读）：[`hmi_p069_humanoid-loco-manipulation-survey.md`](hmi_p069_humanoid-loco-manipulation-survey.md)
