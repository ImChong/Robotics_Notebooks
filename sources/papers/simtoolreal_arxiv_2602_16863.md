# SimToolReal: An Object-Centric Policy for Zero-Shot Dexterous Tool Manipulation（arXiv:2602.16863）

> 来源归档（ingest · 论文一手原文）；核对日期：2026-11-21。

- **标题：** SimToolReal: An Object-Centric Policy for Zero-Shot Dexterous Tool Manipulation
- **arXiv：** <https://arxiv.org/abs/2602.16863>（v3，2026-10-06；23 pages, 16 figures, 3 tables）
- **HTML 全文：** <https://arxiv.org/html/2602.16863v3>
- **项目页：** <https://simtoolreal.github.io/>
- **代码：** <https://github.com/tylerlum/simtoolreal>（MIT License）
- **发表：** Robotics: Science and Systems (RSS) 2026（项目页标注）
- **作者与机构：** Kushal Kedia\*（Cornell University）、Tyler Ga Wei Lum\*（Stanford University）、Jeannette Bohg†（Stanford University）、C. Karen Liu†（Stanford University）；致谢 Sharpa 团队提供硬件与技术支持（含 Kaifeng Zhang 等）
- **入库日期：** 2026-11-21
- **核对范围：** 通读 `https://arxiv.org/html/2602.16863v3` 全文（Abstract、Sec. I–V、Appendix A-A 至 A-B5 奖励/观测/程序化生成/非对称 Critic 与终止条件）及官方项目页 `https://simtoolreal.github.io/`；未在本地 GPU 运行仿真训练或 KUKA iiwa 14 + Sharpa 手真机部署。
- **通读：** 已完成
- **一句话说明：** 通过在仿真中程序化生成手柄–头部组合工具原语并训练单一目标位姿到达 RL 策略（SAPG + 非对称 Critic + 140 维以物体为中心状态），实现无需目标物体建模或任务特定微调的零样本真机灵巧工具操作。

## 论文核心要点（arXiv:2602.16863v3 自报）

### 1) 核心问题与以物体为中心的抽象

- 灵巧工具使用需要从平面抓起薄物体、手内旋转到功能姿态并在受力交互中保持稳定；传统遥操作缺力触觉反馈且存在人–机手运动学鸿沟，而传统 Sim2Real RL 对每个物体和任务都需要独立建模与调参奖励。
- SimToolReal 将工具操作抽象为 **依次到达目标 6D 物体位姿序列** `g^k ∈ SE(3)`：只要策略在仿真中学会把任意形状/惯量工具从桌面抓起并操作到任意随机采样位姿，就会自然涌现抓取、手内重定位与抗扰握持能力。

### 2) 程序化工具生成与非对称 SAPG 训练

- **程序化工具原语（Appendix A-A）：** 每个工具由长方体/胶囊体的手柄（长 `5–30 cm`、宽/径 `1–4 cm`、低密度 `300–600 kg/m³`）与垂直连接的头部（长 `1–15 cm`、宽/径 `0.5–12 cm`、高密度 `300–2000 kg/m³`）构成，覆盖锤、刷、铲、马克笔、板擦、螺丝刀等结构与偏心质量分布。
- **140 维策略观测（Appendix A-B3）：** `29` 关节角 + `29` 关节速度 + `29` 上步关节目标 + `7` 手掌位姿 + `15` 五指尖相对手掌位置 + `4` 物体朝向四元数 + `12` 相对手掌的 4 个抓取框关键点 + `12` 到当前目标的关键点误差 + `3` 抓取框尺度 `s`。动作 `29` 维（7 臂增量控制 `k_arm=0.025` + 22 手绝对映射，EMA 平滑系数 `α=0.1`），控制频率 `60 Hz`。
- **奖励与优化：** `r = r_smooth + r_grasp + I_grasped * r_goal`；其中 `r_goal` 基于 4 个固定尺度关键点（`[0.14, 0.03, 0.03] m`，更关注长轴俯仰与偏航对准）的最远距离进步量及 `ϵ=2 cm` 成功奖励。使用 **SAPG**（Split and Aggregate Policy Gradients）配合 LSTM 骨干与 **非对称 Critic**（读取无噪物体位姿、速度与进度状态）。

### 3) DexToolBench 与真机评测（KUKA iiwa 14 + 22-DoF Sharpa 左手）

- **人类视频目标提取：** 首帧 RGB-D 经 **SAM 3D** 重建真实尺度网格、**SAM 2** 分割抓取区域生成 3D 包围盒、**FoundationPose** 提取并降采样至 `3 Hz` 的 6D 目标序列；推理时 FoundationPose 以 `30 Hz` 实时跟踪。
- **DexToolBench（6 类 × 2 实例 × 2 任务 = 24 任务）：** 涵盖 hammer、marker、eraser、brush、spatula、screwdriver，共 `120` 次真机 rollout（每任务 5 次）。
- **主要结果：** 在任务进度（Task Progress）上超出运动学重定向和固定抓取（Fixed Grasp）基线 **37%**，并在仿真中匹配针对单物体/单轨迹训练的 specialist 策略，同时在物体或轨迹变化时保持泛化（specialist 显著退化）。
- **真机失败分布（Sec. IV-A）：** 位姿跟踪丢失 `43.7%`、物体掉落 `34.5%`、手内旋转未完成 `18.2%`、抓取失败 `3.6%`。
- **消融（Sec. IV-E）：** 将 SAPG 替换为 PPO 或去掉非对称 Critic 均导致训练回报大幅下降。
- **作者自述局限（Sec. V）：** 仅跟踪位姿目标不保证高受力功能完成（如钉入）；仅依赖物体位姿对环境盲目，杂乱场景易碰撞；假设刚体工具（无法描述剪刀等铰接工具）；高层目标序列固定、不在线重规划。

## 对 wiki 的映射

- [SimToolReal](../../wiki/entities/simtoolreal.md)
