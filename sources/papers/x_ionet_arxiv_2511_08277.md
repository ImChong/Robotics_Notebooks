# X-IONet：跨平台惯性里程计网络（行人 + 腿式机器人）（arXiv:2511.08277）

> 论文来源归档（ingest）

- **标题：** X-IONet: Cross-Platform Inertial Odometry Network for Pedestrian and Legged Robot
- **作者：** Dehan Shen, Changhao Chen（通讯作者）
- **机构：** 香港科技大学（广州）智能交通 Thrust；陈昌浩兼 HKUST 新兴交叉学科部
- **类型：** paper / state-estimation / inertial-odometry / deep-learning / quadruped / pedestrian
- **期刊：** IEEE Robotics and Automation Letters, Vol. 11, No. 7, July 2026
- **arXiv：** <https://arxiv.org/abs/2511.08277> · HTML：<https://arxiv.org/html/2511.08277v2>
- **入库日期：** 2026-06-22
- **一句话说明：** 仅用 **单 IMU** 的跨平台惯性里程计：**规则式专家选择** 将行人/四足 IMU 序列路由到平台专属网络；**双阶段 attention** 编码器–解码器回归 **位移 + 协方差**，再经 **EKF** 融合；在 RoNIN、GrandTour 与自采 **Unitree Go2** 上达 SOTA，Go2 上 ATE/RTE 相对最强基线降 **52.8% / 41.3%**。

## 核心摘录（面向 wiki 编译）

### 1) 问题：行人 IO 难迁移到四足

- **要点：** 学习型惯性里程计在行人导航进展显著，但四足 **加减速、侧向/后退、非周期机动** 产生与人体截然不同的惯性签名；单平台模型跨部署严重退化。纯 IMU 导航虽 **不依赖视觉/GNSS**，但积分漂移长期难解。
- **对 wiki 的映射：** [`wiki/entities/paper-x-ionet-cross-platform-inertial-odometry.md`](../../wiki/entities/paper-x-ionet-cross-platform-inertial-odometry.md)、[`wiki/concepts/state-estimation.md`](../../wiki/concepts/state-estimation.md)

### 2) 规则式专家选择 + 平台专属位移网络

- **要点：** 轻量 **1D-CNN 分类器**（6 通道 IMU × 时间窗）判别平台类型，**if–else 路由** 到预训练专家；扩展新平台需补数据并重训 selector + 新 expert。推理时每窗只激活 **一个专家**，部署高效。
- **对 wiki 的映射：** [`wiki/entities/paper-x-ionet-cross-platform-inertial-odometry.md`](../../wiki/entities/paper-x-ionet-cross-platform-inertial-odometry.md)

### 3) 双阶段 attention 位移预测（Crossformer 启发）

- **要点：** 200 Hz、1 s 窗（L=200）；每维划 20 段（L_seg=10）嵌入；**时间 self-attention**（各传感器轴内长程依赖）+ **维度 self-attention**（轴间相关，带 router 降算力）；层次 encoder–decoder + cross-attention 多尺度融合；输出 **3D 位移** 与 **3×3 协方差**。
- **对 wiki 的映射：** [`wiki/entities/paper-x-ionet-cross-platform-inertial-odometry.md`](../../wiki/entities/paper-x-ionet-cross-platform-inertial-odometry.md)

### 4) Huber–Gaussian 损失 + EKF 后融合（TLIO 范式延伸）

- **要点：** 训练用 **Huber + 高斯似然**（λ=1e-4）同时拟合位移与不确定性；输入先经 EKF 姿态 **重力对齐局部系**。EKF 状态含历史位姿、当前 R/v/p 与陀螺/加计 bias；网络位移作量测，**Σ** 进 Kalman 增益。消融：去 EKF、去不确定性、去双阶段 attention 或层次结构均显著变差。
- **对 wiki 的映射：** [`wiki/entities/paper-x-ionet-cross-platform-inertial-odometry.md`](../../wiki/entities/paper-x-ionet-cross-platform-inertial-odometry.md)、[`wiki/formalizations/ekf.md`](../../wiki/formalizations/ekf.md)

### 5) 实验与数据集

- **要点：** **RoNIN**（行人，42.7 h）；**GrandTour** NovAtel CPT7 IMU（36 条四足轨迹）；自采 **Go2**（iPhone 14 Pro Max 刚性安装，LiDAR 里程计作 GT，30 序列约 1.5 h，100→200 Hz 插值）。指标 **ATE / RTE**；相对最强基线：RoNIN **−14.3% / −11.4%**，GrandTour **−11.8% / −9.7%**，Go2 **−52.8% / −41.3%**（Table I，Ours 行 ATE 3.10/4.79/5.37/1.03 m）。
- **对 wiki 的映射：** [`wiki/entities/paper-x-ionet-cross-platform-inertial-odometry.md`](../../wiki/entities/paper-x-ionet-cross-platform-inertial-odometry.md)、[`wiki/entities/unitree.md`](../../wiki/entities/unitree.md)

## 关键数值（Table I 摘要，ATE / RTE 单位 m）

| 数据集 | X-IONet ATE | X-IONet RTE | 相对最强基线 ATE 降幅 |
|--------|-------------|-------------|----------------------|
| RoNIN_seen | 3.10 | 2.42 | 14.3% |
| RoNIN_unseen | 4.79 | 4.05 | — |
| GrandTour | 5.37 | 4.18 | 11.8% |
| Go2（自采） | **1.03** | **0.84** | **52.8%** |

## 当前提炼状态

- [x] 要点摘录与 wiki 映射
- [x] 与 EKF / 状态估计 / 四足平台页交叉引用
