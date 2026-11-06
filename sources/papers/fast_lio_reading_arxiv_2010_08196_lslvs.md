# FAST-LIO 论文通读（LiDAR-惯性里程计选型视角）

> 来源归档（原文通读 · 支撑 `wiki/comparisons/lidar-slam-lio-vio-selection.md`）

- **标题：** FAST-LIO: A Fast, Robust LiDAR-inertial Odometry Package by Tightly-Coupled Iterated Kalman Filter
- **作者：** Wei Xu, Fu Zhang（香港大学 MaRS 实验室）
- **类型：** paper / slam / lidar-inertial-odometry
- **原文：** <https://arxiv.org/abs/2010.08196>（2026-11 通过 arXiv PDF v3 获取全文）
- **一句话说明：** 用紧耦合迭代扩展卡尔曼滤波（iEKF）融合 LiDAR 特征点与 IMU，并给出复杂度只与状态维数（18）相关的等价卡尔曼增益公式，使小型 UAV 机载算力可实时处理上千有效特征点。
- **核对范围：** 通读摘要、§I 引言、§II 相关工作、§III 方法（流形算子、连续/离散模型、LiDAR 预处理、前向/后向传播与运动补偿、残差、迭代更新、新卡尔曼增益、Algorithm 1）、§IV 全部实验（表 II/III）、§V 结论及附录 A/B（Fx̃、Fw 推导与增益等价证明）。论文参考文献未逐篇核对；ikd-Tree 属后续 FAST-LIO2，本文未涉及，页面中相关表述仅依赖 wiki 既有内容。
- **通读：** 已完成

## 核心摘录（均出自原文）

- **问题设定（§I）：** 固态 LiDAR 视场小、环境缺少强特征易退化；一帧含数千特征点，紧耦合融合算力吃紧；点逐个采样带来运动畸变，旋翼振动使 IMU 噪声大。
- **分类（§II）：** 松耦合先配准再与 IMU 融合，忽略位姿与速度等状态相关性，且配准退化会传到后端；紧耦合直接融合特征点，分优化式（LIPS、LIOM）与滤波式（EKF/UKF/iKF，LINS）。LOAM 系（含 LeGO-LOAM、LOAM-Livox）在无特征或小视场下脆弱。
- **流程（§III-A，图 2）：** LiDAR 累积（最小 20 ms，最高 50 Hz 输出）→ 提取平面/边缘特征 → IMU 前向传播 + 后向传播去畸变 → 迭代卡尔曼更新 → 用位姿把特征注册进全局地图；状态估计 10–50 Hz。
- **状态（式 2–3）：** SO(3)×R^15，共 18 维（姿态、位置、速度、陀螺/加计零偏、重力）；误差用切空间 3 维姿态误差表示（最小表示）。
- **运动补偿（§III-C-2，式 9–10）：** 以特征点频率做后向传播，把每点投影到扫描结束时刻。
- **残差（式 11–12）：** 点到最近平面/边缘的距离，阈值如 0.5 m 外视为外点或新观测；用地图点的 KD 树找对应。
- **新增益公式（式 18–20，附录 B）：** K=(HᵀR⁻¹H+P⁻¹)⁻¹HᵀR⁻¹，只需对状态维数矩阵求逆，与常规形式等价；表 II：1453 点时旧公式 1219 ms、新公式 0.59 ms。
- **实验（§IV）：** UAV 室内 50 Hz，平均 270 有效点 / 6.7 ms，闭环漂移 <0.3%（32 m 轨迹 0.08 m）；手持剧烈旋转（>100 deg/s）下 LOAM 59 ms、LOAM+IMU 44 ms、FAST-LIO 23 ms（表 III）；户外 140 m 回到起点漂移 <0.05%（0.07 m），25 ms / 1497 点；对比 LINS：7.3 ms vs 34.5 ms，LINS 为降算力降采样到平均 147 点（FAST-LIO 为 784 点），建图精度更差。
- **结论（§V）：** 前后向传播 + 低复杂度增益，在 UAV、剧烈旋转室内、室外三类测试中结果精确、实时、可靠。
- **局限（读后归纳，非原文明说）：** 本文系统是纯里程计，没有回环/全局位姿图，也不含 GPS 因子；实验为 Livox AVIA / VLP-16 与自测场景，未在公开大规模基准上给出系统性误差表。

## 对 wiki 的映射

- [`wiki/comparisons/lidar-slam-lio-vio-selection.md`](../../wiki/comparisons/lidar-slam-lio-vio-selection.md)
- 仓库索引：[fast_lio.md](../repos/fast_lio.md)
