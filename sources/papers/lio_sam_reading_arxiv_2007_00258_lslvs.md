# LIO-SAM 论文通读（LiDAR-惯性里程计选型视角）

> 来源归档（原文通读 · 支撑 `wiki/comparisons/lidar-slam-lio-vio-selection.md`）

- **标题：** LIO-SAM: Tightly-coupled Lidar Inertial Odometry via Smoothing and Mapping
- **作者：** Tixiao Shan, Brendan Englot, Drew Meyers, Wei Wang, Carlo Ratti, Daniela Rus
- **类型：** paper / slam / factor-graph
- **原文：** <https://arxiv.org/abs/2007.00258>（2026-11 通过 arXiv PDF v3 获取全文）
- **一句话说明：** 在因子图上做紧耦合 LiDAR-惯性里程计，IMU 预积分、激光里程计、GPS、回环作为四类因子，用 iSAM2 增量优化；以关键帧 + 固定大小子关键帧滑窗匹配保证实时。
- **核对范围：** 通读摘要、§I 引言、§II 相关工作、§III 方法（IMU 预积分因子、激光里程计因子含关键帧与子关键帧体素地图、GPS 因子、回环因子）、§IV 全部实验（五个数据集、表 I–IV）、§V 结论与讨论。参考文献未逐篇核对；补充视频与网页材料未看。仓库实际实现细节（如 ROS 2 分支）本文未涉及。
- **通读：** 已完成

## 核心摘录（均出自原文）

- **动机（§I）：** LOAM 把数据存在全局体素地图，难做回环、难引入 GPS 等绝对量；特征丰富时在线优化变慢；大尺度下 scan-matching 本质上会漂移。
- **系统（§III-A，图 1）：** 输入 3D LiDAR、IMU、可选 GPS；状态 x=[R,p,v,b]；MAP 问题在高斯噪声下等价于非线性最小二乘；位姿变化超阈值才加状态节点；插入节点后用 iSAM2 优化。
- **IMU 预积分因子（§III-B）：** 用预积分量（式 7–9）约束相邻节点，IMU 零偏与激光里程计因子联合优化。
- **激光里程计因子（§III-C）：** 提取边缘/平面特征；关键帧阈值 1 m 与 10°；取最近 n=25 个子关键帧拼成体素地图（边缘 0.2 m、平面 0.4 m 降采样）；IMU 预测给初值，用 LOAM 的点到线/面距离做高斯牛顿，得到相对位姿因子。
- **GPS 因子（§III-D）：** 仅当估计位置协方差大于 GPS 协方差时才加，因为 LIO 漂移增长很慢。
- **回环因子（§III-E）：** 朴素的欧氏距离检测（搜索半径 15 m，前后各 m=12 个子关键帧做 scan-matching），可换成点云描述子；仅用 GPS 时高程误差可达近 100 m，回环能修正高度漂移。
- **实验（§IV）：** VLP-16 + MicroStrain IMU + Reach M GPS；Rotation / Walking / Campus / Park / Amsterdam 五个数据集（手持、Jackal UGV、电动船）。对比 LOAM、LIOM。表 II 终点平移误差：Campus 上 LOAM 192.43 m、LIO-odom 9.44 m、LIO-GPS 6.87 m、LIO-SAM 0.12 m；Park 上 LIO-SAM 0.04 m，LIO-GPS 1.21 m（无可靠高程）；Amsterdam 上 LOAM/LIOM/LIO-odom 均失败（桥下退化、阳光误检），LIO-SAM 0.17 m。表 III Park 相对 GPS 的 RMSE：LOAM 47.31，LIO-SAM 0.96。表 IV 单帧耗时：LIO-SAM 约 41.9–100.5 ms，LOAM 83.6–339.8 ms（部分失败），可高达 13 倍速回放。LIOM 仅 0.56–0.67 倍实时。
- **结论（§V）：** 因子图便于接入新传感器（GPS、罗盘、高度计）与位置识别；局部尺度而非全局尺度的 scan-matching 是实时的关键。
- **局限（读后归纳）：** 实验平台为 VLP-16 型旋转 LiDAR，未覆盖固态小视场 LiDAR；回环为欧氏距离的朴素实现；对比对象仅 LOAM / LIOM，未与 FAST-LIO 直接比较。
