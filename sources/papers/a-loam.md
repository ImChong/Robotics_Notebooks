# A-LOAM：LOAM 的 Eigen + Ceres 简化实现（仅代码仓库，无 arXiv 论文）

> 来源归档（paper / repo）：该项目没有独立论文，「原文」即仓库 README 与全部核心源码。

- **标题：** A-LOAM: Advanced implementation of LOAM
- **类型：** repo（无独立论文；其依据的论文是 Zhang & Singh, LOAM: Lidar Odometry and Mapping in Real-time, RSS 2014，见 README §6）
- **代码：** <https://github.com/HKUST-Aerial-Robotics/A-LOAM>
- **读取版本：** `devel` 分支浅克隆，HEAD `e51f88c12b48`，提交时间 2019-03-28（git log 所示；首次发布日期未核实）
- **作者（README 署名 Modifier）：** Tong Qin、Shaozu Cao（README 与源文件头只写姓名与邮箱，未写机构；机构为香港科技大学属推断，来自仓库所属组织 HKUST-Aerial-Robotics 与 Cao 的 connect.ust.hk 邮箱）
- **许可：** BSD 3-Clause，版权头保留 Ji Zhang（CMU）2013 与 Southwest Research Institute 2016
- **核对范围：** 静态通读 README.md、LICENSE、package.xml、CMakeLists.txt、launch/aloam_velodyne_VLP_16.launch、include/aloam_velodyne/common.h，以及核心源码 src/scanRegistration.cpp（503 行）、src/laserOdometry.cpp（601 行）、src/lidarFactor.hpp（171 行）、src/laserMapping.cpp 的关键段落（下采样、cube 地图管理、特征匹配、Ceres 优化、发布；约 938 行中的数据缓存与 ROS 发布样板代码仅扫读）。未通读 src/kittiHelper.cpp（KITTI 转 rosbag 工具）、docker/ 与 rviz 配置。只静态读，未编译未运行，故下文所有结论都是「代码里这样写」，未做精度、速度或鲁棒性实测，也未与 LOAM 原论文全文逐式对照。
- **通读：** 已完成

## 核心摘录（均来自源码，括号为位置）

1. **自述定位（README 首段）：** 「Advanced implementation of LOAM」，使用 Eigen 与 Ceres Solver 简化代码结构；「modified from LOAM and LOAM_NOTED」；「clean and simple without complicated mathematical derivation and redundant operations. It is a good learning material for SLAM beginners」。
2. **三节点流水线（CMakeLists.txt）：** `ascanRegistration`、`alaserOdometry`、`alaserMapping`（另有 `kittiHelper`）。launch 文件：`scan_line`=16、`mapping_skip_frame`=1、`minimum_range`=0.3、`mapping_line_resolution`=0.2、`mapping_plane_resolution`=0.4（VLP_16 版）。
3. **scanRegistration：** 仅支持 16/32/64 线 Velodyne（`scan_line` 之外直接退出）；按仰角手写 scanID 映射；去 NaN 与近距点（`minimum_range`）；每点按方位角算相对时间写入 `intensity = scanID + 0.1 * relTime`；曲率为前后各 5 点的差分平方和；每线分 6 段，段内按曲率排序，每段取曲率最大 2 个为 sharp、最多 20 个为 less sharp，取曲率最小 4 个为 flat，其余（label ≤ 0）体素 0.2 下采样为 less flat；曲率阈值 0.1；选中点邻域 ±5 内（且相邻点距离平方 ≤ 0.05）标记不再选取。未见 LOAM 原文中的遮挡/掠射面剔除的专门代码（本文件内未找到，属对照原文的推断，原文细节未逐条核）。
4. **laserOdometry：** 帧间 scan-to-scan。对 sharp 点在上一帧 corner 点云中做 1 近邻 + 相邻扫描线（NEARBY_SCAN=2.5）内再找一点构成线；对 flat 点同样找三点构成面；距离平方阈值 25（DISTANCE_SQ_THRESHOLD）。残差：点到线（叉积/线长，3 维残差）、点到面（法向点积，1 维），Ceres AutoDiff，Huber(0.1) 核，EigenQuaternionParameterization，DENSE_QR，每次最多 4 次迭代，每帧重新关联并优化 2 轮；`t_w_curr += q_w_curr * t_last_curr; q_w_curr = q_w_curr * q_last_curr` 累积位姿。
5. **运动畸变补偿默认关闭：** `#define DISTORTION 0`（laserOdometry.cpp:59）。为 0 时插值比 `s = 1.0`；为 1 时 `s = (intensity 小数部分)/SCAN_PERIOD`，并在 `lidarFactor.hpp` 里对 q 做 `slerp(s)`、对 t 乘 s（即匀速假设下的线性/球面线性插值）。机制在代码里存在，但默认不启用。`TransformToEnd` 调用处被 `if (0)` 关掉。
6. **laserMapping：** scan-to-map。世界地图分为 21×21×11 个 50 m 立方体（`laserCloudWidth/Height/Depth`），每帧取当前立方体周围 5×5×3=75 个（数组上限 125）作为局部地图并建 KD 树；先用 odom 位姿推出初值（`transformAssociateToMap`）。corner：5 近邻（第 5 近邻距离平方 < 1.0），协方差特征分解，最大特征值 > 3 倍次大才判为直线，取中心 ±0.1·方向两点构成点到线残差；surf：5 近邻最小二乘拟合平面（`colPivHouseholderQr`），任一点到平面距离 > 0.2 则弃用，否则 `LidarPlaneNormFactor`；同样 Huber(0.1)、DENSE_QR、每次最多 4 次迭代、2 轮；地图点按 `mapping_*_resolution` 体素下采样；仅当 corner 地图点 > 10 且 surf > 50 才优化；`frameCount % 5` 发布周边地图、`% 20` 发布整图。
7. **没有回环与后端：** 在 src/*.cpp、*.hpp 中 grep `loop`/`gps`/`imu`（忽略注释与 include）无回环检测、IMU 或图优化实现；仅有 `kdtree`+Ceres 的 scan-to-scan 与 scan-to-map。

## 与综述文章（《激光雷达 SLAM 算法最新综述》转载稿）说法的逐条核对

| 文章说法 | 核对结果 |
|----------|----------|
| 「香港科技大学 Qin 根据 LOAM 论文复现代码，开源 A-LOAM」 | **部分不符。** README 写的是「modified from LOAM and LOAM_NOTED」，署名 Modifier，源文件保留 Ji Zhang（CMU）与 SwRI 版权头：它是改写自既有 LOAM 代码及社区注释版 LOAM_NOTED，并非仅凭论文从零复现。作者为 Tong Qin 与 Shaozu Cao 两人，文章只提 Qin。机构为 HKUST 属实（仓库组织名为 HKUST-Aerial-Robotics，推断）。 |
| 「使用 Eigen 矩阵库和 Ceres-Solver 非线性优化库代替原本复杂的数学推导」 | **基本属实。** README 原话 uses Eigen and Ceres Solver to simplify code structure、without complicated mathematical derivation；代码里各因子用 Ceres AutoDiff，无手推 Jacobian（lidarFactor.hpp）。 |
| 「代码简洁易懂，适合初学者学习」 | **作者自述属实**（README：good learning material for SLAM beginners）；核心三个 cpp 合计约 2000 行，我读下来结构直观，但「易懂」是主观判断。 |
| （同段对 LOAM）「对激光雷达做匀速运动假设，使用线性插值法进行运动补偿」 | 针对 LOAM 的描述；**对 A-LOAM 需要补充：** 补偿代码存在但 `DISTORTION` 默认为 0，默认运行并不补偿。 |
| （同段对 LOAM）「缺少后端优化和回环检测，大场景会漂移」 | 对 A-LOAM 代码核对：确实无回环与位姿图后端；「大场景漂移」是文章判断，仓库内没有任何实验支撑，未验证。 |

## 对 wiki 的映射

- [paper-a-loam](../../wiki/entities/paper-a-loam.md)
