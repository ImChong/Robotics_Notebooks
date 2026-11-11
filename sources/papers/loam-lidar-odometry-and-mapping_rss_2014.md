# LOAM：Lidar Odometry and Mapping in Real-time（RSS 2014）

> 来源归档（paper · 原文通读，支撑 `wiki/entities/paper-loam-lidar-odometry-and-mapping.md`）

- **标题：** LOAM: Lidar Odometry and Mapping in Real-time
- **作者：** Ji Zhang, Sanjiv Singh（Robotics Institute, Carnegie Mellon University）
- **类型：** paper / lidar-slam / odometry
- **原文：** <https://www.roboticsproceedings.org/rss10/p07.pdf>（Robotics: Science and Systems X, 2014；本次下载 PDF 共 9 页，首页标题核对一致）
- **期刊扩展版：** Ji Zhang, Sanjiv Singh, "Low-drift and real-time lidar odometry and mapping", *Autonomous Robots* 41, 401–416, 2017-02，DOI 10.1007/s10514-016-9548-2（出版社页面仅能看到摘要与元数据，全文付费未读）
- **代码：** 作者最初通过 ROS wiki 发布 `loam_velodyne`（论文 §VII 脚注 1、2 给出 wiki.ros.org/loam_back_and_forth、wiki.ros.org/loam_continuous）；GitHub 上 <https://github.com/laboshinl/loam_velodyne> 为第三方镜像（其 README 自述 "All sources were taken from ROS documentation"）。A-LOAM <https://github.com/HKUST-Aerial-Robotics/A-LOAM> 是港科大的改写版（README 自述 modified from LOAM and LOAM_NOTED，用 Eigen 与 Ceres Solver）。仓库只读了 README，未读源码。
- **入库日期：** 2026-11-11
- **一句话说明：** 把「同时定位与建图」拆成两个并行算法：约 10 Hz 的低保真激光里程计（边缘 / 平面特征、点到线 / 点到面距离、匀速假设）与约 1 Hz 的高保真建图（扫描到地图配准），不依赖高精度测距或惯导，不含回环。
- **核对范围：** 通读 RSS 2014 版全文 9 页：摘要、§I 引言、§II 相关工作、§III 记号与任务、§IV 硬件与系统概览、§V 激光里程计（特征提取、对应关系、运动估计、算法 1）、§VI 建图、§VII 实验（室内外、IMU 辅助、KITTI）、§VIII 结论、参考文献列表（仅看条目，未逐篇核对）；图 3、算法 1 页面看了渲染图，其余图表读了 pdftotext 图注与表格文字。未读：Autonomous Robots 2017 期刊版正文（付费墙，仅读摘要）；论文 KITTI 结果页（论文自称因篇幅未收录，仅给榜单链接）；代码仓库源码（只读 README）；补充视频 / 数据集。
- **通读：** 已完成

## 核心摘录（均出自 RSS 2014 版）

- **问题设定（§III）：** 输入为 3D 激光点云序列 P_k（k 为 sweep 序号，sweep 指激光完成一次扫描覆盖），输出每个 sweep 内的自运动并建图；假设雷达已标定；假设角速度与线速度随时间平滑连续、无突变，该假设可由 IMU 放松（§VII-B）。
- **两算法并行（§I、§IV-B、图 3）：** 激光里程计约 10 Hz，估计雷达速度并去畸变；激光建图约 1 Hz，对去畸变点云做精配准；两路位姿变换融合，输出约 10 Hz 的雷达相对地图位姿。摘要称建图频率低一个数量级。
- **自研 3D 激光雷达（§IV-A、图 2）：** Hokuyo UTM-30LX（视场 180°、分辨率 0.25°、40 线 / 秒）加电机，电机以 180°/s 在 -90° 到 90° 间转动，一个 sweep 约 1 s；垂直扫描面方向分辨率 180°/40 = 4.5°（§V-A）。论文称方法「不限于」该硬件。
- **特征点提取（§V-A，式 1）：** 对扫描线上点 i，取同一线上相邻点集 S（两侧各一半，间隔 0.25°），计算局部表面光滑度 c = 1/(|S|·‖X_i‖) · ‖Σ_{j∈S, j≠i}(X_i − X_j)‖；按 c 排序，最大者为边缘点，最小者为平面点；每个 scan 分 4 个等分子区域，每个子区域最多取 2 个边缘点、4 个平面点；还要求 c 超过 / 低于阈值。论文用词是「smoothness」而非「curvature」。
- **剔除不可靠点（§V-A、图 4）：** 已选点的邻点不再选；与激光束近似平行的平面上的点（图 4a 的 B）不选；被遮挡区域边界上的点（图 4b 的 A）不选。
- **对应关系（§V-B、图 6、图 7）：** 上一 sweep 点云 P_k 按运动估计重投影到下一 sweep 起始时刻得到 P̄_k，存入 3D KD 树；边缘点找最近邻 j 及相邻两条扫描线上的最近邻 l，构成边缘线 (j,l)，距离为点到线（式 2）；平面点找 j、同线 l、相邻线 m，构成平面片 (j,l,m)，距离为点到面（式 3）；要求 j、l 来自不同扫描线。
- **运动估计（§V-C，式 4–12，算法 1）：** 假设 sweep 内角速度与线速度恒定，故点 i 的位姿变换由 T_{k+1} 线性插值：T_(k+1,i) = (t_i − t_{k+1})/(t − t_{k+1}) · T_{k+1}（式 4）；旋转用 Rodrigues 公式（式 6）；把所有特征点的距离堆成 f(T)=d，用 Levenberg-Marquardt 迭代（式 12）；算法 1 对每个残差给 bisquare 权重，距离超阈值的视为外点、权重为 0；sweep 结束时把整个点云重投影到 t_{k+2}。
- **建图（§VI、图 8、图 9）：** 特征点数量为里程计的 10 倍；地图点云按 10 m 立方体存储，与当前点云相交的立方体放入 KD 树；对特征点周围点集 S' 求协方差与特征值：一个特征值显著大则为边缘线（取最大特征值对应特征向量为方向），两个大一个小则为平面（取最小特征值对应特征向量为法向），位置取 S' 几何中心；同样用点到线 / 点到面距离，所有点共享同一时间戳 t_{k+2}（即建图阶段不再做 sweep 内插值），仍用鲁棒拟合加 LM；地图用 5 cm 体素网格降采样。
- **计算资源（§VII）：** 2.5 GHz 四核、6 GiB 内存的笔记本，ROS / Linux，里程计与建图各占一核，共两核。
- **室内外实验（§VII-A，图 10、图 11、表 I）：** 走廊、大厅（手推车）与植被道路、果园（车载），速度 0.5 m/s；回到原点的漂移占行程比例：走廊 0.9%（试验 1）/1.1%（试验 2），果园 2.3% / 2.8%（果园真值来自高精度 GPS/INS）；论文总结为室内约 1%、室外约 2.5%。局部地图精度另用静止雷达采集第二份点云做点到面 ICP 对比（图 11）。
- **IMU 辅助（§VII-B，图 12，表 II）：** Xsens MTi-10；用 IMU 姿态把 sweep 内点云旋转对齐，并用加速度部分去除畸变，再交给 LOAM。表 II 的运动估计误差（行程 32 / 27 / 43 / 51 m，真值为卷尺手量）：仅 IMU 16.7% / 11.7% / 13.7% / 11.4%，仅 LOAM 2.1% / 1.7% / 4.4% / 3.7%，LOAM+IMU 0.9% / 1.3% / 2.6% / 2.1%。论文称 IMU 用来抵消非线性运动、LOAM 处理线性运动；IMU 姿态积分 5 分钟漂移超过 25°。
- **KITTI（§VII-C）：** 仅使用 Velodyne 激光数据（10 Hz），测试序列总里程 39.2 km；论文称当时在榜单上排名第一（不分传感器模态，含先进双目视觉里程计），平均位置误差为行程的 0.88%（100 m 到 800 m 的轨迹片段，3D 坐标）。论文自述因篇幅未列结果，读者须去榜单网站核对。
- **局限与未来工作（§I、§VIII）：** 论文明说「目前不做回环」，方法旨在把里程计漂移压到最小；未来工作是做回环以修正漂移，并把输出与 IMU 在卡尔曼滤波器中融合。
- **相关工作对比（§II）：** 与 Bosse & Zlot 的离线批处理建图相比，LOAM 实时给出可引导自主车辆的运动估计；与 Barfoot 等基于激光强度图像视觉特征、需要稠密点云的方法相比，LOAM 在笛卡尔空间提取几何特征，对点云密度要求更低；运动模型与 [18][19] 同为线性（匀速）模型。

## 读后归纳（非论文原话，供 wiki 页「局限」使用）

- 论文所有定量实验是短距离（行程 27–51 m 的 IMU 对照、走廊 / 果园单圈）加 KITTI 榜单；大场景长时间漂移无论文内独立数据，只有 KITTI 平均 0.88% 的榜单自述。
- 表 II 真值为卷尺手量，且 IMU 对照只有 4 个短轨迹。
- 论文假设 sweep 内匀速，高动态运动下（手持、无人机）论文自己用 IMU 来补；无 IMU 的剧烈运动表现论文无专门实验。
- 特征点提取假设相邻点在同一扫描线上有序，按作者自研雷达的 0.25° 线内分辨率与 4.5° 线间分辨率设计；迁移到其他雷达线束 / 固态雷达需改写（推测）。

## 对 wiki 的映射

- [paper-loam-lidar-odometry-and-mapping](../../wiki/entities/paper-loam-lidar-odometry-and-mapping.md)
