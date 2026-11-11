---
type: entity
project_id: loam-lidar-odometry-and-mapping
tags: [paper, lidar, slam, odometry, mapping, cmu, loam]
status: complete
updated: 2026-11-11
paper: https://www.roboticsproceedings.org/rss10/p07.pdf
code: https://github.com/laboshinl/loam_velodyne
original:
  - ../../sources/papers/loam-lidar-odometry-and-mapping_rss_2014.md
sources:
  - ../../sources/papers/loam-lidar-odometry-and-mapping_rss_2014.md
related:
  - ./lego-loam.md
  - ./paper-a-loam.md
  - ./lio-sam.md
  - ./paper-lvi-sam.md
  - ./fast-lio.md
  - ./point-lio-unilidar.md
  - ./cartographer.md
  - ./ceres-solver.md
  - ../concepts/state-estimation.md
  - ../concepts/lidar-sensing.md
  - ../comparisons/lidar-slam-lio-vio-selection.md
  - ../overview/navigation-slam-autonomy-stack.md
summary: "LOAM（Zhang & Singh，RSS 2014）把激光 SLAM 拆成约 10 Hz 的低保真里程计与约 1 Hz 的高保真建图，用边缘 / 平面特征做点到线、点到面配准，sweep 内匀速线性插值去畸变，论文自述不含回环。"
---

# LOAM：Lidar Odometry and Mapping in Real-time

**LOAM**（Ji Zhang, Sanjiv Singh，卡内基梅隆大学机器人研究所，*LOAM: Lidar Odometry and Mapping in Real-time*，Robotics: Science and Systems X，2014，[论文 PDF](https://www.roboticsproceedings.org/rss10/p07.pdf)）。期刊扩展版为 *Low-drift and real-time lidar odometry and mapping*（*Autonomous Robots* 41，401–416，2017，DOI 10.1007/s10514-016-9548-2）；本页只通读了 RSS 2014 版，期刊版正文因付费墙未读。该方法没有 arXiv 论文。

## 一句话定义

**把「同时定位与建图」拆成两个并行算法：约 10 Hz 的低保真激光里程计估计速度并去畸变，约 1 Hz 的高保真建图把点云精配准进地图；特征用边缘点与平面点，论文自述不做回环（RSS 版 §I、§VIII）。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| LOAM | Lidar Odometry and Mapping | 激光里程计与建图，本页主角 |
| SLAM | Simultaneous Localization and Mapping | 同步定位与建图 |
| LM | Levenberg-Marquardt | 非线性最小二乘迭代法，论文用于运动估计（式 12） |
| ICP | Iterative Closest Point | 迭代最近点配准；论文称建图类似 ICP 的批优化（§I） |
| KD-tree | k-dimensional tree | 最近邻检索结构，论文用于对应点搜索（§V-B） |
| IMU | Inertial Measurement Unit | 惯性测量单元，论文可选辅助（§VII-B） |
| KITTI | Karlsruhe Institute of Technology and Toyota Technological Institute dataset | 自动驾驶数据集与里程计榜单（§VII-C） |
| ROS | Robot Operating System | 论文实现所用中间件（§VII） |
| GPS/INS | Global Positioning System / Inertial Navigation System | 论文用作果园实验真值（§VII-A） |

## 为什么重要

- 它给出了后来大量激光 SLAM 沿用的骨架：特征提取、点到线 / 点到面残差、里程计与建图分频。本库的 [LeGO-LOAM](./lego-loam.md) 与 [LIO-SAM](./lio-sam.md) 都以它为起点（LIO-SAM 自述其激光里程计因子沿用 LOAM 的点到线 / 面距离，见该页引用的论文通读归档）。
- 在发表时的 KITTI 里程计榜单上，论文自述排名第一、平均位置误差 0.88%（§VII-C）；这是作者自述的当时榜单位置，本页未独立复核榜单。
- 代码长期是学习与二次开发的基础，例如 A-LOAM 用 [Ceres Solver](./ceres-solver.md) 与 Eigen 改写（A-LOAM README）。

## 核心机制

### 流程总览

```mermaid
flowchart LR
  A[旋转激光雷达点云 一个 sweep] --> B[点云配准到雷达坐标系]
  B --> C[特征提取 光滑度 c 分边缘点与平面点]
  C --> D[激光里程计 约 10 Hz]
  D --> E[匀速假设 线性插值 sweep 内位姿]
  E --> F[点到线 点到面残差 LM 加 bisquare 权重]
  F --> G[去畸变点云 约 1 Hz]
  G --> H[激光建图 扫描到地图配准]
  H --> I[10 Hz 位姿与地图输出]
  D --> I
```

图对应论文图 3（系统框图）：里程计约 10 Hz 输出位姿更新，建图约 1 Hz 输出地图与位姿更新，两者合成约 10 Hz 的雷达位姿。

### 要点（均出自 RSS 2014 版，章节号如括号）

1. **任务与假设（§III）：** 雷达已标定；角速度与线速度平滑连续，该假设可由 IMU 放松。
2. **特征提取（§V-A，式 1）：** 对每个点取同一扫描线上相邻点集，计算光滑度 c（相邻点与该点差向量之和的模，按点到雷达距离与点数归一）；c 最大者为边缘点，最小者为平面点；每条扫描分 4 个子区域，每个子区域至多 2 个边缘点与 4 个平面点，并剔除与激光束近似平行的面和遮挡边界上的点。注意论文用的词是 smoothness，不是 curvature。
3. **对应关系（§V-B）：** 上一 sweep 点云重投影到当前 sweep 起点，存入 KD 树；边缘点匹配到由两点构成的边缘线，平面点匹配到由三点构成的平面片，残差分别是点到线（式 2）与点到面（式 3）距离。
4. **运动补偿与估计（§V-C、算法 1）：** sweep 内假设角速度与线速度恒定，各点位姿由 sweep 总变换按时间比例线性插值（式 4）；用 Levenberg-Marquardt 最小化残差（式 12），每个残差带 bisquare 鲁棒权重，距离过大者权重为 0。
5. **建图（§VI）：** 每 sweep 一次；特征点数是里程计的 10 倍；地图按 10 m 立方体存储；用邻域点集协方差的特征值 / 特征向量判断边缘线或平面片并给出方向；地图用 5 cm 体素滤波。此阶段所有点共享同一时间戳，不再做 sweep 内插值。
6. **实验结果（§VII）：** 室内外 0.5 m/s 下，闭环或 GPS/INS 对照的漂移约占行程的 1%（室内）与 2.5%（室外），表 I；IMU 对照表 II；KITTI 平均 0.88%（§VII-C，结果图未在论文中给出）。

## 工程实践

- **算力：** 论文在 2.5 GHz 四核、6 GiB 内存笔记本上跑 ROS / Linux，里程计与建图各占一核（§VII）。
- **传感器：** 论文原型是 Hokuyo UTM-30LX 加电机的自研旋转雷达，sweep 约 1 s，线内分辨率 0.25°、线间 4.5°（§IV-A、§V-A）；KITTI 实验换用 Velodyne（§VII-C）。
- **可选 IMU：** 用 IMU 姿态预旋转 sweep 内点云，并用加速度部分去畸变；论文在 4 个短轨迹上的误差从仅 LOAM 的 1.7%–4.4% 降到 0.9%–2.6%（表 II，真值为卷尺手量）。
- **代码获取：** 作者最初通过 ROS wiki 发布（§VII 脚注），GitHub 上的 [laboshinl/loam_velodyne](https://github.com/laboshinl/loam_velodyne) 为第三方镜像（README 自述源码取自 ROS 文档，测试环境 ROS Indigo 与 VLP-16），本页未审查其源码；[A-LOAM](https://github.com/HKUST-Aerial-Robotics/A-LOAM) 是港科大基于 LOAM 与 LOAM_NOTED 的改写版，README 自述「modified from LOAM and LOAM_NOTED」，用 Eigen 与 Ceres Solver。本页只读了两个 README，未运行任何代码。
- **选型：** 与后续方法的横向取舍见 [激光 SLAM / LIO / VIO 选型](../comparisons/lidar-slam-lio-vio-selection.md)。

## 与综述说法的核对

下表核对《激光雷达 SLAM 算法最新综述》（刘铭哲等，公众号「新机器视觉」转载，2026-11-11）对 LOAM 的描述，依据是 RSS 2014 版全文。

| 综述说法 | 核对结果 |
|------|------|
| Zhang 等于 2014 年提出 LOAM | 属实，RSS 2014（论文标题页）；期刊扩展版 2017 |
| 采用特征点法 | 属实：边缘点与平面点（§V-A） |
| 用局部点计算曲率，按曲率分边缘 / 平面点 | 基本属实但措辞不严：论文用同一扫描线上邻域差向量之和定义的光滑度 c（式 1），大者为边缘、小者为平面，可视为曲率的代理量，论文并未称曲率 |
| 匀速运动假设，线性插值运动补偿 | 属实：sweep 内角、线速度恒定，位姿线性插值（§V-C，式 4）；补充：建图阶段点共享时间戳，不再插值（§VI） |
| 缺少回环检测 | 属实，论文自述目前不做回环，列为未来工作（§I、§VIII） |
| 缺少后端优化 | 需限定：论文没有位姿图 / 因子图式后端，但建图模块本身是对地图做的批优化，论文称类似 ICP（§I）；若「后端」指全局一致性优化，则确实没有 |
| 大规模场景下漂移、精度大幅下降 | 论文本身不含这一结论：其数据是短轨迹漂移约 1%–2.5%（表 I）与 KITTI 平均 0.88%；大场景漂移是后续工作（如 LIO-SAM §I）的论述，不是 LOAM 论文自己的实验结果 |
| A-LOAM 是港科大 Qin 根据 LOAM 论文复现的代码 | 不完全准确：A-LOAM README 称改自 LOAM 与 LOAM_NOTED，而非仅据论文复现 |

## 局限

论文自己承认或可直接看出的：

- **没有回环**（§I、§VIII），长时间运行的累计漂移无法被全局修正。
- **依赖速度平滑假设**（§III）；速度突变要靠 IMU 补（§VII-B），论文未给出无 IMU 下剧烈运动的专门实验。

读后看到的未验证之处（作者自述与本页判断分开）：

- 论文的定量验证集中在短轨迹（表 II 行程 27–51 m）与 KITTI；KITTI 结果在论文中没有展示，只给榜单链接（§VII-C），大场景长时漂移没有论文内数据。
- 表 II 的真值是卷尺手量，样本仅 4 条轨迹。
- 特征提取按作者自研雷达的线内 / 线间分辨率设计（§V-A）；固态或小视场雷达是否适用，论文没有验证（推测需要改写）。
- 本页只通读 RSS 2014 版，未读期刊版正文；期刊版是否补充了回环或更多实验，本页未核实。
- 若干数字（表 I、表 II、KITTI 0.88%）是论文自述，未复现。

## 关联页面

- [LeGO-LOAM](./lego-loam.md)：在 LOAM 之上加入地面优化的轻量版本
- [A-LOAM](./paper-a-loam.md)：港科大基于 LOAM 的 Eigen + Ceres 改写版
- [LIO-SAM](./lio-sam.md)：因子图紧耦合激光惯性方案，里程计部分沿用 LOAM 思想
- [LVI-SAM](./paper-lvi-sam.md)：激光-视觉-惯性紧耦合
- [FAST-LIO](./fast-lio.md)：另一条以迭代卡尔曼滤波为核心的激光惯性路线
- [Point-LIO（UniLiDAR）](./point-lio-unilidar.md)
- [Cartographer](./cartographer.md)：带回环与子图的另一类激光 SLAM
- [Ceres Solver](./ceres-solver.md)：A-LOAM 使用的非线性优化库
- [状态估计](../concepts/state-estimation.md)
- [激光雷达感知](../concepts/lidar-sensing.md)
- [激光 SLAM / LIO / VIO 选型](../comparisons/lidar-slam-lio-vio-selection.md)
- [导航·SLAM·自主栈总览](../overview/navigation-slam-autonomy-stack.md)

## 参考来源

- [loam-lidar-odometry-and-mapping_rss_2014.md](../../sources/papers/loam-lidar-odometry-and-mapping_rss_2014.md)：RSS 2014 全文通读归档
- [Zhang & Singh, RSS 2014](https://www.roboticsproceedings.org/rss10/p07.pdf)
- [Zhang & Singh, Autonomous Robots 2017（DOI）](https://doi.org/10.1007/s10514-016-9548-2)

## 继续阅读

- [LeGO-LOAM](./lego-loam.md)
- [LIO-SAM](./lio-sam.md)
- [激光 SLAM / LIO / VIO 选型](../comparisons/lidar-slam-lio-vio-selection.md)
- [A-LOAM 仓库](https://github.com/HKUST-Aerial-Robotics/A-LOAM)
