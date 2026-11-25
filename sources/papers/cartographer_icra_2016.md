# Real-Time Loop Closure in 2D LIDAR SLAM（Google Cartographer，ICRA 2016）

> 论文来源归档（ingest）

- **标题：** Real-Time Loop Closure in 2D LIDAR SLAM
- **类型：** paper / 2d-slam / loop-closure / branch-and-bound / scan-matching
- **出处：** IEEE International Conference on Robotics and Automation (ICRA 2016), pp. 1271–1278
- **原文链接：** <https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/45466.pdf>
- **作者：** Wolfgang Hess, Damon Kohler, Holger Rapp, Daniel Andor（Google Inc.）
- **官方仓库：** <https://github.com/cartographer-project/cartographer>
- **入库日期：** 2026-11-21
- **通读：** 已完成
- **一句话说明：** 提出基于概率子图（Submap）分层位姿图优化与分支定界（Branch-and-Bound, BnB）相关扫描匹配的实时 2D 激光闭环 SLAM 算法，在桌面 CPU（Intel Xeon E5-1650）上于约 2253 m 德国博物馆轨迹实现 5.3x 实时运行与厘米级地图一致精度（仅使用 5 cm 分辨率栅格与 49 € 级 Neato Revo LDS 也可稳定建图）。

---

## 核心摘录（面向 wiki 编译）

### 1) Local SLAM：子图构建与连续 Ceres 扫描匹配
- 局部前端连续将近邻帧激光扫描经运动滤波（距离/角度/时间阈值）后通过 **Ceres 非线性最小二乘扫描匹配器** 对齐至当前活跃概率栅格子图 $M$（grid resolution 可达 5 cm），并利用相邻像素双三次插值使损失函数可微。
- 为防止单子图内部持续累计里程计漂移，累积一定数量（如 $N_{\text{scans}}$）扫描后即冻结子图，随后仅在全局闭环阶段整体刚体调整该子图位姿。

### 2) Global SLAM：Sparse Pose Adjustment（SPA）与实时分支定界闭环
- 所有已完成子图与历史扫描点共同构成后台稀疏位姿调整（SPA）优化图：$\arg\min_{\Xi^m, \Xi^s} \frac{1}{2} \sum_{ij} \rho\left( E^2(\xi_i^m, \xi_j^s; \Sigma_{ij}, \xi_{ij}) \right)$（采用 Huber Loss $\rho$ 抑止误闭环离群点）。
- **分支定界（BnB）加速全局窗口搜索：** 传统暴力相关扫描匹配（CSM）在百米搜索窗内复杂度过高；论文构造 7 层预计算多分辨率最大概率金字塔栅格（Precomputed Grids，步长 $2^h r$），高层单元格存储子栅格区域的最大概率上界，作为 BnB 深度优先剪枝的严格上界，实现无需初值先验的毫秒级全局闭环检索。

### 3) 关键基准评测数据
- **Deutsches Museum（2253 m，双 TiM551）：** 单核/多线程桌面 CPU 实现 **5.3x 实时** 建图（对比 Karto SLAM、GMapping 的较大漂移或实时崩溃），可构建 30,000+ m² 连贯平面图。
- **Radish 基准与低成本雷达：** 在 MIT CSAIL、ACES3 等标准数据集上绝对误差优于对照，同时在低转速（5 Hz）、仅 1 组旋转扫描的低价位 **Neato Revo LDS**（49 €）室内长环测试中依然实现高精度闭环。

---

## 对 wiki 的映射

- 实体页：[`wiki/entities/cartographer.md`](../../wiki/entities/cartographer.md)
- 仓储归档：[`sources/repos/cartographer.md`](../repos/cartographer.md)
- 选型对比：[`wiki/comparisons/lidar-slam-lio-vio-selection.md`](../../wiki/comparisons/lidar-slam-lio-vio-selection.md)
