# puma_icra_2021_vizzo

> 来源归档（ingest）

- **标题：** Poisson Surface Reconstruction for LiDAR Odometry and Mapping
- **短名：** PUMA
- **类型：** paper
- **来源：** ICRA 2021 / IPB Bonn PDF
- **原始链接：**
  - <https://www.ipb.uni-bonn.de/pdfs/vizzo2021icra.pdf>
  - <https://youtu.be/7yWtYWaO5Nk>（演示视频）
- **项目页 / 作者：** <https://www.ipb.uni-bonn.de/people/ignacio-vizzo/index.html> — 归档见 [`sources/sites/ipb-puma-vizzo.md`](../sites/ipb-puma-vizzo.md)
- **作者：** Ignacio Vizzo, Xieyuanli Chen, Nived Chebrolu, Jens Behley, Cyrill Stachniss
- **机构：** 波恩大学（University of Bonn），Photogrammetry & Robotics Lab（PRBonn / IPB）
- **出处：** IEEE Intl. Conf. on Robotics & Automation (ICRA) 2021
- **入库日期：** 2026-09-15
- **一句话说明：** 以 **Poisson 曲面重建** 将 LiDAR 地图表示为 **三角 mesh**，通过 **射线投射（ray casting）** 建立 **scan-to-mesh** 对应，并以 **point-to-plane** 配准估计 6DoF 位姿，形成 frame-to-mesh 里程计与建图管线。

## 核心摘录

### 1) 问题与动机
- 经典 LiDAR 建图多用 **点云 / surfel / TSDF 体素**；mesh 地图在自动驾驶场景中可提供 **更紧凑、更平滑** 的表面，并利于可视化和部分规划接口。
- 难点在于：如何在 **增量 mesh 地图** 上与 **新扫描** 建立稳定对应并完成位姿估计，而非只做离线重建。

### 2) 方法要点
1. **地图表示：** 用 [Poisson Surface Reconstruction (PSR)](http://sites.fas.harvard.edu/~cs277/papers/poissonrecon.pdf) 从累积点云生成 **三角 mesh** 作为全局地图。
2. **Scan-to-mesh 对应：** 对输入扫描中每个点，向 mesh 做 **ray-to-triangle** 求交；用 [Intel Embree](https://www.embree.org/)（经 [pyembree](https://github.com/scopatz/pyembree)）加速射线投射。
3. **位姿估计：** **Point-to-plane (P2L)** 迭代配准估计 LiDAR 6DoF 位姿（frame-to-mesh / icp_frame_2_mesh）。
4. **管线：** `pipelines/slam/puma_pipeline.py` 串联数据转换、建图与里程计；对比基线含 surfel 与 TSDF 地图（论文 Fig. / README 定性对比 KITTI `00`）。

### 3) 实验（论文 / 仓库摘要）
| 基准 | 设定 | 读法 |
|------|------|------|
| KITTI Odometry | 64-beam Velodyne 类；序列如 `00`、`07` | 官方 Docker 流程：`bin2ply` → `puma_pipeline.py` |
| Mai City | Bonn 发布数据集 | 与 KITTI 同类传感器；见 IPB 数据页 |
| 地图形态 | Surfels / TSDF / **PUMA mesh** | README 表：mesh 在表面连续性与噪声上相对 surfel/TSDF 有视觉优势 |

- **应用取向：** 主要面向 **自动驾驶** 车辆 LiDAR 里程计与建图研究原型，而非 ROS 实时产品栈。

### 4) 开源核查（步骤 2.5）
- **GitHub：** [`PRBonn/puma`](https://github.com/PRBonn/puma) — README、`INSTALL.md`、`docker/`、`apps/pipelines/slam/puma_pipeline.py` 等 **可运行入口齐全** → **已开源**。
- **项目页：** 作者 IPB 页链到论文 PDF 与视频；代码入口以 GitHub 为准。
- **依赖：** Docker 推荐路径；本地安装见 `INSTALL.md`；数据集需自备 KITTI / Mai City。

## 对 wiki 的映射

- 升格 [PUMA 论文实体](../../wiki/entities/paper-puma-lidar-mesh-odometry.md)
- 交叉更新 [里程计与激光雷达融合](../../wiki/methods/lidar-odometry-fusion.md)、[LiDAR / LIO / VIO 选型](../../wiki/comparisons/lidar-slam-lio-vio-selection.md)、[Points as Tori（PSR 对照）](../../wiki/entities/paper-points-as-tori.md)

## 当前提炼状态

- [x] 摘要 + scan-to-mesh + P2L + 开源边界
- [x] wiki 实体页与交叉引用
- [x] `sources/sites/` + `sources/repos/`
