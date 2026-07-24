---
type: entity
tags: [repo, unitree, unitreerobotics, lidar, slam, localization]
status: complete
updated: 2026-07-24
related:
  - ./unitree.md
  - ./unilidar-sdk2.md
  - ../tasks/locomotion.md
  - ../concepts/sim2real.md
sources:
  - ../../sources/repos/point_lio_unilidar.md
  - ../../sources/repos/unitree.md
summary: "point_lio_unilidar 将 Point-LIO 适配到 Unitree L1/L2 雷达，提供高带宽激光惯性里程计与建图能力；推荐 Ubuntu 20.04 + ROS Noetic。"
---

# point_lio_unilidar

**point_lio_unilidar** 把学术界 **Point-LIO**（稳健高带宽 LiDAR-Inertial Odometry）接到宇树 **L1 / L2** 雷达产品。

## 一句话定义

官方维护的 UniLidar + Point-LIO 组合——在振动与剧烈运动下仍追求准确高频里程计与可靠建图。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| LIO | LiDAR-Inertial Odometry | 激光惯性里程计 |
| LiDAR | Light Detection and Ranging | L1/L2 传感器 |
| IMU | Inertial Measurement Unit | 惯性量测 |
| SLAM | Simultaneous Localization and Mapping | 定位建图总称 |
| ROS | Robot Operating System | 推荐 Noetic |
| FOV | Field of View | L1/L2 约 360°×90° |

## 为什么重要

- 给 Unitree 移动平台一条**可复现的开源 LIO 基线**，而不是只卖硬件。
- Point-LIO 论文强调高带宽与抗剧烈运动，契合足式/轮足颠簸场景。
- 与 [`unilidar-sdk2`](./unilidar-sdk2.md) 形成「驱动 → 算法」闭环。

## 核心原理

上游算法来自 HKU MARS Lab [Point-LIO](https://github.com/hku-mars/Point-LIO)。本仓负责 Unitree 雷达驱动对接、话题与参数。L1/L2 共性：大视场、非重复扫描、低成本、适低速移动机器人。

## 工程实践

1. Ubuntu **20.04** + ROS **Noetic**（上游劝阻 18.04 及更旧）。
2. 安装 `ros-$DISTRO-pcl-conversions`、`libeigen3-dev`。
3. 先用 UniLidar SDK 确认点云/IMU 正常，再启动本仓 LIO。
4. 参考仓库视频 demo（L1/L2）核对室外/室内表现预期。

## 局限与风险

- 低速平台假设：高速或极端运动需重调参数。
- ROS1 Noetic 栈与实验室 ROS2 主线并存时，注意桥接成本。
- 建图质量强烈依赖外参与时间同步，需按 SDK 坐标定义校准。

## 关联页面

- [UniLidar SDK](./unilidar-sdk2.md)
- [Locomotion](../tasks/locomotion.md)
- [Unitree](./unitree.md)

## 参考来源

- [sources/repos/point_lio_unilidar.md](../../sources/repos/point_lio_unilidar.md)
- Point-LIO 论文：<https://onlinelibrary.wiley.com/doi/epdf/10.1002/aisy.202200459>
- 上游：<https://github.com/unitreerobotics/point_lio_unilidar>

## 推荐继续阅读

- Point-LIO 官方仓：<https://github.com/hku-mars/Point-LIO>

