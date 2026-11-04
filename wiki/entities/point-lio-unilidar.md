---
type: entity
tags: [repo, unitree, unitreerobotics, lidar, slam, localization]
status: complete
updated: 2026-07-26
related:
  - ./unitree.md
  - ./unilidar-sdk2.md
  - ./autonomy-stack-go2.md
  - ../tasks/locomotion.md
  - ../concepts/sim2real.md
  - ../queries/go2-3d-semantic-mapping-sam-pipeline.md
sources:
  - ../../sources/repos/point_lio_unilidar.md
  - ../../sources/repos/unitree.md
  - ../../sources/repos/autonomy_stack_go2.md
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
- **运动时重影/墙面变厚**：优先查时间同步、逐点时间戳、去畸变、外参与 IMU 初始化，再谈叠加 SAM 语义；见 [GO2 三维语义建图 Query](../queries/go2-3d-semantic-mapping-sam-pipeline.md)。

## 源码运行时序图

入口：`roslaunch point_lio_unilidar mapping_unilidar_l1.launch`（ROS1 节点 `pointlio_mapping`，节点名 `laserMapping`；L2 使用 `mapping_unilidar_l2.launch`，默认 `use_imu_as_input=0`）

```mermaid
sequenceDiagram
    participant DRV as Unitree 雷达驱动话题 /unilidar/cloud 与 /unilidar/imu
    participant CB as src/laserMapping.cpp 回调
    participant PRE as src/preprocess.cpp Preprocess
    participant SYNC as sync_packages
    participant IMU as src/IMU_Processing.hpp ImuProcess
    participant KF as src/Estimator.h kf_output 迭代卡尔曼滤波
    participant IK as include/ikd-Tree KD_TREE
    participant PUB as ROS 发布器 odom、path 与点云

    DRV->>CB: 点云消息进入 standard_pcl_cbk
    CB->>PRE: p_pre->process(msg, ptr)
    PRE->>PRE: lidar_type 为 UNILIDAR 时调用 unilidar_handler
    DRV->>CB: IMU 消息进入 imu_cbk 并入队
    loop main 中 while ros::ok 每次迭代
        CB->>CB: ros::spinOnce 处理回调
        CB->>SYNC: sync_packages(Measures) 把点云与 IMU 对齐成一组
        CB->>IMU: p_imu->Process(Measures, feats_undistort) 用 IMU 做运动去畸变
        CB->>IK: 首次且地图为空时 ikdtree.Build 建初始地图
        CB->>CB: lasermap_fov_segment 按视场裁剪地图
        CB->>CB: 按 filter_size_surf 对本体系点云下采样
        loop 每个时间段的 IEKF 迭代
            CB->>KF: kf_output.predict 用 IMU 输入传播状态与协方差
            CB->>KF: update_iterated_dyn_share_modified 调用 h_model_output
            KF->>IK: Nearest_Search 查找每个点的近邻平面
            KF-->>CB: 返回更新后的位姿状态
        end
        CB->>IK: map_incremental 把新点加入 ikd-Tree
        CB->>PUB: publish_path、publish_frame_world 与 publish_odometry
    end
```

图依据：unitreerobotics/point_lio_unilidar@18ed597，读过 launch/mapping_unilidar_l1.launch、CMakeLists.txt（target 段）、config/unilidar_l1.yaml（部分）、src/laserMapping.cpp（main、sync_packages、imu_cbk、map_incremental、publish 段）、src/preprocess.cpp（process 分派，unilidar_handler 仅见位置）、src/Estimator.cpp（Nearest_Search 所在函数）。未编译、未在 ROS 中运行。

## 关联页面

- [UniLidar SDK](./unilidar-sdk2.md)
- [autonomy_stack_go2](./autonomy-stack-go2.md)
- [Locomotion](../tasks/locomotion.md)
- [Unitree](./unitree.md)
- [GO2 三维语义建图与 SAM 流水线](../queries/go2-3d-semantic-mapping-sam-pipeline.md)

## 参考来源

- [sources/repos/point_lio_unilidar.md](../../sources/repos/point_lio_unilidar.md)
- [sources/repos/autonomy_stack_go2.md](../../sources/repos/autonomy_stack_go2.md) — CMU GO2 几何自主导航栈中的 Point-LIO 用法
- Point-LIO 论文：<https://onlinelibrary.wiley.com/doi/epdf/10.1002/aisy.202200459>
- 上游：<https://github.com/unitreerobotics/point_lio_unilidar>
- 源码核对（2026-11-02）：unitreerobotics/point_lio_unilidar@18ed597，读过 launch/mapping_unilidar_l1.launch、CMakeLists.txt、config/unilidar_l1.yaml、src/laserMapping.cpp、src/preprocess.cpp、src/Estimator.cpp；时序图依据以上文件，未运行代码。

## 推荐继续阅读

- Point-LIO 官方仓：<https://github.com/hku-mars/Point-LIO>
- CMU `autonomy_stack_go2`：<https://github.com/jizhang-cmu/autonomy_stack_go2>

