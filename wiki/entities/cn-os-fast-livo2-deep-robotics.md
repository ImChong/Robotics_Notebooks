---
type: entity
tags: [repo, china-embodied-opensource, open-source, project]
status: draft
updated: 2026-09-06
related:
  - ../overview/china-domestic-embodied-opensource-76-companies-technology-map.md
  - ../entities/humanoid-motion-intelligence.md
  - ../queries/china-domestic-opensource-424-coverage.md
sources:
  - ../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md
  - ../../sources/repos/fast-livo2-deep-robotics.md
summary: "云深处科技 开源项目 fast-livo2-deep-robotics（工程与工具）：感知/定位/建图模块：为导航与控制提供环境状态…"
institutions:
  - deeprobotics
---

# fast-livo2-deep-robotics

## 一句话定义

**fast-livo2-deep-robotics** 是 [云深处科技](https://github.com/DeepRoboticsLab) 公开的 **工程与工具** 开源项目：感知/定位/建图模块：为导航与控制提供环境状态

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| SDK | Software Development Kit | 真机控制与状态读取接口 |
| RL | Reinforcement Learning | 强化学习训练与策略优化 |
| VLA | Vision-Language-Action | 视觉–语言–动作统一策略 |
| Sim2Real | Simulation to Real | 仿真策略迁移真机 |
| URDF | Unified Robot Description Format | 机器人描述与仿真资产 |

## 为什么重要

- 收录于 [国内具身智能开源全景（76 家 · 424 项）](../overview/china-domestic-embodied-opensource-76-companies-technology-map.md) 的 **第一层** 分组。
- 与 [Humanoid Motion Intelligence](../entities/humanoid-motion-intelligence.md) 同源策展；本页为 **独立详情节点**，便于从公司清单跳到机制与入口说明。

## 核心原理

| 字段 | 内容 |
|------|------|
| 机构 | 云深处科技 |
| 类别 | 工程与工具 |
| 官方组织 | https://github.com/DeepRoboticsLab |

## 工程实践

1. 从官方 GitHub/Gitee 组织检索 `fast-livo2-deep-robotics` 仓库并核对 README 许可与依赖。
2. 对照本库 [424 项覆盖索引](../queries/china-domestic-opensource-424-coverage.md) 查看同公司其它入口是否共用训练/部署链路。
3. 若与既有方法页（如 RL 框架、VLA、SDK）主题相同，优先读关联页中的「开源入口」小节，避免重复维护平行叙事。

## 局限与风险

- 公众号清单为 **策展快照**（2026-09-06）；仓库更名、归档或许可证变化须回官方组织页核实。
- **开源状态**：以仓库 README 与 release 为准（入库日按文章描述归纳，未逐仓 clone 验证）。

## 源码运行时序图

入口：终端 1 `ros2 launch livox_ros_driver2 msg_MID360s_launch.py`（Mid-360 用 `msg_MID360_launch.py`）；终端 2 `ros2 launch realsense2_camera rs_launch.py ...`；终端 3 `ros2 launch fast_livo mapping_avia.launch.py use_rviz:=True`（数据集回放时改为 `ros2 bag play Retail_Street.db3`）。建图节点可执行文件为 `fastlivo_mapping`，入口 `src/FAST-LIVO2/src/main.cpp`。

```mermaid
sequenceDiagram
    participant L as livox_ros_driver2 (msg_MID360s_launch.py)
    participant C as realsense2_camera 或 ros2 bag play
    participant Lc as mapping_avia.launch.py
    participant Mn as main.cpp main()
    participant LM as LIVMapper (LIVMapper.cpp)
    participant IMU as ImuProcess::Process2
    participant VM as VoxelMapManager (LIO)
    participant VIO as VIOManager::processFrame
    participant Rv as RViz2 / 下游话题

    L-->>LM: /livox/lidar (CustomMsg), /livox/imu
    C-->>LM: 相机图像话题 (img_topic，来自 config/avia.yaml)
    Lc->>Mn: 启动 fastlivo_mapping（参数 avia.yaml + camera_pinhole.yaml），并可选启动 image_transport republish 与 rviz2
    Mn->>LM: LIVMapper(nh, "laserMapping", options)
    Mn->>LM: initializeSubscribersAndPublishers()（订阅 lidar/imu/img，发布 /cloud_registered、/aft_mapped_to_init、/path 等）
    Mn->>LM: run()
    loop rclcpp::ok()，rate 5000 Hz
        LM->>LM: rclcpp::spin_some() 触发 livox_pcl_cbk / imu_cbk / img_cbk 入缓冲
        LM->>LM: sync_packages(LidarMeasures)，数据不齐则 sleep 继续
        LM->>LM: handleFirstFrame()
        LM->>IMU: processImu() → Process2()（IMU 前向传播与点云去畸变，可选重力对齐）
        alt lio_vio_flg == VIO
            LM->>VIO: handleVIO() → processFrame(图像, 体素地图)
            VIO-->>Rv: publish_frame_world、publish_img_rgb
        else LIO 或 LO
            LM->>VM: handleLIO()：降采样 → BuildVoxelMap（首帧）/ StateEstimation(state_propagat)
            VM-->>LM: 更新后的状态 (_state)
            LM->>VM: UpdateVoxelMap()，必要时 mapSliding()
            LM-->>Rv: publish_odometry (/aft_mapped_to_init)、publish_path，无相机时 publish_frame_world
        end
    end
    LM->>LM: 退出循环后 savePCD()
```

图依据：DeepRoboticsLab/fast-livo2-deep-robotics@f6762c9，读过 README.md、src/FAST-LIVO2/src/main.cpp、src/FAST-LIVO2/src/LIVMapper.cpp（run、processImu、handleVIO、handleLIO、initializeSubscribersAndPublishers）、src/FAST-LIVO2/launch/mapping_avia.launch.py；ImuProcess、VoxelMapManager、VIOManager 内部实现未读，仅按 LIVMapper 的调用关系绘制。

## 关联页面

- [国内具身开源全景技术地图](../overview/china-domestic-embodied-opensource-76-companies-technology-map.md)
- [HMI 开源项目主表导读](../queries/hmi-opensource-projects-coverage.md)
- [Humanoid Motion Intelligence](../entities/humanoid-motion-intelligence.md)

## 参考来源

- [fast-livo2-deep-robotics 源码归档](../../sources/repos/fast-livo2-deep-robotics.md)（<https://github.com/DeepRoboticsLab/fast-livo2-deep-robotics>）

- [国内具身智能开源全景（微信公众号）](../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md)
- 源码核对（2026-11-02）：DeepRoboticsLab/fast-livo2-deep-robotics@f6762c9，读过 README.md、src/FAST-LIVO2/src/main.cpp、src/FAST-LIVO2/src/LIVMapper.cpp（部分函数）、src/FAST-LIVO2/launch/mapping_avia.launch.py；时序图依据以上文件，未运行代码。

## 推荐继续阅读

- [云深处科技 官方组织](https://github.com/DeepRoboticsLab)
