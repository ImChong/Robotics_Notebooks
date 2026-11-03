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
  - ../../sources/repos/hesailidar_ros_2_0.md
summary: "禾赛科技 开源项目 HesaiLidar_ROS_2.0（工程与工具）：把禾赛激光雷达接入ROS与ROS2，发布点云和设备状态供机器人系统使用。…"
institutions:
  - hesai
---

# HesaiLidar_ROS_2.0

## 一句话定义

**HesaiLidar_ROS_2.0** 是 [禾赛科技](https://github.com/HesaiTechnology) 公开的 **工程与工具** 开源项目：把禾赛激光雷达接入ROS与ROS2，发布点云和设备状态供机器人系统使用。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| SDK | Software Development Kit | 真机控制与状态读取接口 |
| RL | Reinforcement Learning | 强化学习训练与策略优化 |
| VLA | Vision-Language-Action | 视觉–语言–动作统一策略 |
| Sim2Real | Simulation to Real | 仿真策略迁移真机 |
| URDF | Unified Robot Description Format | 机器人描述与仿真资产 |

## 为什么重要

- 收录于 [国内具身智能开源全景（76 家 · 424 项）](../overview/china-domestic-embodied-opensource-76-companies-technology-map.md) 的 **第五层** 分组。
- 与 [Humanoid Motion Intelligence](../entities/humanoid-motion-intelligence.md) 同源策展；本页为 **独立详情节点**，便于从公司清单跳到机制与入口说明。

## 核心原理

| 字段 | 内容 |
|------|------|
| 机构 | 禾赛科技 |
| 类别 | 工程与工具 |
| 官方组织 | https://github.com/HesaiTechnology |

## 工程实践

1. 从官方 GitHub/Gitee 组织检索 `HesaiLidar_ROS_2.0` 仓库并核对 README 许可与依赖。
2. 对照本库 [424 项覆盖索引](../queries/china-domestic-opensource-424-coverage.md) 查看同公司其它入口是否共用训练/部署链路。
3. 若与既有方法页（如 RL 框架、VLA、SDK）主题相同，优先读关联页中的「开源入口」小节，避免重复维护平行叙事。

## 局限与风险

- 公众号清单为 **策展快照**（2026-09-06）；仓库更名、归档或许可证变化须回官方组织页核实。
- **开源状态**：以仓库 README 与 release 为准（入库日按文章描述归纳，未逐仓 clone 验证）。

## 源码运行时序图

入口：`ros2 launch hesai_ros_driver start.py`（ROS 1 为 `roslaunch hesai_ros_driver start.launch`；Dashing 用 `dashing_start.py`），拉起节点 `hesai_ros_driver_node` 与 rviz2。下图按 ROS 2 路径（`source_driver_ros2.hpp`）、`source_type: 1`（实时雷达）绘制。

```mermaid
sequenceDiagram
    participant L as launch/start.py
    participant N as node/hesai_ros_driver_node.cc main()
    participant M as src/manager/node_manager.cc NodeManager
    participant S as src/manager/source_driver_ros2.hpp SourceDriver
    participant SDK as HesaiLidar_SDK_2.0 HesaiLidarSdk（子模块 src/driver）
    participant ROS as ROS 2 话题
    L->>N: 启动 hesai_ros_driver_node（命名空间 hesai_ros_driver）
    N->>N: rclcpp::init；config_path 默认 config/config.yaml，可由参数 config_path 覆盖；YAML::LoadFile
    N->>M: Init(config)
    M->>S: 对 config 中 lidar 每一项 SourceDriver(DATA_FROM_LIDAR)->Init(lidar_config[i])
    S->>S: GetDriveYamlParam 填 DriverParam；创建 rclcpp::Node 与各 publisher（PointCloud2、IMU、丢包、PTP 等，按配置开关）
    S->>SDK: RegRecvCallback(SendPointCloud 等)；driver_ptr_->Init(driver_param)
    N->>M: Start()
    M->>S: Start()
    S->>SDK: HesaiLidarSdk::Start()：等待初始化完成后起 Run 线程
    loop SDK 的 Run 线程
        SDK->>SDK: GetOnePacket → DecodePacket → ComputeXYZI，凑满一帧 scan_complete
        SDK-->>S: point_cloud_cb_(frame) 即 SendPointCloud(frame)
        S->>ROS: pub_->publish(ToRosMsg(frame, frame_id))，字段 x y z intensity ring timestamp
    end
    N->>N: 主线程循环等待 IsPlayEnded() 或 SIGINT
    N->>M: Stop()，进而 SourceDriver::Stop → HesaiLidarSdk::Stop
```

图依据：HesaiTechnology/HesaiLidar_ROS_2.0@3d54078（SDK 子模块未随浅克隆拉取，SDK 内部调用以 HesaiLidar_SDK_2.0@fbcd94a 的 `driver/hesai_lidar_sdk.hpp` 核对），读过 README.md、config/config.yaml、launch/start.py、node/hesai_ros_driver_node.cc、src/manager/node_manager.cc、src/manager/source_driver_ros2.hpp、.gitmodules。`hesai_ros_driver_node.cu`（GPU 版）、`source_driver_ros1.hpp` 与 pcap / rosbag / 串口数据源分支未细读，图中未画。

## 关联页面

- [国内具身开源全景技术地图](../overview/china-domestic-embodied-opensource-76-companies-technology-map.md)
- [HMI 开源项目主表导读](../queries/hmi-opensource-projects-coverage.md)
- [Humanoid Motion Intelligence](../entities/humanoid-motion-intelligence.md)

## 参考来源

- [HesaiLidar_ROS_2.0 源码归档](../../sources/repos/hesailidar_ros_2_0.md)（<https://github.com/HesaiTechnology/HesaiLidar_ROS_2.0>）

- [国内具身智能开源全景（微信公众号）](../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md)
- 源码核对（2026-11-02）：HesaiTechnology/HesaiLidar_ROS_2.0@3d54078，读过 README.md、config/config.yaml、launch/start.py、node/hesai_ros_driver_node.cc、src/manager/node_manager.cc、src/manager/source_driver_ros2.hpp、.gitmodules；SDK 子模块调用另核对 HesaiLidar_SDK_2.0@fbcd94a 的 driver/hesai_lidar_sdk.hpp；时序图依据以上文件，未运行代码。

## 推荐继续阅读

- [禾赛科技 官方组织](https://github.com/HesaiTechnology)
