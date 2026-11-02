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
  - ../../sources/repos/rslidar_sdk.md
summary: "速腾聚创 开源项目 rslidar_sdk（SDK/驱动）：提供速腾聚创激光雷达ROS与ROS2 SDK，连接雷达数据、点云与机器人感知系统。…"
institutions:
  - robosense
---

# rslidar_sdk

## 一句话定义

**rslidar_sdk** 是 [速腾聚创](https://github.com/RoboSense-LiDAR) 公开的 **SDK/驱动** 开源项目：提供速腾聚创激光雷达ROS与ROS2 SDK，连接雷达数据、点云与机器人感知系统。

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
| 机构 | 速腾聚创 |
| 类别 | SDK/驱动 |
| 官方组织 | https://github.com/RoboSense-LiDAR |

## 工程实践

1. 从官方 GitHub/Gitee 组织检索 `rslidar_sdk` 仓库并核对 README 许可与依赖。
2. 对照本库 [424 项覆盖索引](../queries/china-domestic-opensource-424-coverage.md) 查看同公司其它入口是否共用训练/部署链路。
3. 若与既有方法页（如 RL 框架、VLA、SDK）主题相同，优先读关联页中的「开源入口」小节，避免重复维护平行叙事。

## 局限与风险

- 公众号清单为 **策展快照**（2026-09-06）；仓库更名、归档或许可证变化须回官方组织页核实。
- **开源状态**：以仓库 README 与 release 为准（入库日按文章描述归纳，未逐仓 clone 验证）。

## 源码运行时序图

入口：ROS2 下 `ros2 launch rslidar_sdk start.py`（`launch/start.py` 启动可执行文件 `rslidar_sdk_node`，参数 `config_path` 指向 `config/config.yaml`）；ROS1 为 `launch/start.launch`。下图以 `msg_source: 1`（在线雷达）+ `send_point_cloud_ros: true` 的默认配置为例；`rs_driver` 是子模块（`src/rs_driver`），图中只画 `rslidar_sdk` 侧对它的调用。

```mermaid
sequenceDiagram
    participant Launch as launch/start.py
    participant Main as node/rslidar_sdk_node.cpp main()
    participant NM as NodeManager (src/manager/node_manager.cpp)
    participant SD as SourceDriver (source_driver.hpp)
    participant RSD as rs_driver LidarDriver 子模块
    participant Dst as DestinationPointCloudRos (source_pointcloud_ros.hpp)

    Launch->>Main: 启动 rslidar_sdk_node，传 config_path
    Main->>Main: rclcpp::init；YAML::LoadFile(config.yaml)
    Main->>NM: init(config)
    NM->>NM: 读 common.msg_source / send_point_cloud_ros
    loop 遍历 config 的 lidar 列表
        NM->>SD: make_shared SourceDriver(MSG_FROM_LIDAR) 后 init(lidar_config)
        SD->>SD: 解析 driver 段为 RSDriverParam（msop_port、lidar_type 等）
        SD->>RSD: regPointCloudCallback(getPointCloud, putPointCloud)
        SD->>SD: 启动 point_cloud_process_thread_
        SD->>RSD: init(driver_param)
        NM->>Dst: init(lidar_config) 创建 PointCloud2 publisher
        NM->>SD: regPointCloudCallback(dst)
    end
    Main->>NM: start()
    NM->>SD: start()
    SD->>RSD: start()
    Note over Main: 之后 g_cv.wait 阻塞，直到 SIGINT
    RSD-->>SD: putPointCloud(msg) 推入 point_cloud_queue_
    SD->>SD: processPointCloud 线程 popWait 取出
    SD->>Dst: sendPointCloud(*msg)（Source::sendPointCloud 遍历 pc_cb_vec_）
    Dst-->>Dst: toRosMsg 后 pub_->publish，话题默认 /rslidar_points
    SD->>SD: free_point_cloud_queue_.push(msg) 回收缓冲
```

图依据：RoboSense-LiDAR/rslidar_sdk@8b4b4b7，读过 `node/rslidar_sdk_node.cpp`、`src/manager/node_manager.cpp`、`src/source/source.hpp`、`src/source/source_driver.hpp`、`src/source/source_pointcloud_ros.hpp`（ROS2 分支）、`config/config.yaml`、`launch/start.py`、`.gitmodules`；`rs_driver` 子模块内部（UDP 收包与解码）未读，故不画。

## 关联页面

- [国内具身开源全景技术地图](../overview/china-domestic-embodied-opensource-76-companies-technology-map.md)
- [HMI 开源项目主表导读](../queries/hmi-opensource-projects-coverage.md)
- [Humanoid Motion Intelligence](../entities/humanoid-motion-intelligence.md)

## 参考来源

- [rslidar_sdk 源码归档](../../sources/repos/rslidar_sdk.md)（<https://github.com/RoboSense-LiDAR/rslidar_sdk>）

- [国内具身智能开源全景（微信公众号）](../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md)

- 源码核对（2026-11-02）：RoboSense-LiDAR/rslidar_sdk@8b4b4b7，读过 node/rslidar_sdk_node.cpp、src/manager/node_manager.cpp、src/source/source.hpp、src/source/source_driver.hpp、src/source/source_pointcloud_ros.hpp、config/config.yaml、launch/start.py；时序图依据以上文件，未运行代码。

## 推荐继续阅读

- [速腾聚创 官方组织](https://github.com/RoboSense-LiDAR)
