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
  - ../../sources/repos/orbbecsdk_ros2.md
summary: "奥比中光 开源项目 OrbbecSDK_ROS2（SDK/驱动）：把奥比中光相机接入ROS2，发布图像、深度和相机参数供感知与机器人应用使用。…"
institutions:
  - orbbec
---

# OrbbecSDK_ROS2

## 一句话定义

**OrbbecSDK_ROS2** 是 [奥比中光](https://github.com/orbbec) 公开的 **SDK/驱动** 开源项目：把奥比中光相机接入ROS2，发布图像、深度和相机参数供感知与机器人应用使用。

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
| 机构 | 奥比中光 |
| 类别 | SDK/驱动 |
| 官方组织 | https://github.com/orbbec |

## 工程实践

1. 从官方 GitHub/Gitee 组织检索 `OrbbecSDK_ROS2` 仓库并核对 README 许可与依赖。
2. 对照本库 [424 项覆盖索引](../queries/china-domestic-opensource-424-coverage.md) 查看同公司其它入口是否共用训练/部署链路。
3. 若与既有方法页（如 RL 框架、VLA、SDK）主题相同，优先读关联页中的「开源入口」小节，避免重复维护平行叙事。

## 局限与风险

- 公众号清单为 **策展快照**（2026-09-06）；仓库更名、归档或许可证变化须回官方组织页核实。
- **开源状态**：以仓库 README 与 release 为准（入库日按文章描述归纳，未逐仓 clone 验证）。

## 关联页面

- [国内具身开源全景技术地图](../overview/china-domestic-embodied-opensource-76-companies-technology-map.md)
- [HMI 开源项目主表导读](../queries/hmi-opensource-projects-coverage.md)
- [Humanoid Motion Intelligence](../entities/humanoid-motion-intelligence.md)

## 源码运行时序图

入口：`ros2 launch orbbec_camera gemini2.launch.py`（机型对应的 launch 文件见 `orbbec_camera/launch/`）。launch 文件既可起独立 Node（可执行文件 `orbbec_camera_node`），也可起 `ComposableNodeContainer` 加载插件 `orbbec_camera::OBCameraNodeDriver`。

```mermaid
sequenceDiagram
    autonumber
    participant U as 用户（ros2 launch）
    participant L as launch/gemini2.launch.py
    participant C as component_container / orbbec_camera_node
    participant D as OBCameraNodeDriver（ob_camera_node_driver.cpp）
    participant SDK as libOrbbecSDK（ob::Context / ob::Pipeline）
    participant N as OBCameraNode（ob_camera_node.cpp）
    participant R as ROS2 话题（image / pointcloud / imu / tf）
    U->>L: 启动 launch 文件
    L->>C: 创建容器或独立节点
    C->>D: 构造并调用 init()
    D->>SDK: 创建 ob::Context（扩展目录、日志、设备枚举）
    D->>SDK: registerDeviceChangedCallback
    D->>D: create_wall_timer → checkConnectTimer；query_thread_ → queryDevice
    D->>SDK: ctx_->queryDeviceList()
    SDK-->>D: DeviceList
    D->>D: onDeviceConnected / queryDevice → startDevice → selectDevice
    D->>D: initializeDevice(device)
    D->>N: make_unique OBCameraNode(this, device_, parameters_)
    N->>N: setupDevices，loadConfigJson，setupProfiles，setupPublishers
    D->>N: startIMU()，startStreams()
    N->>SDK: pipeline_->start(config, onNewFrameSetCallback)
    loop 每个 FrameSet
        SDK-->>N: onNewFrameSetCallback(frame_set)
        N->>N: processDepthFrameFilter / processColorFrameFilter 等后处理
        N->>N: onNewFrameCallback(frame, stream_index)
        N->>R: image_publishers_ publish（点云、元数据同理）
    end
```

图依据：OrbbecSDK/OrbbecSDK_ROS2@c153462，读过 `README.MD`、`orbbec_camera/launch/gemini2.launch.py`、`orbbec_camera/src/ob_camera_node_driver.cpp`、`orbbec_camera/src/ob_camera_node.cpp`（未运行）。

## 参考来源

- [OrbbecSDK_ROS2 源码归档](../../sources/repos/orbbecsdk_ros2.md)（<https://github.com/orbbec/OrbbecSDK_ROS2>）

- [国内具身智能开源全景（微信公众号）](../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md)
- 源码核对（2026-11-02）：orbbec/OrbbecSDK_ROS2@c153462，读过 `README.MD`、`orbbec_camera/launch/gemini2.launch.py`、`orbbec_camera/src/ob_camera_node_driver.cpp`、`orbbec_camera/src/ob_camera_node.cpp`；时序图依据以上文件，未运行代码。

## 推荐继续阅读

- [奥比中光 官方组织](https://github.com/orbbec)
