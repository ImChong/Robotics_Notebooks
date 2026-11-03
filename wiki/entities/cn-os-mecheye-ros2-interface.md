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
  - ../../sources/repos/mecheye_ros2_interface.md
summary: "梅卡曼德 开源项目 mecheye_ros2_interface（工程与工具）：为Mech-Eye工业3D相机提供ROS2接口，使机器人系统能够获取图像、深度和点云并接入感知与操作流程。…"
institutions:
  - mech-mind
---

# mecheye_ros2_interface

## 一句话定义

**mecheye_ros2_interface** 是 [梅卡曼德](https://github.com/MechMindRobotics) 公开的 **工程与工具** 开源项目：为Mech-Eye工业3D相机提供ROS2接口，使机器人系统能够获取图像、深度和点云并接入感知与操作流程。

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
| 机构 | 梅卡曼德 |
| 类别 | 工程与工具 |
| 官方组织 | https://github.com/MechMindRobotics |

## 工程实践

1. 从官方 GitHub/Gitee 组织检索 `mecheye_ros2_interface` 仓库并核对 README 许可与依赖。
2. 对照本库 [424 项覆盖索引](../queries/china-domestic-opensource-424-coverage.md) 查看同公司其它入口是否共用训练/部署链路。
3. 若与既有方法页（如 RL 框架、VLA、SDK）主题相同，优先读关联页中的「开源入口」小节，避免重复维护平行叙事。

## 局限与风险

- 公众号清单为 **策展快照**（2026-09-06）；仓库更名、归档或许可证变化须回官方组织页核实。
- **开源状态**：以仓库 README 与 release 为准（入库日按文章描述归纳，未逐仓 clone 验证）。

## 源码运行时序图

入口：`colcon build` 后 `ros2 run mecheye_ros_interface start`，或 `ros2 launch ~/colcon_ws/src/mecheye_ros2_interface/launch/start_camera.py`；另开终端 `ros2 service call /capture_point_cloud mecheye_ros_interface/srv/CapturePointCloud`。依赖闭源的 Mech-Eye SDK（`mmind::eye`，不在本仓库内）。

```mermaid
sequenceDiagram
    participant L as launch/start_camera.py
    participant Main as src/main.cpp main()
    participant Node as MechMindCamera (src/MechMindCamera.cpp)
    participant SDK as mmind::eye::Camera（Mech-Eye SDK，闭源）
    participant Cli as ros2 service call（用户终端）
    participant Sub as 订阅方 / RViz
    L->>Main: Node(package=mecheye_ros_interface, executable=start, parameters=save_file/camera_ip/fx/fy/u/v)
    L->>L: 另起两个 tf2_ros static_transform_publisher（map → point_cloud 坐标系）
    Main->>Node: rclcpp::init；构造 MechMindCamera
    Node->>Node: declare_parameter/get_parameter；创建 /mechmind/* 七个 publisher
    Node->>SDK: findAndConnect(camera)（失败则抛 Camera not found.）
    Node->>SDK: getCameraInfo；getCameraIntrinsics（或使用 fx/fy/u/v）；setPointCloudUnit(Meter)
    Node->>Node: create_service 注册 capture_* / 用户集 / 各类参数读写服务
    Main->>Main: MultiThreadedExecutor.add_node(node) → spin()
    Cli->>Node: capture_point_cloud 请求
    Node->>SDK: capture3D(Frame3D)
    SDK-->>Node: frame.getUntexturedPointCloud()
    Node->>Node: publishPointCloud(pointCloud)
    Node-->>Sub: /mechmind/point_cloud (PointCloud2)
    Node->>Node: save_file 为 true 时 saveUntexturedPointCloud(PLY, /tmp/point_cloud.ply)
    Node-->>Cli: Response(error_code, error_description)
```

图依据：mecheye_ros2_interface@0123b8b，读过 README.md、src/main.cpp、src/MechMindCamera.cpp（构造函数、capture_depth_map_callback、capture_point_cloud_callback，其余回调仅看注册名）、launch/start_camera.py。其余 capture_* 与参数读写回调内部未读。

## 关联页面

- [国内具身开源全景技术地图](../overview/china-domestic-embodied-opensource-76-companies-technology-map.md)
- [HMI 开源项目主表导读](../queries/hmi-opensource-projects-coverage.md)
- [Humanoid Motion Intelligence](../entities/humanoid-motion-intelligence.md)

## 参考来源

- [mecheye_ros2_interface 源码归档](../../sources/repos/mecheye_ros2_interface.md)（<https://github.com/MechMindRobotics/mecheye_ros2_interface>）

- [国内具身智能开源全景（微信公众号）](../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md)
- 源码核对（2026-11-02）：MechMindRobotics/mecheye_ros2_interface@0123b8b，读过 README.md、src/main.cpp、src/MechMindCamera.cpp（构造函数与 capture_depth_map / capture_point_cloud 回调）、launch/start_camera.py；时序图依据以上文件，未运行代码。

## 推荐继续阅读

- [梅卡曼德 官方组织](https://github.com/MechMindRobotics)
