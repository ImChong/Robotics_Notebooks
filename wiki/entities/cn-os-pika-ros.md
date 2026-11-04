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
  - ../../sources/repos/pika_ros.md
summary: "松灵机器人 开源项目 pika_ros（遥操作与数据采集）：遥操作与数据采集：人体/设备输入映射为机器人动作并记录示范数据…"
institutions:
  - agilex
---

# pika_ros

## 一句话定义

**pika_ros** 是 [松灵机器人](https://github.com/agilexrobotics) 公开的 **遥操作与数据采集** 开源项目：遥操作与数据采集：人体/设备输入映射为机器人动作并记录示范数据

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
| 机构 | 松灵机器人 |
| 类别 | 遥操作与数据采集 |
| 官方组织 | https://github.com/agilexrobotics |

## 工程实践

1. 从官方 GitHub/Gitee 组织检索 `pika_ros` 仓库并核对 README 许可与依赖。
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

入口：`bash scripts/start_single_sensor.bash`（脚本内以 `roslaunch sensor_tools open_single_sensor.launch` 启动传感端）。本图只覆盖传感端启动链路；仓库中 `src/data_tools`、`src/data_msgs`、`src/PikaAnyArm` 为未展开的 git 子模块，`pika_locator` 功能包不在仓库内，相关环节不展开。

```mermaid
sequenceDiagram
    autonumber
    participant U as 用户
    participant SH as scripts/start_single_sensor.bash
    participant RL as roslaunch sensor_tools open_single_sensor.launch
    participant GI as serial_gripper_imu（src/sensor_tools/src/serial_gripper_imu.cpp）
    participant CAM as camera_fisheye（src/sensor_tools/scripts/usb_camera.py）
    participant LOC as pika_locator launch（仓库外，未核对）
    participant RS as realsense2_camera rs_camera.launch
    participant ROS as ROS1 话题
    U->>SH: 运行启动脚本
    SH->>SH: sudo 写入 udev 规则，赋予 ttyUSB* / video* 权限
    SH->>RL: serial_port、fisheye_port、camera_fps 等参数
    RL->>GI: 启动节点 serial_gripper_imu（respawn）
    GI->>GI: NodeHandle("~")，读取 serial_port，以 460800 波特率打开串口
    GI->>ROS: advertise /imu/data、/gripper/data、/gripper/joint_state、/joint_state_gripper
    RL->>CAM: 启动节点 camera_fisheye（respawn）
    CAM->>CAM: init_camera：cv2.VideoCapture，按 camera_fps 循环取帧
    CAM->>ROS: 经 remap 发布 /camera_fisheye/color/image_raw 与 camera_info
    RL->>LOC: include pika_single_locator.launch
    RL->>RS: include rs_camera.launch（color/depth 帧率与分辨率同参数）
    RS->>ROS: RealSense 彩色 / 深度话题
```

图依据：agilexrobotics/pika_ros@1ec7788，读过 `README.md`、`scripts/start_single_sensor.bash`、`src/sensor_tools/launch/open_single_sensor.launch`、`src/sensor_tools/src/serial_gripper_imu.cpp`（部分）、`src/sensor_tools/scripts/usb_camera.py`（部分）、`.gitmodules`；未运行代码。

## 参考来源

- [pika_ros 源码归档](../../sources/repos/pika_ros.md)（<https://github.com/agilexrobotics/pika_ros>）

- [国内具身智能开源全景（微信公众号）](../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md)
- 源码核对（2026-11-02）：agilexrobotics/pika_ros@1ec7788，读过 `README.md`、`scripts/start_single_sensor.bash`、`src/sensor_tools/launch/open_single_sensor.launch`、`src/sensor_tools/src/serial_gripper_imu.cpp`（部分）、`src/sensor_tools/scripts/usb_camera.py`（部分）、`.gitmodules`；时序图依据以上文件，未运行代码。

## 推荐继续阅读

- [松灵机器人 官方组织](https://github.com/agilexrobotics)
