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
  - ../../sources/repos/xense-ros.md
summary: "千觉机器人 开源项目 xense-ros（SDK/驱动）：Xense 触觉设备 ROS 集成与示例…"
institutions:
  - qianjue
---

# xense-ros

## 一句话定义

**xense-ros** 是 [千觉机器人](https://github.com/XenseRobotics-AI) 公开的 **SDK/驱动** 开源项目：Xense 触觉设备 ROS 集成与示例

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| SDK | Software Development Kit | 真机控制与状态读取接口 |
| RL | Reinforcement Learning | 强化学习训练与策略优化 |
| VLA | Vision-Language-Action | 视觉–语言–动作统一策略 |
| Sim2Real | Simulation to Real | 仿真策略迁移真机 |
| URDF | Unified Robot Description Format | 机器人描述与仿真资产 |

## 为什么重要

- 收录于 [国内具身智能开源全景（76 家 · 424 项）](../overview/china-domestic-embodied-opensource-76-companies-technology-map.md) 的 **第三层** 分组。
- 与 [Humanoid Motion Intelligence](../entities/humanoid-motion-intelligence.md) 同源策展；本页为 **独立详情节点**，便于从公司清单跳到机制与入口说明。

## 核心原理

| 字段 | 内容 |
|------|------|
| 机构 | 千觉机器人 |
| 类别 | SDK/驱动 |
| 官方组织 | https://github.com/XenseRobotics-AI、https://github.com/XenseRobotics |

## 工程实践

1. 从官方 GitHub/Gitee 组织检索 `xense-ros` 仓库并核对 README 许可与依赖。
2. 对照本库 [424 项覆盖索引](../queries/china-domestic-opensource-424-coverage.md) 查看同公司其它入口是否共用训练/部署链路。
3. 若与既有方法页（如 RL 框架、VLA、SDK）主题相同，优先读关联页中的「开源入口」小节，避免重复维护平行叙事。

## 局限与风险

- 公众号清单为 **策展快照**（2026-09-06）；仓库更名、归档或许可证变化须回官方组织页核实。
- **开源状态**：以仓库 README 与 release 为准（入库日按文章描述归纳，未逐仓 clone 验证）。

## 源码运行时序图

入口（ROS1）：`roslaunch xense_ros xense_publisher.launch serial_number:=<SN> rectify_width:=200 rectify_height:=350 publish_rate:=30.0`（ROS2 入口 `ros2 launch xense_ros2 xense_publisher_launch.py ...` 对应 `xense_publisher_ros2.py` 的 `XensePublisher` 节点，结构相同但用 timer 回调代替 `rospy.Rate` 循环，此处未单独成图）。

```mermaid
sequenceDiagram
    autonumber
    participant RL as xense_publisher.launch roslaunch
    participant N as xense_publisher.py run()
    participant CB as cv_bridge CvBridge
    participant SDK as xensesdk Sensor
    participant T as ROS 话题 /xense/SN/

    RL->>N: 启动节点 xense_publisher，注入 ~serial_number 等参数
    N->>N: rospy.init_node，rospy.get_param 读取参数
    N->>N: 校验 serial_number、rectify 尺寸、Sensor 与 CvBridge 是否可用
    N->>T: 创建 force、force_resultant、rectify、depth、marker2d、status 发布者
    N->>SDK: Sensor.create(serial, rectify_size)
    alt 传感器初始化失败
        N->>T: status 发布 error 信息
        N->>N: rospy.signal_shutdown 后返回
    else 初始化成功
        loop rospy.Rate(publish_rate)，直到 rospy.is_shutdown
            N->>SDK: selectSensorInfo(Force, ForceResultant, Rectify, Depth, Marker2D)
            SDK-->>N: force, res_force, src, depth, marker2d
            N->>T: force 与 marker2d 以 Float32MultiArray 发布
            N->>T: force_resultant 以 WrenchStamped 发布（前三个为力，后三个为力矩）
            N->>CB: cv2_to_imgmsg(src, bgr8) 与 cv2_to_imgmsg(depth, 32FC1)
            CB-->>N: sensor_msgs/Image
            N->>T: rectify 与 depth 图像发布
            N->>T: status 发布 ok
        end
        N->>SDK: sensor.release()
    end
```

图依据：xense-ros@c8395f5，实际读过 README.md、src/xense_ros/launch/xense_publisher.launch、src/xense_ros/scripts/xense_publisher.py（全文）；另读过 src/xense_ros2/xense_ros2/xense_publisher_ros2.py 与 launch 文件，仅用于对比 ROS2 入口，未出图。xensesdk 与 cv_bridge 为外部依赖，未在本仓库内读到实现，SDK 调用只按本仓库调用点画。

## 关联页面

- [国内具身开源全景技术地图](../overview/china-domestic-embodied-opensource-76-companies-technology-map.md)
- [HMI 开源项目主表导读](../queries/hmi-opensource-projects-coverage.md)
- [Humanoid Motion Intelligence](../entities/humanoid-motion-intelligence.md)

## 参考来源

- [xense-ros 源码归档](../../sources/repos/xense-ros.md)（<https://github.com/XenseRobotics/xense-ros>）

- [国内具身智能开源全景（微信公众号）](../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md)

- 源码核对（2026-11-02）：XenseRobotics/xense-ros@c8395f5，读过 README.md、src/xense_ros/launch/xense_publisher.launch、src/xense_ros/scripts/xense_publisher.py、src/xense_ros2/xense_ros2/xense_publisher_ros2.py、src/xense_ros2/launch/xense_publisher_launch.py、src/xense_ros2/setup.py；时序图依据以上文件，未运行代码。

## 推荐继续阅读

- [千觉机器人 官方组织](https://github.com/XenseRobotics-AI)
