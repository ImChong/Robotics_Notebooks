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
  - ../../sources/repos/booster_robotics_sdk.md
  - ../../sources/repos/booster_robotics_sdk_ros2.md
summary: "加速进化 开源项目 booster_robotics_sdk（SDK/驱动）：官方 SDK：真机控制与状态读取的统一接入层…"
institutions:
  - booster
---

# booster_robotics_sdk

## 一句话定义

**booster_robotics_sdk** 是 [加速进化](https://github.com/BoosterRobotics) 公开的 **SDK/驱动** 开源项目：官方 SDK：真机控制与状态读取的统一接入层

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
| 机构 | 加速进化 |
| 类别 | SDK/驱动 |
| 官方组织 | https://github.com/BoosterRobotics |

## 工程实践

1. 从官方 GitHub/Gitee 组织检索 `booster_robotics_sdk` 仓库并核对 README 许可与依赖。
2. 对照本库 [424 项覆盖索引](../queries/china-domestic-opensource-424-coverage.md) 查看同公司其它入口是否共用训练/部署链路。
3. 若与既有方法页（如 RL 框架、VLA、SDK）主题相同，优先读关联页中的「开源入口」小节，避免重复维护平行叙事。

## 局限与风险

- 公众号清单为 **策展快照**（2026-09-06）；仓库更名、归档或许可证变化须回官方组织页核实。
- **开源状态**：以仓库 README 与 release 为准（入库日按文章描述归纳，未逐仓 clone 验证）。

## 源码运行时序图

入口：`cd build && cmake .. && make` 后运行示例，如 `./b1_loco_example_client 127.0.0.1`（高层运动 RPC）、`./b1_low_level_subscriber`、`./b1_low_level_publisher`（低层 DDS 话题）。本仓库提供头文件、示例源码与预编译静态库 `lib/<arch>/libbooster_robotics_sdk.a`；库内部实现（RPC 与 DDS 传输）为二进制，下图只画示例与头文件中可见的调用，库内部不展开。

```mermaid
sequenceDiagram
    participant EX1 as example/high_level/b1_loco_example_client.cpp
    participant CF as ChannelFactory (channel/channel_factory.hpp)
    participant LC as B1LocoClient (b1/b1_loco_client.hpp)
    participant SVC as 机器人端运动服务（rt/LocoApiTopic）
    participant EX2 as example/low_level/low_level_subscriber.cpp
    participant EX3 as example/low_level/low_level_publisher.cpp
    participant BUS as DDS 话题 rt/low_state、rt/joint_ctrl

    EX1->>CF: ChannelFactory::Instance()->Init(0, argv[1])
    EX1->>LC: client.Init() 绑定 rt/LocoApiTopic
    loop 读标准输入命令
        EX1->>LC: ChangeMode(kPrepare/kDamping/kWalking/kCustom)
        LC->>SVC: SendApiRequest(kChangeMode, JSON)
        SVC-->>LC: 状态码
        EX1->>LC: Move(vx, vy, vyaw) / RotateHead / ControlDexterousHand
        LC->>SVC: SendApiRequest(kMove 等, JSON)
        SVC-->>LC: 状态码（0 为成功）
    end
    Note over EX2,BUS: 低层：均先 ChannelFactory::Instance()->Init(0)
    EX2->>BUS: ChannelSubscriber<LowState>("rt/low_state", Handler).InitChannel()
    BUS-->>EX2: Handler 打印 IMU 与电机状态
    EX3->>BUS: ChannelPublisher<LowCmd>(kTopicJointCtrl).InitChannel()
    loop 周期发布
        EX3->>BUS: 填 MotorCmd（q/dq/tau/kp/kd/weight）后 Write(&low_cmd)
    end
```

图依据：BoosterRobotics/booster_robotics_sdk@87a9a26，读过 `README.md`、`CMakeLists.txt`、`example/high_level/b1_loco_example_client.cpp`、`example/low_level/{low_level_subscriber,low_level_publisher}.cpp`、`include/booster/robot/b1/{b1_loco_client,b1_api_const}.hpp`、`include/booster/robot/rpc/rpc_client.hpp`。预编译库内部（RPC 与 DDS 实现）无源码可读，未画；机器人端服务一侧同样不在本仓库内。

## 关联页面

- [国内具身开源全景技术地图](../overview/china-domestic-embodied-opensource-76-companies-technology-map.md)
- [HMI 开源项目主表导读](../queries/hmi-opensource-projects-coverage.md)
- [Humanoid Motion Intelligence](../entities/humanoid-motion-intelligence.md)

## 参考来源

- [booster_robotics_sdk_ros2 源码归档](../../sources/repos/booster_robotics_sdk_ros2.md)（<https://github.com/BoosterRobotics/booster_robotics_sdk_ros2>）

- [booster_robotics_sdk 源码归档](../../sources/repos/booster_robotics_sdk.md)（<https://github.com/BoosterRobotics/booster_robotics_sdk>）

- [国内具身智能开源全景（微信公众号）](../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md)
- 源码核对（2026-11-02）：BoosterRobotics/booster_robotics_sdk@87a9a26，读过 README.md、CMakeLists.txt、example/high_level/b1_loco_example_client.cpp、example/low_level/{low_level_subscriber,low_level_publisher}.cpp、include/booster/robot/b1/{b1_loco_client,b1_api_const}.hpp、include/booster/robot/rpc/rpc_client.hpp；时序图依据以上文件，预编译库 libbooster_robotics_sdk.a 内部未读，未运行代码。

## 推荐继续阅读

- [加速进化 官方组织](https://github.com/BoosterRobotics)
