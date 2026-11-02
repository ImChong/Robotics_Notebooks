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
  - ../../sources/repos/elite_robots_cs_sdk.md
summary: "艾利特机器人 开源项目 Elite_Robots_CS_SDK（SDK/驱动）：提供艾利特CS系列机器人C++访问与控制库，为ROS2驱动和自定义应用提供底层接口。…"
institutions:
  - elite-robot
---

# Elite_Robots_CS_SDK

## 一句话定义

**Elite_Robots_CS_SDK** 是 [艾利特机器人](https://github.com/Elite-Robots) 公开的 **SDK/驱动** 开源项目：提供艾利特CS系列机器人C++访问与控制库，为ROS2驱动和自定义应用提供底层接口。

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
| 机构 | 艾利特机器人 |
| 类别 | SDK/驱动 |
| 官方组织 | https://github.com/Elite-Robots |

## 工程实践

1. 从官方 GitHub/Gitee 组织检索 `Elite_Robots_CS_SDK` 仓库并核对 README 许可与依赖。
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

入口：`./servoj_example --robot-ip=<ip> --use-headless-mode=true`（`example/servoj_example.cpp`，由 `example/CMakeLists.txt` 构建；库本身无独立可执行入口）

```mermaid
sequenceDiagram
    participant M as servoj_example.cpp main()
    participant D as ELITE::EliteDriver (source/Elite/EliteDriver.cpp)
    participant P as PrimaryPortInterface (source/Primary)
    participant S as TcpServer 侧接口 ReverseInterface / TrajectoryInterface / ScriptCommandInterface
    participant RT as RtsiIOInterface (source/Rtsi)
    participant R as 艾利特 CS 控制器
    participant X as external_control.script (在机器人端运行)

    M->>D: EliteDriver(config) -> init()
    D->>P: connect(robot_ip, PRIMARY_PORT)，取 getLocalIP()
    D->>D: readScriptFile + scriptParamWrite（替换 IP/端口/servoj 参数占位符）
    D->>S: 创建 reverse / trajectory / script_command 三个 TCP 服务端
    alt headless_mode
        D->>P: sendExternalControlScript() -> sendScript(externalControl 脚本)
        P->>R: 经 primary 端口发送脚本
    else 非 headless
        D->>S: 创建 ScriptSender（等待机器人侧 External Control 插件取脚本）
    end
    M->>RT: RtsiIOInterface(output/input recipe).connect(robot_ip)
    M->>P: powerOn() / brakeRelease()
    R->>X: 运行脚本
    X->>S: socket_open 连接 reverse/trajectory/script_command 端口
    M->>D: isRobotConnected() 轮询直到为真
    loop 每 4ms，直到正反向转到 ±3 rad
        M->>RT: getActualJointPositions()
        M->>D: writeServoj(target_joint, 100, false)
        D->>S: ReverseInterface::writeJointCommand(MODE_SERVOJ)
        S->>X: reverse_socket 发送整数编码的关节目标
        X->>R: servoj(joints, t=SERVOJ_TIME, ...)
    end
    M->>D: stopControl()
```

图依据：Elite-Robots/Elite_Robots_CS_SDK@75bb7ea，读过 `example/servoj_example.cpp`、`source/Elite/EliteDriver.cpp`（init、writeServoj、sendExternalControlScript）、`source/resources/external_control.script`（socket_open、servoj 相关行）。`ReverseInterface.cpp`、`RtsiClient.cpp`、`PrimaryPortInterface.cpp` 内部实现未逐行读，相应消息按 EliteDriver.cpp 的调用关系画出。

## 参考来源

- [Elite_Robots_CS_SDK 源码归档](../../sources/repos/elite_robots_cs_sdk.md)（<https://github.com/Elite-Robots/Elite_Robots_CS_SDK>）

- [国内具身智能开源全景（微信公众号）](../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md)
- 源码核对（2026-11-02）：Elite-Robots/Elite_Robots_CS_SDK@75bb7ea，读过 example/servoj_example.cpp、source/Elite/EliteDriver.cpp、source/resources/external_control.script（部分行）；时序图依据以上文件，未运行代码。

## 推荐继续阅读

- [艾利特机器人 官方组织](https://github.com/Elite-Robots)
