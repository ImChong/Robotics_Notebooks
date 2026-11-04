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
  - ../../sources/repos/genisom_robot_sdk.md
summary: "智身科技 开源项目 genisom_robot_sdk（SDK/驱动）：官方 SDK：真机控制与状态读取的统一接入层…"
institutions:
  - zsibot
---

# genisom_robot_sdk

## 一句话定义

**genisom_robot_sdk** 是 [智身科技](https://github.com/zsibot) 公开的 **SDK/驱动** 开源项目：官方 SDK：真机控制与状态读取的统一接入层

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
| 机构 | 智身科技 |
| 类别 | SDK/驱动 |
| 官方组织 | https://github.com/zsibot |

## 工程实践

1. 从官方 GitHub/Gitee 组织检索 `genisom_robot_sdk` 仓库并核对 README 许可与依赖。
2. 对照本库 [424 项覆盖索引](../queries/china-domestic-opensource-424-coverage.md) 查看同公司其它入口是否共用训练/部署链路。
3. 若与既有方法页（如 RL 框架、VLA、SDK）主题相同，优先读关联页中的「开源入口」小节，避免重复维护平行叙事。

## 局限与风险

- 公众号清单为 **策展快照**（2026-09-06）；仓库更名、归档或许可证变化须回官方组织页核实。
- **开源状态**：以仓库 README 与 release 为准（入库日按文章描述归纳，未逐仓 clone 验证）。

## 源码运行时序图

入口：`./quick_start 192.168.234.1 8082`（`example/quick_start.cpp` 的 `main()`，只读示例，不改变机器人状态；交互控制入口为 `./control <ip> <port>`）。SDK 核心以预编译共享库 `lib/<arch>/librobot_sdk.so` 提供，仓库内无其源码。

```mermaid
sequenceDiagram
    participant M as main()（example/quick_start.cpp）
    participant C as SDKClient（include/robot_sdk/sdk_client.hpp）
    participant S as librobot_sdk.so（闭源预编译实现）
    participant R as 运动控制主机（UDP 8082）
    participant D as DataCallback（应用侧回调）
    M->>C: SDKClient(错误回调)
    M->>C: SetDataCallback(data_callback)
    M->>C: Connect(ip, port, true)
    C->>S: 建立连接（默认 UDP）
    S->>R: 连接请求
    R-->>S: 状态与设备信息
    M->>C: GetDeviceInfo() / Version() / ProtocolVersion() / SystemVersion()
    C-->>M: 打印 SDK 与机器人版本、序列号
    loop 最多 5 秒，每 100ms 检查（ReceivedState）
        R-->>S: 周期性状态数据（约 1 Hz）
        S-->>D: OnRobotStateData(RobotState)（SDK I/O 线程）
        D-->>M: 置位 received_state_ 标志
    end
    M->>C: Disconnect(true)
    C->>S: 断开
    M-->>M: 返回 EXIT_SUCCESS / EXIT_FAILURE
```

图依据：genisom_robot_sdk@e8f959a，读过 README.md、example/quick_start.cpp、include/robot_sdk/sdk_client.hpp（声明）、docs/en/sdk_network_en.md、docs/en/sdk_callback_en.md；回调线程与频率取自 sdk_callback_en.md，SDK 内部收发实现未开源，未运行代码。

## 关联页面

- [国内具身开源全景技术地图](../overview/china-domestic-embodied-opensource-76-companies-technology-map.md)
- [HMI 开源项目主表导读](../queries/hmi-opensource-projects-coverage.md)
- [Humanoid Motion Intelligence](../entities/humanoid-motion-intelligence.md)

## 参考来源

- [genisom_robot_sdk 源码归档](../../sources/repos/genisom_robot_sdk.md)（<https://github.com/zsibot/genisom_robot_sdk>）

- [国内具身智能开源全景（微信公众号）](../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md)

- 源码核对（2026-11-02）：zsibot/genisom_robot_sdk@e8f959a，读过 README.md、example/quick_start.cpp、include/robot_sdk/sdk_client.hpp、docs/en/sdk_network_en.md、docs/en/sdk_callback_en.md；时序图依据以上文件，未运行代码。

## 推荐继续阅读

- [智身科技 官方组织](https://github.com/zsibot)
