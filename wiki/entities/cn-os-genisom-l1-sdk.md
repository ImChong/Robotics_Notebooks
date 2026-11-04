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
  - ../../sources/repos/genisom_l1_sdk.md
summary: "智身科技 开源项目 genisom_L1_sdk（SDK/驱动）：官方 SDK：真机控制与状态读取的统一接入层…"
institutions:
  - zsibot
---

# genisom_L1_sdk

## 一句话定义

**genisom_L1_sdk** 是 [智身科技](https://github.com/zsibot) 公开的 **SDK/驱动** 开源项目：官方 SDK：真机控制与状态读取的统一接入层

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

1. 从官方 GitHub/Gitee 组织检索 `genisom_L1_sdk` 仓库并核对 README 许可与依赖。
2. 对照本库 [424 项覆盖索引](../queries/china-domestic-opensource-424-coverage.md) 查看同公司其它入口是否共用训练/部署链路。
3. 若与既有方法页（如 RL 框架、VLA、SDK）主题相同，优先读关联页中的「开源入口」小节，避免重复维护平行叙事。

## 局限与风险

- 公众号清单为 **策展快照**（2026-09-06）；仓库更名、归档或许可证变化须回官方组织页核实。
- **开源状态**：以仓库 README 与 release 为准（入库日按文章描述归纳，未逐仓 clone 验证）。

## 源码运行时序图

入口：`./build/sdk_control`（`example/sdk_control.cpp` 的 `main()`，SDK 模式；遥控模式对应 `remote_control.cpp`）。SDK 核心实现以预编译静态库 `lib/<arch>/libzsibot.a` 提供，仓库内无其源码。

```mermaid
sequenceDiagram
    participant U as main()（example/sdk_control.cpp）
    participant E as ZsibotExecutor（include/zsibot_sdk/zsibot_api.h）
    participant L as libzsibot.a（闭源预编译实现）
    participant R as 机器狗（UDP 192.168.234.1）
    U->>E: ZsibotExecutor(Role::ROLE_SDK)，默认 send 8081 / recv 8080
    E->>L: 构造 m_impl（Pimpl）
    loop 连接维持（protocol.md：每 50ms）
        L->>R: heartbeat JSON
    end
    U->>E: SetCmd(CMD_SDK_CONTROL_RIGHT)（先申请控制权）
    E->>L: 命令 {"type":"cmd","cmd":...}
    L->>R: UDP 发送
    loop 键盘循环 while(true) + kbhit()
        U->>E: SetRemote(joystick[4], button[14])（如 w/s/a/d）
        E->>L: {"type":"remote", joystick, button}
        L->>R: UDP 发送遥控量
    end
    R-->>L: dog_state / feedback / imu_info / leg_joint_info 等 UDP 回包
    U->>E: GetPower() / GetBatteryInfo() / GetRPY() / GetLegHipJoint()
    E-->>U: 返回 SDK 已接收的状态值
    U->>E: SetCmd(CMD_EMERGENCY_STOP)（按键 0，软急停）
```

图依据：genisom_L1_sdk@f7ccbf3，读过 README.md、example/sdk_control.cpp、example/remote_control.cpp、include/zsibot_sdk/zsibot_api.h、docs/protocol/protocol.md、example/CMakeLists.txt、CMakeLists.txt（头文件与示例为源码；SDK 内部收发线程实现未开源，仅知 protocol.md 描述的消息格式与线程划分）。

## 关联页面

- [国内具身开源全景技术地图](../overview/china-domestic-embodied-opensource-76-companies-technology-map.md)
- [HMI 开源项目主表导读](../queries/hmi-opensource-projects-coverage.md)
- [Humanoid Motion Intelligence](../entities/humanoid-motion-intelligence.md)

## 参考来源

- [genisom_L1_sdk 源码归档](../../sources/repos/genisom_l1_sdk.md)（<https://github.com/zsibot/genisom_L1_sdk>）

- [国内具身智能开源全景（微信公众号）](../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md)

- 源码核对（2026-11-02）：zsibot/genisom_L1_sdk@f7ccbf3，读过 README.md、example/sdk_control.cpp、example/remote_control.cpp、include/zsibot_sdk/zsibot_api.h、docs/protocol/protocol.md、CMakeLists.txt、example/CMakeLists.txt；时序图依据以上文件，未运行代码。

## 推荐继续阅读

- [智身科技 官方组织](https://github.com/zsibot)
