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
  - ../../sources/repos/limxsdk-lowlevel.md
summary: "逐际动力 开源项目 limxsdk-lowlevel（SDK/驱动）：官方 SDK：真机控制与状态读取的统一接入层…"
institutions:
  - limx
---

# limxsdk-lowlevel

## 一句话定义

**limxsdk-lowlevel** 是 [逐际动力](https://github.com/limxdynamics) 公开的 **SDK/驱动** 开源项目：官方 SDK：真机控制与状态读取的统一接入层

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
| 机构 | 逐际动力 |
| 类别 | SDK/驱动 |
| 官方组织 | https://github.com/limxdynamics、https://github.com/FluxVLA |

## 工程实践

1. 从官方 GitHub/Gitee 组织检索 `limxsdk-lowlevel` 仓库并核对 README 许可与依赖。
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

入口：`python3 python3/examples/api/example_humanoid.py [robot_ip]`（先按 README 执行 `pip install python3/amd64/limxsdk-*.whl`；不传参数时连接 `127.0.0.1`，即本机仿真器）。开源仓库只含示例与预编译 SDK（`lib/`、`python3/*.whl`），SDK 源码不在仓库内，因此下图只画示例脚本与 SDK 公开接口之间的调用与数据流，SDK 内部的传输实现未画。

```mermaid
sequenceDiagram
    participant E as example_humanoid.py (__main__)
    participant R as limxsdk Robot(RobotType.Humanoid) (闭源 wheel)
    participant X as 仿真器 / 真机 (init 的 IP)
    participant CB as RobotReceiver 回调

    E->>R: Robot(RobotType.Humanoid), init(robot_ip)
    R-->>X: 建立连接 (127.0.0.1 或机器人 IP)
    E->>R: getMotorNumber(), getMotorNames()
    E->>R: subscribeImuData / subscribeRobotState / subscribeSensorJoy / subscribeDiagnosticValue
    loop 1000 Hz (Rate(1000) + rate.sleep)
        E->>E: 构造 RobotCmd (mode, q, dq, tau, Kp, Kd, motor_names, stamp)
        E->>R: publishRobotCmd(cmd_msg)
        R-->>X: RobotCmd (逐关节 q / dq / tau / Kp / Kd)
    end
    Note over X,CB: 异步: 机器人或仿真器推送状态, SDK 调用已注册回调
    X-->>R: RobotState, ImuData, SensorJoy, DiagnosticValue
    R-->>CB: imuDataCallback / robotStateCallback / sensorJoyCallback / diagnosticValueCallback
```

图依据：limxdynamics/limxsdk-lowlevel@02ed0e3，`README.md`、`python3/examples/api/example_humanoid.py`（另读 `examples/test_humanoid.cpp` 作为 C++ 接口的对照，它是 gtest 用例而非运行入口）。

## 参考来源

- [limxsdk-lowlevel 源码归档](../../sources/repos/limxsdk-lowlevel.md)（<https://github.com/limxdynamics/limxsdk-lowlevel>）

- [国内具身智能开源全景（微信公众号）](../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md)

## 推荐继续阅读

- [逐际动力 官方组织](https://github.com/limxdynamics)
