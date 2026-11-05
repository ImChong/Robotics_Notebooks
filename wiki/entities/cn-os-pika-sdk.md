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
  - ../../sources/repos/pika_sdk.md
summary: "松灵机器人 开源项目 pika_sdk（SDK/驱动）：官方 SDK：真机控制与状态读取的统一接入层…"
institutions:
  - agilex
---

# pika_sdk

## 一句话定义

**pika_sdk** 是 [松灵机器人](https://github.com/agilexrobotics) 公开的 **SDK/驱动** 开源项目：官方 SDK：真机控制与状态读取的统一接入层

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
| 类别 | SDK/驱动 |
| 官方组织 | https://github.com/agilexrobotics |

## 工程实践

1. 从官方 GitHub/Gitee 组织检索 `pika_sdk` 仓库并核对 README 许可与依赖。
2. 对照本库 [424 项覆盖索引](../queries/china-domestic-opensource-424-coverage.md) 查看同公司其它入口是否共用训练/部署链路。
3. 若与既有方法页（如 RL 框架、VLA、SDK）主题相同，优先读关联页中的「开源入口」小节，避免重复维护平行叙事。

## 局限与风险

- 公众号清单为 **策展快照**（2026-09-06）；仓库更名、归档或许可证变化须回官方组织页核实。
- **开源状态**：以仓库 README 与 release 为准（入库日按文章描述归纳，未逐仓 clone 验证）。

## 源码运行时序图

入口：`python examples/sense_control_gripper.py`（Sense 编码器遥操作 Gripper；需 `/dev/ttyUSB0`、`/dev/ttyUSB1` 两个串口设备）

```mermaid
sequenceDiagram
    participant Ex as examples/sense_control_gripper.py main()
    participant Se as pika/sense.py Sense
    participant GrSC as pika/serial_comm.py SerialComm（Sense 串口）
    participant Gr as pika/gripper.py Gripper
    participant GrSC2 as SerialComm（Gripper 串口）
    Ex->>Se: sense('/dev/ttyUSB0').connect()
    Se->>GrSC: connect() 打开串口 460800 8N1
    Se->>GrSC: start_reading_thread(callback=_data_callback)
    Note over GrSC: 后台线程 _reading_thread_func：read_data → _find_json 括号配对解析
    GrSC-->>Se: _data_callback(json)，写入 encoder_data（AS5047 的 angle、rad）
    Ex->>Gr: Gripper('/dev/ttyUSB1').connect()
    Gr->>GrSC2: connect() 并启动读线程（电机状态回调）
    Ex->>Gr: enable()
    Gr->>GrSC2: send_command(CommandType.ENABLE)
    loop while True，每 0.03 s
        Ex->>Se: get_encoder_data() / get_gripper_distance()
        Se-->>Ex: encoder_data['rad']、开口距离 mm
        Ex->>Gr: get_motor_data()
        Gr-->>Ex: 电机状态（含 Current）
        Ex->>Gr: set_motor_angle(encoder_data['rad'])
        Gr->>GrSC2: send_command(POSITION_CTRL, rad)，rad 小于 0 时截为 0
    end
```

图依据：agilexrobotics/pika_sdk@902b476，读过 README.md、examples/sense_control_gripper.py、pika/__init__.py、pika/sense.py、pika/serial_comm.py、pika/gripper.py（enable、set_motor_angle 部分）。

## 关联页面

- [国内具身开源全景技术地图](../overview/china-domestic-embodied-opensource-76-companies-technology-map.md)
- [HMI 开源项目主表导读](../queries/hmi-opensource-projects-coverage.md)
- [Humanoid Motion Intelligence](../entities/humanoid-motion-intelligence.md)

## 参考来源

- [pika_sdk 源码归档](../../sources/repos/pika_sdk.md)（<https://github.com/agilexrobotics/pika_sdk>）

- [国内具身智能开源全景（微信公众号）](../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md)
- 源码核对（2026-11-02）：agilexrobotics/pika_sdk@902b476，读过 README.md、examples/sense_control_gripper.py、pika/__init__.py、pika/sense.py、pika/serial_comm.py、pika/gripper.py（部分）；时序图依据以上文件，未运行代码。

## 推荐继续阅读

- [松灵机器人 官方组织](https://github.com/agilexrobotics)
