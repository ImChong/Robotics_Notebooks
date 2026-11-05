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
  - ../../sources/repos/electronic-skin-ml.md
summary: "玄雅科技 开源项目 Electronic-Skin-ML（SDK/驱动）：ML 电子皮肤 SDK…"
institutions:
  - xuanya
---

# Electronic-Skin-ML

## 一句话定义

**Electronic-Skin-ML** 是 [玄雅科技](https://github.com/Synria-Robotics) 公开的 **SDK/驱动** 开源项目：ML 电子皮肤 SDK

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
| 机构 | 玄雅科技 |
| 类别 | SDK/驱动 |
| 官方组织 | https://github.com/Synria-Robotics |

## 工程实践

1. 从官方 GitHub/Gitee 组织检索 `Electronic-Skin-ML` 仓库并核对 README 许可与依赖。
2. 对照本库 [424 项覆盖索引](../queries/china-domestic-opensource-424-coverage.md) 查看同公司其它入口是否共用训练/部署链路。
3. 若与既有方法页（如 RL 框架、VLA、SDK）主题相同，优先读关联页中的「开源入口」小节，避免重复维护平行叙事。

## 局限与风险

- 公众号清单为 **策展快照**（2026-09-06）；仓库更名、归档或许可证变化须回官方组织页核实。
- **开源状态**：以仓库 README 与 release 为准（入库日按文章描述归纳，未逐仓 clone 验证）。

## 源码运行时序图

入口：`python examples/04_demo_read_pressure.py`（需先把脚本里的 `PORT`、`SLAVE_ADDRESS` 改成实际值；SDK 门面为 `tactile_sdk.TactilePressureSDK`）。

```mermaid
sequenceDiagram
    participant EX as examples/04_demo_read_pressure.py
    participant SDK as TactilePressureSDK（tactile_sdk/client.py）
    participant CF as ConfigAPI（api/config_api.py）
    participant PA as PressureAPI（api/pressure_api.py）
    participant MB as ModbusRTU（protocol/modbus_rtu.py）
    participant TR as SerialTransport（transport/serial_transport.py）
    participant HW as 60 点触觉压力传感器（Modbus RTU 从站）
    EX->>SDK: with TactilePressureSDK(port, slave_address) as sdk
    SDK->>TR: SerialTransport(port, baudrate, timeout)
    SDK->>MB: ModbusRTU(transport)；创建 device / config / pressure / calibration 四个 API 对象
    SDK->>TR: __enter__ → connect() → open() 打开串口
    EX->>CF: get_pressure_point_count()
    EX->>CF: set_pressure_value_type(1)（1=标定值 mN）
    loop 目标 200 Hz 定时循环
        EX->>PA: read_fast()
        PA->>MB: read_all_pressure_values_fast(slave_address)
        MB->>TR: reset_input_buffer()，write(_build_frame(功能码 READ_ALL_PRESSURE_VALUES 0x42))
        TR->>HW: Modbus RTU 请求帧
        HW-->>MB: _read_response_fast_0x42 → CRC16 校验 → 小端 uint16 解析
        MB-->>PA: 压力值列表（失败返回 None，不抛异常）
        PA-->>EX: 若做过 trigger_dynamic_zero 则扣除 _zero_offsets
    end
    EX->>SDK: 退出 with → __exit__ → disconnect()
```

图依据：Synria-Robotics/Electronic-Skin-ML@e9d536f，读过 README.md（开头）、examples/04_demo_read_pressure.py、tactile_sdk/client.py、tactile_sdk/api/pressure_api.py、tactile_sdk/protocol/modbus_rtu.py（read_all_pressure_values_fast 等部分）；config_api.py、serial_transport.py 仅读函数签名，未读其实现。

## 关联页面

- [国内具身开源全景技术地图](../overview/china-domestic-embodied-opensource-76-companies-technology-map.md)
- [HMI 开源项目主表导读](../queries/hmi-opensource-projects-coverage.md)
- [Humanoid Motion Intelligence](../entities/humanoid-motion-intelligence.md)

## 参考来源

- [Electronic-Skin-ML 源码归档](../../sources/repos/electronic-skin-ml.md)（<https://github.com/Synria-Robotics/Electronic-Skin-ML>）

- [国内具身智能开源全景（微信公众号）](../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md)
- 源码核对（2026-11-02）：Synria-Robotics/Electronic-Skin-ML@e9d536f，读过 README.md（开头）、examples/04_demo_read_pressure.py、tactile_sdk/client.py、tactile_sdk/api/pressure_api.py、tactile_sdk/protocol/modbus_rtu.py（部分）；config_api.py、serial_transport.py 仅读签名；时序图依据以上文件，未运行代码。

## 推荐继续阅读

- [玄雅科技 官方组织](https://github.com/Synria-Robotics)
