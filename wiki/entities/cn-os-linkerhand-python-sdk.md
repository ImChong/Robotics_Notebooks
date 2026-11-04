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
  - ../../sources/repos/linkerhand-python-sdk.md
summary: "灵心巧手 开源项目 linkerhand-python-sdk（SDK/驱动）：提供LinkerHand的Python控制接口，用于关节命令、状态读取和机器人应用集成。…"
institutions:
  - linkerbot
---

# linkerhand-python-sdk

## 一句话定义

**linkerhand-python-sdk** 是 [灵心巧手](https://github.com/linker-bot) 公开的 **SDK/驱动** 开源项目：提供LinkerHand的Python控制接口，用于关节命令、状态读取和机器人应用集成。

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
| 机构 | 灵心巧手 |
| 类别 | SDK/驱动 |
| 官方组织 | https://github.com/linker-bot |

## 工程实践

1. 从官方 GitHub/Gitee 组织检索 `linkerhand-python-sdk` 仓库并核对 README 许可与依赖。
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

入口：`python3 gui_control.py`（在 `examples/gui_control/` 目录运行；先按 README `pip3 install -r requirements.txt`，并在 `LinkerHand/config/setting.yaml` 中设置 `CAN: "can0"`、`MODBUS: "None"`、`JOINT` 与手型号，再执行 `sudo ip link set can0 up type can bitrate 1000000`）。

```mermaid
sequenceDiagram
    participant G as gui_control.py: HandControlGUI
    participant AM as gui_control.py: HandApiManager
    participant API as LinkerHand/linker_hand_api.py: LinkerHandApi
    participant L10 as core/can/linker_hand_l10_can.py: LinkerHandL10Can
    participant OC as utils/open_can.py: OpenCan
    participant CAN as can0 (python-can socketcan)

    G->>AM: 启动: HandApiManager.init_api()
    AM->>API: LinkerHandApi(hand_joint, hand_type, modbus, can)
    API->>L10: LinkerHandL10Can(can_id=0x28 或 0x27, can_channel, yaml)
    L10->>CAN: init_can_bus: can.interface.Bus(socketcan, bitrate)
    API->>OC: open_can(can0) 与 is_can_up_sysfs (Linux 且 MODBUS 为 None)
    API->>API: get_embedded_version, get_serial_number
    loop 每 30 ms (publish_timer)
        G->>AM: publish_joint_state(滑块值列表)
        AM->>API: finger_move(pose), 校验 0 到 255
        API->>L10: set_joint_positions(pose)
        L10->>CAN: send_frame(JOINT_POSITION2_RCO, pose[6:])
        L10->>CAN: send_frame(JOINT_POSITION_RCO, pose[:6])
    end
    loop 每 500 ms (matrix_timer)
        AM->>API: get_thumb / index / middle / ring / little _matrix_touch
        API->>L10: 触觉矩阵读取 (收发细节未逐行核对)
        AM-->>G: matrix_data_updated 信号, 刷新矩阵显示
    end
    alt CanError 发送失败
        L10->>OC: open_can(channel), is_can_up_sysfs
        L10->>CAN: 重建 can.interface.Bus
    end
```

图依据：linker-bot/linkerhand-python-sdk@0cc0585，`README.md`、`LinkerHand/linker_hand_api.py`、`LinkerHand/core/can/linker_hand_l10_can.py`（`__init__`、`init_can_bus`、`send_frame`、`set_joint_positions`）、`LinkerHand/utils/open_can.py`（节选）、`LinkerHand/config/setting.yaml`、`example/gui_control/gui_control.py`（节选）。

## 参考来源

- [linkerhand-python-sdk 源码归档](../../sources/repos/linkerhand-python-sdk.md)（<https://github.com/linker-bot/linkerhand-python-sdk>）

- [国内具身智能开源全景（微信公众号）](../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md)

## 推荐继续阅读

- [灵心巧手 官方组织](https://github.com/linker-bot)
