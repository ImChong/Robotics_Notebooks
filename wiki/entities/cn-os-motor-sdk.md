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
  - ../../sources/repos/motor-sdk.md
summary: "达妙科技 开源项目 motor-sdk（SDK/驱动）：提供达妙电机设备控制与通信示例，服务关节执行器调试和机器人底层接入。…"
institutions:
  - dmbots
---

# motor-sdk

## 一句话定义

**motor-sdk** 是 [达妙科技](https://github.com/dmBots) 公开的 **SDK/驱动** 开源项目：提供达妙电机设备控制与通信示例，服务关节执行器调试和机器人底层接入。

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
| 机构 | 达妙科技 |
| 类别 | SDK/驱动 |
| 官方组织 | https://github.com/dmBots |

## 工程实践

1. 从官方 GitHub/Gitee 组织检索 `motor-sdk` 仓库并核对 README 许可与依赖。
2. 对照本库 [424 项覆盖索引](../queries/china-domestic-opensource-424-coverage.md) 查看同公司其它入口是否共用训练/部署链路。
3. 若与既有方法页（如 RL 框架、VLA、SDK）主题相同，优先读关联页中的「开源入口」小节，避免重复维护平行叙事。

## 局限与风险

- 公众号清单为 **策展快照**（2026-09-06）；仓库更名、归档或许可证变化须回官方组织页核实。
- **开源状态**：以仓库 README 与 release 为准（入库日按文章描述归纳，未逐仓 clone 验证）。

## 源码运行时序图

入口：`python DM_Motor_Test.py`（在 `Python例程/u2can/` 目录下运行，串口示例为 `/dev/ttyACM0`，波特率 921600）

仓库还含 C#、C++（u2can / u2canfd）、Matlab 等多套例程，本图只取 Python u2can 路线，其余路线未画。

```mermaid
sequenceDiagram
    participant U as 用户终端
    participant T as DM_Motor_Test.py 主程序
    participant O as Motor 对象
    participant S as 串口 serial
    participant C as MotorControl（DM_CAN.py）
    U->>T: python DM_Motor_Test.py
    T->>O: Motor(DM_Motor_Type.DM4310, SlaveID 0x01, MasterID 0x11)
    T->>S: serial.Serial('/dev/ttyACM0', 921600)
    T->>C: MotorControl(serial_device)，构造时打开串口
    T->>C: addMotor(Motor1)，按 SlaveID 与 MasterID 登记
    T->>C: switchControlMode(Motor1, POS_VEL)
    C->>S: __write_motor_param 写 RID 10，帧 ID 0x7FF
    loop 轮询最多 10 次
        C->>S: recv_set_param_data() 读回包
        C->>O: 写入 temp_param_dict[RID 10]
    end
    T->>C: read_motor_param 读取 sub_ver Gr PMAX MST_ID VMAX TMAX
    C->>S: __read_RID_param 发读取帧，轮询 temp_param_dict
    T->>C: save_motor_param(Motor1)
    C->>C: disable(Motor1) 先失能
    C->>S: 发保存帧 0xAA，帧 ID 0x7FF
    T->>C: enable(Motor1)
    C->>S: __control_cmd 发使能 0xFC
    loop 10000 次，每次 sleep 1 ms
        T->>C: control_Pos_Vel(Motor1, sin(t) * 8, 30)
        C->>S: __send_data，帧 ID 为 0x100 加 SlaveID
        C->>S: recv() 读串口
        C->>O: __process_packet（CMD 0x11）调用 recv_data(q, dq, tau, err)
        T->>O: getPosition / getVelocity / getTorque / getError 打印状态
    end
    T->>S: serial_device.close()
```

图依据：dmBots/motor-sdk@0b2ede4，读过 README.md、Python例程/README.md、Python例程/u2can/README.md、Python例程/u2can/USAGE.md、Python例程/u2can/DM_Motor_Test.py、Python例程/u2can/DM_CAN.py（MotorControl 的 __init__、control_Pos_Vel、enable、recv、__process_packet、addMotor、__send_data、switchControlMode、read_motor_param、save_motor_param 等方法）。

## 关联页面

- [国内具身开源全景技术地图](../overview/china-domestic-embodied-opensource-76-companies-technology-map.md)
- [HMI 开源项目主表导读](../queries/hmi-opensource-projects-coverage.md)
- [Humanoid Motion Intelligence](../entities/humanoid-motion-intelligence.md)

## 参考来源

- [motor-sdk 源码归档](../../sources/repos/motor-sdk.md)（<https://github.com/dmBots/motor-sdk>）

- [国内具身智能开源全景（微信公众号）](../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md)

- 源码核对（2026-11-02）：dmBots/motor-sdk@0b2ede4，读过 README.md、Python例程/README.md、Python例程/u2can/README.md、Python例程/u2can/USAGE.md、Python例程/u2can/DM_Motor_Test.py、Python例程/u2can/DM_CAN.py；时序图依据以上文件，未运行代码。

## 推荐继续阅读

- [达妙科技 官方组织](https://github.com/dmBots)
