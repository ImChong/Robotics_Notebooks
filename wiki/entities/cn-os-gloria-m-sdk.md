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
  - ../../sources/repos/gloria-m-sdk.md
summary: "玄雅科技 开源项目 Gloria-M-SDK（SDK/驱动）：云犀夹爪 Python SDK…"
institutions:
  - xuanya
---

# Gloria-M-SDK

## 一句话定义

**Gloria-M-SDK** 是 [玄雅科技](https://github.com/Synria-Robotics) 公开的 **SDK/驱动** 开源项目：云犀夹爪 Python SDK

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

1. 从官方 GitHub/Gitee 组织检索 `Gloria-M-SDK` 仓库并核对 README 许可与依赖。
2. 对照本库 [424 项覆盖索引](../queries/china-domestic-opensource-424-coverage.md) 查看同公司其它入口是否共用训练/部署链路。
3. 若与既有方法页（如 RL 框架、VLA、SDK）主题相同，优先读关联页中的「开源入口」小节，避免重复维护平行叙事。

## 局限与风险

- 公众号清单为 **策展快照**（2026-09-06）；仓库更名、归档或许可证变化须回官方组织页核实。
- **开源状态**：以仓库 README 与 release 为准（入库日按文章描述归纳，未逐仓 clone 验证）。

## 源码运行时序图

入口：`python demos/02_pv_control.py [--port auto] [--id 0x01] [--close-q ...] [--close-vel ...]`（PV 模式夹爪柔和闭合示例；SDK 门面为 `gloria_m_sdk.GloriaGripper`）。

```mermaid
sequenceDiagram
    participant D as demos/02_pv_control.py
    participant G as GloriaGripper（client.py）
    participant MO as MotorAPI / MotionAPI（api/）
    participant C as CanController（controller.py）
    participant A as SerialCanAdapter（serial_can_adapter.py）
    participant M as 电机固件（串口转 CAN）
    D->>G: with GloriaGripper(port, ...) as g
    G->>G: __enter__ → connect()
    G->>A: SerialCanAdapter(port, baudrate=921600)
    G->>C: CanController(adapter)，register(Actuator)
    G->>C: apply_limits_and_save(act, limits)
    C->>A: send(参数写入帧)
    D->>MO: g.motor.set_mode(ControlMode.POS_VEL)
    MO->>C: set_control_mode(act, mode)（带重试）
    D->>MO: g.motor.enable()
    MO->>C: enable(act)
    C->>A: send(使能帧)
    loop 每个控制周期（先开口，再慢速闭合）
        D->>MO: g.motion.send_pos_vel(position, velocity, poll=True)
        MO->>C: send_pos_vel(act, position, velocity)
        C->>A: send(0x100+command_id, pack_f32(位置)+pack_f32(速度))
        A->>M: CAN 帧（PV 指令）
        C->>A: poll() → read_packets()
        M-->>C: 反馈帧 → _handle_packet → unpack_mit_feedback
        C-->>D: 更新 Actuator 状态，g.state.position
        D->>D: 位置误差 < SETTLE_THRESHOLD 并持续 SETTLE_TIME 视为到位
    end
    D->>G: 退出 with → disconnect()（示例注明退出时电机自动失能）
```

图依据：Synria-Robotics/Gloria-M-SDK@f034f26，读过 demos/02_pv_control.py、src/gloria_m_sdk/client.py（部分）、controller.py（部分）、api/motion_api.py（仅签名）、serial_can_adapter.py（仅签名）、transport.py（仅签名）；未读 api/motor_api.py，MotorAPI 内部到 controller 的调用为按命名推断。

## 关联页面

- [国内具身开源全景技术地图](../overview/china-domestic-embodied-opensource-76-companies-technology-map.md)
- [HMI 开源项目主表导读](../queries/hmi-opensource-projects-coverage.md)
- [Humanoid Motion Intelligence](../entities/humanoid-motion-intelligence.md)

## 参考来源

- [Gloria-M-SDK 源码归档](../../sources/repos/gloria-m-sdk.md)（<https://github.com/Synria-Robotics/Gloria-M-SDK>）

- [国内具身智能开源全景（微信公众号）](../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md)
- 源码核对（2026-11-02）：Synria-Robotics/Gloria-M-SDK@f034f26，读过 demos/02_pv_control.py、src/gloria_m_sdk/client.py（部分）、controller.py（部分）；motion_api.py、serial_can_adapter.py、transport.py 仅读签名，未读 motor_api.py；时序图依据以上文件，未运行代码。

## 推荐继续阅读

- [玄雅科技 官方组织](https://github.com/Synria-Robotics)
