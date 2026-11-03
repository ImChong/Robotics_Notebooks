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
  - ../../sources/repos/taccap-gripper.md
summary: "千觉机器人 开源项目 TacCap-Gripper（SDK/驱动）：TacCap 多模态触觉数据采集夹爪 SDK（C++17/Python，支持主从与腕部相机）…"
institutions:
  - qianjue
---

# TacCap-Gripper

## 一句话定义

**TacCap-Gripper** 是 [千觉机器人](https://github.com/XenseRobotics-AI) 公开的 **SDK/驱动** 开源项目：TacCap 多模态触觉数据采集夹爪 SDK（C++17/Python，支持主从与腕部相机）

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
| 机构 | 千觉机器人 |
| 类别 | SDK/驱动 |
| 官方组织 | https://github.com/XenseRobotics-AI、https://github.com/XenseRobotics |

## 工程实践

1. 从官方 GitHub/Gitee 组织检索 `TacCap-Gripper` 仓库并核对 README 许可与依赖。
2. 对照本库 [424 项覆盖索引](../queries/china-domestic-opensource-424-coverage.md) 查看同公司其它入口是否共用训练/部署链路。
3. 若与既有方法页（如 RL 框架、VLA、SDK）主题相同，优先读关联页中的「开源入口」小节，避免重复维护平行叙事。

## 局限与风险

- 公众号清单为 **策展快照**（2026-09-06）；仓库更名、归档或许可证变化须回官方组织页核实。
- **开源状态**：以仓库 README 与 release 为准（入库日按文章描述归纳，未逐仓 clone 验证）。

## 源码运行时序图

入口：`python python/examples/force_position_control.py left`（抓取控制；同目录另有 `follower_status.py`、`impedance_control.py`、`gripper_console.py` 等；C++ 对应 `cpp/examples/follower_force_position.cpp`，本图只画 Python 版）。

```mermaid
sequenceDiagram
    participant U as 用户 / force_position_control.py
    participant T as _target.py<br/>open_follower
    participant F as FollowerGripper<br/>(xense.taccap，pybind11 绑定 C++)
    participant C as ForcePositionController<br/>cpp/src/force_position_controller.cpp
    participant TR as bus::Transport<br/>(串口 TC-GU-01 协议)
    participant MCU as 夹爪 MCU / 电机固件

    U->>T: open_follower("left")
    T->>T: resolve_target → scan_grippers() 按固件 SN 判左右/主从
    T->>F: FollowerGripper(mcu_device=endpoint.mcu_device)
    U->>F: motor.get_spec() → ForcePositionConfig.for_spec(spec)
    U->>C: ForcePositionController(g, cfg)
    U->>F: motor.clear_fault()
    U->>C: start()
    C->>F: motor().get_startup_limit_torque()，与 motion_torque_limit_nm 比对
    C->>F: motor().read_status() 取初值，创建 ForcePositionPolicy
    C->>F: motor().on_status(回调 on_status_)；start_streaming(motor_stream_hz)
    C->>C: 起控制线程 run_()
    U->>F: motor.enable()
    U->>C: set_target(p)（非阻塞，入队）
    loop 每个电机状态帧
        MCU-->>TR: 电机状态流
        TR-->>C: on_status_(sample)：更新 snapshot 观测，唤醒 run_
        C->>C: run_：取队列命令 → policy.step(sample) 得阻抗指令
        C->>TR: motor().submit(MotorImpedanceCtrl)（send_cmd_no_ack）
        TR->>MCU: 指令帧；固件侧做运动包络/堵转/温度保护
    end
    U->>C: snapshot()（无总线流量），检查 holding / arrived
    U->>C: finally: stop()
    C->>F: submit 零力矩 → motor().disable() → 取消订阅、停流
```

图依据：XenseRobotics-AI/TacCap-Gripper@e595763，读过 README.md、docs/ARCHITECTURE.md、python/examples/force_position_control.py、python/examples/_target.py、cpp/src/force_position_controller.cpp（start/stop/on_status_/run_ 等）、cpp/src/components/motor.cpp（相关片段）、cpp/src/follower_gripper.cpp（相关片段）；pybind11 绑定文件（python/bindings/）未读。

## 关联页面

- [国内具身开源全景技术地图](../overview/china-domestic-embodied-opensource-76-companies-technology-map.md)
- [HMI 开源项目主表导读](../queries/hmi-opensource-projects-coverage.md)
- [Humanoid Motion Intelligence](../entities/humanoid-motion-intelligence.md)

## 参考来源

- [TacCap-Gripper 源码归档](../../sources/repos/taccap-gripper.md)（<https://github.com/XenseRobotics-AI/TacCap-Gripper>）

- [国内具身智能开源全景（微信公众号）](../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md)
- 源码核对（2026-11-02）：XenseRobotics-AI/TacCap-Gripper@e595763，读过 README.md、docs/ARCHITECTURE.md、python/examples/force_position_control.py、python/examples/_target.py、cpp/src/force_position_controller.cpp、cpp/src/components/motor.cpp（片段）、cpp/src/follower_gripper.cpp（片段）；时序图依据以上文件，未运行代码。

## 推荐继续阅读

- [千觉机器人 官方组织](https://github.com/XenseRobotics-AI)
