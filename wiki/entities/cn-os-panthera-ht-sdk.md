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
  - ../../sources/repos/panthera-ht_sdk.md
summary: "高擎机电 开源项目 Panthera-HT_SDK（SDK/驱动）：官方 SDK：真机控制与状态读取的统一接入层…"
institutions:
  - hightorque
---

# Panthera-HT_SDK

## 一句话定义

**Panthera-HT_SDK** 是 [高擎机电](https://github.com/HighTorque-Robotics) 公开的 **SDK/驱动** 开源项目：官方 SDK：真机控制与状态读取的统一接入层

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
| 机构 | 高擎机电 |
| 类别 | SDK/驱动 |
| 官方组织 | https://github.com/HighTorque-Robotics |

## 工程实践

1. 从官方 GitHub/Gitee 组织检索 `Panthera-HT_SDK` 仓库并核对 README 许可与依赖。
2. 对照本库 [424 项覆盖索引](../queries/china-domestic-opensource-424-coverage.md) 查看同公司其它入口是否共用训练/部署链路。
3. 若与既有方法页（如 RL 框架、VLA、SDK）主题相同，优先读关联页中的「开源入口」小节，避免重复维护平行叙事。

## 局限与风险

- 公众号清单为 **策展快照**（2026-09-06）；仓库更名、归档或许可证变化须回官方组织页核实。
- **开源状态**：以仓库 README 与 release 为准（入库日按文章描述归纳，未逐仓 clone 验证）。

## 源码运行时序图

入口：`python scripts/1_moveJ_control.py`（在 `panthera_python/` 下；需先按 README 安装 `hightorque_robot` whl，或用 `panthera_cpp/motor_cpp` + `src/bindings.cpp` 源码编译）。

```mermaid
sequenceDiagram
    participant U as 用户 / scripts/1_moveJ_control.py
    participant P as scripts/Panthera_lib/Panthera.py<br/>class Panthera(htr.Robot)
    participant B as hightorque_robot (pybind11)<br/>src/bindings.cpp
    participant R as motor_cpp/src/hardware/robot.cpp<br/>robot
    participant M as hardware/motor.cpp<br/>motor
    participant CB as canboard.cpp → canport.cpp
    participant SD as serial_driver.cpp
    participant HW as 串口 CAN 板 / 关节电机

    U->>P: Panthera()（默认 robot_param/Follower.yaml）
    P->>P: _load_config_file；_load_joint_limits；_prepare_sdk_config
    P->>B: super().__init__(sdk_config_path)
    B->>R: robot(config_path) → init_robot
    R->>SD: 打开串口，启动 recv_1for6_42 接收线程
    R->>HW: send_get_motor_state_cmd（探测电机）
    P->>P: _init_motors；_load_motor_parameters；_load_urdf_model（pinocchio）
    U->>P: send_get_motor_state_cmd；motor_send_cmd（sleep 0.5 s ×2）
    P->>R: robot::send_get_motor_state_cmd / motor_send_cmd
    R->>CB: canboard → canport
    CB->>SD: send_2(&cdc_tr_message)
    SD->>HW: 串口写出
    HW-->>SD: 电机状态帧 → recv 线程更新 motor 状态

    U->>P: moveJ(pos, duration, max_tqu, iswait=True)
    P->>P: 检查关节限位；vel = (pos - get_current_pos()) / duration
    P->>P: Joint_Pos_Vel(pos, vel, max_tqu)
    loop 每个关节 i
        P->>M: Motors[i].pos_vel_MAXtqe(pos, vel, torque_max)
    end
    P->>R: motor_send_cmd()
    R->>CB: canboard::motor_send_cmd → canport::motor_send_cmd
    CB->>SD: send_2 → 串口帧 (MODE_POS_VEL_TQE)
    SD->>HW: 位置 / 速度 / 最大力矩指令
    P->>P: wait_for_position(pos, tolerance, timeout)
    P-->>U: True / False

    U->>P: gripper_close() / gripper_open()
    Note over P,M: 脚本中依次 moveJ(zero→pos1→pos2→pos1→zero)，并穿插夹爪开合
    Note over U,R: C++ 路径对应 panthera_cpp/robot_cpp（Panthera.cpp + example/*.cpp），<br/>与 motor_cpp 同属 hightorque_robot 库；本图未逐行核对 C++ 例程
```

图依据：HighTorque-Robotics/Panthera-HT_SDK@7c506f8，读过 README.md、panthera_python/README.md、panthera_python/scripts/1_moveJ_control.py、scripts/Panthera_lib/Panthera.py（__init__、moveJ、Joint_Pos_Vel）、panthera_python/src/bindings.cpp（include 与类注册）、panthera_cpp/motor_cpp/src/hardware/{robot,motor,canboard,canport}.cpp 与 src/serial_driver.cpp（相关函数段）。

## 关联页面

- [国内具身开源全景技术地图](../overview/china-domestic-embodied-opensource-76-companies-technology-map.md)
- [HMI 开源项目主表导读](../queries/hmi-opensource-projects-coverage.md)
- [Humanoid Motion Intelligence](../entities/humanoid-motion-intelligence.md)

## 参考来源

- [Panthera-HT_SDK 源码归档](../../sources/repos/panthera-ht_sdk.md)（<https://github.com/HighTorque-Robotics/Panthera-HT_SDK>）

- [国内具身智能开源全景（微信公众号）](../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md)
- 源码核对（2026-11-02）：HighTorque-Robotics/Panthera-HT_SDK@7c506f8，读过 README.md、panthera_python/README.md、panthera_python/scripts/1_moveJ_control.py、scripts/Panthera_lib/Panthera.py、panthera_python/src/bindings.cpp、panthera_cpp/motor_cpp/src/hardware/{robot,motor,canboard,canport}.cpp、src/serial_driver.cpp（局部）；C++ 例程未逐行核对；时序图依据以上文件，未运行代码。

## 推荐继续阅读

- [高擎机电 官方组织](https://github.com/HighTorque-Robotics)
