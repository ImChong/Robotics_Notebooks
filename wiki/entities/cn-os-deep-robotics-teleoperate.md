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
  - ../../sources/repos/deep-robotics-teleoperate.md
summary: "云深处科技 开源项目 deep-robotics-teleoperate（遥操作与数据采集）：遥操作与数据采集：人体/设备输入映射为机器人动作并记录示范数据…"
institutions:
  - deeprobotics
---

# deep-robotics-teleoperate

## 一句话定义

**deep-robotics-teleoperate** 是 [云深处科技](https://github.com/DeepRoboticsLab) 公开的 **遥操作与数据采集** 开源项目：遥操作与数据采集：人体/设备输入映射为机器人动作并记录示范数据

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
| 机构 | 云深处科技 |
| 类别 | 遥操作与数据采集 |
| 官方组织 | https://github.com/DeepRoboticsLab |

## 工程实践

1. 从官方 GitHub/Gitee 组织检索 `deep-robotics-teleoperate` 仓库并核对 README 许可与依赖。
2. 对照本库 [424 项覆盖索引](../queries/china-domestic-opensource-424-coverage.md) 查看同公司其它入口是否共用训练/部署链路。
3. 若与既有方法页（如 RL 框架、VLA、SDK）主题相同，优先读关联页中的「开源入口」小节，避免重复维护平行叙事。

## 局限与风险

- 公众号清单为 **策展快照**（2026-09-06）；仓库更名、归档或许可证变化须回官方组织页核实。
- **开源状态**：以仓库 README 与 release 为准（入库日按文章描述归纳，未逐仓 clone 验证）。

## 源码运行时序图

入口：`ros2 run dr02_tele state_machine [--log] [--record] [--record-dir PATH] [--ee zhixing]`（对应 `src/dr02_pro/state_machine/main.cpp`）

```mermaid
sequenceDiagram
    participant U as 用户 / PICO 头显与手柄
    participant M as state_machine/main.cpp main()
    participant F as HumanoidStateMachine
    participant RI as DR2ProInterface (robot_interface.hpp)
    participant T as TeleoperationState
    participant PR as PicoDataReceiver (PXREA SDK)
    participant TC as PicoTeleoperationController
    participant IK as Dr2ArmIK (Pinocchio)
    participant R as 机器人 / MuJoCo (/JOINTS_DATA, /JOINTS_CMD)

    U->>M: ros2 run dr02_tele state_machine
    M->>M: ParseRuntimeOptions()（--log/--record/--ee）
    M->>F: HumanoidStateMachine(DR2PRO, ...)，rclcpp::init
    M->>F: Start()
    F->>RI: 创建节点 sdk_deploy，订阅 /JOINTS_DATA，发布 /JOINTS_CMD
    F->>F: 创建 Emergency/ZeroPos/Teleoperation/Down 四个状态，初始为 Emergency
    M->>F: Run()（2 ms 循环 + 后台 rclcpp::spin）
    loop 每 2 ms，仅当 /JOINTS_DATA 时间戳更新
        R-->>RI: Joints 反馈
        F->>F: current_controller_->Run()，检查 GetNextStateName()
    end
    U->>F: 键盘 / 手柄 z、s（/GAMEPAD_KEY）切到 ZeroPos，再切 Teleoperation
    F->>T: OnEnter()
    T->>PR: Start() → PXREAInit（dlopen libPXREARobotSDK.so）
    PR-->>PR: OnPicoCallback 持续缓存 PicoState
    loop Teleoperation 状态下每个反馈帧
        T->>PR: Snapshot()
        T->>TC: Update(current, pico)
        alt 按住扳机且有新的 PICO 数据包
            TC->>IK: Solve(左右目标位姿, seed_q)
            IK-->>TC: 14 维手臂关节解
        end
        TC-->>T: Command（腰部来自头显姿态，手臂来自 IK + 平滑）
        T->>RI: SetJointCommand（kp/kd + 位置，腿颈保持进入时目标）
        RI->>R: publish /JOINTS_CMD
    end
    U->>F: e/R1 → Emergency，d/R2 → Down
    F->>T: OnExit() → PicoDataReceiver::Stop()
```

图依据：DeepRoboticsLab/deep-robotics-teleoperate@840fcec，读过 README.md、src/dr02_pro/README.md、docs/STATE_MACHINE.md、state_machine/main.cpp、humanoid_state_machine.hpp、state/teleoperation_state.hpp、interface/robot_interface.hpp、manipulation/src/pico_data_receiver.cpp、pico_teleoperation.cpp、dr2_arm_ik.cpp（部分）。

## 关联页面

- [国内具身开源全景技术地图](../overview/china-domestic-embodied-opensource-76-companies-technology-map.md)
- [HMI 开源项目主表导读](../queries/hmi-opensource-projects-coverage.md)
- [Humanoid Motion Intelligence](../entities/humanoid-motion-intelligence.md)

## 参考来源

- [deep-robotics-teleoperate 源码归档](../../sources/repos/deep-robotics-teleoperate.md)（<https://github.com/DeepRoboticsLab/deep-robotics-teleoperate>）

- [国内具身智能开源全景（微信公众号）](../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md)
- 源码核对（2026-11-02）：DeepRoboticsLab/deep-robotics-teleoperate@840fcec，读过 README.md、src/dr02_pro/README.md、docs/STATE_MACHINE.md、state_machine/main.cpp、humanoid_state_machine.hpp、state/teleoperation_state.hpp、interface/robot_interface.hpp、manipulation/src/pico_data_receiver.cpp、pico_teleoperation.cpp、dr2_arm_ik.cpp（部分）；时序图依据以上文件，未运行代码。

## 推荐继续阅读

- [云深处科技 官方组织](https://github.com/DeepRoboticsLab)
