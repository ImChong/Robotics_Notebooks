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
  - ../../sources/repos/deep-robotics-sdk2.md
summary: "云深处科技 开源项目 deep-robotics-sdk2（SDK/驱动）：官方 SDK：真机控制与状态读取的统一接入层…"
institutions:
  - deeprobotics
---

# deep-robotics-sdk2

## 一句话定义

**deep-robotics-sdk2** 是 [云深处科技](https://github.com/DeepRoboticsLab) 公开的 **SDK/驱动** 开源项目：官方 SDK：真机控制与状态读取的统一接入层

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
| 类别 | SDK/驱动 |
| 官方组织 | https://github.com/DeepRoboticsLab |

## 工程实践

1. 从官方 GitHub/Gitee 组织检索 `deep-robotics-sdk2` 仓库并核对 README 许可与依赖。
2. 对照本库 [424 项覆盖索引](../queries/china-domestic-opensource-424-coverage.md) 查看同公司其它入口是否共用训练/部署链路。
3. 若与既有方法页（如 RL 框架、VLA、SDK）主题相同，优先读关联页中的「开源入口」小节，避免重复维护平行叙事。

## 局限与风险

- 公众号清单为 **策展快照**（2026-09-06）；仓库更名、归档或许可证变化须回官方组织页核实。
- **开源状态**：以仓库 README 与 release 为准（入库日按文章描述归纳，未逐仓 clone 验证）。

## 源码运行时序图

入口：`ros2 run dr02_pro state_machine`（`src/dr02_pro/docs/STATE_MACHINE.md`；可执行文件由 `src/dr02_pro/state_machine/CMakeLists.txt` 的 `add_executable(state_machine main.cpp)` 生成）。本仓库是 ROS 2 SDK，含 `dr02_pro`、`dr02_std` 两个产品包以及大量单话题示例（`high_level/`、`low_level/`、`peripherals/` 等）；下图只取 `dr02_pro` 状态机这一条主路径，`dr02_std` 与各示例未核对。消息类型来自外部仓库 `deep-robotics-msg`（包名 `drdds`）。

```mermaid
sequenceDiagram
    participant Main as main.cpp main()
    participant FSM as HumanoidStateMachine
    participant UC as UserCommandInterface
    participant RI as DR2ProInterface（RobotInterface）
    participant St as 状态：Idle、ZeroPos、StandUp、RLControlAMP / Mimic
    participant PR as AMP / Mimic PolicyRunner（ONNX Runtime）
    participant Bus as ROS 2 话题（机器人或 MuJoCo 仿真）

    Main->>FSM: rclcpp::init，构造 HumanoidStateMachine(DR2PRO)
    Main->>FSM: Start()
    FSM->>UC: 创建订阅 /GAMEPAD_KEY、/STEER、/REAL_STEER
    FSM->>RI: 创建 DR2ProInterface(node_)
    FSM->>St: 构造各 State，current_controller_ = IdleState
    FSM->>RI: Start()：发布 /JOINTS_CMD，订阅 /JOINTS_DATA、/IMU_DATA_HEAD、/IMU_DATA_BASE
    RI->>Bus: WaitDataUpdate 等首帧关节数据
    FSM->>St: current_controller_.OnEnter()
    Main->>FSM: Run()
    FSM->>FSM: 后台线程 rclcpp::spin(node_)
    loop 2 ms LoopTimer
        FSM->>RI: RefreshRobotData()
        FSM->>St: current_controller_.Run()
        FSM->>St: GetNextStateName()
        opt 状态变化
            FSM->>St: OnExit()，新状态 OnEnter()
        end
    end
    Note over St,PR: 进入 RLControlAMP 后
    St->>St: OnEnter 启动 PolicyRunner 线程
    St->>PR: GetRobotAction(robot_state, user_command) 每 decimation 个周期一次
    PR-->>St: 21 维动作
    St->>St: MapToJointCommand 转成 31 维关节指令
    St->>RI: SetJointCommand(joint_cmd)
    RI->>Bus: 发布 /JOINTS_CMD（kp、position、kd、velocity、torque）
```

图依据：DeepRoboticsLab/deep-robotics-sdk2@c892523，读过 `README.md`、`src/dr02_pro/docs/STATE_MACHINE.md`、`src/dr02_pro/state_machine/main.cpp`、`src/dr02_pro/state_machine/humanoid_state_machine.hpp`、`src/dr02_pro/state_machine/interface/robot_interface.hpp`（`Start`、`SetJointCommand`）、`src/dr02_pro/state_machine/interface/user_command_interface.hpp`（订阅部分）、`src/dr02_pro/state_machine/state/rl_control_amp_state.hpp`（`OnEnter`、`Run`、`PolicyRunner` 片段）、`src/dr02_pro/state_machine/run_policy/amp_policy_runner.hpp`（成员声明片段）；`Mimic` 状态与 `AMP` 状态的差异、`SafeController`、`StandUp`/`ZeroPos` 的具体插值、`PolicyRunner` 内部观测拼装与 `session_.Run` 调用细节未读；`/JOINTS_DATA` 等话题的另一端（机器人驱动或 `deep-robotics-simulation`）不在本仓库。

## 关联页面

- [国内具身开源全景技术地图](../overview/china-domestic-embodied-opensource-76-companies-technology-map.md)
- [HMI 开源项目主表导读](../queries/hmi-opensource-projects-coverage.md)
- [Humanoid Motion Intelligence](../entities/humanoid-motion-intelligence.md)

## 参考来源

- [deep-robotics-sdk2 源码归档](../../sources/repos/deep-robotics-sdk2.md)（<https://github.com/DeepRoboticsLab/deep-robotics-sdk2>）

- [国内具身智能开源全景（微信公众号）](../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md)
- 源码核对（2026-11-02）：DeepRoboticsLab/deep-robotics-sdk2@c892523，读过 README.md、src/dr02_pro/docs/STATE_MACHINE.md、src/dr02_pro/state_machine/{main.cpp,humanoid_state_machine.hpp}、interface/{robot_interface,user_command_interface}.hpp、state/rl_control_amp_state.hpp（部分）、run_policy/amp_policy_runner.hpp（部分）；时序图依据以上文件，未运行代码。

## 推荐继续阅读

- [云深处科技 官方组织](https://github.com/DeepRoboticsLab)
