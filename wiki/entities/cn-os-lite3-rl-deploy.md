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
  - ../../sources/repos/lite3_rl_deploy.md
summary: "云深处科技 开源项目 Lite3 RL Deploy（仿真环境）：把训练得到的PyTorch策略转换为ONNX，并通过策略运行器、机器人接口和Idle、StandUp、RL、JointDamping状态机接入MuJoCo或PyBullet；切换编译平台与网络配置后可连接Lite3运动主机，形成仿真检查到实…"
institutions:
  - deeprobotics
---

# Lite3 RL Deploy

## 一句话定义

**Lite3 RL Deploy** 是 [云深处科技](https://github.com/DeepRoboticsLab) 公开的 **仿真环境** 开源项目：把训练得到的PyTorch策略转换为ONNX，并通过策略运行器、机器人接口和Idle、StandUp、RL、JointDamping状态机接入MuJoCo或PyBullet；切换编译平台与网络配置后可连接Lite3运动主机，形成仿真检查到实机执行的连续入口。

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
| 类别 | 仿真环境 |
| 官方组织 | https://github.com/DeepRoboticsLab |

## 工程实践

1. 从官方 GitHub/Gitee 组织检索 `Lite3 RL Deploy` 仓库并核对 README 许可与依赖。
2. 对照本库 [424 项覆盖索引](../queries/china-domestic-opensource-424-coverage.md) 查看同公司其它入口是否共用训练/部署链路。
3. 若与既有方法页（如 RL 框架、VLA、SDK）主题相同，优先读关联页中的「开源入口」小节，避免重复维护平行叙事。

## 局限与风险

- 公众号清单为 **策展快照**（2026-09-06）；仓库更名、归档或许可证变化须回官方组织页核实。
- **开源状态**：以仓库 README 与 release 为准（入库日按文章描述归纳，未逐仓 clone 验证）。

## 源码运行时序图

入口：仿真对仿真时，终端 1 `python3 interface/robot/simulation/mujoco_simulation.py`（或 `pybullet_simulation.py`），终端 2 `build/rl_deploy`（`cmake .. -DBUILD_PLATFORM=x86 -DBUILD_SIM=ON` 构建，`main.cpp` 中 `StateMachine(RobotType::Lite3).Run()`）。下图取 MuJoCo 脚本与 `USE_PYBULLET` 下的 `SimulationInterface` 的 UDP 配对；实机时 `RobotInterface` 换成 `HardwareInterface`（用 `Receiver`/`Sender` 与运动主机通信，未展开）。

```mermaid
sequenceDiagram
    participant Sim as mujoco_simulation.py（MuJoCoSimulation）
    participant RI as SimulationInterface（simulation_interface.hpp）
    participant SM as StateMachine（state_machine.hpp）
    participant St as 状态机各 State（idle、standup、rl_control、joint_damping、liedown）
    participant RL as RLControlStateONNX（rl_control_state_onnx.hpp）
    participant P as Lite3TestPolicyRunnerONNX（policy/ppo/policy.onnx）
    participant UC as UserCommandInterface（键盘或 Xbox 手柄）

    Sim->>RI: UDP 发送关节、IMU 观测（_send_robot_state）
    SM->>RI: 构造 SimulationInterface("Lite3")，Start()（接收线程与发送线程）
    SM->>UC: 构造 Keyboard / XboxGamepadInterface，Start()
    SM->>St: current_controller_ = IdleState，OnEnter()
    loop Run()，时间戳变化时
        SM->>St: Run()
        SM->>St: LoseControlJudge() / GetNextStateName()
        opt 状态切换
            SM->>St: OnExit()，下一状态 OnEnter()
        end
    end
    UC-->>SM: 按键 z / c / x / r 设置 target_mode
    Note over SM,RL: 切入 kRLControl 后
    RL->>RL: OnEnter() 启动 PolicyRunner 线程
    RL->>RI: Run() 里 UpdateRobotObservation：读 IMU、关节、用户速度指令
    RL->>P: GetRobotAction(rbs_) 每 decimation 个周期一次
    P->>P: 组 45 维观测，session_.Run（ONNX Runtime）
    P-->>RL: 12 维动作转成关节指令
    RL->>RI: SetJointCommand(res)
    RI->>Sim: UDP 发送 kp、pos、kd、vel、tau 共 5x12 个 float
    Sim->>Sim: _udp_receiver 解包，_apply_joint_torque，mj_step
```

图依据：DeepRoboticsLab/Lite3_rl_deploy@1c4a8f1，读过 `README.md`、`main.cpp`、`state_machine/state_machine.hpp`、`state_machine/rl_control_state_onnx.hpp`、`run_policy/lite3_test_policy_runner_onnx.hpp`（`GetRobotAction` 及成员声明）、`interface/robot/simulation/simulation_interface.hpp`（Start 与 UDP 收发片段）、`interface/robot/simulation/mujoco_simulation.py`（`start`、`_udp_receiver` 片段）；`IdleState`/`StandUpState` 等具体动作、`HardwareInterface` 与 `third_party/Lite3_MotionSDK` 内部、`pybullet_simulation.py`、`policy/pt2onnx.py` 未读。`Sim->>RI` 的观测数据包布局与 `SimulationInterface` 接收线程的对应关系依代码中 UDP 绑定/`recvfrom` 的存在推断，未逐字段核对。

## 关联页面

- [国内具身开源全景技术地图](../overview/china-domestic-embodied-opensource-76-companies-technology-map.md)
- [HMI 开源项目主表导读](../queries/hmi-opensource-projects-coverage.md)
- [Humanoid Motion Intelligence](../entities/humanoid-motion-intelligence.md)

## 参考来源

- [Lite3 RL Deploy 源码归档](../../sources/repos/lite3_rl_deploy.md)（<https://github.com/DeepRoboticsLab/Lite3_rl_deploy>）

- [国内具身智能开源全景（微信公众号）](../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md)
- 源码核对（2026-11-02）：DeepRoboticsLab/Lite3_rl_deploy@1c4a8f1，读过 README.md、main.cpp、state_machine/state_machine.hpp、state_machine/rl_control_state_onnx.hpp、run_policy/lite3_test_policy_runner_onnx.hpp、interface/robot/simulation/simulation_interface.hpp（部分）、interface/robot/simulation/mujoco_simulation.py（部分）；时序图依据以上文件，未运行代码。

## 推荐继续阅读

- [云深处科技 官方组织](https://github.com/DeepRoboticsLab)
