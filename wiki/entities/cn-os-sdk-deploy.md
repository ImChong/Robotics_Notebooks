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
  - ../../sources/repos/sdk_deploy.md
summary: "云深处科技 开源项目 sdk_deploy（SDK/驱动）：策略部署运行时：加载训练策略、下发关节命令…"
institutions:
  - deeprobotics
---

# sdk_deploy

## 一句话定义

**sdk_deploy** 是 [云深处科技](https://github.com/DeepRoboticsLab) 公开的 **SDK/驱动** 开源项目：策略部署运行时：加载训练策略、下发关节命令

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

1. 从官方 GitHub/Gitee 组织检索 `sdk_deploy` 仓库并核对 README 许可与依赖。
2. 对照本库 [424 项覆盖索引](../queries/china-domestic-opensource-424-coverage.md) 查看同公司其它入口是否共用训练/部署链路。
3. 若与既有方法页（如 RL 框架、VLA、SDK）主题相同，优先读关联页中的「开源入口」小节，避免重复维护平行叙事。

## 局限与风险

- 公众号清单为 **策展快照**（2026-09-06）；仓库更名、归档或许可证变化须回官方组织页核实。
- **开源状态**：以仓库 README 与 release 为准（入库日按文章描述归纳，未逐仓 clone 验证）。

## 源码运行时序图

本仓库按机型分目录（`Lite3_sdk_deploy`、`M20_sdk_deploy`、`M20S_sdk_deploy`、`S10_sdk_deploy`、`lite3_sdk_service`、`lite3_transfer`、`drdds`）。下图只覆盖已读的 Lite3 部署链路（仿真闭环），其余机型目录未读，不在图内。

入口：`ros2 run lite3_sdk_deploy rl_deploy`（`src/Lite3_sdk_deploy/main.cpp`）；仿真侧另起 `python3 src/Lite3_sdk_deploy/interface/robot/simulation/mujoco_simulation_ros2.py`。

```mermaid
sequenceDiagram
    participant U as 用户（键盘 / 手柄 / ROS 2 命令）
    participant M as main.cpp main()
    participant Q as q::QStateMachine / StateMachineBase
    participant RI as Lite3Interface (ri_ptr_)
    participant RL as RLControlState
    participant PR as Lite3PolicyRunner (policy/policy.onnx)
    participant S as mujoco_simulation_ros2.py 或 /lite3 实机

    M->>Q: QStateMachine(Lite3, RemoteCommandType::kKeyBoard)
    M->>Q: Start()
    Q->>RI: 创建 Lite3Interface、ControlParameters、JointsDataShadowSubscriber
    Q->>Q: 创建 KeyboardInterface、SafeController、Idle/StandUp/RLControl/JointDamping/LieDown 五个状态
    Q->>RI: ri_ptr_->Start()，uc_ptr_->Start()，sc_ptr_->Start()，Idle.OnEnter()
    M->>Q: Run()（后台线程 RunThread + rclcpp::spin）
    S-->>RI: /IMU_DATA 与 /JOINTS_DATA（MuJoCo 每 5 步发布一次）
    loop RunThread 每 5 ms
        Q->>RI: RefreshRobotData()
        Q->>Q: current_controller_->Run()，LoseControlJudge / GetNextStateName
        opt 状态切换
            Q->>Q: OnExit() → OnEnter()
        end
    end
    U->>Q: 命令 Idle → StandUp → RLControl
    Q->>RL: OnEnter()（启动 PolicyRunner 线程）
    loop RLControl 状态
        RL->>RI: Run() → UpdateRobotObservation()：IMU rpy/omega/acc、关节 pos/vel/tau
        RL->>PR: 每 decimation 个周期 getRobotAction(rbs_, UserCommand)
        PR->>PR: 拼 obs → ONNX Runtime session_.Run(obs → actions)
        PR-->>RL: RobotAction（kp、kd、目标关节位置）
        RL->>RI: SetJointCommand()
        RI->>S: 发布 /JOINTS_CMD
        S->>S: _cmd_callback → _apply_joint_torque：tau = kp*(pos_cmd-q) + kd*(vel_cmd-dq) + tau_ff，mujoco.mj_step
    end
    U->>Q: 目标模式 LieDown / JointDamping（或急停）→ 状态切换
```

图依据：DeepRoboticsLab/sdk_deploy@2027907，读过 README.md、src/Lite3_sdk_deploy/{README.md、main.cpp、state_machine/state_machine_base.h、state_machine/quadruped/q_state_machine.hpp、rl_control_state.hpp、run_policy/lite3_policy_runner.hpp（部分）、interface/robot/simulation/mujoco_simulation_ros2.py}。

## 关联页面

- [国内具身开源全景技术地图](../overview/china-domestic-embodied-opensource-76-companies-technology-map.md)
- [HMI 开源项目主表导读](../queries/hmi-opensource-projects-coverage.md)
- [Humanoid Motion Intelligence](../entities/humanoid-motion-intelligence.md)

## 参考来源

- [sdk_deploy 源码归档](../../sources/repos/sdk_deploy.md)（<https://github.com/DeepRoboticsLab/sdk_deploy>）

- [国内具身智能开源全景（微信公众号）](../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md)
- 源码核对（2026-11-02）：DeepRoboticsLab/sdk_deploy@2027907，读过 README.md、src/Lite3_sdk_deploy/README.md、main.cpp、state_machine/state_machine_base.h、state_machine/quadruped/q_state_machine.hpp、rl_control_state.hpp、run_policy/lite3_policy_runner.hpp（部分）、interface/robot/simulation/mujoco_simulation_ros2.py；仅覆盖 Lite3 目录，M20/M20S/S10/lite3_sdk_service/lite3_transfer/drdds 未读；时序图依据以上文件，未运行代码。

## 推荐继续阅读

- [云深处科技 官方组织](https://github.com/DeepRoboticsLab)
