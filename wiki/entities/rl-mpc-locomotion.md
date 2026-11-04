---
type: entity
project_id: rl-mpc-locomotion
tags:
  - repo
  - independent
  - quadruped
  - rl
  - mpc
  - isaac-gym
  - aliengo
status: complete
updated: 2026-11-01
code: https://github.com/silvery107/rl-mpc-locomotion
related:
  - ../methods/reinforcement-learning.md
  - ../methods/model-predictive-control.md
  - ../entities/isaac-gym.md
  - ../tasks/locomotion.md
  - ../overview/quadruped-open-source-projects-technology-map.md
sources:
  - ../../sources/repos/rl_mpc_locomotion.md
  - ../../sources/blogs/wechat_guyueju_quadruped_opensource_deep_dive_2026-11-01.md
summary: "rl-mpc-locomotion 提供快速仿真与 RL 训练框架：高层策略网络动态预测 MPC 控制器的权重参数，低层是参考 MIT Cheetah Software、用 Python 重写的 MPC。"
---

# rl-mpc-locomotion

**rl-mpc-locomotion**（[GitHub](https://github.com/silvery107/rl-mpc-locomotion)）来自 [古月居转载的四足机器人开源项目综述](https://mp.weixin.qq.com/s/pRGoPm41qYCNyHPWR6Dl6A)，属「强化学习」组。

## 一句话定义

**rl-mpc-locomotion 提供快速仿真与 RL 训练框架：高层策略网络动态预测 MPC 控制器的权重参数，低层是参考 MIT Cheetah Software、用 Python 重写的 MPC。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RL | Reinforcement Learning | 强化学习 |
| MPC | Model Predictive Control | 模型预测控制 |

## 为什么重要

- 综述把它放在「强化学习」这条技术路线下，作为可读源码的参照项目；横向对比见 [四足机器人开源项目技术地图：从极简运动学到研究级硬件（36 个 GitHub 项目）](../overview/quadruped-open-source-projects-technology-map.md)。
- 文中星标：946（文章表述，入库日未复核）。

## 核心信息

| 项 | 内容 |
|----|------|
| **仓库** | [silvery107/rl-mpc-locomotion](https://github.com/silvery107/rl-mpc-locomotion) |
| **最近提交** | 2026-04-26 |
| **可达性** | `git ls-remote` 可达（2026-11-01） |
| **文章归类** | 强化学习 |

## 文章说法核对

- **文章：** 文章称它是完整教学框架（配套 UG-Thesis-EN 教学 PPT），实现凸 MPC、WBC，并用 PPO 学习 MPC 权重，基于 Isaac Gym 并行训练。
- **README / 核查：** README 称 RL 训练使用 NVIDIA Isaac Gym 并行、使用 Unitree Aliengo 模型，并在真实 Aliengo 上实现 sim2real（**sim2real 不在本代码库内**）；依赖 Python 3.8、PyTorch 1.10、Isaac Gym Preview 4。
- **差异与保留：** 文章提到的 WBC 与教学 PPT 细节未在 README 前段核对。

## 源码运行时序图

入口：`python RL_MPC_Locomotion.py --robot=Aliengo`（默认 `--mode=Fsm`；`--mode=Min` 为无 FSM 的最小 MPC；`--mode=Policy --checkpoint=...` 使用 RL 权重策略生成 MPC 权重）。依赖 Isaac Gym 仿真器，`gymapi` 与 `isaacgym` 为外部依赖，未读。

```mermaid
sequenceDiagram
    autonumber
    participant U as 用户（手柄）
    participant MAIN as RL_MPC_Locomotion.py<br/>main()
    participant GP as RL_Environment/gamepad_reader.py<br/>Gamepad
    participant SIM as Isaac Gym 仿真<br/>sim_utils.py（acquire_sim / create_envs）
    participant RR as MPC_Controller/robot_runner/<br/>RobotRunnerFSM / RobotRunnerPolicy
    participant SE as MPC_Controller/common/StateEstimator.py
    participant LC as MPC_Controller/common/LegController.py
    participant DSC as MPC_Controller/common/DesiredStateCommand.py
    participant WP as RL_Environment/WeightPolicy.py<br/>（仅 Policy 模式）
    participant FSM as MPC_Controller/FSM_states/ControlFSM.py
    participant CMPC as MPC_Controller/convex_MPC/ConvexMPCLocomotion.py
    participant CPP as mpc_osqp（C++ pybind）<br/>MPC_Controller/convex_MPC/mpc_osqp.cc

    U->>MAIN: python RL_MPC_Locomotion.py --robot --mode --checkpoint
    MAIN->>GP: Gamepad(...)（未加 --disable-gamepad 时）
    MAIN->>SIM: acquire_sim，add_ground，add_terrain（slope、stair），create_envs，add_viewer
    loop 每台机器人（num_envs）
        MAIN->>SIM: 把 DOF 驱动模式设为 effort，刚度与阻尼置 0
        MAIN->>RR: RobotRunnerFSM() / RobotRunnerMin() / RobotRunnerPolicy(checkpoint)，init(robot)
    end

    loop 直到 viewer 关闭
        MAIN->>SIM: gym.simulate(sim)，fetch_results
        MAIN->>GP: get_command() 得到线速度、角速度、e_stop；get_gait()、get_mode()
        MAIN->>SIM: get_actor_dof_states，get_actor_rigid_body_states
        MAIN->>RR: run(dof_states, body_states, commands)
        RR->>LC: updateData(dof_states)，zeroCommand()
        RR->>SE: update(body_states)
        alt Policy 模式
            RR->>WP: compute_observations(dof, 状态估计, commands, weights)
            WP-->>RR: step() 得到 12 维 MPC 权重
            RR->>DSC: updateCommand(commands, weights)
        else FSM 或 Min 模式
            RR->>DSC: updateCommand(commands)
        end
        RR->>FSM: runFSM()（FSM 与 Policy 模式）
        FSM->>CMPC: Locomotion 状态调用 LocomotionControlStep，即 cMPC.run(data)
        CMPC->>CMPC: __SetupCommand；按步态计算摆动时间与足端参考
        CMPC->>CMPC: updateMPCIfNeeded，按 MPC 周期调用 solveDenseMPC
        CMPC->>CPP: compute_contact_forces(mpc_weight, 质心位姿与速度, ...)
        CPP-->>CMPC: 预测接触力
        RR->>LC: updateCommand() 得到 12 个关节力矩
        RR-->>MAIN: legTorques（12，float32）
        MAIN->>SIM: apply_actor_dof_efforts(env, actor, legTorques)
    end
```

图依据：silvery107/rl-mpc-locomotion@df944ef，实际读过 `README.md`、`RL_MPC_Locomotion.py`（参数与主循环）、`MPC_Controller/robot_runner/RobotRunnerFSM.py`（`run`）、`MPC_Controller/robot_runner/RobotRunnerPolicy.py`（`run`）、`MPC_Controller/FSM_states/ControlFSM.py`（`runFSM` 的状态切换结构）、`MPC_Controller/FSM_states/FSM_State_Locomotion.py`（`LocomotionControlStep`）、`MPC_Controller/convex_MPC/ConvexMPCLocomotion.py`（`solveDenseMPC`、`run` 中 `updateMPCIfNeeded` 调用）、`MPC_Controller/common/DesiredStateCommand.py`（`updateCommand`）、`RL_Environment/WeightPolicy.py`（构造与加载）、`setup.py`（`mpc_osqp` 扩展源文件）、`MPC_Controller/convex_MPC/mpc_osqp.cc`（求解器相关函数，未核对 pybind 导出列表）；未读 `RobotRunnerMin.py`（图中 Min 仅按 `RL_MPC_Locomotion.py` 的分派示意）；`FSM_State_Locomotion.run` 到 `LocomotionControlStep` 的完整调用链、`sim_utils.py` 与 `train.py` 训练流程未逐行核对；未运行代码。

## 关联页面

- [reinforcement-learning](../methods/reinforcement-learning.md)
- [model-predictive-control](../methods/model-predictive-control.md)
- [isaac-gym](../entities/isaac-gym.md)
- [locomotion](../tasks/locomotion.md)
- [quadruped-open-source-projects-technology-map](../overview/quadruped-open-source-projects-technology-map.md)

## 参考来源

- [rl_mpc_locomotion.md](../../sources/repos/rl_mpc_locomotion.md)
- [wechat_guyueju_quadruped_opensource_deep_dive_2026-11-01.md](../../sources/blogs/wechat_guyueju_quadruped_opensource_deep_dive_2026-11-01.md)
- [原文（微信公众号）](https://mp.weixin.qq.com/s/pRGoPm41qYCNyHPWR6Dl6A)
- 源码核对（2026-11-02）：silvery107/rl-mpc-locomotion@df944ef，读过 README.md、RL_MPC_Locomotion.py、MPC_Controller/robot_runner/RobotRunnerFSM.py、MPC_Controller/robot_runner/RobotRunnerPolicy.py、MPC_Controller/FSM_states/ControlFSM.py、MPC_Controller/FSM_states/FSM_State_Locomotion.py、MPC_Controller/convex_MPC/ConvexMPCLocomotion.py、MPC_Controller/common/DesiredStateCommand.py、RL_Environment/WeightPolicy.py、setup.py、MPC_Controller/convex_MPC/mpc_osqp.cc；时序图依据以上文件，未运行代码。
