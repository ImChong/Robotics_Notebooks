---
type: entity
tags: [repo, unitree, unitreerobotics, isaac-lab, teleoperation, simulation, imitation-learning]
status: complete
updated: 2026-10-06
related:
  - ./unitree.md
  - ./xr-teleoperate.md
  - ./unitree-lerobot.md
  - ./unitree-rl-lab.md
  - ./unitree-dexterous-hand-services.md
  - ../tasks/teleoperation.md
  - ../concepts/sim2real.md
sources:
  - ../../sources/repos/unitree_sim_isaaclab.md
  - ../../sources/repos/unitree.md
summary: "unitree_sim_isaaclab 在 Isaac Lab 上仿真 G1/H1-2 多执行器任务，用与真机相同的 DDS 主题支持遥操作采数、回放与模型验证；常与 xr_teleoperate 联用。"
---

# unitree_sim_isaaclab

**unitree_sim_isaaclab** 基于 Isaac Lab，为 Unitree **G1 / H1-2**（夹爪、Dex3、Inspire 等执行器组合）提供多任务仿真，服务数据采集、回放、生成与策略验证。

## 一句话定义

「长得像真机 DDS」的 Isaac Lab 仿真场——XR 遥操作或策略节点不用改话题就能在仿真里采数/验模型。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Isaac Lab | NVIDIA Isaac Lab | 仿真与学习框架 |
| DDS | Data Distribution Service | 与真机同构通信 |
| XR | Extended Reality | 常经 xr_teleoperate 接入 |
| DoF | Degrees of Freedom | 如 G1-29dof |
| IL | Imitation Learning | 采数下游用途 |
| Sim2Real | Simulation to Real | 仿真到真机 |

## 项目时间边界

官方仓库在 **2025-06-24** 的代码提交已包含实现，本路线将其标为「代码历史起点」；这不证明正式首发或首次公开日，也不表示当日已具备本页介绍的全部现有功能。提交证据见 [原始仓库时间核对](../../sources/repos/unitree_sim_isaaclab.md)。

## 为什么重要

- 解决「没真机 / 真机档期不足」时的人形操作数据瓶颈。
- **Wholebody** 任务名表示可移动操作，不仅是固定基座桌面。
- 与 [`unitree_rl_lab`](./unitree-rl-lab.md) 同属 Isaac 生态，但本仓侧重 **遥操作采数与任务场景**，不是 locomotion RL 环境包。

## 核心原理

| 要点 | 说明 |
|------|------|
| 通信 | 启动后收发与真机相同 DDS 主题 |
| 机型×末端 | G1-29dof × Dex1/Dex3/Inspire；H1-2 × Inspire 等 |
| 任务示例 | `Isaac-PickPlace-Cylinder-G129-*-Joint`、`Isaac-PickPlace-RedBlock-...` |
| 权重 | 仓库提供的权重**仅供仿真测试** |

## 工程实践

1. 硬件按 Isaac Lab 官方推荐；已测 GPU：RTX 3080/3090/4090；50 系需 Isaac Sim 5.0.0（上游说明）。
2. 与 [`xr_teleoperate`](./xr-teleoperate.md) 联用采数；DDS 用法对照 SDK2 Python G1 示例与 Dex3 示例。
3. 同网有真机时必须隔离，防止仿真节点误控实机。
4. 首启加载慢属预期；主视角需在 Isaac 中切到 PerspectiveCamera。

## 局限与风险

- 仿真权重 ≠ 真机可部署策略。
- GPU/Isaac 版本矩阵变化快，以 README「Important Notes」为准。
- 执行器（夹爪 vs 多指手）与 [`unitree_lerobot`](./unitree-lerobot.md) 转换脚本必须一致。

## 源码运行时序图

官方实现 [unitreerobotics/unitree_sim_isaaclab](https://github.com/unitreerobotics/unitree_sim_isaaclab)（短 SHA `e30c25b`）：基于 Isaac Lab 的仿真程序，主入口 `sim_main.py`。主线程循环中 `RobotController.step()` 向环境下发动作；动作来自 `action_provider`（默认 `action_source=dds`，即 `DDSActionProvider` 读取 DDS 缓存的 `rt/lowcmd`）。关节状态由观测函数 `tasks/common_observations/g1_29dof_state.py` 中的 `get_robot_boy_joint_states` 节流后发布到 `rt/lowstate`，与真机使用同一套 DDS 话题。

入口：`python sim_main.py --device cpu --enable_cameras --task Isaac-PickPlace-Cylinder-G129-Dex1-Joint --enable_dex1_dds --robot_type g129`

```mermaid
sequenceDiagram
    autonumber
    actor U as 控制程序 / xr_teleoperate
    participant MAIN as sim_main.py<br/>main()
    participant ENV as Isaac Lab 环境<br/>parse_env_cfg / gym.make（tasks/）
    participant IMG as run_isaacsim_server<br/>teleimager（外部）
    participant DDS as dds_create.py<br/>G1RobotDDS / Dex1 / ResetPose / SimState
    participant AP as DDSActionProvider<br/>action_provider/action_provider_dds.py
    participant RC as RobotController<br/>layeredcontrol/robot_control_system.py
    participant OBS as 观测函数<br/>tasks/common_observations/g1_29dof_state.py
    U->>MAIN: python sim_main.py --task ... --enable_dex1_dds --robot_type g129
    MAIN->>MAIN: AppLauncher 启动 Isaac Sim，再导入 layeredcontrol 与 tasks
    MAIN->>ENV: parse_env_cfg(task, device, num_envs=1)；gym.make(task, cfg).unwrapped
    MAIN->>ENV: 设置渲染、物理步长等参数；env.sim.reset()；env.reset()
    MAIN->>IMG: run_isaacsim_server()（相机图像服务）
    MAIN->>DDS: create_dds_objects(args, env)
    MAIN->>AP: create_action_provider(env, args)（action_source=dds）
    MAIN->>RC: RobotController(env, config)；set_action_provider(provider)
    MAIN->>RC: start()（启动 action provider 等异步组件）
    loop 主线程循环：simulation_app.is_running() 且 controller.is_running
        MAIN->>ENV: env.scene.get_state() → sim_state_to_json
        MAIN->>DDS: sim_state_dds.write_sim_state_data(sim_state)
        MAIN->>DDS: reset_pose_dds.get_reset_pose_command()（按需复位）
        MAIN->>RC: step()
        RC->>AP: get_action(env)
        AP->>DDS: robot_dds.get_robot_command()（rt/lowcmd 缓存）；夹爪与灵巧手命令
        AP-->>RC: action（本帧无新命令则沿用上一帧）
        RC->>ENV: env.step(action)（Isaac Lab 内部未读）
        ENV->>OBS: 计算观测（含 get_robot_boy_joint_states）
        OBS->>DDS: G1RobotDDS.write_robot_state → rt/lowstate（节流发布）
        RC->>RC: 按控制频率节流 sleep
    end
    U->>DDS: 发布 rt/lowcmd（与真机相同的话题）
```

图依据：unitree_sim_isaaclab@e30c25b，读过 `README.md`（运行命令与参数）、`sim_main.py`（`main` 主流程与主循环，部分）、`layeredcontrol/robot_control_system.py`（`start`、`step`，部分）、`action_provider/action_provider_dds.py`（`get_action`，部分）、`dds/dds_create.py`（部分）、`dds/g1_robot_dds.py`（订阅与发布话题，部分）、`tasks/common_observations/g1_29dof_state.py`（`write_robot_state` 调用位置，部分）。未运行代码；Isaac Lab 的 `env.step` 与观测管理器内部未读，图中"观测计算触发发布"为依据调用位置的推断。`action_source` 的其它取值（file / trajectory / policy / replay）未画。

## 关联页面

- [xr_teleoperate](./xr-teleoperate.md)
- [unitree_lerobot](./unitree-lerobot.md)
- [unitree_rl_lab](./unitree-rl-lab.md)
- [Teleoperation](../tasks/teleoperation.md)
- [Unitree](./unitree.md)

## 参考来源

- [sources/repos/unitree_sim_isaaclab.md](../../sources/repos/unitree_sim_isaaclab.md)
- 上游：<https://github.com/unitreerobotics/unitree_sim_isaaclab>
- 源码核对（2026-11-02）：unitreerobotics/unitree_sim_isaaclab@e30c25b，读过 README.md、sim_main.py（部分）、layeredcontrol/robot_control_system.py（部分）、action_provider/action_provider_dds.py（部分）、dds/dds_create.py（部分）、dds/g1_robot_dds.py（部分）、tasks/common_observations/g1_29dof_state.py（部分）；时序图依据以上文件，未运行代码。

## 推荐继续阅读

- 仓内中文 README：`README_zh-CN.md`

