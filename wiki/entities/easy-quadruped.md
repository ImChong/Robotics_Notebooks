---
type: entity
project_id: easy-quadruped
code: https://github.com/Xzgz718/easy_quadruped
tags: [quadruped, open-source, stanford, mujoco, model-based-control, education, pupper]
status: complete
updated: 2026-11-22
original:
  - ../../sources/repos/easy_quadruped.md
related:
  - ./stanford-doggo-and-pupper.md
  - ./quadruped-robot.md
  - ./mujoco.md
  - ./legged-gym.md
  - ../concepts/gait-generation.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/repos/easy_quadruped.md
summary: "easy_quadruped 是 StanfordQuadruped（Pupper）控制栈的独立 fork：模型式 Trot 步态 + IK + MuJoCo 浮动机身闭环，附带任务调度与舵机标定，适合低成本四足控制入门。"
---

# easy_quadruped（Pupper 控制栈 + MuJoCo 闭环）

## 一句话定义

**easy_quadruped** 是在 [StanfordQuadruped](https://github.com/stanfordroboticsclub/StanfordQuadruped)（[Stanford Pupper](./stanford-doggo-and-pupper.md) 软件 lineage）基础上独立维护的二次开发仓库：保留 **步态调度、支撑/摆腿足端规划、行为状态机与逆运动学**，并补齐 **MuJoCo 浮动机身仿真、观测回填与可脚本化任务序列**，便于在不依赖大规模 RL 训练的前提下理解四足控制闭环。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| IK | Inverse Kinematics | 满足末端/姿态约束求解关节角的运动学逆解 |
| MuJoCo | Multi-Joint dynamics with Contact | 接触丰富的刚体物理仿真引擎 |
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略的范式 |
| PD | Proportional–Derivative | 关节位置/阻抗底层控制，策略输出常为其 setpoint |
| ROS 2 | Robot Operating System 2 | 机器人系统集成与通信的常用中间件 |
| VLM | Vision-Language Model | 视觉-语言多模态理解模型，VLA 的上游 |
| MJCF | MuJoCo XML Format | MuJoCo 的模型与场景描述格式 |
| Sim2Real | Simulation to Real | 把仿真中学到的策略迁移落地真机的工程主线 |
| DR | Domain Randomization | 训练时随机化仿真参数以提升跨域鲁棒迁移 |
| BOM | Bill of Materials | 物料清单，硬件零部件列表 |
| legged_gym | Legged Gym | 足式机器人 RL 训练的常用开源框架 |
| Isaac Gym | NVIDIA Isaac Gym | GPU 并行刚体仿真训练环境 |
| Locomotion | Robot Locomotion | 足式/人形等无轮移动能力的总称 |

## 为什么重要

- **与 RL 训练栈形成对照**：本库 [Legged Gym](./legged-gym.md)、[robot_lab](https://github.com/fan-ziqi/robot_lab) 等解决的是「如何在并行仿真里训策略」；easy_quadruped 展示的是「**时钟式步态 + 足端轨迹 + IK + 关节 PD**」这条经典模型控制链，对读懂四足底层控制更有直接性。
- **低成本教学载体**：对应 **早期 Pupper / StanfordQuadruped**（树莓派 + 舵机、模型步态），与当前 **[Pupper v3](./stanford-doggo-and-pupper.md)**（无刷 GIM4305 + ROS 2 monorepo + RL/VLM）不是同一代软件栈；公开快照把仿真入口收敛到 `run_floating_base.py`，降低上手摩擦。
- **MuJoCo 闭环样板**：`TaskScheduler → Controller → IK → PD → MuJoCo → State` 是小型四足在 [MuJoCo](./mujoco.md) 里做集成测试与调参的常见形态，可与 Menagerie / RL 环境区分使用场景。

> **非官方声明：** 维护方 Xzgz718 在 `NOTICE` 中明确本仓非 Stanford Student Robotics 官方发布；名称仅用于标明上游来源。

## 核心原理

`easy_quadruped` 以参数化相位步态调度 + 逆运动学（IK）+ 关节 PD 替代端到端强化学习策略：每个控制周期先从 MuJoCo 浮动机身读取躯干姿态、速度与触地状态回填 `State`，再由 `Controller` 按 `DEACTIVATED` / `REST` / `TROT` / `HOP` 状态机分派支撑相（`StanceController`）与摆动相（`SwingLegController`）足端目标位置，最后经 `four_legs_inverse_kinematics` 求解 12 个关节角送入 MuJoCo 位置/PD 执行器推进仿真。

### 控制架构概览

```mermaid
flowchart LR
  TS[TaskScheduler<br/>rest / trot 序列]
  CMD[Command<br/>速度 / 高度 / 姿态]
  GC[GaitController<br/>相位与接触模式]
  SC[StanceController]
  SW[SwingLegController]
  IK[逆运动学]
  PD[关节 PD 力矩]
  MJ[MuJoCo 浮动机身]
  OBS[观测回填 State]

  TS --> CMD
  CMD --> GC
  GC --> SC
  GC --> SW
  SC --> IK
  SW --> IK
  IK --> PD
  PD --> MJ
  MJ --> OBS
  OBS --> GC
```

**行为状态：** `REST`、`TROT`、`HOP` 等通过状态机切换；公开仿真以 **对角 Trot** 与静止 **Rest** 为主，`sim/task_scheduler.py` 支持段间平滑过渡步态与反馈增益（`vx`、`z_clearance`、`swing_time`、`attitude_kp` 等）。

## 公开快照包含什么

| 目录 | 职责 |
|------|------|
| `src/` | `Controller` 整合步态与足端规划；`Gaits` 计算接触模式 |
| `pupper/` | 机器人几何、`Config` 控制周期与步态 tick、舵机标定、硬件接口 |
| `sim/` | MJCF 生成、浮动机身闭环、`sim_robot` 观测桥接、任务调度 |
| `calibrate_servos.py` | 实机舵机零位标定 |

刻意未纳入：上游完整部署残留、本地 IDE/日志/课件草稿（见上游 README 说明）。

## 快速验证（仓库根目录）

```bash
pip install mujoco transforms3d numpy
python -m sim.build_floating_base_mjcf
python sim/run_floating_base.py --mode trot --duration 20
```

无界面任务序列示例：

```bash
python sim/run_floating_base.py --headless --duration 8 --task-sequence "rest:1.0,trot:4.0,rest"
```

更细参数与任务语法见上游 [`sim/README.md`](https://github.com/Xzgz718/easy_quadruped/blob/main/sim/README.md)。

## 源码运行时序图

图依据：`Xzgz718/easy_quadruped@main`（2026-11-22 核对），基于 `sim/run_floating_base.py` 与 `src/Controller.py` 源码整理，未在本地执行 MuJoCo viewer。

```mermaid
sequenceDiagram
  autonumber
  actor User as 用户 CLI
  participant Main as sim/run_floating_base.py:main
  participant MJCF as build_floating_base_mjcf.py
  participant Runner as run_loop
  participant Obs as SimObservationInterface / SimIMU
  participant Cmd as command_source / apply_feedback
  participant Ctrl as src/Controller.py:Controller
  participant HW as SimHardwareInterface (MuJoCo)
  User->>Main: --mode / --task-sequence / --kp --kd
  Main->>MJCF: 按需 build_mjcf(pupper_floating.xml)
  Main->>Runner: 加载 MjModel/MjData 并启动闭环
  loop 每个控制 tick
    Runner->>Obs: sync_state() + read_orientation()
    Obs-->>Runner: 更新 State（关节、触地、姿态）
    Runner->>Cmd: apply_step_config() + get_command()
    Cmd-->>Runner: Command（含可选 attitude/velocity 反馈）
    Runner->>Ctrl: run(state, command) → step_gait() + IK
    Ctrl-->>Runner: state.joint_angles
    Runner->>HW: set_actuator_postions() + step()
  end
  Runner-->>User: 输出位移、足端误差与 |roll|/|pitch| 统计
```

## 工程实践

1. **先建模后无头验证：** 修改几何或执行器参数后先加 `--rebuild` 重新生成 `pupper_floating.xml`，再以 `--headless --duration 8 --task-sequence "rest:1.0,trot:4.0,rest"` 跑通闭环与稳定检查（`qpos[2] >= 0.05` 且无非有限值）。
2. **调参边界对齐：** 默认 `--kp 24.0`、`--kd 2.2`、`--torque-limit 3.0` 与 `--z-clearance 0.03`、`--swing-time 0.11` 绑定于 Pupper 几何尺度，迁移到不同负载或更长步态周期时应先固定无反馈基线再开启 `--attitude-kp` / `--velocity-kp`。
3. **版本鲜度确认（2026-11-22 复核）：** 上游仍聚焦早期 StanfordQuadruped 模型控制与 MuJoCo 闭环入口，未合入 Pupper v3 的 ROS 2 / GIM4305 / RL-VLM 代码路径，选型时与 [Stanford Pupper v3](./stanford-doggo-and-pupper.md) 保持明确区分。

## 局限与风险

- **无动力学全身优化与地形感知：** 采用运动学步态 + IK 与可选轻量姿态/速度反馈，不含 WBC、MPC 或视觉地形感知，复杂崎岖地形鲁棒性有限。
- **单机非并行仿真：** 仅面向单机器人 MuJoCo 教学与调参，不提供 Isaac Gym / Isaac Lab 大规模 GPU 并行环境。
- **未在本地真机实测：** 实机部署需另外完成树莓派 GPIO PWM 接口适配与 `calibrate_servos.py` 零位标定；本页结论基于仓库源码阅读，未对真实 Pupper 硬件开展联调。

## 与 Stanford Pupper / RL 生态的关系

| 路线 | 代表 | 本库角色 |
|------|------|----------|
| 开源硬件 + 模型控制 | [Stanford Doggo / Pupper](./stanford-doggo-and-pupper.md) | easy_quadruped 承接 **软件控制与 MuJoCo 仿真** 切片 |
| 并行 RL + sim2real | [Legged Gym](./legged-gym.md)、Isaac 系 | 互补：学 reward/DR 之前可先弄清步态相位与足端轨迹 |
| 步态理论 | [Gait Generation](../concepts/gait-generation.md) | 仓库内 **Trot 对角支撑** 是参数化步态调度器的具体实现 |

## 常见误区

1. **误当作官方 Stanford 发布** — 须读 `NOTICE`；硬件与 BOM 仍以社区 [StanfordPupper](https://github.com/stanfordroboticsclub/StanfordPupper) 等为准。
2. **与 legged_gym 混为一谈** — 本仓无 Isaac Gym 并行环境，也不训练神经网络 locomotion 策略。
3. **期待完整 ROS2/Gazebo 镜像** — 公开快照以 MuJoCo `run_floating_base` 为主；GitHub 描述中的 ros2/gazebo 可能指上游或其它分支，以仓库当前树为准。

## 关联页面

- [Stanford Doggo / Pupper](./stanford-doggo-and-pupper.md)
- [四足机器人](./quadruped-robot.md)
- [MuJoCo](./mujoco.md)
- [Legged Gym](./legged-gym.md)
- [Gait Generation（步态生成）](../concepts/gait-generation.md)
- [Locomotion 任务](../tasks/locomotion.md)

## 推荐继续阅读

- 上游官方 lineage：[StanfordQuadruped](https://github.com/stanfordroboticsclub/StanfordQuadruped)
- 本 fork 仓库：<https://github.com/Xzgz718/easy_quadruped>
- Pupper 社区硬件文档（交叉核对）：[StanfordPupper](https://github.com/stanfordroboticsclub/StanfordPupper)

## 参考来源

- [easy_quadruped（sources 归档）](../../sources/repos/easy_quadruped.md)
- 上游 MIT 项目：Stanford Student Robotics, [StanfordQuadruped](https://github.com/stanfordroboticsclub/StanfordQuadruped)（2020）
