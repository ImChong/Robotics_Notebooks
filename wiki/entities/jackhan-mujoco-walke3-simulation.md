---
type: entity
title: Mujoco-WalkerE3-Simulation（Walker 泰山手柄仿真）
tags: [mujoco, humanoid, pytorch, gamepad, locomotion, jackhan-sdu, walke3]
summary: "MuJoCo 中加载 E3 场景与预训练 PyTorch 策略，用手柄发送平面速度指令，支持行走/跑步/扰动测试与相机跟踪；README 同时列出 MuJoCo 接触与静立打滑的已知局限。"
updated: 2026-06-05
status: complete
related:
  - ./jackhan-walke3-e3-ecosystem.md
  - ./jackhan-walke3-controller.md
  - ./mujoco.md
  - ../methods/reinforcement-learning.md
  - ../tasks/locomotion.md
  - ../tasks/stair-obstacle-perceptive-locomotion.md
sources:
  - ../../sources/repos/jackhan-mujoco-walke3-simulation.md
---

# Mujoco-WalkerE3-Simulation（Walker 泰山手柄仿真）

本仓库在 MuJoCo 中加载 E3 地形与预训练 PyTorch 策略，用手柄发送速度指令，支持行走、跑步与扰动测试模式。

**定位**：面向「可立即运行」的 MuJoCo 人形演示——仓库内置 **Walker 泰山** 预训练策略（`.pt`）、`e3.yaml` 与手柄标定脚本，强调速度指令跟踪、楼梯与扰动模式切换。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| MuJoCo | Multi-Joint dynamics with Contact | 接触丰富的刚体物理仿真引擎 |
| PD | Proportional–Derivative | 关节位置/阻抗底层控制，策略输出常为其 setpoint |
| Locomotion | Robot Locomotion | 足式/人形等无轮移动能力的总称 |

## 核心机制（工程切片）

- **控制回路**：配置中的 `simulation_dt`、`control_decimation` 与 PD `kps`/`kds` 共同决定仿真步长、控制频率与关节阻抗；观测/动作维度在 `e3.yaml` 中显式给出（README 示例为 72 维观测、21 维动作）。
- **输入层**：右摇杆平面速度、左摇杆偏航角速度；LB 循环模式；可选扰动模式下左摇杆映射到躯干外力。
- **可视化**：接触力向量、接触布尔、骨盆相机跟踪等开关与 README 中的按键表一一对应。

## 流程总览

```mermaid
flowchart TD
  cfg["e3.yaml<br/>模型 / 策略 / PD / 缩放"]
  gp["手柄输入<br/>标定 JSON"]
  obs["观测构造<br/>关节状态 + 指令等"]
  pol["PyTorch 策略<br/>motion_*.pt"]
  pd["PD 执行层<br/>kps / kds"]
  mj["MuJoCo 步进<br/>scene_terrain.xml"]

  cfg --> obs
  gp --> obs
  obs --> pol --> pd --> mj
  mj --> obs
```

## 源码运行时序图

以下时序图基于官方仓库 `JackHan-Sdu/Mujoco-WalkerE3-Simulation`（`main` 分支）核读的源码与配置梳理：`README_CN.md`、`configs/e3.yaml` 与主脚本 `deploy_mujoco_gamepad.py`。

```mermaid
sequenceDiagram
    autonumber
    participant MAIN as 部署主脚本<br/>deploy_mujoco_gamepad.py
    participant PAD as 手柄与扰动<br/>gamepad_utils / disturbance_utils
    participant POL as TorchScript 策略<br/>policy/motion_5000.pt
    participant MJ as MuJoCo 引擎与视图<br/>MjModel / MjData / viewer

    MAIN->>MAIN: 读取 configs/e3.yaml（simulation_dt=0.002, control_decimation=10）
    MAIN->>MJ: MjModel.from_xml_path("model/e3/scene_terrain.xml") 与 MjData(m)
    MAIN->>POL: torch.jit.load("policy/motion_5000.pt")
    MAIN->>PAD: init_gamepad(config) 与 set_disturbance_body_id(m)
    MAIN->>MJ: mujoco.viewer.launch_passive(m, d)

    loop 主仿真循环（viewer.is_running()）
        opt viewer_utils.reset_requested
            MAIN->>MJ: 恢复 initial_qpos/initial_qvel/initial_cmd，清空 xfrc_applied 与 reset_recording()
        end
        MAIN->>PAD: 50 Hz 调用 update_cmd_from_gamepad(cmd)
        MAIN->>PAD: apply_disturbance_force(m, d)（扰动模式下施加躯干外力）
        MAIN->>MJ: pd_control(target_dof_pos, d.qpos[7:], kps, ..., d.qvel[6:], kds) -> d.ctrl[:] = tau -> mujoco.mj_step(m, d)
        opt counter % control_decimation == 0 (50 Hz)
            MAIN->>MAIN: 读取 qpos/qvel 与 get_gravity_orientation(quat)，组装 72 维 obs
            MAIN->>POL: policy(obs_tensor)
            POL-->>MAIN: action (21-DoF)
            MAIN->>MAIN: target_dof_pos = action * action_scale + default_angles
            opt plot_utils.recording (5s 窗口)
                MAIN->>MAIN: 记录 tau/dqj 并在结束后调用 plot_data()
            end
        end
        MAIN->>MJ: viewer_utils.update_viewer_settings() 与 viewer.sync()
    end
```

- **关键复现路径**：执行 `python deploy_mujoco_gamepad.py e3.yaml`；仿真步进频率 500 Hz（`dt=0.002`），手柄指令采样与策略推理均按 `control_decimation=10`（50 Hz）驱动预训练 TorchScript 模型 `policy/motion_5000.pt` 输出 21 自由度目标关节角，再由底层 PD 控制器转化为执行器力矩。

## 常见误区或局限

- **README 已知问题**：楼梯首步可能穿模、长时间站立可能打滑——属于仿真接触与摩擦建模边界，不宜直接当作硬件结论。
- **手柄标定**：非罗技/北通品牌需运行 `calibrate_gamepad.py`，否则漂移会导致「策略正常但指令脏」的假阳性故障。

## 与其他页面的关系

- **[生态总览](./jackhan-walke3-e3-ecosystem.md)**：把本仓与数据集、控制器、FEAP 线放在同一张图里。
- **[WalkE3 控制器](./jackhan-walke3-controller.md)**：若要从「单机演示」升级到「仿真进程 + 控制进程」分离栈，可对照阅读。

## 参考来源

- [Mujoco-WalkerE3-Simulation 仓库归档](../../sources/repos/jackhan-mujoco-walke3-simulation.md)

## 关联页面

- [JackHan-Sdu WalkE3 / HumanoidE3 工具链生态](./jackhan-walke3-e3-ecosystem.md)
- [MuJoCo (物理引擎)](./mujoco.md)
- [Locomotion](../tasks/locomotion.md)

## 推荐继续阅读

- 上游仓库 README：<https://github.com/JackHan-Sdu/Mujoco-WalkerE3-Simulation>
