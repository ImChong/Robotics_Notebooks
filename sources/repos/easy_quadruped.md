# easy_quadruped

> 来源归档

- **标题：** easy_quadruped
- **类型：** repo
- **来源：** Xzgz718（GitHub 个人维护，StanfordQuadruped 二次开发）
- **链接：** https://github.com/Xzgz718/easy_quadruped
- **上游：** https://github.com/stanfordroboticsclub/StanfordQuadruped（MIT，Stanford Student Robotics，2020）
- **Stars：** ~2（2026-05，GitHub API）
- **入库日期：** 2026-05-24
- **一句话说明：** 基于 Stanford Pupper 控制栈的独立 fork：保留步态/支撑/摆腿/状态机与逆运动学核心，补齐 MuJoCo 浮动机身闭环仿真、任务调度器与舵机标定脚本，适合低成本四足模型控制入门与调参。
- **沉淀到 wiki：** 是 → [`wiki/entities/easy-quadruped.md`](../../wiki/entities/easy-quadruped.md)

---

## 定位与声明

- 本仓库为 **独立维护的 derivative work**，非 Stanford Student Robotics 官方发布。
- 公开快照刻意精简：聚焦 `src/` 控制器、`pupper/` 运动学与硬件抽象、`sim/` MuJoCo 链路，不含上游完整镜像、本地 IDE 配置与个人草稿。
- 许可证：上游 MIT 保留于 `LICENSE`；二次开发部分默认 MIT（见 `NOTICE`）。

---

## 技术栈

| 层级 | 内容 |
|------|------|
| 控制 | 对角 **Trot**、**Rest**、**Hop** 行为状态机；`GaitController` 相位调度；`StanceController` / `SwingLegController` 足端轨迹 |
| 运动学 | `pupper/Kinematics.py` 逆运动学；`Config.py` 几何与 PWM/舵机标定参数 |
| 仿真 | MuJoCo `freejoint` 浮动机身 + 关节 PD 力矩；`SimObservationInterface` 回填 `State`（姿态、速度、触地、关节） |
| 调度 | `TaskScheduler` 解析 `rest` / `trot` 任务序列，支持段间参数平滑过渡（`vx`、`height`、`z_clearance`、`swing_time` 等） |
| 实机 | `calibrate_servos.py`、`HardwareInterface`、树莓派 GPIO PWM（Pupper 主线，非 `woofer/`） |

**最少依赖：** `mujoco`、`transforms3d`、`numpy`

---

## 仓库结构（公开快照）

```
easy_quadruped/
├── src/
│   ├── Controller.py          # 主控制器：步态 + IK + 行为状态
│   ├── Gaits.py               # 步态相位与接触模式
│   ├── StanceController.py
│   ├── SwingLegController.py
│   ├── State.py / Command.py
│   └── JoystickInterface.py   # 实机手柄（仿真侧多用 TaskScheduler 替代）
├── pupper/
│   ├── Config.py              # 控制周期、步态 tick、几何尺寸
│   ├── Kinematics.py
│   ├── HardwareInterface.py
│   └── ServoCalibration.py
├── sim/
│   ├── run_floating_base.py   # 主仿真入口
│   ├── sim_robot.py           # MuJoCo ↔ State 桥接
│   ├── task_scheduler.py
│   └── build_floating_base_mjcf.py
└── calibrate_servos.py
```

---

## 仿真闭环（公开版主路径）

```text
TaskScheduler → Controller → IK → PD torque → MuJoCo step → Observation → State
```

典型命令：

```bash
python -m sim.build_floating_base_mjcf
python sim/run_floating_base.py --mode trot --duration 20
python sim/run_floating_base.py --headless --duration 8 --task-sequence "rest:1.0,trot:4.0,rest"
```

任务序列语法：`mode[:duration][@key=value;...]`，仅支持 `rest` 与 `trot`；详见仓库 `sim/README.md`。

---

## 与 RL 训练栈的区别

| 维度 | easy_quadruped | legged_gym / robot_lab 等 |
|------|----------------|---------------------------|
| 范式 | **模型式步态 + 足端轨迹 + IK + PD** | **并行仿真 + PPO 等 RL** |
| 目标 | 理解 Pupper 级四足控制结构与 MuJoCo 闭环调参 | 大规模域随机化与 sim2real 策略训练 |
| 算力 | 单机 MuJoCo，数千行 Python | GPU 并行 Isaac Gym / Lab |

二者互补：前者适合「控制栈长什么样」，后者适合「策略怎么训出来」。

---

## 与本项目其他资料的关系

| 资料 | 关系 |
|------|------|
| [stanford-doggo-and-pupper 微信策展](../../wiki/entities/stanford-doggo-and-pupper.md) | 上游硬件与社区叙事 |
| [legged_gym.md](legged_gym.md) | 四足 RL 工程范式对照 |
| [mujoco.md](../../wiki/entities/mujoco.md) | 仿真底座 |
| [gait-generation.md](../../wiki/concepts/gait-generation.md) | Trot 相位调度与参数化步态实例 |

---

## 对 wiki 的映射

| 主题 | 目标页 |
|------|--------|
| 实体与架构 | `wiki/entities/easy-quadruped.md` |
| 四足平台语境 | 交叉更新 `wiki/entities/quadruped-robot.md`、`wiki/entities/stanford-doggo-and-pupper.md` |
| 步态概念 | `wiki/concepts/gait-generation.md` |
| 仿真索引 | `references/repos/simulation.md` |
