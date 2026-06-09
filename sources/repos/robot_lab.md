# robot_lab

> 来源归档

- **标题：** robot_lab
- **类型：** repo
- **来源：** fan-ziqi（GitHub 个人项目）
- **链接：** https://github.com/fan-ziqi/robot_lab
- **Stars / Forks：** ~1.75k / 183+（2026-06）
- **入库日期：** 2026-04-20
- **最近复核：** 2026-06-09
- **一句话说明：** 基于 IsaacLab 的机器人 RL 扩展训练框架，在独立仓库中注册 24+ 速度跟踪环境与 BeyondMimic/AMP 等实验任务，覆盖四足 / 轮足 / 人形多厂商机型。
- **沉淀到 wiki：** 是 → [`wiki/entities/robot-lab.md`](../../wiki/entities/robot-lab.md)

---

## 核心定位

**robot_lab** 是建立在 NVIDIA **IsaacLab** 之上的 RL 扩展库（*RL Extension Library for Robots*），允许用户在 **IsaacLab 核心仓库之外** 独立开发机器人资产、Gym 环境与训练脚本，不污染上游代码。

典型工作流：安装 Isaac Lab → `pip install -e source/robot_lab` → `python scripts/tools/list_envs.py` 验证注册 → 用 RSL-RL / CusRL / SKRL 训练 → 真机或 Gazebo 部署走配套项目 **[rl_sar](https://github.com/fan-ziqi/rl_sar)**。

---

## 版本依赖（README 矩阵）

| robot_lab | Isaac Lab | Isaac Sim |
|-----------|-----------|-----------|
| `main` / `v2.3.2` | `main` / `v2.3.2` | 4.5 / 5.0 / **5.1** |
| `v2.2.2` | `v2.2.1` | 4.5 / 5.0 |
| `v2.1.1` | `v2.1.1` | 4.5 |
| `v1.1` | `v1.4.1` | 4.2 |

当前 badge 默认：**Isaac Lab 2.3.2 · Isaac Sim 5.1 · Python 3.11 · Linux / Windows**。

---

## 支持机器人与环境（Velocity-Rough 主干表）

Gym 命名规范：`RobotLab-Isaac-[Task]-[Terrain]-[Robot]-v0`（`Rough` 可换 `Flat`）。

| 类别 | 机器人型号 | 环境 ID 示例 |
|------|-----------|-------------|
| **四足** | Anymal D、Unitree Go2/B2/A1、Deeprobotics Lite3、**Zsibot ZSL1**、**Magiclab MagicDog**、**Agibot D1** | `RobotLab-Isaac-Velocity-Rough-Unitree-Go2-v0` |
| **轮足** | Unitree Go2W/B2W、Deeprobotics M20、DDTRobot Tita、**Zsibot ZSL1W**、**Magiclab MagicDog-W** | `RobotLab-Isaac-Velocity-Rough-Unitree-Go2W-v0` |
| **人形** | Unitree G1/H1、FFTAI GR1T1/GR1T2、Booster T1、RobotEra Xbot、**Openloong Loong**、**RoboParty ATOM01**、**Magiclab MagicBot-Gen1/Z1** | `RobotLab-Isaac-Velocity-Rough-Unitree-G1-v0` |

除上表外，README 还列出 **BeyondMimic（G1 平地模仿）**、**G1 AMP Dance（SKRL Direct）**、**A1 Handstand**、**Anymal D 对称增广 / 蒸馏** 等实验任务。

---

## 仓库结构

```
robot_lab/
├── source/robot_lab/robot_lab/
│   ├── assets/              # 机器人 URDF/USD 与配置（如 unitree.py）
│   └── tasks/
│       ├── manager_based/
│       │   ├── locomotion/velocity/   # 速度指令跟踪
│       │   └── beyondmimic/           # 人形动作模仿
│       └── direct/                    # Direct RL（如 AMP Dance）
├── scripts/
│   ├── reinforcement_learning/   # rsl_rl / cusrl / skrl train & play
│   └── tools/                    # list_envs, zero/random agent, beyondmimic 工具链
├── docker/                       # 基于 isaac-lab-base 的 robot-lab 镜像
└── docs/imgs/
```

扩展新机器人时遵循 Isaac Lab 惯例：`assets/` 定义 embodiment → `tasks/.../config/<robot>/` 下 `flat_env_cfg.py` / `rough_env_cfg.py` + agent 配置 → `__init__.py` 里 `gym.register(...)`。

---

## 支持的 RL 框架与典型命令

| 框架 | 角色 | 训练入口 |
|------|------|---------|
| **RSL-RL** | 主训练器（单 / 多 GPU / 多节点） | `scripts/reinforcement_learning/rsl_rl/train.py --task=<TASK> --headless` |
| **CusRL** | 实验性替代 | `scripts/reinforcement_learning/cusrl/train.py` |
| **SKRL** | AMP Dance 等 Direct 任务 | `scripts/reinforcement_learning/skrl/train.py --algorithm AMP` |

调试：`zero_agent.py` / `random_agent.py` 验证环境 wiring；`play.py` 支持 `--keyboard` 单机体感控制、`--video` 录屏、checkpoint 恢复与 `--distributed` 多卡。

---

## BeyondMimic（G1）数据与工具链

1. 收集重定向动作（需遵守各数据集许可）：Unitree LAFAN1（HuggingFace）、KungfuBot sidekick、ASAP 庆祝动作、HuB 平衡动作等。
2. `scripts/tools/beyondmimic/csv_to_npz.py` — CSV → 含 body pose/vel/acc 的 npz。
3. `replay_npz.py` — Isaac Sim 回放。
4. 训练 / 评估：`RobotLab-Isaac-BeyondMimic-Flat-Unitree-G1-v0`。

---

## Sim2Real 与社区

- **真机 / Gazebo 部署**：官方 README 指向 **[rl_sar](https://github.com/fan-ziqi/rl_sar)**，与 robot_lab 训练产物衔接。
- **讨论区**：[GitHub Discussions](https://github.com/fan-ziqi/robot_lab/discussions)、[Discord](http://www.robotsfan.com/dc_robot_lab)。

---

## 与本项目其他资料的关系

| 资料 | 关系 |
|------|------|
| [isaac_gym_isaac_lab.md](isaac_gym_isaac_lab.md) | robot_lab 建立在 IsaacLab 之上 |
| [legged_gym.md](legged_gym.md) | 同类足式 RL 训练栈；legged_gym 为 Isaac Gym 时代工程范本 |
| [unitree.md](unitree.md) | Go2/G1/H1 等是 robot_lab 主力机型 |
| [openloong.md](../repos/openloong.md) | Openloong Loong 已注册 Velocity 环境 |
| [atom01_train.md](atom01_train.md) | RoboParty ATOM01 在 robot_lab 有 Rough 环境注册 |
