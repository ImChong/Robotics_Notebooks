# robot_lab

> 来源归档

- **标题：** robot_lab
- **类型：** repo
- **来源：** fan-ziqi（GitHub 个人项目）
- **链接：** https://github.com/fan-ziqi/robot_lab
- **Stars / Forks：** ~1.6k / 183（2026-04）
- **入库日期：** 2026-04-20
- **一句话说明：** 基于 IsaacLab 的机器人 RL 扩展训练框架，支持 26+ 机器人（四足 / 轮足 / 人形），可独立于 IsaacLab 核心仓库开发。
- **沉淀到 wiki：** 待定

---

## 核心定位

robot_lab 是一个建立在 NVIDIA **IsaacLab** 之上的 RL 扩展库，允许用户在隔离的仓库中开发机器人任务，不污染 IsaacLab 核心代码。

依赖版本（截至 2026-04）：
- Isaac Lab 2.3.2 / Isaac Sim 5.1.0
- Python 3.11，Linux / Windows

---

## 支持机器人（26+）

| 类别 | 典型型号 |
|------|---------|
| 四足（Quadruped） | Anymal D、Unitree Go2 / A1 / B2、Deeprobotics Lite3 |
| 轮足（Wheeled） | Unitree Go2W / B2W、Deeprobotics M20、DDTRobot Tita |
| 人形（Humanoid） | Unitree G1 / H1、FFTAI GR1T1 / GR1T2、Booster T1、RobotEra Xbot |

---

## 仓库结构

```
robot_lab/
├── source/robot_lab/robot_lab/
│   ├── assets/           # 机器人资产定义（URDF/USD + 配置）
│   └── tasks/
│       ├── manager_based/
│       │   ├── locomotion/velocity/   # 速度指令跟踪 locomotion 任务
│       │   └── beyondmimic/           # 人形动作模仿（BeyondMimic）
│       └── direct/                    # Direct RL 任务
├── scripts/              # 训练 / 评估脚本
├── docker/               # Docker 部署配置
└── docs/imgs/            # 文档图片
```

---

## 支持的 RL 框架

| 框架 | 用途 |
|------|------|
| RSL-RL | 主训练框架（单 / 多 GPU） |
| CusRL | 实验性替代训练器 |
| SKRL | AMP Dance 等特殊任务 |
| BeyondMimic | 人形机器人动作模仿 |

---

## 关键特性

- **Gym 注册规范：** `RobotLab-Isaac-[Task]-[Environment]-[Robot]-v[X]`
- **模块化扩展：** 新机器人只需添加资产配置 + 任务配置 + 环境注册
- **多 GPU 分布式训练**、TensorBoard 可视化、视频录制
- **AMP Dance**：用 SKRL 驱动的动作风格迁移示例
- **BeyondMimic**：人形机器人动作模仿，manager_based 任务

---

## 与本项目其他资料的关系

| 资料 | 关系 |
|------|------|
| [isaac_gym_isaac_lab.md](isaac_gym_isaac_lab.md) | robot_lab 建立在 IsaacLab 之上 |
| [legged_gym.md](legged_gym.md) | 同类训练框架，legged_gym 是前身生态参考 |
| [unitree.md](unitree.md) | Go2、G1、H1 等 Unitree 机器人是 robot_lab 主要支持对象 |
