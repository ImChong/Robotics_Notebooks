# Isaac Lab Cartpole 环境族（Isaac-Cartpole-v0 等）

- **标题：** Isaac Lab Available Environments / Quickstart — Cartpole
- **类型：** site / 官方文档 + 源码核对
- **环境总表：** <https://isaac-sim.github.io/IsaacLab/main/source/overview/environments.html>
- **Quickstart：** <https://isaac-sim.github.io/IsaacLab/main/source/setup/quickstart.html>
- **仓库：** <https://github.com/isaac-sim/IsaacLab>（已开源，BSD-3-Clause）
- **配套仓库归档：** [`sources/repos/isaac_lab.md`](../repos/isaac_lab.md)
- **入库日期：** 2026-08-16
- **一句话说明：** NVIDIA Isaac Lab 把经典 cart-pole 做成 GPU 并行 RL 教学任务：manager-based `Isaac-Cartpole-v0` 与 direct `Isaac-Cartpole-Direct-v0`，并派生 RGB/Depth/冻结视觉编码器变体。
- **代码：** 已开源 → manager cfg [`cartpole_env_cfg.py`](https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/isaaclab_tasks/manager_based/classic/cartpole/cartpole_env_cfg.py)；direct env [`cartpole_env.py`](https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/isaaclab_tasks/direct/cartpole/cartpole_env.py)；资产 [`cartpole.py`](https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_assets/isaaclab_assets/robots/cartpole.py)
- **沉淀到 wiki：** 是 → [`wiki/concepts/cartpole.md`](../../wiki/concepts/cartpole.md)

---

## 官方环境 ID（文档表，main 分支）

| ID | 工作流 | 说明 |
|----|--------|------|
| `Isaac-Cartpole-v0` | Manager Based | 移动小车使杆保持向上；经典 cartpole |
| `Isaac-Cartpole-Direct-v0` | Direct | 同一任务的单类 `DirectRLEnv` 实现 |
| `Isaac-Cartpole-RGB-v0` / `Isaac-Cartpole-Depth-v0` | Manager | 感知输入；需 `--enable_cameras` |
| `Isaac-Cartpole-RGB-Camera-Direct-v0` / `Isaac-Cartpole-Depth-Camera-Direct-v0` | Direct | 同上 |
| `Isaac-Cartpole-RGB-ResNet18-v0` / `Isaac-Cartpole-RGB-TheiaTiny-v0` | Manager | 冻结预训练视觉编码器特征；需 `--enable_cameras` |

文档描述原文：*Move the cart to keep the pole upwards in the classic cartpole control*。

支持的训练后端（状态观测版）：**rl_games / rsl_rl / skrl / sb3** 的 PPO。相机版后端子集更窄（以文档表为准）。

## 源码核对（manager-based `Isaac-Cartpole-v0`，2026-08-16）

`gym.register(id="Isaac-Cartpole-v0", entry_point="isaaclab.envs:ManagerBasedRLEnv", ...)`。

| 项 | 源码值 |
|----|--------|
| 并行数 | `num_envs=4096`，`env_spacing=4.0` |
| 仿真步长 | `sim.dt = 1/120`，`decimation = 2` → 控制约 60 Hz |
| 回合时长 | `episode_length_s = 5` |
| 动作 | `JointEffortAction`，关节 `slider_to_cart`，`scale=100.0` N（连续力，不是 Gym 的左右离散推） |
| 观测 | `joint_pos_rel` + `joint_vel_rel` 拼接（相对默认位姿的关节位置/速度） |
| 奖励 | `alive` +1；`is_terminated` −2；杆角相对 0 的 L2 −1；小车速度 L1 −0.01；杆角速度 L1 −0.005 |
| 终止 | 超时；小车关节越出 **±3.0 m**。Manager 版 **不** 因杆角单独 done（杆倒用奖励 shaping） |
| Reset | 小车位置 ±1.0 m、速度 ±0.5；杆位置/速度 ±0.25π |

## Direct 工作流差异（`Isaac-Cartpole-Direct-v0`）

- 观测拼接顺序：**杆位置、杆速度、小车位置、小车速度**（与 Gymnasium 的「车、车速、杆、杆速」不同）。
- 终止：`|cart| > 3.0` **或** `|pole| > π/2`，外加 5 s 超时。
- 奖励尺度与 manager 版同名：`rew_scale_alive=1`、`terminated=-2`、`pole_pos=-1`、`cart_vel=-0.01`、`pole_vel=-0.005`。
- `action_scale = 100` N；`action_space = 1`（连续）。

## 资产与执行器（`CARTPOLE_CFG`）

- USD：`ISAACLAB_NUCLEUS_DIR/Robots/Classic/Cartpole/cartpole.usd`
- 关节：`slider_to_cart`（车）、`cart_to_pole`（杆）
- **Implicit actuator**：车 `stiffness=0`、`damping=10`；杆 `stiffness=0`、`damping=0`；`effort_limit_sim=400` N
- 初始根位置 `(0, 0, 2)` m（相对地面抬起，便于并排 clone）

## 官方运行入口（文档）

```bash
# 零动作冒烟
./isaaclab.sh -p scripts/environments/zero_agent.py --task Isaac-Cartpole-Direct-v0 --num_envs 128

# RSL-RL 训练（文档示例用 Direct；v0 manager 同形换 task 名）
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Cartpole-v0 --num_envs 4096
```

Quickstart 还给出 Newton MJWarp / PhysX 后端切换与相机任务 `presets=rgb` 示例。列出全部注册环境：`python scripts/environments/list_envs.py`。

## 开源核查（步骤 2.5）

- 项目页即 Isaac Lab 文档站；Code 为官方仓 [isaac-sim/IsaacLab](https://github.com/isaac-sim/IsaacLab)。
- **已开源、可运行**：任务 cfg、Direct 环境类、PPO yaml、USD 资产路径均在仓内。运行依赖 Isaac Sim / 文档所述 physics backend，不是 `pip install` 单文件。

## 对 wiki 的映射

- [Cartpole 问题](../../wiki/concepts/cartpole.md)
- [Isaac Lab](../../wiki/entities/isaac-lab.md)
- [Implicit / Explicit 执行器建模](../../wiki/concepts/implicit-explicit-actuator-modeling.md)
- [Gymnasium](../../wiki/entities/gymnasium.md) — Lab 用 Gymnasium `gym.register` 挂 id，但物理与 MDP 数字不同

## 为什么值得保留

- `Isaac-Cartpole-v0` 是 Lab 文档 Quickstart 的默认教学任务，也是从 CPU 玩具 CartPole 跨到 **GPU 并行 + manager MDP** 的最小台阶。
- 与 Gymnasium 同名「CartPole」但动作连续、奖励 shaping、终止阈值、观测顺序均不同；混用 checkpoint / 超参会静默失败。
