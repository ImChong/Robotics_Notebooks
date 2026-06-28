# Gymnasium

> 来源归档

- **标题：** Gymnasium — 单智能体强化学习环境 API 标准与参考环境集合
- **类型：** repo / 官方文档
- **来源：** Farama Foundation
- **链接：** https://gymnasium.farama.org/
- **代码仓库：** https://github.com/Farama-Foundation/Gymnasium
- **前身：** OpenAI Gym（Gymnasium 为社区维护的延续 fork）
- **入库日期：** 2026-06-28
- **一句话说明：** 定义单智能体 RL 环境与训练代码之间的标准 Python 接口（`make` / `reset` / `step` / `render`），并附带经典控制、MuJoCo、Atari 等参考环境；多智能体场景由姊妹项目 PettingZoo 覆盖。
- **沉淀到 wiki：** 是 → [`wiki/entities/gymnasium.md`](../wiki/entities/gymnasium.md)

---

## 官方定位（文档摘录）

- **API 标准**：`gymnasium.Env` 封装 MDP 式交互；核心方法为 `reset()`、`step()`、`render()`、`close()`。
- **与旧版 Gym 的关键差异（v0.25+ / v0.26+）**：
  - `reset()` 返回 `(observation, info)` 元组；
  - `step()` 用 `terminated`（任务内终止）与 `truncated`（时间限制等 MDP 外截断）取代单一 `done`。
- **空间约定**：`action_space` / `observation_space` 基于 `gymnasium.spaces`（`Box`、`Discrete`、`Dict`、`Tuple` 等）。
- **默认包装**：`gymnasium.make()` 通常自动叠加 `TimeLimit`、`OrderEnforcing`、`PassiveEnvChecker` 等 Wrapper。
- **并行训练入口**：`gymnasium.make_vec()` 创建向量化环境（与 Isaac Gym 类 GPU 物理并行不同，侧重 API 层批量 `step`）。
- **自定义环境**：继承 `gym.Env`，实现 `reset` / `step`，可用 `gym.register` 注册后用 `make` 实例化；`gymnasium.utils.env_checker.check_env` 用于调试。

## 内置参考环境（可选 extra 安装）

文档与注册表涵盖多类任务族，机器人相关常见入口包括：

- **classic-control**：CartPole、Pendulum、MountainCar 等教学基准；
- **mujoco**：Ant、HalfCheetah、Humanoid 等连续控制（依赖 MuJoCo）；
- **box2d**：LunarLander 等；
- **atari**：离散动作游戏基准（算法验证常用，非机器人本体）。

完整列表以官方 `pprint_registry()` 与文档为准。

## 为什么值得保留

- **算法与仿真解耦**：Stable-Baselines3、CleanRL 等主流 RL 库以 Gymnasium API 为默认环境契约；读懂接口即可迁移到 MuJoCo / PyBullet / 自研仿真。
- **基准可比性**：经典控制与 MuJoCo 域长期用于 PPO、SAC 等算法横向对比；与 [dm_control](../wiki/entities/dm-control.md) 等「另一套约定更统一的 MuJoCo 基准」并行存在。
- **机器人项目常见用法**：环境实现层（如 [gym-pybullet-drones](../wiki/entities/gym-pybullet-drones.md)、各 sim 的 Gym 封装）对齐本 API，训练 loop 与论文复现更一致。
- **维护状态**：OpenAI 停止维护 Gym 后，Farama Foundation 接手；新代码应使用 `import gymnasium as gym`，旧 Gym 环境需按官方 migration guide 适配 `terminated`/`truncated` 语义。

## 对 wiki 的映射

- [gymnasium](../../wiki/entities/gymnasium.md) — 实体页（API 层 / 基准环境注册表）
- 交叉：[mujoco](../../wiki/entities/mujoco.md)、[dm-control](../../wiki/entities/dm-control.md)、[reinforcement-learning](../../wiki/methods/reinforcement-learning.md)、[gym-pybullet-drones](../../wiki/entities/gym-pybullet-drones.md)
