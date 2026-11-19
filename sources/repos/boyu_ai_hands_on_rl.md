# boyu-ai/Hands-on-RL

> 来源归档

- **标题：** Hands-on Reinforcement Learning（动手学强化学习 / 蘑菇书）
- **类型：** repo
- **链接：** <https://github.com/boyu-ai/Hands-on-RL>
- **在线阅读（推荐）：** <https://hrl.boyuai.com/>
- **入库日期：** 2026-05-21（2026-11-19 全仓代码 notebook 通读复核）
- **核对范围：** 2026-11-19 通读仓库 [`README.md`](https://github.com/boyu-ai/Hands-on-RL/blob/main/README.md)、共用工具模块 [`rl_utils.py`](https://github.com/boyu-ai/Hands-on-RL/blob/main/rl_utils.py) 与全部 20 个 Jupyter Notebook（`第2章-多臂老虎机问题.ipynb` 至 `第21章-多智能体强化学习进阶.ipynb`；仓库从第 2 章开始提供 `.ipynb`，无独立的第 1 章 `.ipynb` 文件，在线站点另含第 1 章文本导论）；未在本地安装 `gym==0.18.3` 运行训练脚本。
- **通读：** 已完成
- **一句话说明：** 上海交通大学 APEX 实验室主导的**中文**强化学习开源教材与 Jupyter 代码仓：用统一的 `rl_utils.py` 骨架串联表格型 RL、深度 RL（DQN/TRPO/PPO/DDPG/SAC）与前沿专题（IL/MPC/MBPO/CQL/HER/MARL）。
- **沉淀到 wiki：** 是 → [`wiki/entities/hands-on-rl-book.md`](../../wiki/entities/hands-on-rl-book.md)

## 核心摘录（源码通读核对）

### 1) 共用训练底座（`rl_utils.py`）

- **`ReplayBuffer`：** 基于 `collections.deque(maxlen=capacity)` 的经验回放池，提供 `add` / `sample` / `size`，返回 `(np.array(state), action, reward, np.array(next_state), done)`。
- **两条统一训练主循环：**
  - `train_on_policy_agent(env, agent, num_episodes)`：每个 episode 清空 `transition_dict`（`states` / `actions` / `next_states` / `rewards` / `dones`），循环执行 `action = agent.take_action(state)` → `next_state, reward, done, _ = env.step(action)`，整条轨迹结束后调用 `agent.update(transition_dict)`。
  - `train_off_policy_agent(env, agent, num_episodes, replay_buffer, minimal_size, batch_size)`：每步把转移写入 `replay_buffer`；一旦 `replay_buffer.size() > minimal_size`，每步采样 minibatch 调用 `agent.update(transition_dict)`。
- **广义优势估计（GAE）：** `compute_advantage(gamma, lmbda, td_delta)` 对 `td_delta[::-1]` 反向递推计算优势张量，供 TRPO 与 PPO 等章节调用。
- **Gym 接口版本：** `README.md` 注明需 `gym==0.18.3`，训练循环按旧版 Gym 4 元组 `(next_state, reward, done, _)` 解包 `env.step(action)`（与 Gymnasium 的 5 元组 `(obs, reward, terminated, truncated, info)` 不同）。

### 2) 全部 20 个代码章节目录（第 2–21 章 `.ipynb` 实核）

| 篇目 | 章节文件 | 核心类 / 算法实现 | 演示环境与备注 |
|------|---------|-------------------|---------------|
| 基础篇 | `第2章-多臂老虎机问题.ipynb` | `BernoulliBandit`、`Solver`、`EpsilonGreedy`、`DecayingEpsilonGreedy`、`UCB`、`ThompsonSampling` | 多臂老虎机探索-利用对照 |
| 基础篇 | `第3章-马尔可夫决策过程.ipynb` | `sample`、`MC`、状态-动作占用度量 `occupancy` | 离散马尔可夫链与 MDP 价值/占用度计算 |
| 基础篇 | `第4章-动态规划算法.ipynb` | `CliffWalkingEnv`、`PolicyIteration`、`ValueIteration` | 悬崖漫步网格与 `FrozenLake-v0` |
| 基础篇 | `第5章-时序差分算法.ipynb` | `Sarsa`、`nstep_Sarsa`、`QLearning` | 自建 `CliffWalking` 环境 |
| 基础篇 | `第6章-Dyna-Q算法.ipynb` | `DynaQ`、`DynaQ_CliffWalking` | 引入规划步的模型学习对照 |
| 进阶篇 | `第7章-DQN算法.ipynb` | `Qnet`、`DQN`、`ConvolutionalQnet` | `CartPole-v0`（含图像输入卷积版 `ConvolutionalQnet`） |
| 进阶篇 | `第8章-DQN改进算法.ipynb` | `VAnet`、`DQN`（支持 `VanillaDQN` / `DoubleDQN` / `DuelingDQN`） | `Pendulum-v0`（动作离散化） |
| 进阶篇 | `第9章-策略梯度算法.ipynb` | `REINFORCE` | `CartPole-v0` 蒙特卡洛策略梯度 |
| 进阶篇 | `第10章-Actor-Critic算法.ipynb` | `ActorCritic` | `CartPole-v0` 单步 TD 演员-评论员 |
| 进阶篇 | `第11章-TRPO算法.ipynb` | `TRPO`、`TRPOContinuous` | `CartPole-v0`（离散）与 `Pendulum-v0`（连续）；共轭梯度 + 线性搜索 |
| 进阶篇 | `第12章-PPO算法.ipynb` | `PPO`、`PPOContinuous` | `CartPole-v0`、`Pendulum-v0`；PPO-Clip 结合 `rl_utils.compute_advantage` |
| 进阶篇 | `第13章-DDPG算法.ipynb` | `DDPG` | `Pendulum-v0` 确定性策略梯度 + 软更新 |
| 进阶篇 | `第14章-SAC算法.ipynb` | `SAC`、`SACContinuous` | `Pendulum-v0`、`CartPole-v0`；双 Q、重参数化与自动温度调节 |
| 前沿篇 | `第15章-模仿学习.ipynb` | `BehaviorClone`、`GAIL` | `CartPole-v0` 行为克隆与生成对抗模仿学习 |
| 前沿篇 | `第16章-模型预测控制.ipynb` | `CEM`、`PETS` | `Pendulum-v0` 概率集成动力学模型 + 交叉熵打靶 |
| 前沿篇 | `第17章-基于模型的策略优化.ipynb` | `MBPO` | 动力学模型短时展开混合回放池训练 SAC |
| 前沿篇 | `第18章-离线强化学习.ipynb` | `CQL` | `Pendulum-v0` 保守 Q 学习抑制分布外高估 |
| 前沿篇 | `第19章-目标导向的强化学习.ipynb` | `DDPG` + `HER`、`WorldEnv` | 自建二维网格世界、`future` 事后经验回放重标注 |
| 前沿篇 | `第20章-多智能体强化学习入门.ipynb` | `IPPO` | `boyu-ai/ma-gym` 的 `Combat` 环境 |
| 前沿篇 | `第21章-多智能体强化学习进阶.ipynb` | `MADDPG` | `boyu-ai/multiagent-particle-envs` 的 `simple_adversary` 环境 |

### 3) 与机器人研究栈的边界

- 全仓代码围绕经典低维 `gym==0.18.3`（`CartPole-v0`、`Pendulum-v0`、`FrozenLake-v0`）、自建网格或轻量多智能体仓，**不包含** URDF/MJCF 刚体机器人、并行 GPU 仿真（如 Isaac Lab）或 Sim2Real 域随机化。
- 对本知识库而言，它最适合作为学习与教学底座，衔接 [`roadmap/depth-rl-locomotion.md`](../../roadmap/depth-rl-locomotion.md) 与 [`wiki/entities/hands-on-rl-book.md`](../../wiki/entities/hands-on-rl-book.md) 之后再过渡到 [Gymnasium](../../wiki/entities/gymnasium.md) 与机器人仿真器。

## 对 wiki 的映射

- [`wiki/entities/hands-on-rl-book.md`](../../wiki/entities/hands-on-rl-book.md) — 实体全书与源码结构解析
- [`wiki/methods/ppo.md`](../../wiki/methods/ppo.md) · [`wiki/methods/sac.md`](../../wiki/methods/sac.md) · [`wiki/comparisons/ppo-vs-sac.md`](../../wiki/comparisons/ppo-vs-sac.md)
- [`wiki/methods/model-based-rl.md`](../../wiki/methods/model-based-rl.md) · [`wiki/methods/mbpo.md`](../../wiki/methods/mbpo.md) · [`wiki/methods/her.md`](../../wiki/methods/her.md) · [`wiki/methods/marl.md`](../../wiki/methods/marl.md)

## 关联原始资料

- [`sources/sites/hrl-boyuai-hands-on-rl.md`](../sites/hrl-boyuai-hands-on-rl.md) — 在线书站与章节目录
- [`sources/courses/boyuai_hands_on_rl_elites_course.md`](../courses/boyuai_hands_on_rl_elites_course.md) — 伯禹平台免费视频课

## 推荐继续阅读（外部）

- Sutton & Barto, *Reinforcement Learning: An Introduction*
- [OpenAI Spinning Up](https://spinningup.openai.com)
