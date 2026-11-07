# boyu-ai/Hands-on-RL

> 来源归档

- **标题：** Hands-on Reinforcement Learning（动手学强化学习 / 蘑菇书）
- **类型：** repo
- **链接：** <https://github.com/boyu-ai/Hands-on-RL>
- **在线阅读（推荐）：** <https://hrl.boyuai.com/>
- **入库日期：** 2026-05-21（2026-11-07 补通读核对）
- **核对范围：** 2026-11-07 读完仓库 `README.md` 全文（含 gym==0.18.3 提示、notebook 渲染说明、书籍与视频课入口）；在线站点首页与章节目录（前言 + 基础篇 6 章 + 进阶篇 8 章 + 前沿篇 7 章 + 总结与展望）；并通过抓取摘要核对了四章正文：基础篇《马尔可夫决策过程》、进阶篇《PPO 算法》《SAC 算法》、前沿篇《目标导向的强化学习》（摘要由抓取工具生成，非逐字阅读，数学公式在页面上未渲染）。**没有读**其余章节的正文，没有下载或运行仓库中的 notebook。
- **通读：** 未完成
- **一句话说明：** 上海交通大学 APEX 实验室主导的**中文**强化学习开源教材与 Jupyter 代码仓：从表格型 RL 到 DQN/PPO/SAC，再到模仿学习、MPC、离线 RL 与 MARL；配套免费视频课与 Dumi 站点在线运行 notebook。
- **沉淀到 wiki：** 是 → [`wiki/entities/hands-on-rl-book.md`](../wiki/entities/hands-on-rl-book.md)

## 核对要点（2026-11-07）

- README：该系列"从强化学习的定义等基础讲起，一步步由浅入深"；每章是一个 Jupyter Notebook；GitHub 渲染有限，推荐在 hrl.boyuai.com 浏览，仓库提供纯代码版；运行 gym 环境报错时请 `pip install gym==0.18.3`；配套视频课在伯禹学习平台免费开放。
- 章节：基础篇（初探强化学习、多臂老虎机、马尔可夫决策过程、动态规划算法、时序差分算法、Dyna-Q 算法）；进阶篇（DQN、DQN 改进、策略梯度、Actor-Critic、TRPO、PPO、DDPG、SAC）；前沿篇（模仿学习、模型预测控制、基于模型的策略优化、离线强化学习、目标导向的强化学习、多智能体强化学习入门 / 进阶）。
- PPO 章：讲 PPO-Penalty 与 PPO-Clip（以 Clip 为主）；代码用 `CartPole-v0`（离散）与 `Pendulum-v0`（连续，奖励被缩放），使用旧版 gym API；代码里调用了 `compute_advantage(gamma, lmbda, td_delta)`，正文未点名 GAE。
- SAC 章：最大熵目标、软策略迭代（仅表格情形有收敛保证）、双 Q、重参数化、自动温度（目标熵 −动作维数；离散 CartPole 取 −1）；Pendulum-v0 回报从约 −1500 升至约 −170，CartPole-v0 约第 160 回合达到满分 200；规模均为小基准。
- 目标导向 RL 章：目标条件策略 + HER（DDPG 上用 "future" 策略、重标注比例 0.8）；环境是自建 2D 网格世界；有 HER 回报升至约 −3.6~−4.5，无 HER 在约 −31.5；每种配置只跑一次、无方差估计。
- MDP 章：马尔可夫过程 → 马尔可夫奖励过程 → MDP；$(S,A,P,R,\gamma)$；回报、V/Q、Bellman 期望与最优方程；占用度量及"两策略占用度量相同当且仅当策略相同"；用 6 状态马尔可夫链与 5 状态 MDP 举例。

## 为什么值得保留

- 国内读者学习 RL 的**默认中文入口**之一（社区称「蘑菇书」），与 Sutton & Barto、Spinning Up 形成互补。
- **进阶篇**覆盖 PPO、SAC、Actor-Critic，与本知识库人形 locomotion 主线的 Stage 0 算法栈直接对齐。
- **前沿篇**含模仿学习、MPC、基于模型的策略优化、离线 RL、多智能体 RL，便于和 `wiki/methods/`、`wiki/comparisons/` 交叉索引。

## 仓库结构（2026-05 检索）

| 资源 | 说明 |
|------|------|
| `第2章`–`第6章` | 基础篇 notebook：多臂老虎机、MDP、动态规划、TD、Dyna-Q |
| `第7章`–`第14章` | 进阶篇：DQN 及改进、策略梯度、Actor-Critic、TRPO、**PPO**、DDPG、**SAC** |
| `第15章`–`第21章` | 前沿篇：模仿学习、MPC、MBPO、离线 RL、目标导向 RL、MARL 入门/进阶 |
| `README.md` | 指向在线站、京东购书、伯禹视频课；**gym 建议 `pip install gym==0.18.3`** |

## 核心摘录

### 1) 三部分学习路径

- **要点：** 基础篇 = 有限状态/动作（tabular）；进阶篇 = 连续空间 + 神经网络；前沿篇 = IL / 模型类 / 离线 / 多智能体等扩展方向。
- **对 wiki 的映射：** [`wiki/entities/hands-on-rl-book.md`](../../wiki/entities/hands-on-rl-book.md)

### 2) PPO 与 SAC 实践章节

- **要点：** 第 12 章 PPO、第 14 章 SAC 含可运行实现，适合在进 IsaacGym/legged_gym 前建立 on-policy vs off-policy 直觉。
- **对 wiki 的映射：** [`wiki/methods/reinforcement-learning.md`](../../wiki/methods/reinforcement-learning.md)、[`wiki/comparisons/ppo-vs-sac.md`](../../wiki/comparisons/ppo-vs-sac.md)

### 3) 与机器人栈的边界

- **要点：** 教材以经典 Gym 环境与算法教学为主，**不覆盖**人形 WBC、sim2real、AMP 等机器人专有管线；读完应接本库 [`roadmap/depth-rl-locomotion.md`](../../roadmap/depth-rl-locomotion.md) Stage 1+。
- **对 wiki 的映射：** [`roadmap/depth-rl-locomotion.md`](../../roadmap/depth-rl-locomotion.md)

## 关联原始资料

- [`sources/sites/hrl-boyuai-hands-on-rl.md`](../sites/hrl-boyuai-hands-on-rl.md) — 在线书站与章节目录
- [`sources/courses/boyuai_hands_on_rl_elites_course.md`](../courses/boyuai_hands_on_rl_elites_course.md) — 伯禹平台免费视频课

## 推荐继续阅读（外部）

- Sutton & Barto, *Reinforcement Learning: An Introduction*
- [OpenAI Spinning Up](https://spinningup.openai.com)
