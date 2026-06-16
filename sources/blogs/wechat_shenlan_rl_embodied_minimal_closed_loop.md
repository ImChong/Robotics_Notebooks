# 强化学习必备知识①：跑通一个具身控制的最小闭环

> 来源归档（blog / 微信公众号）

- **标题：** 强化学习必备知识①：跑通一个具身控制的最小闭环
- **类型：** blog
- **作者：** 深蓝具身智能（微信公众号；《具身智能基础》专栏第 4 篇）
- **原始链接：** https://mp.weixin.qq.com/s/hHkQqLfIOTn0CoAZNuLWJA
- **发表日期：** 2026-06-11（frontmatter）
- **入库日期：** 2026-06-16
- **抓取方式：** [Agent Reach](https://github.com/Panniantong/Agent-Reach) v1.5.0 + [wechat-article-for-ai](https://github.com/bzd6661/wechat-article-for-ai)（Camoufox）；正文约 6000 字 / 22 图；Jina Reader 对该链接触发微信 CAPTCHA，未采用
- **专栏姊妹篇：** [李群、李代数、四元数](wechat_shenlan_lie_group_lie_algebra_quaternion.md)（`JviRH2LW-fkCHA5gY7Qflw`）；[三维世界坐标变换](wechat_shenlan_3d_coordinate_transforms.md)（`P5Jm7bMhaTHsytHStFbbLg`）；[黎曼流形与切空间](wechat_shenlan_riemannian_manifold_tangent_space.md)（`uFTKN5FDvlHQxOSspvxVZw`）
- **一句话说明：** 从「岔路口路牌」直觉出发，串起策略、MDP 五元组、POMDP、PPO/SAC 分工与 PyBullet KUKA 定点到达最小闭环；强调具身场景以连续动作为主、仿真先行、策略更新幅度需受约束。

## 核心摘录（归纳，非全文）

### RL 历史脉络（文内简史）

- 2013 DeepMind DQN（Atari）→ 2016 AlphaGo → 2017 PPO 稳定训练 → 2018 OpenAI 灵巧手 → 2020+ GPU 并行仿真使人形/四足进入可部署阶段。
- 文旨：RL 理论框架数十年，**工程价值在近十年**；本篇只切「具身最小闭环」一口。

### 策略直觉（岔路口思维实验）

- 无标准答案标注，靠环境反馈迭代；存在**延迟因果**与**探索–利用**平衡。
- **策略** $\pi(a|s)$：状态→动作的决策手册；工程上常用神经网络参数化。
- 围棋为**离散动作**；具身为**连续控制**（关节角、力矩）——载体同为策略，动作空间不同。

### MDP 五元组与 POMDP

| 要素 | 具身含义 |
|------|----------|
| $S$ | IMU、关节角、相机等观测 |
| $A$ | 力矩、关节目标、步态参数等连续动作 |
| $R$ | 抓取成功、摔倒、靠近目标等即时标量 |
| $P$ | 物理引擎或真机动力学决定的转移 |
| $\gamma$ | 远期奖励折扣 |

- 标准 MDP 假设全状态可观测；真机多为 **POMDP**——用 CNN/Transformer 将多帧观测编码为隐状态再端到端训练。

### PPO vs SAC（具身分工）

| 算法 | 文内定位 | 典型场景 |
|------|----------|----------|
| **PPO** | 工业落地首选；Clip 限制策略更新幅度 | 四足/人形行走、常规 loco |
| **SAC** | 最大熵、鲁棒性高 | 灵巧手、精细抓取、扰动敏感任务 |

- 共同准则：**策略单次迭代不能变化太大**，否则训练震荡或真机失控。
- 文内提及 **FlashSAC** 在宇树 G1 户外行走的快速部署案例（分钟级仿真训练）。

### PyBullet 最小闭环（KUKA 定点到达）

- 依赖 `pybullet` + `numpy`；`p.connect(GUI)` → 加载 `plane.urdf` 与 `kuka_iiwa` → 红球目标点。
- 循环内：`getLinkState` 得末端位置 $S$ → 奖励 $R=-\|ee-target\|$ → 速度控制指令 $A$ → `stepSimulation` 隐式实现 $P$。
- 教学点：先用手写启发式控制跑通 **S–A–R–P 闭环**，再换 PPO/SAC 学策略。

### 典型应用场景（文内列举）

1. **四足/人形 loco** — PPO；IMU + 关节 + 深度为状态，腿关节力矩/目标为动作。
2. **工业机械臂操作** — SAC；无序分拣、随机位姿抓取。
3. **移动机器人导航** — 激光/视觉状态，轮速动作，碰撞惩罚。
4. **仿生灵巧手** — SAC + [HER](../wiki/methods/her.md) 缓解稀疏奖励。
5. **虚拟预训练** — Isaac Sim / Habitat 训策略再蒸馏上真机。

### 前沿三趋势（文末）

- **LLM + RL**：语义层 LLM，执行层 RL。
- **多任务统一策略**：单模型覆盖行走/抓取/避障。
- **无奖励自驱动**：内在好奇心类内在奖励（见 [Intrinsic reward 预训练](../wiki/overview/bfm-category-03-intrinsic-reward-pretraining.md)）。

## 对 wiki 的映射

- **不入库为独立 wiki 节点**（用户要求）；知识并入 [`roadmap/motion-control.md`](../../roadmap/motion-control.md) L5 段。
- 新建 [`wiki/entities/pybullet.md`](../../wiki/entities/pybullet.md)、[`wiki/concepts/embodied-rl-minimal-closed-loop.md`](../../wiki/concepts/embodied-rl-minimal-closed-loop.md)。
- 交叉：[`wiki/formalizations/mdp.md`](../../wiki/formalizations/mdp.md)、[`wiki/formalizations/pomdp.md`](../../wiki/formalizations/pomdp.md)、[`wiki/methods/reinforcement-learning.md`](../../wiki/methods/reinforcement-learning.md)、[`wiki/comparisons/ppo-vs-sac.md`](../../wiki/comparisons/ppo-vs-sac.md)、[`wiki/overview/shenlan-embodied-ai-fundamentals-series.md`](../../wiki/overview/shenlan-embodied-ai-fundamentals-series.md)。
