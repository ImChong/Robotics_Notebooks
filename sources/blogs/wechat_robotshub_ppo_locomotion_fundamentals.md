# 【人形机器人运控基础】万字长文解析最常用的强化学习运控算法 PPO

> 来源归档（blog / 微信公众号）

- **标题：** 【人形机器人运控基础】万字长文解析最常用的强化学习运控算法 PPO
- **类型：** blog
- **作者：** RobotsHub（微信公众号）
- **原始链接：** https://mp.weixin.qq.com/s/MJQYYyOBSLirVr0vH1-AZg
- **发表日期：** 2026-07-16（frontmatter `2026-07-16 09:42:20`）
- **入库日期：** 2026-08-17
- **抓取方式：** [Agent Reach](https://github.com/Panniantong/Agent-Reach) v1.5.0 + `wechat-article-for-ai`（Camoufox）；`--no-images`；Jina Reader 对 `mp.weixin.qq.com` 返回 CAPTCHA，未采用
- **原始抓取落盘：** [`sources/raw/wechat_robotshub_ppo_locomotion_2026-07-16.md`](../raw/wechat_robotshub_ppo_locomotion_2026-07-16.md)
- **一句话说明：** 以人形 locomotion 为贯穿例子，把 MDP → 策略梯度 / REINFORCE → Actor-Critic / GAE → PPO clip → 连续高斯策略与动作量纲 → 课程 / DR / 非对称 AC / 师生蒸馏串成一条可对照代码的原理链；**不是新论文**，升格方式是加深既有 [PPO](../../wiki/methods/ppo.md) / [GAE](../../wiki/methods/gae.md) 等方法页，不新建重复节点。
- **步骤 2.5：** 本文是教学长文，无独立项目页。文末指向已公开资料：PPO / GAE / TRPO 论文、[Rudin et al. 2021](https://arxiv.org/abs/2109.11978)、[rsl_rl](https://github.com/leggedrobotics/rsl_rl)（ETH RSL，**已开源**）、Isaac Lab 特权信息文档。本库已有对应 wiki 节点，不另建仓页。

## 核心摘录（归纳，非全文）

文内立场：运控只是例子，主线是把几个公式接到训练现象上。Python / PyTorch / 仿真器用法不在讨论范围。

### 因果链（文内一页总结）

observation / action / reward 定义问题 → 高斯策略用 `log_prob` 量化「这个动作有多符合当前策略」→ critic 的 $V$ 与 [GAE](../../wiki/methods/gae.md) 给出「比平均好多少」→ 策略梯度按优势加减概率 → [PPO](../../wiki/methods/ppo.md) 的 clip / KL / 熵决定改多快、多稳、多敢探索 → 课程、域随机化、模仿 / 师生决定仿真策略能否上真机。

### 值得写入 wiki 的独特判断

1. **$\gamma$ 的有效视野按步数计，不按秒计。** 约 $1/(1-\gamma)$ 步；控制频率升高时，同样 $\gamma$ 覆盖的真实时间变短。想「看未来两秒」必须随频率一起调 $\gamma$。
2. **horizon ≠ episode。** `num_steps_per_env` 是 rollout 窗口；episode 可因摔倒提前结束，PPO 仍截固定 $T$ 步并用末状态 $V$ bootstrap。
3. **`old_log_prob` 必须在采样时 detach 存下。** REINFORCE「采一次更一次」时重算与否数值相同；PPO 对同一批做 $K$ 个 epoch，$\theta$ 一直在变，ratio 必须是 `exp(log_prob - old_log_prob)`。
4. **clip 不是硬 KL 约束。** 它只让「朝有利方向走过头」的额外收益变平、梯度归零；ratio 仍可能出界。是 TRPO 信赖域的廉价一阶近似。
5. **value loss 下降 ≠ 策略变好。** critic 只拟合当前策略的价值：奖励黑客、actor 停滞、采样分布差时，value loss 仍可漂亮收敛。真正该盯 episode return、摔倒率、速度跟踪。
6. **$\gamma$ 与 $\lambda$ 不要混。** $\gamma$ 管「看多远」；$\lambda$ 管「多大程度信任 critic 概括未来」。运控常用 $\gamma\approx 0.99$、$\lambda\approx 0.95$。
7. **动作不是力矩的同义词。** 运控主流是 `q_target = q_default + action_scale · a`，再交给 PD；`action_scale` 过大乱甩、过小学不动步。
8. **训练采样、部署用均值。** 高斯 $\sigma$ 是探索范围；健康训练里 $\sigma$ 应随学习下降。
9. **非对称 Actor-Critic ≠ Teacher-Student。** 前者同一轮训练里 critic 看特权、actor 只看可部署观测，部署丢掉 critic；后者先训特权 teacher，再蒸馏可部署 student。
10. **探索 ≠ 乱动。** 人形高维下随机动作几乎必摔；有效探索要落在可恢复范围内，靠合理 `action_scale`、熵与初始化，而不是把 std 开到最大。

### 文内算法对照（与本库选型一致）

大规模并行人形 locomotion 里 PPO 是主流，因为只需采样、吃得下 GPU 海量 on-policy 数据、对密集 reward 稳健。SAC / TD3 样本效率更高，但高维接触上更难稳；仿真采样足够便宜时，「稳 + 简单」压过样本效率。PPO 对稀疏奖励不友好。

### rsl_rl 代码映射（文内附录）

| 代码概念 | 算法概念 |
|----------|----------|
| `actor_critic.act` | 高斯策略采样 |
| `actor_critic.evaluate` | critic $V(o)$ |
| rollout storage | on-policy 轨迹缓冲 |
| `compute_returns` | GAE 优势与回报目标 |
| `ratio` | $\pi_\theta/\pi_{\theta_\text{old}}$ |
| surrogate / clip | PPO clipped objective |
| `desired_kl` / adaptive lr | KL 监控与自适应学习率 |
| `action_scale` | 动作量纲映射 |

## 对 wiki 的映射

- **主沉淀：** [PPO](../../wiki/methods/ppo.md) — clip 误区、`old_log_prob`、clip fraction、高斯动作与 `action_scale`、训练循环图。
- **配套加深：** [GAE](../../wiki/methods/gae.md)（$\gamma$ vs $\lambda$、value loss 陷阱）、[MDP](../../wiki/formalizations/mdp.md)（有效视野 vs 控制频率）、[RL 超参指南](../../wiki/queries/rl-hyperparameter-guide.md)。
- **交叉：** [五模块训练栈](../../wiki/overview/humanoid-rl-policy-training-five-modules.md)（深蓝体系文的算法原理姊妹篇）、[RL Runner](../../wiki/concepts/rl-runner.md)、[特权训练](../../wiki/concepts/privileged-training.md)、[奖励设计](../../wiki/concepts/reward-design.md)、[Humanoid RL Cookbook](../../wiki/queries/humanoid-rl-cookbook.md)、[PPO vs SAC](../../wiki/comparisons/ppo-vs-sac.md)。
- **不新建** 独立「PPO 原理」实体页：本库已有 complete 的 `wiki/methods/ppo.md`。

## 当前提炼状态

- [x] 公众号正文抓取与 raw 归档
- [x] 与既有 PPO / GAE / 五模块节点对照（0 新建方法页）
- [x] 文内引用仓库开源状态核查（rsl_rl 已开源；本文本身无项目页）
