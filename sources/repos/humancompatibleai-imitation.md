# HumanCompatibleAI / imitation

> 来源归档

- **标题：** imitation（Clean Imitation Learning Implementations）
- **类型：** repo
- **链接：** https://github.com/HumanCompatibleAI/imitation
- **文档：** https://imitation.readthedocs.io/
- **PyPI：** `pip install imitation`
- **配套论文：** Gleave et al., *imitation: Clean Imitation Learning Implementations*, [arXiv:2211.11972](https://arxiv.org/abs/2211.11972)
- **维护者：** Center for Human-Compatible AI（UC Berkeley）
- **License：** MIT
- **Stars：** ~1.8k（2026-08-28）
- **入库日期：** 2026-08-28
- **最后核对：** 2026-11-19
- **核对范围：** 通读克隆仓库 `README.md` 与核心算法源码 `src/imitation/algorithms/adversarial/gail.py`、`airl.py`、`common.py`、`src/imitation/algorithms/mce_irl.py`、`src/imitation/algorithms/density.py`（核对类签名、判别器 logits、MDP/MCE-IRL 更新与交替训练主循环；未在本地执行实际训练脚本）。
- **通读：** 已完成
- **一句话说明：** PyTorch + Gymnasium 的模仿 / 奖励学习参考实现，覆盖 BC、DAgger、最大因果熵 IRL、GAIL、AIRL 与偏好比较；是复现经典 IRL 论文的现代可运行入口。
- **沉淀到 wiki：** 是 → [`wiki/methods/inverse-reinforcement-learning.md`](../../wiki/methods/inverse-reinforcement-learning.md)

## 开源核查（2026-08-28 / 2026-11-19 源码核对）

**已开源** — 可运行训练入口，不是占位 README。

| 项 | 内容 |
|----|------|
| 安装 | `pip install imitation`（Python 3.8+；**只支持 Gymnasium，不支持旧 `gym` API**） |
| 源码安装 | `pip install -e ".[dev]"` |
| CLI | Sacred：`python -m imitation.scripts.train_rl`、`python -m imitation.scripts.train_adversarial gail\|airl` |
| 算法模块 | `algorithms.bc` / `dagger` / `density` / `mce_irl` / `airl` / `gail` / `preference_comparisons` / `sqil` |
| 文档 | [算法页](https://imitation.readthedocs.io/en/latest/)；[benchmark 摘要](https://imitation.readthedocs.io/en/latest/main-concepts/benchmark_summary.html) |
| 最后推送 | 2025-01-07（截至核查日仓未归档） |
| 对照官方论文仓 | GAIL [openai/imitation](https://github.com/openai/imitation) **已归档**；AIRL [justinjfu/inverse_rl](https://github.com/justinjfu/inverse_rl) 为 TF1 时代实现 |

README 声明的动作空间覆盖：MCE-IRL 与 SQIL **仅离散**；GAIL / AIRL / BC / DAgger **离散 + 连续**。

## 核心模块源码结构（实读摘录）

- **对抗式共同基类 `AdversarialTrainer`（`algorithms/adversarial/common.py`）：**
  - 用 `wrappers.BufferingWrapper(venv)` 与 `reward_wrapper.RewardVecEnvWrapper(..., reward_fn=self.reward_train.predict_processed)` 包装环境，把环境交互奖励替换为判别器 / 奖励网输出。
  - `train(total_timesteps)` 按 `n_rounds = total_timesteps // gen_train_timesteps` 交替执行：`train_gen()`（调用 SB3 `gen_algo.learn(...)`，收集轨迹存入 `_gen_replay_buffer`）与 `n_disc_updates_per_round` 次 `train_disc()`（混采专家与生成器 minibatch，以 `F.binary_cross_entropy_with_logits` 更新判别器，专家标 1、生成器标 0）。
- **GAIL（`algorithms/adversarial/gail.py`）：**
  - `logits_expert_is_high` 直接返回 `self._reward_net(state, action, next_state, done)`，忽略 `log_policy_act_prob`。
  - `RewardNetFromDiscriminatorLogit` 对原始 logit $L$ 计算 `-F.logsigmoid(-logits)` 即 $R = -\log(1 - D)$；`reward_train` 与 `reward_test` 均返回该判别器衍生信号，不可作为解耦环境动力学的可迁移奖励。
- **AIRL（`algorithms/adversarial/airl.py`）：**
  - `logits_expert_is_high` 计算 `self._reward_net(state, action, next_state, done) - log_policy_act_prob`，对应论文 $D_\theta(s,a) = \exp f_\theta / (\exp f_\theta + \pi(a|s))$；构造时强制检查 `gen_algo.policy` 为 `ActorCriticPolicy` 或 `SACPolicy` 等随机策略以获取 $\log \pi(a|s)$。
  - `reward_train` 返回带势函数塑形包装的 `_reward_net`，而 `reward_test` 递归剥离 `RewardNetWrapper` 返回底层基础奖励网络，供换环境再优化测试。
- **最大因果熵 IRL `MCEIRL`（`algorithms/mce_irl.py`）：**
  - 面向有限视野、已知转移矩阵的表格型 `TabularModelPOMDP`：每步 `_train_step` 调用 `mce_partition_fh` 执行软 Bellman 备份（`logsumexp` 计算 $V^{\text{soft}}, Q^{\text{soft}}, \pi$），再调用 `mce_occupancy_measures` 前向递推状态访问频率 `visitations`，并以 `loss = th.dot(visitations - demo_state_om, predicted_r)` 回传梯度 $\mathbb{E}_\pi[\nabla r_\theta] - \mathbb{E}_D[\nabla r_\theta]$。
- **密度基线 `DensityAlgorithm`（`algorithms/density.py`）：**
  - 用 `sklearn.neighbors.KernelDensity` 对演示的 $s$、$(s,a)$ 或 $(s,s')$ 拟合非参数核密度，以对数密度直接充当固定奖励给 `rl_algo` 优化（支持平稳与分时间步非平稳两种模式）。

## 和论文仓的关系

- **不要**把本库当成 Ho & Ermon 2016 或 Fu et al. 2018 的官方论文附录。它是 CHAI 对同一算法族的 **清洁再实现**，依赖与 API 已换到 PyTorch / Gymnasium / Stable-Baselines3 生态。
- 论文级行为以原 PDF 为准；本库用于跑通占用匹配 / 对抗奖励学习闭环，并对照文档中的 benchmark。

## 对 wiki 的映射

- [逆强化学习](../../wiki/methods/inverse-reinforcement-learning.md) — 方法页工程入口
- [Imitation Learning](../../wiki/methods/imitation-learning.md)
- [Behavior Cloning](../../wiki/methods/behavior-cloning.md)
- [DAgger](../../wiki/methods/dagger.md)
- [Gymnasium](../../wiki/entities/gymnasium.md)
- [IRL 一手论文索引](../papers/inverse_reinforcement_learning_primary_refs.md)
