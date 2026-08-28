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
- **一句话说明：** PyTorch + Gymnasium 的模仿 / 奖励学习参考实现，覆盖 BC、DAgger、最大因果熵 IRL、GAIL、AIRL 与偏好比较；是复现经典 IRL 论文的现代可运行入口。
- **沉淀到 wiki：** 是 → [`wiki/methods/inverse-reinforcement-learning.md`](../../wiki/methods/inverse-reinforcement-learning.md)

## 开源核查（2026-08-28）

**已开源** — 可运行训练入口，不是占位 README。

| 项 | 内容 |
|----|------|
| 安装 | `pip install imitation`（Python 3.8+；**只支持 Gymnasium，不支持旧 `gym` API**） |
| 源码安装 | `pip install -e ".[dev]"` |
| CLI | Sacred：`python -m imitation.scripts.train_rl`、`python -m imitation.scripts.train_adversarial gail|airl` |
| 算法模块 | `algorithms.bc` / `dagger` / `density` / `mce_irl` / `airl` / `gail` / `preference_comparisons` / `sqil` |
| 文档 | [算法页](https://imitation.readthedocs.io/en/latest/)；[benchmark 摘要](https://imitation.readthedocs.io/en/latest/main-concepts/benchmark_summary.html) |
| 最后推送 | 2025-01-07（截至核查日仓未归档） |
| 对照官方论文仓 | GAIL [openai/imitation](https://github.com/openai/imitation) **已归档**；AIRL [justinjfu/inverse_rl](https://github.com/justinjfu/inverse_rl) 为 TF1 时代实现 |

README 声明的动作空间覆盖：MCE-IRL 与 SQIL **仅离散**；GAIL / AIRL / BC / DAgger **离散 + 连续**。

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
