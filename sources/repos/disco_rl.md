# google-deepmind/disco_rl

- **类型：** repo / JAX / meta-learning
- **仓库：** <https://github.com/google-deepmind/disco_rl>
- **项目：** <https://google-deepmind.github.io/disco_rl/>
- **论文：** <https://www.nature.com/articles/s41586-025-09761-x>
- **核查日期：** 2026-10-12
- **核查提交：** `9059a29f7121d60948f25ef165e08e050e9399c8`
- **许可证：** 软件 Apache-2.0；其他材料 CC-BY
- **配套归档：** [论文](../papers/discorl_nature_2025.md)、[项目页](../sites/discorl.md)

## 开放程度

已公开最小训练框架、元训练和元评估示例，以及 `disco_rl/update_rules/weights/disco_103.npz`。README 声明之后不会主动维护；不能据此承诺持续适配新 JAX。

## 可辨识运行入口

| 文件 / 模块 | 作用 |
|---|---|
| `colabs/eval.ipynb` | 下载 / 解包 Disco103 元参数，创建 Catch 环境和新智能体，训练并绘制奖励 |
| `colabs/meta_train.ipynb` | 从随机参数或 checkpoint 开始，展开内层更新、计算并汇总元梯度 |
| `disco_rl/agent.py` | `Agent.actor_step` 采样动作；`learner_step` 计算目标、损失及参数更新 |
| `disco_rl/update_rules/disco.py` | `DiscoUpdateRule.unroll_meta_net` 构造目标；`agent_loss` 与 `agent_loss_no_meta` 提供损失 |
| `disco_rl/networks/meta_nets.py` | LSTM 元网络实现 |
| `disco_rl/environments/` | 环境接口、CPU 与可 JIT 的 Catch 实现 |
| `disco_rl/update_rules/weights/disco_103.npz` | 更新规则权重，不是预训练动作策略 |

## 复现提示

`pyproject.toml` 要求 Python ≥3.11，依赖 JAX、Haiku、Optax、RLax 等；GPU 后端需按设备另行配置。先使用官方 Colab，不把 notebook 内的 `/content/` 路径直接当本地 CLI。元训练 notebook 的默认两智能体 / Catch 设置是机制演示，与论文的多环境 TPU 发现预算不同。本轮只核验源码，未安装或执行训练依赖。

## 对 wiki 的映射

- [DiscoRL](../../wiki/entities/paper-discorl.md)：源码时序图、最小复现路径与维护风险。
