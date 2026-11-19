# Gated Memory Policy（real-stanford/gated-memory-policy）

- **URL：** <https://github.com/real-stanford/gated-memory-policy>
- **核心子模块：** `imitation-learning-policies/`、`mujoco-env/`、`mikasa-robo-env/`、`real-env/`（`gated-memory-policy` 分支）
- **许可：** MIT License
- **语言：** Python（PyTorch / Hydra / SigLIP2 / RobotMQ）
- **配套论文：** [arXiv:2604.18933](https://arxiv.org/abs/2604.18933)
- **项目页：** <https://gated-memory-policy.github.io/>
- **入库日期：** 2026-11-19
- **核对范围：** 通读根目录 `README.md`、`LICENSE`、`imitation-learning-policies/README.md`、训练/门控/评估/服务脚本（`scripts/` 与 `shell_scripts/`）及核心类源码（`HistoryDenoisingPolicy`、`MemoryTransformer`、`MemoryGate`、`PolicyServer`）。
- **通读：** 已完成

## 仓库核心流程与入口（源码核对）

| 阶段 | 脚本 / 模块 | 作用与关键参数 |
|------|-------------|----------------|
| 无记忆 / 全记忆基线训练 | `imitation-learning-policies/scripts/train_policy.py`（参考 `shell_scripts/train_sim.sh`） | 在重叠轨迹上训练 `HistoryDenoisingPolicy`（历史动作加末步扩散噪声 `last_step`，缓存冻结 SigLIP2 特征） |
| 门控伪标签标定 | `imitation-learning-policies/scripts/generate_gate_labels.py` (`get_gate_labels`) | 加载无记忆与有记忆双 checkpoint，在验证集按 20 步窗口与阈值 $\theta$ 计算误差差值并生成 `.h5`/标签 |
| 门控网络训练 | `imitation-learning-policies/scripts/train_memory_gate.py` | 用 BCE 训练 `MemoryGate` 二分类判断当前观测是否开启历史记忆 |
| 门控策略二阶段训练 | `scripts/train_policy.py`（配置 `diffusion_gated_transformer`） | 冻结已训练 `MemoryGate`，联合微调 `MemoryTransformer`（支持 `BinaryGatingSTE`） |
| 仿真评测 | `imitation-learning-policies/scripts/eval_model.py` (`eval_model`) | 在 MemMimic / MIKASA-Robo / RoboMimic 环境中并发 rollout 评估 |
| 真机服务端 | `imitation-learning-policies/scripts/run_policy_server.py` → `imitation_learning/envs/policy_server.py` (`PolicyServer`) | 基于 `robotmq.RMQServer` 接收观测与状态，通过 `self.policy.predict_action` 缓存历史视觉特征并输出动作块 |

## 工程注意（源码边界）

- 根目录依赖多个 git submodule（`imitation-learning-policies`、`real-env`、`mujoco-env`、`mikasa-robo-env`），克隆时须 `git submodule update --init`。
- 论文 Finding 8.9 指出：真机 **ARX X5（T3 Flip & Place Back）** 实验为最大化长时记忆覆盖直接使用了**不带门控的纯记忆版本**；高频强反应操作（如 RoboMimic Tool Hang）才是二进制门控避免掉点的关键场景。

## 交叉

- 论文归档：[`sources/papers/gated_memory_policy_arxiv_2604_18933.md`](../papers/gated_memory_policy_arxiv_2604_18933.md)
- 项目页归档：[`sources/sites/gated-memory-policy-github-io.md`](../sites/gated-memory-policy-github-io.md)
- wiki：[`wiki/entities/paper-gated-memory-policy.md`](../../wiki/entities/paper-gated-memory-policy.md)
