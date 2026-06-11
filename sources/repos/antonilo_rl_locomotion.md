# antonilo/rl_locomotion

> 来源归档

- **标题：** rl_locomotion — RMA 系四足 locomotion RL 训练代码（RaiSim）
- **类型：** repo
- **组织：** Antonio Loquercio（antonilo）；基于 Kumar et al. RMA 代码扩展
- **链接：** <https://github.com/antonilo/rl_locomotion>
- **Stars：** ~346（截至入库日）
- **入库日期：** 2026-06-11
- **一句话说明：** 在 **RaiSim + raisimGymTorch** 上复现/扩展 **RMA 特权 locomotion 训练**；README 标明同时服务 **CMS（Learning Visual Locomotion with Cross-Modal Supervision, ICRA 2023）**；真机部署见 [`vision_locomotion`](https://github.com/antonilo/vision_locomotion)。

## 仓库定位

| 维度 | 说明 |
|------|------|
| **上游** | [RMA 项目页](https://ashish-kmr.github.io/rma-legged-robots/) / [arXiv:2107.04034](https://arxiv.org/abs/2107.04034) |
| **扩展论文** | [Learning Visual Locomotion with Cross-Modal Supervision](https://antonilo.github.io/vision_locomotion/)（ICRA 2023） |
| **仿真器** | **RaiSim**（CPU）；需 checkout commit `f0bb440762c09a9cc93cf6ad3a7f8552c6a4f858`（**不支持最新版**） |
| **安装形态** | 克隆到 `raisimLib` 后 **重命名为 `raisimGymTorch`**，`python setup.py develop` |
| **平台** | A1 任务环境 `rsg_a1_task`；特权信息含质量、速度、电机强度、地形几何等 |

## 训练与评测命令（README 摘要）

| 步骤 | 命令 / 路径 |
|------|-------------|
| 特权策略训练 | `env/envs/rsg_a1_task/runner.py --name random --gpu 1 --exptid 1` |
| 收敛量级 | ~**4K iterations** 可得可用策略 |
| 模仿强度 | `runner.py` 内 **`RL_coeff`** 调节对 [`data/base_policy`](https://github.com/antonilo/rl_locomotion/tree/master/data/base_policy) 的模仿 |
| 可视化 | `raisimUnity` + `viz_policy.py` |
| 基准评测 | `evaluate_policy.py` → `eval_scripts/compute_results.py` |

## 与本仓库知识的关系

| 主题 | 关系 |
|------|------|
| [RMA 论文实体](../../wiki/entities/paper-rma-rapid-motor-adaptation.md) | 方法正文与两阶段管线 |
| [Privileged Training](../../wiki/concepts/privileged-training.md) | 特权 $e_t \to z_t$ + 历史 $\phi$ 蒸馏范式 |
| [legged_gym](./legged_gym.md) | 另一主流栈（Isaac Gym）；Extreme Parkour 等走 legged_gym，本文走 **RaiSim** |
| CMS 视觉扩展 | 同作者 [`vision_locomotion`](https://github.com/antonilo/vision_locomotion) 负责真机视觉策略 |

## 对 wiki 的映射

- 与论文 / 项目页成对维护：**[`sources/papers/rma_arxiv_2107_04034.md`](../papers/rma_arxiv_2107_04034.md)**、**[`sources/sites/rma-legged-robots-github-io.md`](../sites/rma-legged-robots-github-io.md)** → **[`wiki/entities/paper-rma-rapid-motor-adaptation.md`](../../wiki/entities/paper-rma-rapid-motor-adaptation.md)**
