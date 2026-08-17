# DaivdYuan/SpeedTuning

- **标题：** SpeedTuning 官方仿真复现
- **类型：** repo
- **URL：** <https://github.com/DaivdYuan/SpeedTuning>
- **许可：** MIT
- **配套论文：** [arXiv:2608.09138](https://arxiv.org/abs/2608.09138) — [`sources/papers/speedtuning_arxiv_2608_09138.md`](../papers/speedtuning_arxiv_2608_09138.md)
- **项目页：** <https://daivdyuan.github.io/speed-tuning/>
- **入库日期：** 2026-08-17

## 一句话说明

冻结脚本化基座策略，Rainbow DQN 学离散速度倍率；含 pick-and-place / insertion / tea-bag 仿真环。

## 仓库状态（2026-08-17 核查）

| 项 | 内容 |
|----|------|
| 训练 | `scripts/train_speed_policy.py`、`rl/rainbowDQN` |
| 评测 | `scripts/eval_speed_policy.py`、`speed_evaluation.py` |
| 仿真 | `scripts/run_sim.py`、`sim_env.py`、`ee_sim_env.py` |
| 真机钩子 | `act_integration.py`（ACT 集成，无随仓真机数据） |

最短复现：`uv` 按 README 装 Python 3.10 → `python scripts/train_speed_policy.py`（脚本化 tea-bag）→ `python scripts/eval_speed_policy.py`。

## 与 wiki 的关系

- 实体页：[paper-speedtuning](../../wiki/entities/paper-speedtuning.md) — 含源码运行时序图。
