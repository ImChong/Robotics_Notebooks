# TSIL（Temporal Self-Imitation Learning）

> 来源归档

- **标题：** TSIL
- **类型：** repo
- **来源：** General Robotics Lab（Duke University）
- **链接：** <https://github.com/generalroboticslab/TSIL>
- **Stars / Forks：** ~13 / 0（2026-07-01）
- **许可证：** Apache-2.0（仓库代码）；捆绑 **Isaac Gym Preview 4** 与 **MTBench** 子模块保留各自许可
- **入库日期：** 2026-07-01
- **一句话说明：** TSIL 论文官方实现：核心 PPO+TSIL 训练代码、Hydra 启动脚本、捆绑 Isaac Gym Preview 4 + MTBench 子集、Colab 迷你 demo 与论文图表复现管线。
- **沉淀到 wiki：** [`wiki/entities/paper-tsil-temporal-self-imitation-learning.md`](../../wiki/entities/paper-tsil-temporal-self-imitation-learning.md)

---

## 核心定位

**TSIL** 仓库是 [arXiv:2606.19752](https://arxiv.org/abs/2606.19752) 的公开代码入口，与 [项目页](https://generalroboticslab.com/TSIL) 配套。发布内容包含：

| 组件 | 说明 |
|------|------|
| `core/` | 共享 TSIL 实现：`agents/`、`training/algo/tsil/`（memory / replay loss）、评估与绘图工具 |
| `projects/TSIL/` | 训练 / 评测 / 绘图 Python 入口（`train.py`、`eval.py`、`plot.py`） |
| `exec/TSIL/` | MT01 / MT15 的 Hydra shell 启动器与实验配置 |
| `MTBench/` | 捆绑的 MTBench 环境子集（`isaacgymenvs` 操作任务） |
| `isaacgym/` | 捆绑 **Isaac Gym Preview 4** Python 包 |
| `notebooks/TSIL_demo.ipynb` | 迷宫迷你 demo（[Colab](https://colab.research.google.com/github/generalroboticslab/TSIL/blob/main/notebooks/TSIL_demo.ipynb)） |

---

## 环境与安装（README 摘要）

```bash
conda env create -f tsil.yml
conda activate tsil
pip install -e ./isaacgym/python
pip install -e ./MTBench
```

推荐在仓库根目录设置：

```bash
export PYTHONPATH="$(pwd):$(pwd)/MTBench:$(pwd)/isaacgym/python:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
```

冒烟检查：

```bash
python -B exec/TSIL/train/launcher.py --help
python -B -m projects.TSIL.eval --help
```

---

## 训练 / 评测 / 复现入口

| 用途 | 入口 |
|------|------|
| 单任务 MT01 时间基线对比 | `exec/TSIL/train/entrypoints/mt01/compare_temporal/scratch/all.sh` |
| 单任务 MT01 TSIL 消融 | `exec/TSIL/train/entrypoints/mt01/compare_tsil/scratch/all.sh` |
| MT15 全基准训练 | `exec/TSIL/train/entrypoints/mt15/compare_temporal/scratch/all.sh` 等 |
| MT15 训练扰动实验 | `policy_grad_noise`、`dense_dropout`、`sweep_clip`、`sweep_lr` 子目录 |
| 评测 2000 trials | `python -m projects.TSIL.eval --task_name MT15_T28 --checkpoint CKPT --index_episode best_suc_tail ...` |
| 论文图表 | `exec/TSIL/plot/paper/figures/`、`tables/`（读 `results/TSIL/train_res`） |

Hydra 方法命名速查：

- `method/temporal=ih` — 无限时域 PPO 基线
- `method/temporal=attl` — 自适应时间目标（主时间基线）
- `method/sil=tsil` — 快速成功轨迹回放
- `method/sil=sil_trans` — 标准 SIL 高回报回放

训练输出：`results/TSIL/train_res/`；评测：`results/TSIL/eval_res/`；论文产物：`results/TSIL/paper_artifacts/`。

---

## 对 wiki 的映射

- 论文实体：[paper-tsil-temporal-self-imitation-learning.md](../../wiki/entities/paper-tsil-temporal-self-imitation-learning.md)
- 论文摘录：[tsil_arxiv_2606_19752.md](../papers/tsil_arxiv_2606_19752.md)
- 项目页归档：[generalroboticslab-tsil.md](../sites/generalroboticslab-tsil.md)
- 方法页：[Reinforcement Learning](../../wiki/methods/reinforcement-learning.md)、[Imitation Learning](../../wiki/methods/imitation-learning.md)
- 任务页：[Manipulation](../../wiki/tasks/manipulation.md)
