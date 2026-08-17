# ChinmayMundane/PGIF_MPPI

- **标题：** PGIF-MPPI 官方实现
- **类型：** repo
- **URL：** <https://github.com/ChinmayMundane/PGIF_MPPI>
- **许可：** MIT
- **配套论文：** [arXiv:2608.08323](https://arxiv.org/abs/2608.08323) — [`sources/papers/pgif_mppi_arxiv_2608_08323.md`](../papers/pgif_mppi_arxiv_2608_08323.md)
- **入库日期：** 2026-08-17

## 一句话说明

JAX 实现的人群走廊 MPPI：各向异性高斯行人代价 vs vanilla 静态点障碍；含 100-seed 评测脚本。

## 仓库状态（2026-08-17 核查）

| 项 | 内容 |
|----|------|
| 核心 | `mppi_dynamic_humans.py` |
| 评测 | `evaluate_mppi.py`（`use_gaussian_cost` 切换 PGIF / vanilla） |
| 作图 | `plot_paper_figures.py` |
| 依赖 | `pip install numpy matplotlib jax jaxlib` |

最短复现：`python mppi_dynamic_humans.py`（可视化单局）或改 `evaluate_mppi.py` 底部调用后 `python evaluate_mppi.py`。

## 与 wiki 的关系

- 实体页：[paper-pgif-mppi](../../wiki/entities/paper-pgif-mppi.md) — 含源码运行时序图。
