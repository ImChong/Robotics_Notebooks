# tasl-lab/nav-ps-balance

- **标题：** 接近–安全约束分解跟随官方实现
- **类型：** repo
- **URL：** <https://github.com/tasl-lab/nav-ps-balance>
- **许可：** MIT
- **配套论文：** [arXiv:2608.10056](https://arxiv.org/abs/2608.10056) — [`sources/papers/nav_ps_balance_arxiv_2608_10056.md`](../papers/nav_ps_balance_arxiv_2608_10056.md)
- **入库日期：** 2026-08-18

## 一句话说明

CrowdNav 扩展静态障碍 + DtACI 不确定性 + 多 critic PPO-Lagrangian；含预训练权重。

## 仓库状态（2026-08-18 核查）

| 项 | 内容 |
|----|------|
| 评测 | `python test.py` |
| 可视化 | `python visualize.py` |
| 训练 | `python train.py`（改 `arguments.py` / `crowd_nav/configs/config.py`） |
| 依赖 | PyTorch 1.12.1、`baselines`、Python-RVO2、OGM C++ 扩展；numpy 1.23.5 |

最短复现：按 README 装环境 → `python test.py` 跑 `trained_models/`。

## 与 wiki 的关系

- 实体页：[paper-nav-ps-balance](../../wiki/entities/paper-nav-ps-balance.md) — 含源码运行时序图。
