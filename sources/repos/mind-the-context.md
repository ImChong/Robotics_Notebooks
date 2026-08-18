# Cambridge-AFAR/Mind-the-Context

- **标题：** Mind the Context / EDD 官方实现
- **类型：** repo
- **URL：** <https://github.com/Cambridge-AFAR/Mind-the-Context>
- **许可：** 仓内无 SPDX LICENSE
- **默认分支：** `iros2026`
- **配套论文：** [arXiv:2608.13448](https://arxiv.org/abs/2608.13448) — [`sources/papers/mind_the_context_arxiv_2608_13448.md`](../papers/mind_the_context_arxiv_2608_13448.md)
- **入库日期：** 2026-08-18

## 一句话说明

社交适当性持续学习：环境/社会双分支 + rehearsal；训练与评测走 notebook。

## 仓库状态（2026-08-18 核查）

| 项 | 内容 |
|----|------|
| 模型 | `models/heuristicSplitModel.py`、`buffers.py` |
| 训练 | `experiments/training.ipynb` |
| 评测 | `experiments/evaluation.ipynb` |
| 数据 | MANNERSDB+ / OFFICE-MANNERSDB，需自备目录结构 |

最短复现：按 README 摆好数据集 → 跑 `data_processing/build_data.py` → `experiments/training.ipynb`。

## 与 wiki 的关系

- 实体页：[paper-mind-the-context](../../wiki/entities/paper-mind-the-context.md) — 含源码运行时序图。
