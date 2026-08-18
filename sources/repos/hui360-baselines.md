# hucebot/HUI360-Baselines

- **标题：** HUI360 官方基线
- **类型：** repo
- **URL：** <https://github.com/hucebot/HUI360-Baselines>
- **许可：** MIT
- **配套论文：** [arXiv:2608.11051](https://arxiv.org/abs/2608.11051) — [`sources/papers/hui360_arxiv_2608_11051.md`](../papers/hui360_arxiv_2608_11051.md)
- **标注流水线：** [RaphaelLorenzo/Interact360](https://github.com/RaphaelLorenzo/Interact360)
- **入库日期：** 2026-08-18

## 一句话说明

360° 人机交互预测基线：`training.py` / `infer.py`，首次运行从 HF 拉约 59GB 骨架标注。

## 仓库状态（2026-08-18 核查）

| 项 | 内容 |
|----|------|
| 训练 | `python training.py -hp ./experiments/configs/in_hui/lstm_base.yaml --save_model` |
| 推理 | `python infer.py --model_path ./checkpoints/...` |
| 论文数字 | `legacy` 分支 |
| 权重桶 | Hugging Face `rlorlou/hui360-baselines-checkpoints` |

最短复现：`conda` Python 3.10 → `pip install -r requirements.txt` → 按 README 跑 LSTM 配置（可 CPU）。

## 与 wiki 的关系

- 实体页：[paper-hui360](../../wiki/entities/paper-hui360.md) — 含源码运行时序图。
