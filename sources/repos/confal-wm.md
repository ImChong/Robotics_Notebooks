# ConfAL-WM（ConfAL-WM/ConfAL-WM）

- **URL：** <https://github.com/ConfAL-WM/ConfAL-WM>
- **许可：** 未在根目录声明 SPDX
- **配套论文：** [arXiv:2608.25572](https://arxiv.org/abs/2608.25572)
- **权重：** <https://huggingface.co/anonymous89793/ConfAL-WM>
- **数据：** <https://huggingface.co/datasets/anonymous89793/ConfAL-WM-Dataset>

## 状态（2026-08-28）

| 项 | 状态 |
|----|------|
| 选择 / 打分 / 再训练 | `al_pipeline/` + `trainer/train_evac_with_al.py` |
| 评测 | `eval/al_results/` |
| HF 权重与预计算产物 | 已发布 |

可运行路径对齐：`al_pipeline/build_external_al_splits.py` → `trainer/train_c3_probe.py` → `trainer/train_evac_with_al.py` → `eval/al_results/evaluate_al_round.py`。

## wiki

- [`wiki/entities/paper-confal-wm.md`](../../wiki/entities/paper-confal-wm.md)
