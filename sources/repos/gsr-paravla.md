# AutoLab-SAI-SJTU/GSR-ParaVLA

- **标题：** GSR / ParaVLA 官方实现
- **类型：** repo
- **URL：** <https://github.com/AutoLab-SAI-SJTU/GSR-ParaVLA>
- **许可：** MIT
- **配套论文：** [arXiv:2608.02497](https://arxiv.org/abs/2608.02497) — [`sources/papers/gsr_paravla_arxiv_2608_02497.md`](../papers/gsr_paravla_arxiv_2608_02497.md)
- **权重：** <https://huggingface.co/AutoLab-SJTU/GSR>
- **入库日期：** 2026-08-15

## 一句话说明

完整训练与 LIBERO-Para 评测配方：ParaVLA、VLA-Adapter GSR、SmolVLA GSR、π₀.₅ GSR；配套 HF checkpoint。

## 仓库状态（2026-08-15 核查）

| 项 | 内容 |
|----|------|
| 训练 | `recipes/train_paravla.sh`、`train_vla_adapter_gsr.sh`、`train_smolvla_gsr.sh`、`train_pi05_gsr.sh` |
| 评测 | `recipes/eval_*_libero_goal.sh`、`eval_*_libero_para.sh` |
| 环境 | `environments/lerobot.yml`、`environments/vla_adapter.yml` |
| 后端 | 内嵌 `lerobot_backend/` |

最短复现：按 README 建 conda 环境 → `STAGE=smoke bash recipes/train_paravla.sh` → `bash recipes/eval_lerobot_libero_para.sh`。

## 与 wiki 的关系

- 实体页：[paper-gsr-paravla](../../wiki/entities/paper-gsr-paravla.md) — 含源码运行时序图。
