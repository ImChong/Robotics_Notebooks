# RoMAN-Flow（konnyaku28/RoMAN-Flow）

- **URL：** <https://github.com/konnyaku28/RoMAN-Flow>
- **权重：** <https://huggingface.co/wangshaoxuan/RoMAN-Flow>
- **论文：** [arXiv:2608.20208](https://arxiv.org/abs/2608.20208)

## 入口（README）

| 阶段 | 命令/脚本 |
|------|-----------|
| 环境 | `bash setup_env.sh` |
| LIBERO buffer | `scripts/prepare_libero_buffer.py` |
| 训练 | `main_torch.py`（IL → IQL → One-Step） |
| 批量评测 | `scripts/evaluate.py` + `WEIGHTS_ROOT` manifest |

## 状态（2026-08-22）

**已开源、可运行**（LIBERO + RoboMimic；MetaWorld 数据准备待更新）。

## wiki

- [`wiki/entities/paper-roman-flow.md`](../../wiki/entities/paper-roman-flow.md)
