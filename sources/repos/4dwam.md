# lishanyqy/4DWAM

- **标题：** 4D-WAM 官方后训练实现
- **类型：** repo
- **URL：** <https://github.com/lishanyqy/4DWAM>
- **许可：** 仓内无根级 SPDX LICENSE；`FastWAM/` 子树含 LICENSE 文件
- **配套论文：** [arXiv:2608.08023](https://arxiv.org/abs/2608.08023) — [`sources/papers/4d_wam_arxiv_2608_08023.md`](../papers/4d_wam_arxiv_2608_08023.md)
- **入库日期：** 2026-08-17

## 一句话说明

在 FastWAM 与 Lingbot-VA 上注入轨迹场 alignment；含 Trace Anything 预处理、DeepSpeed 训练与 LIBERO / RoboTwin 评测入口。

## 仓库状态（2026-08-17 核查）

| 项 | 内容 |
|----|------|
| FastWAM 训练 | `FastWAM/scripts/train.py`、`train_zero1.sh`、`train_zero2.sh` |
| 预处理 | `preprocess_action_dit_backbone.py`、`precompute_text_embeds.py` |
| 评测 | `FastWAM/experiments/libero/run_libero_manager.py`、`robotwin/run_robotwin_manager.py` |
| Lingbot-VA | `lingbot-va/` + `TraceAnything/` |

最短复现：按 `FastWAM/README.md` 建 `fastwam` conda → 缓存轨迹特征 → `bash scripts/train_zero1.sh`。

## 与 wiki 的关系

- 实体页：[paper-4d-wam](../../wiki/entities/paper-4d-wam.md) — 含源码运行时序图。
