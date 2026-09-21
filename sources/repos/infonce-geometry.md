# InfoNCE_Geometry 官方仓库

> 来源归档（repo）

- **标题：** The Geometric Mechanics of Contrastive Representation Learning
- **类型：** repo
- **链接：** https://github.com/YichaoCai1/InfoNCE_Geometry
- **arXiv：** <https://arxiv.org/abs/2601.19597>
- **项目页：** <https://yichaocai.com/nce_geo.github.io/>
- **入库日期：** 2026-09-21
- **一句话说明：** 论文数值与 COCO/CLIP 实验的可复现脚本（ICML 2026）。
- **沉淀到 wiki：** [`wiki/entities/paper-infonce-geometry.md`](../../wiki/entities/paper-infonce-geometry.md)

## 开源状态（步骤 2.5，2026-09-21）

- **已开源（实验复现）：** 各子目录自包含脚本，输出 PDF/图至当前目录或配置 output dir。
- **非完整 CLIP 预训练栈：** 以论文验证实验为主，非 production training framework。

## 仓库布局

| 目录 | 内容 |
|------|------|
| `numerical_val/grad_consistency/` | `large_batch_consistency.py` — 梯度 vs negatives |
| `numerical_val/unimodal_gibbs/` | `unimodal_gibbs.py` — 球面 Gibbs 均衡 |
| `numerical_val/modality_gap/` | `multimodal_modality_gap.py` — 多模态 gap toy |
| `coco_experiments/` | COCO same-category index、corruption dataset、pretrained gap、train_exp2 |

## 依赖

Python 3.10+、PyTorch、numpy、matplotlib、pillow、tqdm、open_clip_torch、pycocotools；动画可选 ffmpeg。
