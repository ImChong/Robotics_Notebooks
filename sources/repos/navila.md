# NaVILA（AnjieCheng/NaVILA）

> 来源归档

- **标题：** NaVILA
- **类型：** repo
- **来源：** UC San Diego / USC / NVIDIA
- **链接：** <https://github.com/AnjieCheng/NaVILA>
- **项目页：** <https://navila-bot.github.io/>
- **论文：** <https://arxiv.org/abs/2412.04453>
- **许可：** Apache-2.0
- **入库日期：** 2026-07-28
- **一句话说明：** NaVILA VLA 的数据处理、SFT、R2R-CE 评测、checkpoint 与标注；腿式 benchmark / locomotion 由配套仓库提供。
- **开源状态：** **已开源**；YouTube 原视频受版权限制，仅发布 IDs 与 annotations。
- **沉淀到 wiki：** [`paper-notebook-navila-legged-robot-vision-language-action-model.md`](../../wiki/entities/paper-notebook-navila-legged-robot-vision-language-action-model.md)

## 仓库概况（2026-07-28）

| 字段 | 值 |
|------|-----|
| 环境入口 | `environment_setup.sh navila` |
| 视频处理 | `scripts/extract_rawframes.py` |
| 训练入口 | `scripts/train/sft_8frames.sh` |
| R2R 评测 | `evaluation/scripts/eval/r2r.sh` |
| 汇总 | `evaluation/scripts/eval_jsons.py` |
| 数据 / 权重 | Hugging Face `a8cheng/NaVILA-Dataset` 与 NaVILA checkpoints |
| 配套 | `yang-zj1026/NaVILA-Bench`、`yang-zj1026/legged-loco` |

## 对 wiki 的映射

- [NaVILA 论文实体](../../wiki/entities/paper-notebook-navila-legged-robot-vision-language-action-model.md)
- [NaVILA 项目页](../sites/navila.md)
- [论文 source](../papers/humanoid_pnb_navila-legged-robot-vision-language-action-model.md)
