# RynnWorld-Latent 官方仓库

> 来源归档（repo）

- **标题：** RynnWorld-Latent: A Cross-Embodiment World Model Conditioned on Universal Latent Actions
- **类型：** repo
- **链接：** https://github.com/alibaba-damo-academy/RynnWorld-Latent
- **arXiv：** <https://arxiv.org/abs/2610.36791>
- **入库日期：** 2026-10-28
- **一句话说明：** 以三维落地潜在动作对齐人类、机器人与仿真视频，学习可控未来，并通过数字遥操作和动作迁移合成示范。
- **沉淀到 wiki：** [`wiki/entities/paper-rynnworld-latent.md`](../../wiki/entities/paper-rynnworld-latent.md)

## 开源状态

- **已开源**：仓库可达（git ls-remote）。项目页 HTTP 200。 仓库内容以 README 与 release 为准（本次仅用 `git ls-remote` 核实可达，未逐个阅读 README）。

## 2026-11-02 核查

- 官方 README 列出推理、训练、数据与 RynnLAM 源码；[模型权重](https://huggingface.co/Alibaba-DAMO-Academy/RynnWorld-Latent)单独发布，基础模型权重有独立访问条件。
- 已阅读 [`scripts/inference/rollout.py`](https://github.com/alibaba-damo-academy/RynnWorld-Latent/blob/main/scripts/inference/rollout.py)：入口构造数据集，加载模型，逐样本生成、解码并计算 PSNR；`--dry-run` 只检查数据准备。内部模块与实际运行尚待核查。
- 论文 v1 报告 23 个数据集、约 5.535 亿帧；README 报告 32 个数据集、超过 2.65 亿帧，差异原因尚未核实。
