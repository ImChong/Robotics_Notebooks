# Zero-WAM 项目页

- **论文：** [arXiv:2608.26103](https://arxiv.org/abs/2608.26103)
- **项目页：** <https://robbyant-research.github.io/Zero-WAM/>
- **代码：** <https://github.com/robbyant-research/Zero-WAM>（Apache-2.0，训练/评测代码、权重与部分 HumanGen 数据已上线）
- **权重与数据：** <https://huggingface.co/robbyant-research/zero-wam-pretrain>、<https://huggingface.co/robbyant-research/zero-wam-posttrain-robotwin>、<https://huggingface.co/datasets/robbyant-research/HumanGen>
- **机构：** 蚂蚁灵波科技（Robbyant）；香港科技大学广州校区；香港科技大学
- **入库日期：** 2026-08-28
- **再核日期：** 2026-11-19
- **核对范围：** 通读 Zero-WAM 项目页全文与 Hugging Face/GitHub Release 链接
- **通读：** 已完成

## 开源核查（步骤 2.5）

- 项目页与 GitHub 仓均已于 `2026-08-28` 上线；`2026-09-04` 初核时仍仅占位。
- **`2026-11-19` 再核：** GitHub 已完整开放 `wan_va/`、`script/train.sh` 与 `evaluation/robotwin/`；Hugging Face 已上线 `zero-wam-pretrain`、`zero-wam-posttrain-robotwin` 与 `HumanGen`（包含 RoboTwin 与 5 个外部数据集潜变量；in-house/real 子集与数据合成管线源码未公开）。
- **结论：** **已开源（核心模型与训练/RoboTwin 评测代码、预训练与后训练权重、公开子集数据均已发布）**。

## 交叉

- 论文归档：[`sources/papers/zero_wam_arxiv_2608_26103.md`](../papers/zero_wam_arxiv_2608_26103.md)
- 仓库归档：[`sources/repos/zero-wam.md`](../repos/zero-wam.md)
- wiki：[`wiki/entities/paper-zero-wam.md`](../../wiki/entities/paper-zero-wam.md)
