# VNT-PA 作者主页（官方研究链接）

- **类型：** site / author-primary-source
- **URL：** <https://beimingli0626.github.io/>
- **项目：** Visual Navigation Transformer with Pose Attention（VNT-PA）
- **论文：** <https://arxiv.org/abs/2609.21212>；见 [论文归档](../papers/vnt-pa_arxiv_2609_21212.md)
- **代码：** <https://github.com/beimingli0626/VNT-PA>；见 [代码归档](../repos/vnt-pa.md)
- **机构：** 宾夕法尼亚大学 GRASP Laboratory
- **入库日期：** 2026-11-08
- **核对范围：** 作者主页 Preprints 列表、VNT-PA 的 Arxiv / Code 指向及项目简述。
- **通读：** 已完成

## 一手来源判断

该链接是第一作者 Beiming Li 的研究主页，**不是独立的 VNT-PA 产品项目页**。Preprints 中的论文标题与 arXiv 2609.21212 相符，并由 Code 链直接指向 `beimingli0626/VNT-PA`。因此官方开源状态按该链接与仓库 README 联合判断。

## 开源核查（2026-11-08）

- **源码：** 已开源，MIT；`src/pan/collection/` 与 `src/pan/planner/` 含可辨识可运行命令。
- **模型和数据：** README 要求从 Hugging Face 拉取 DeFM 冻结编码器权重，并自行准备 **HM3D v0.2**（全部 split 约 240GB）；README 未提供可直接下载的 VNT-PA 训练完成权重。
- **复现边界：** 代码公开 ≠ 已在本地跑通/复现论文 93.3% SR。本次为静态代码与文档核验。

## 对 wiki 的映射

- [VNT-PA 论文实体](../../wiki/entities/paper-vnt-pa.md)
