# vidihand.github.io（ViDiHand 项目页）

- **标题：** ViDiHand — The Surprising Effectiveness of Video Diffusion Models for Hand Motion Reconstruction
- **类型：** site / project-page
- **URL：** <https://vidihand.github.io/>
- **配套论文：** [arXiv:2606.30308](https://arxiv.org/abs/2606.30308) — [`sources/papers/vidihand_arxiv_2606_30308.md`](../papers/vidihand_arxiv_2606_30308.md)
- **代码：** <https://github.com/NTUYWANG103/ViDiHand> — [`sources/repos/vidihand.md`](../repos/vidihand.md)（待发布）
- **入库 / 复核日期：** 2026-08-13

## 一句话摘要

NTU / SJTU 官方站点：展示 video diffusion（Wan2.1-VACE）驱动的 egocentric 双手 4D MANO 重建；含多视角对比视频与 ARCTIC / HOT3D / HOI4D 全表。

## 公开信息要点（截至复核日）

- **页首按钮：** Code → GitHub；ArXiv → 2606.30308。
- **方法三块：** VACE hand-overlay 微调（冻结 base DiT）→ 双分支 decoder → 单次 VACE 推理。
- **定量：** 三基准检测 / 3D pose / 朝向位置 / jitter 全面领先；ARCTIC Jitter **3.183**、HOT3D **3.741**、HOI4D **4.010**。
- **开源（步骤 2.5）：** 有 GitHub 链接，但仓内仅 README 写「Code will be released soon」→ **代码待发布**。

## 为何值得保留

- 步骤 2.5 主入口：区分「有 Code 按钮」与「可运行实现」。
- 全量对照表与 in-the-wild 可视化补论文摘要数字。

## 关联资料

- 论文：[`sources/papers/vidihand_arxiv_2606_30308.md`](../papers/vidihand_arxiv_2606_30308.md)
- 代码：[`sources/repos/vidihand.md`](../repos/vidihand.md)
- Wiki：[wiki/entities/paper-vidihand.md](../../wiki/entities/paper-vidihand.md)
