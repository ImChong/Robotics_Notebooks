# ARROW — 官方项目页

> 来源归档（ingest · 项目页）

- **标题：** ARROW: Arbitrary Reconstruction and Tracking of 4D Observations in the Wild
- **类型：** site / project-page
- **项目页：** <https://ifradlin.github.io/arrow/>
- **论文：** <https://arxiv.org/abs/2610.01314>
- **代码：** <https://github.com/ifradlin/arrow>
- **权重：** <https://huggingface.co/ifradlin/arrow>
- **机构：** 亚琛工业大学（RWTH Aachen University）
- **入库日期：** 2026-10-26
- **一句话说明：** 用同一共享场景表示处理无序图像、视频与多视角观测，并通过查询预测 3D 轨迹和重建几何。

## 项目页要点

- **顺序无关：** 任意图像集合都映射到共享上下文，不要求输入具有固定时间顺序。
- **内容派生身份：** 每个观测生成 identity token，查询借此关联源、目标和参考观测。
- **统一预测：** 同一编码可服务稀疏点跟踪与稠密 3D 重建；项目页提供交互式结果示例。
- **评测主张：** 项目页与论文报告在 WorldTrack、TAPVid-3D 的 3D 跟踪达到新的最佳结果，并在多视角 RGB-only 基准上超过专用跟踪器。

## 开源核查

官方 GitHub README 列出推理代码、预训练权重和交互式查看器，README 更新记为 2026-10-12。查看器可选择图像或视频后直接运行任务，也能打开已保存的预测结果。代码与权重许可不同，详见 [仓库归档](../repos/arrow.md)。

## 对 wiki 的映射

- [ARROW 实体页](../../wiki/entities/paper-arrow-4d-reconstruction.md)
- [ARROW 论文摘录](../papers/arrow_arxiv_2610_01314.md)
- [ARROW 官方仓库](../repos/arrow.md)
