# VGGT（arXiv:2503.11651）

> 来源归档（paper）

- **标题：** VGGT: Visual Geometry Grounded Transformer
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2503.11651>
- **PDF：** <https://arxiv.org/pdf/2503.11651>
- **项目页：** <https://vgg-t.github.io/>
- **代码：** <https://github.com/facebookresearch/vggt>
- **入库日期：** 2026-11-01
- **线索来源：** [四年十篇顶会Best Paper，谁在持续霸榜？](../blogs/wechat_shenlan_cv_best_papers_2023_2026_labs.md)（深蓝具身智能，2026-11-01）
- **一句话说明：** VGGT 是前馈神经网络，直接从一张、几张或上百张视图推出相机参数、点图、深度图与 3D 点轨迹等关键 3D 属性。

## 开源状态

- **已开源**（步骤 2.5 核查，2026-11-01）：仓库 facebookresearch/vggt 经 git ls-remote 可达；项目页 HTTP 200。

## 核心摘录（公众号转述，未核验原文表格）

- 统一此前各自专用的 3D 视觉任务；摘要称重建不到 1 秒，且无需几何优化后处理仍优于需要后处理的方法（公众号归属：牛津 VGG 组 + Meta AI）。
- 单个 Transformer 前馈输出相机参数、多视图深度、稠密点云与点跟踪（arXiv 摘要）。
- 指标：arXiv 摘要称在相机参数估计、多视图深度估计、稠密点云重建、3D 点跟踪等任务达到 SOTA；具体数字未转录。

## 对 wiki 的映射

- [paper-vggt-visual-geometry-grounded-transformer](../../wiki/entities/paper-vggt-visual-geometry-grounded-transformer.md)
- [从视觉编程到 4D 重建：CV 三大会 2023–2026 Best Paper 阅读坐标](../../wiki/overview/cv-best-papers-2023-2026-technology-map.md)
