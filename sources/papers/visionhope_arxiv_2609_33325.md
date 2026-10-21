# VisionHOPE 论文来源归档

> 来源归档（ingest）

- **标题：** VisionHOPE: Visual Backbones as Self-Modifying Learning Systems
- **类型：** paper
- **作者：** Siran Peng、Tianshuo Zhang、Tianyu Fu、Weisong Zhao、Haoyuan Zhang、Jiankuo Zhao、Minghui Wu、Ping Jiang、Xiangyu Zhu、Chenxu Zhao、Zhen Lei
- **机构：** Mininglamp Technology；MAIS, CASIA（中科院自动化所）；SAI, UCAS（中国科学院大学）；M.U.S.T.（部分作者在明略实习期间完成工作）
- **来源：** arXiv cs.CV，v1，2026-09-27
- **论文：** <https://arxiv.org/abs/2609.33325>
- **Hugging Face 论文页：** <https://huggingface.co/papers/2609.33325>
- **代码：** <https://github.com/PSRben/VisionHOPE>（MIT；源码归档见 [sources/repos/visionhope.md](../repos/visionhope.md)）
- **模型权重：** <https://huggingface.co/PSRben/VisionHOPE>（ImageNet-1K、COCO、ADE20K checkpoints）
- **入库日期：** 2026-10-18
- **一句话说明：** 提出把视觉骨干设计为自修改学习系统，让记忆状态与更新规则在图像内共同演化，并在 ImageNet-1K、COCO 和 ADE20K 上评测。

## 核心论文摘录（MVP）

### 1) 自修改视觉骨干
- **核心贡献：** 论文将 Nested Learning 的自指构造用于视觉骨干：模型处理图像时，记住的内容与学习方式共同演化；五种耦合记忆分别存储内容、生成键和值，并控制学习率与保留。
- **对 wiki 的映射：** [VisionHOPE 论文实体页](../../wiki/entities/paper-visionhope.md)

### 2) 稳定性控制与空间扫描
- **核心贡献：** 无约束的自指更新可能不稳定。方法以自指注入软上限和保留记忆转移的谱裁剪控制步长，并在二维特征图上按行列对齐 chunk，使用四个方向扫描。

### 3) 多任务评测
- **核心贡献：** 论文报告在 ImageNet-1K 分类、COCO 检测与实例分割、ADE20K 语义分割上的结果。主表（arXiv HTML v1，自动摘录，未逐格核验）：ImageNet-1K 层级式 Top-1 84.1 / 85.2 / 85.6（T/S/B）；COCO Mask R-CNN 1× AP^b 47.9 / 49.5 / 50.5；ADE20K UPerNet mIoU 49.4 / 50.3 / 51.8。消融见实体页。
- **对 wiki 的映射：** [VisionHOPE 论文实体页](../../wiki/entities/paper-visionhope.md)

## 当前提炼状态

- [x] 论文摘要与方法线索已整理
- [x] wiki 页面映射已确认
- [x] 官方源码与权重开放状态已核查
