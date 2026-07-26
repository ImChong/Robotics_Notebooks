# Segment Anything（SAM，arXiv:2304.02643）

> 来源归档（ingest）

- **标题：** Segment Anything
- **缩写 / 模型：** **SAM**（Segment Anything Model）；项目亦称 **SA**
- **类型：** paper / foundation-model / promptable-segmentation / computer-vision
- **arXiv：** <https://arxiv.org/abs/2304.02643>（PDF：<https://arxiv.org/pdf/2304.02643>）
- **项目页：** <https://segment-anything.com/> — 归档见 [`sources/sites/segment-anything-com.md`](../sites/segment-anything-com.md)
- **代码：** <https://github.com/facebookresearch/segment-anything>（Apache-2.0）— 归档见 [`sources/repos/segment-anything.md`](../repos/segment-anything.md)
- **数据集：** SA-1B（11M 图 / 1.1B masks）— <https://ai.facebook.com/datasets/segment-anything/>
- **作者：** Alexander Kirillov、Eric Mintun、Nikhila Ravi、Hanzi Mao、Chloe Rolland、Laura Gustafson、Tete Xiao、Spencer Whitehead、Alexander C. Berg、Wan-Yen Lo、Piotr Dollár、Ross Girshick
- **机构：** Meta AI Research（FAIR）
- **入库日期：** 2026-07-26
- **一句话说明：** 提出可提示分割任务 + SAM（ViT 图像编码 + 轻量提示/掩码解码）+ data engine，建成 SA-1B，实现跨分布零样本分割。

## 开源状态（步骤 2.5）

- **项目页 / 仓库核查（2026-07-26）：** 论文与 README 指向 <https://segment-anything.com/>；官方仓 [facebookresearch/segment-anything](https://github.com/facebookresearch/segment-anything) 公开推理代码、ViT-B/L/H 权重、自动掩码脚本与 ONNX 导出（Apache-2.0）。SA-1B 需接受研究许可后下载。
- **结论：** **已开源**（推理 / checkpoint / 示例 notebook / ONNX；训练代码未作为本仓主入口发布）。后续视频版见 [SAM 2](sam2_arxiv_2408_00714.md) / [facebookresearch/sam2](../repos/sam2.md)。

## 摘录 1：三件套主张（§1–§2）

- **任务：** Promptable segmentation — 任意提示（点/框/掩码/文本）须返回**至少一个**合理有效 mask（含歧义场景多 mask）。
- **模型：** 重图像编码器一次算 embedding；轻量 prompt encoder + mask decoder ≈50 ms（浏览器，amortized）。
- **数据引擎：** assisted-manual → semi-automatic → fully automatic；最终 SA-1B 约 99.1% 掩码全自动生成。

**对 wiki 的映射：** 升格 [`wiki/entities/paper-segment-anything.md`](../../wiki/entities/paper-segment-anything.md)；机器人侧作「2D mask 基元」链到语义建图 Query。

## 摘录 2：架构与 SA-1B（§3–§5）

- **图像编码器：** MAE 预训练 ViT（B/L/H），高分辨率适配。
- **提示：** 稀疏（点/框；文本用 CLIP）与稠密（掩码卷积嵌入）。
- **歧义：** 单点可输出多个有效 mask + IoU 分数。
- **SA-1B：** 11M 许可/隐私保护图，1.1B masks；约 100 masks/图；相对当时最大分割集约 **400×** 掩码规模。

**对 wiki 的映射：** 实体页画「encode once → prompt → decode」流程图与推理时序图。

## 摘录 3：评测要点（§7）

- **单点质量：** 23 个分割数据集上，单前景点 mask 常接近人工标注。
- **零样本迁移：** 边缘检测、proposal、实例分割、初步 text-to-mask 等经 prompt engineering。
- **许可：** 模型 Apache-2.0；SA-1B 为研究用途许可。

**对 wiki 的映射：** 强调「提示工程可组合进检测器/机器人标注管线」，但 **不** 自带语义类别名。

## 建议 wiki 动作

- 新建 **`wiki/entities/paper-segment-anything.md`**（含流程总览 + 源码运行时序图）。
- 新建 **`sources/repos/segment-anything.md`**、**`sources/sites/segment-anything-com.md`**。
- 交叉更新 GO2 SAM 流水线 Query、OVO / DualMap / OV-SAM3D；与 SAM 2 互链。
