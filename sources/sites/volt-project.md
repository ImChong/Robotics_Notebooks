# Volt / Volume Transformer 作者项目页

- **项目页：** <https://yilmazkadir.github.io/Volt/>
- **论文：** [arXiv:2604.19609](../papers/volt_arxiv_2604_19609.md)；ECCV 2026 Long Oral
- **代码：** <https://github.com/YilmazKadir/Volt>（MIT，已公开实现与训练入口）
- **权重：** <https://huggingface.co/KadirYilmaz/Volt>（公开 checkpoint 与实验目录）
- **入库/复核日期：** 2026-11-13
- **一句话说明：** 全局 attention + volumetric patches + 3D RoPE 的标准 Transformer 3D 骨干。
- **源码开放程度：** **已开源**（官方 README、模型实现、训练脚本及权重可访问）。

## 关键资源

- 论文 Figure 2：体积 patch token 化 → Transformer → 单层轻量上采样 → 稠密语义分割头。
- 训练与复现：仓库 README 提供 uv sync、数据预处理和训练 shell 命令。
- 权重与数据：模型权重已发布；ScanNet/nuScenes/Waymo 等数据需按各数据集许可自行获取并预处理。

## 交叉链接

- [论文全文归档](../papers/volt_arxiv_2604_19609.md)
- [官方代码运行归档](../repos/yilmazkadir-volt.md)
- [主 wiki 节点](../../wiki/entities/paper-volt.md)
- [VINO 项目页](./vino-project.md)
