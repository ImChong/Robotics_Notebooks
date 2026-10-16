# VT-Bridge 项目页

> 来源归档（官方项目页）

- **标题：** VT-Bridge: Bridging Pretrained Foundation VLAs to VTLAs via Lightweight Residual Adaptation
- **类型：** project page
- **项目页：** https://hoxnocha.github.io/vt-bridge-web/
- **网页源码：** https://github.com/hoxnocha/vt-bridge-web
- **论文：** [arXiv:2609.22606](https://arxiv.org/abs/2609.22606)
- **研究代码：** 截至 2026-10-16 未公开。项目页的 Code 按钮显示 “Coming soon”；仓库 README 说明训练代码由单独仓库维护，但未给出该仓库链接。
- **数据与权重：** 项目页未提供公开下载入口；网站仓库 README 将训练代码、模型权重和完整数据集列为单独管理内容。

## 页面核查

项目页概述一种慢快结构：预训练 VLA 低频产生 action chunk，轻量 residual adapter 读取最新力觉/触觉反馈，在约 30 Hz 执行频率修正动作。页面展示方法图、结果图和四项任务的实验视频。

论文与项目页报告：每任务最多 50 条视触觉示范；adapter 约 0.98M 参数；三种 VLA 主干（π₀、π₀.₅、SmolVLA）在四项真实接触丰富操作任务上的平均完成率从 11.7% 提升至 62.9%。项目页给出的残差推理耗时为 0.7 ms；此处仅记录页面报告值，不代表独立复测。

## 来源映射

- 论文原始资料：[VT-Bridge arXiv 归档](../papers/vt_bridge_arxiv_2609_22606.md)
- 网站仓库：[hoxnocha/vt-bridge-web](../repos/hoxnocha-vt-bridge-web.md)
- wiki：[VT-Bridge 实体页](../../wiki/entities/paper-vt-bridge.md)
