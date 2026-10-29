# FluxVLA Engine Hugging Face 模型卡

> 来源归档；核查日期：2026-10-28

- **发布者：** LimX Dynamics
- **模型卡和文件：** <https://huggingface.co/limxdynamics/FluxVLAEngine>
- **论文：** <https://arxiv.org/abs/2609.17210v2>
- **代码：** <https://github.com/FluxVLA/FluxVLA>
- **定位：** 官方模型卡提供上游预训练模型和 FluxVLA 训练 checkpoint 的链接，以及 LIBERO 结果和推理速度表。具体权重的许可与使用条件应分别查看对应页面。

## 模型卡记录

- 列出 PI0.5 PaliGemma LIBERO-10、Cosmos3-Edge 和 GR00T Eagle 3B 等 checkpoint 入口。
- 模型卡报告 π0.5 在 LIBERO Spatial / Object / Goal / Long 分别为 98.6 / 99.0 / 97.8 / 96.0%，四套件平均 97.85%；[GitHub README](https://github.com/FluxVLA/FluxVLA) 报告的对应平均为 97.95%，分项也不同。不要将两张表作为同一实验。
- 模型卡报告 RTX 5090 上 GR00T N1.5 优化前 13.4 Hz、优化后 42.8 Hz；这是该模型卡给定硬件和实现条件下的推理速度，不代表机器人闭环控制频率。

## 对 wiki 的映射

- [FluxVLA Engine 实体](../../wiki/entities/fluxvla-engine.md)
- [论文归档](../papers/fluxvla_arxiv_2609_17210.md) · [源码归档](../repos/fluxvla.md)
