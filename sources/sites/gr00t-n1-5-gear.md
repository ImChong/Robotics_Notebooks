# GR00T N1.5 项目页（NVIDIA GEAR）

- **URL：** <https://research.nvidia.com/labs/gear/gr00t-n1_5/>
- **前序：** [GR00T N1](../../wiki/entities/paper-hrl-stack-34-gr00t_n1.md)（arXiv:2503.14734）
- **代码 / 权重：** <https://github.com/NVIDIA/Isaac-GR00T> · [HF GR00T-N1.5-3B](https://huggingface.co/nvidia/GR00T-N1.5-3B)
- **平台页：** [isaac-gr00t.md](../../wiki/entities/isaac-gr00t.md)
- **入库日期：** 2026-09-28（2026-11-07 全文复核）
- **版本：** GEAR 项目页（页面日期 2025-06-11）；HF 模型卡 `nvidia/GR00T-N1.5-3B` 的 `README.md`（Version 1.5），2026-11-07 抓取
- **核对范围：** 2026-11-07 通读 GEAR 项目页全文（架构改动、Eagle 2.5 grounding 对比表、FLARE、训练设置、架构验证、数据受限后训练、真机语言跟随、新物体/人类视频、DreamGen 新动词、Unitree G1 后训练、讨论）与 HF 模型卡全文（架构、输入输出、许可、软件集成、资源）；页面无独立论文、无逐任务附录，指标均为项目页汇总值，训练数据混合比例仅有饼图未给数字；未读 FLARE / DreamGen 论文原文，未读仓库 `n1d5` 分支源码。
- **通读：** 已完成
- **说明：** 截至入库日 **无独立 arXiv 条目**；技术报告以 GEAR 网页 + Isaac-GR00T 仓库 `n1d5` 分支为准。

## 一句话说明

GR00T N1.5：冻结 **Eagle VLM** + 简化 vision–LLM adapter；**DiT** cross-attend VLM embedding，对 **state + 带噪 action chunk** 做 **flow matching 速度预测**；预训练加 **FLARE** 未来 latent 对齐；相对 N1 显著改善语言跟随与少样本 post-train。

## 开源核查（2026-09-28）

| 资源 | 状态 |
|------|------|
| [NVIDIA/Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T) | **已开源**（Apache 2.0；N1.5 权重 HF） |
| [nvidia/GR00T-N1.5-3B](https://huggingface.co/nvidia/GR00T-N1.5-3B) | **已发布**；2026-11-07 抓取的模型卡写明「ready for non-commercial use」，许可链接为 NVIDIA OneWay Noncommercial License（与 N1.7 的 Open Model License 不同，使用前以模型卡现行文本为准） |
