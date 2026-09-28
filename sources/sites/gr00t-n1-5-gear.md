# GR00T N1.5 项目页（NVIDIA GEAR）

- **URL：** <https://research.nvidia.com/labs/gear/gr00t-n1_5/>
- **前序：** [GR00T N1](../../wiki/entities/paper-hrl-stack-34-gr00t_n1.md)（arXiv:2503.14734）
- **代码 / 权重：** <https://github.com/NVIDIA/Isaac-GR00T> · [HF GR00T-N1.5-3B](https://huggingface.co/nvidia/GR00T-N1.5-3B)
- **平台页：** [isaac-gr00t.md](../../wiki/entities/isaac-gr00t.md)
- **入库日期：** 2026-09-28
- **说明：** 截至入库日 **无独立 arXiv 条目**；技术报告以 GEAR 网页 + Isaac-GR00T 仓库 `n1d5` 分支为准。

## 一句话说明

GR00T N1.5：冻结 **Eagle VLM** + 简化 vision–LLM adapter；**DiT** cross-attend VLM embedding，对 **state + 带噪 action chunk** 做 **flow matching 速度预测**；预训练加 **FLARE** 未来 latent 对齐；相对 N1 显著改善语言跟随与少样本 post-train。

## 开源核查（2026-09-28）

| 资源 | 状态 |
|------|------|
| [NVIDIA/Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T) | **已开源**（Apache 2.0；N1.5 权重 HF） |
| [nvidia/GR00T-N1.5-3B](https://huggingface.co/nvidia/GR00T-N1.5-3B) | **已发布**（NVIDIA Open Model License） |
