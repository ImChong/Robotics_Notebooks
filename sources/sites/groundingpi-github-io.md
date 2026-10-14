# GroundingPI 项目主页（groundingpi.github.io）

- **类型：** 网站 / 研究项目主页
- **入口：** <https://groundingpi.github.io/>
- **维护方：** GroundingPI 作者团队
- **论文：** [arXiv:2609.39601](https://arxiv.org/abs/2609.39601)；[Hugging Face 论文页](https://huggingface.co/papers/2609.39601)
- **代码：** <https://github.com/groundingpi/GroundingPI>（官方项目已公开代码，见[仓库归档](../repos/groundingpi.md)）
- **模型权重：** <https://huggingface.co/GroundingPI/GroundingPI>
- **核查日期：** 2026-10-14
- **一句话说明：** GroundingPI 官方项目页介绍以视觉基元为基础的 4B grounding 模型，并连接论文、GitHub 实现与 Hugging Face 权重。

## 项目与开放状态核查

官方项目主页和 GitHub README 将 GroundingPI 定位为面向 physical intelligence 的 grounding foundation model。GitHub README 列出开源源码、vLLM / Transformers 推理、批量标注、训练和 34 项 benchmark 评测指南；Hugging Face 提供 grounding 视觉语言 checkpoint。原始项目代码声明 Apache 2.0，但组合模型包含 Kimi K3 派生部分及第三方组件，使用或再分发权重时需按其上游许可和第三方说明逐项核对。

当前公开权重是 grounding 视觉语言 checkpoint。README 中的 StarVLA / OpenWAM 集成是骨干比较与下游策略训练工作流，需各自准备环境和策略 checkpoint；仓库说明并未提供一个开箱即用、直接输出动作的 GroundingPI 通用策略。

## 对 wiki 的映射

- 主实体页：[GroundingPI](../../wiki/entities/paper-groundingpi.md)
- 原始论文：[sources/papers/groundingpi_arxiv_2609_39601.md](../papers/groundingpi_arxiv_2609_39601.md)
- 官方代码：[sources/repos/groundingpi.md](../repos/groundingpi.md)

## 参考链接

- 项目页：<https://groundingpi.github.io/>
- 论文：<https://arxiv.org/abs/2609.39601>
- GitHub：<https://github.com/groundingpi/GroundingPI>
- 模型：<https://huggingface.co/GroundingPI/GroundingPI>
- Hugging Face 论文页：<https://huggingface.co/papers/2609.39601>
