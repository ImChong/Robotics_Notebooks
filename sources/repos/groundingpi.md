# GroundingPI 官方代码仓库

> 来源归档

- **标题：** GroundingPI: A Grounding Foundation Model towards Physical Intelligence with Visual Primitives — Official Code
- **类型：** repo / grounding / visual-foundation-model
- **代码：** <https://github.com/groundingpi/GroundingPI>
- **项目页：** <https://groundingpi.github.io/>
- **论文：** <https://arxiv.org/abs/2609.39601>
- **模型权重：** <https://huggingface.co/GroundingPI/GroundingPI>
- **Hugging Face 论文页：** <https://huggingface.co/papers/2609.39601>
- **检查日期：** 2026-10-14
- **一句话说明：** 官方仓库包含 GroundingPI 的模型实现、训练与评测配置、vLLM/Transformers 服务、客户端示例和批量图像标注工作流。
- **沉淀到 wiki：** [GroundingPI 论文实体页](../../wiki/entities/paper-groundingpi.md)

## 仓库能力与复现入口

- **推理服务：** `run.py` 提供环境准备和服务启动入口；默认指南使用 vLLM Transformers backend，也提供原生 Transformers 参考服务。服务暴露 OpenAI-compatible 图文 API。
- **客户端：** `grounding_pi/` 提供 HTTP 客户端与可视化；`examples/` 包含单图预测和 JSONL 批处理示例。
- **评测：** `eval/` 与 `configs/eval/` 覆盖论文报告的 34 项 grounding benchmark；默认 recipe 为 smoke test，完整套件按 Evaluation Guide 配置运行。
- **训练：** `train/` 与 `configs/train/`、`configs/release/` 提供训练流程与配置。README 指出完整输入缓存、manifest 和相符的运行时需由使用者准备；默认训练指南面向对应的 PPU 环境。
- **模型结构：** MoonViT-V2（Kimi K3）视觉编码器、多模态投影器和 Qwen3-4B 语言骨干。
- **依赖与环境：** README 的 serving workflow 目标为 Linux x86_64 / Python 3.12；需先准备兼容 accelerator 的 Torch 与 vLLM 0.18.x 运行时，再安装项目依赖。
- **开放边界：** 仓库提供 grounding VLM checkpoint 的推理/训练/评测代码和配置；下游策略对比需要 StarVLA / OpenWAM 各自环境及策略 checkpoint，不等同于公开了直接执行机器人动作的通用策略。

## 许可提示

仓库原始贡献声明 Apache 2.0。组合权重及依赖还包含 Kimi K3 和其他第三方材料，其许可条件仍然适用；使用及再分发前应阅读根目录 `LICENSE` 与 `THIRD_PARTY_NOTICES.md`。

## 对 wiki 的映射

- 主实体页：[wiki/entities/paper-groundingpi.md](../../wiki/entities/paper-groundingpi.md)
- 项目主页归档：[sources/sites/groundingpi-github-io.md](../sites/groundingpi-github-io.md)
- 论文摘要归档：[sources/papers/groundingpi_arxiv_2609_39601.md](../papers/groundingpi_arxiv_2609_39601.md)
