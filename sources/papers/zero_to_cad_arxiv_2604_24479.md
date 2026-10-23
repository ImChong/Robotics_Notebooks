# Zero-to-CAD: Agentic Synthesis of Interpretable CAD Programs at Million-Scale Without Real Data（arXiv:2604.24479）

> 来源归档（paper / arXiv；核查日期：2026-10-23）

- **标题：** Zero-to-CAD: Agentic Synthesis of Interpretable CAD Programs at Million-Scale Without Real Data
- **作者：** Mohammadmehdi Ataei、Farzaneh Askari、Kamal Rahimi Malekshan、Pradeep Kumar Jayaraman
- **机构：** Autodesk Research
- **arXiv：** <https://arxiv.org/abs/2604.24479>（2026-04-27 提交）
- **官方模型：** <https://huggingface.co/ADSKAILab/Zero-To-CAD-Qwen3-VL-2B>（[模型归档](../models/zero-to-cad-qwen3-vl-2b.md)）
- **精选数据集：** <https://huggingface.co/datasets/ADSKAILab/Zero-To-CAD-100k>（[数据集归档](../datasets/zero-to-cad-100k.md)）
- **项目实体：** [Zero-to-CAD](../../wiki/entities/zero-to-cad.md)

## 论文摘要级要点

- 研究目标是生成带有参数化构造历史的可执行 CAD 程序，弥补只包含 B-rep 或网格、缺少设计步骤的大规模 3D 数据资源。
- 方法将大语言模型置于带反馈的 CAD 环境中，反复生成、执行和验证程序，并调用工具与文档检索来提高几何有效性和操作多样性。
- 作者报告合成约 **100 万**条可读、可编辑的 CAD 构造序列，并另行发布基于几何多样性挑选的 **100K 子集**。
- 配套视觉语言模型展示了从多视角图像重建可编辑 CAD 程序的下游用法；发布模型卡提供了推理入口与评测数字。

## 开源核查

截至 2026-10-23，官方 Hugging Face 已公开 Qwen3-VL-2B 微调权重和 Zero-to-CAD-100k 数据集，两个页面均标注 Apache-2.0。当前归档依据未发现独立的 GitHub 源码仓库链接；开放模型权重与数据集不等同于发布了论文中完整的 agentic 数据合成系统实现。

## 对 Robotics_Notebooks 的映射

- [Zero-to-CAD 实体页](../../wiki/entities/zero-to-cad.md)
- [文字生成 CAD](../../wiki/concepts/text-to-cad.md)
