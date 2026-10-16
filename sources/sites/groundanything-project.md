# GroundAnything 项目主页

- **类型：** 网站 / 研究项目主页
- **入口：** <https://groundingpi.github.io/groundanything/>
- **维护方：** GroundAnything 作者团队
- **论文：** [arXiv:2609.39600](https://arxiv.org/abs/2609.39600)；[Hugging Face 论文页](https://huggingface.co/papers/2609.39600)
- **代码：** <https://github.com/groundingpi/GroundAnything>（已公开；见[仓库归档](../repos/groundanything.md)）
- **模型权重（DLM）：** <https://huggingface.co/GroundingPI/GroundAnything>
- **模型权重（自回归 VLM）：** <https://huggingface.co/GroundingPI/GroundAnything-VLM>
- **HF Demo：** <https://huggingface.co/spaces/GroundingPI/GroundAnything-VLM>
- **核查日期：** 2026-10-16
- **一句话说明：** 官方项目页汇总 GroundAnything 论文、模型权重、演示、解码分析和 benchmark 结果，并链接到已发布的源码仓库。

## 项目与开放状态核查

项目页列出论文、GitHub 仓库、两种 Hugging Face checkpoint 和在线 Demo。新闻记录注明源码、部署与批量标注指南、完整评测工作流已于 2026-10-03 发布；DLM 与自回归 VLM 权重于 2026-10-01 发布。因此项目属于**已开源**，具备模型权重、推理、训练与评测资源。

论文与项目页描述的 GroundAnything 主版本使用分块扩散和熵引导解码；GroundAnything-VLM 则是独立的自回归 checkpoint。公开资源提供的是图像 grounding 能力，不是直接输出机器人动作的控制策略。

## 对 wiki 的映射

- 主实体页：[GroundAnything](../../wiki/entities/paper-groundanything.md)
- 原始论文：[sources/papers/groundanything_arxiv_2609_39600.md](../papers/groundanything_arxiv_2609_39600.md)
- 官方代码：[sources/repos/groundanything.md](../repos/groundanything.md)

## 参考链接

- 项目主页：<https://groundingpi.github.io/groundanything/>
- 论文：<https://arxiv.org/abs/2609.39600>
- GitHub：<https://github.com/groundingpi/GroundAnything>
- GroundAnything DLM：<https://huggingface.co/GroundingPI/GroundAnything>
- GroundAnything-VLM：<https://huggingface.co/GroundingPI/GroundAnything-VLM>
- Hugging Face 论文页：<https://huggingface.co/papers/2609.39600>
