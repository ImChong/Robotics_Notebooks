# GroundAnything 官方代码仓库

> 来源归档

- **标题：** GroundAnything: Reconciling Parallel Decoding with Precise Visual Grounding at Flash Speed — Official Code
- **类型：** repo / visual-grounding / diffusion-language-model
- **代码：** <https://github.com/groundingpi/GroundAnything>
- **项目页：** <https://groundingpi.github.io/groundanything/>
- **论文：** <https://arxiv.org/abs/2609.39600>
- **模型权重（DLM）：** <https://huggingface.co/GroundingPI/GroundAnything>
- **模型权重（自回归 VLM）：** <https://huggingface.co/GroundingPI/GroundAnything-VLM>
- **Hugging Face 论文页：** <https://huggingface.co/papers/2609.39600>
- **检查日期：** 2026-10-16
- **一句话说明：** 官方仓库提供 GroundAnything 的模型服务、图像 grounding 客户端、训练配置和 30 项 benchmark 评测工作流。

## 仓库能力与复现入口

- **推理服务：** `run.py` 提供安装与服务启动入口；GroundAnything DLM 使用定制 SGLang serving 和熵引导 / 自推测解码，GroundAnything-VLM 可走同仓 SGLang 自回归服务，也提供 vLLM Transformers backend 路径。
- **客户端：** `grounding_anything/` 提供 HTTP 客户端与可视化，`examples/` 包含单图预测和 JSONL 批量标注示例；接口接受图像与文本指令，返回点或边界框等结构化坐标。
- **任务范围：** README 列出指代表达、目标定位、文本区域 grounding、文档版面、点定位、GUI grounding 等任务。
- **评测：** `eval/` 与配置覆盖论文报告的 30 项 benchmark；完整评测需依照说明准备数据与路径，默认 recipe 是少量样本 smoke test。
- **训练：** `train/`、`configs/train/` 与 `configs/release/` 包含分阶段训练工作流；训练缓存、manifest 和相符硬件环境需另行准备。
- **运行环境：** 服务指南目标为 Linux x86_64 / Python 3.12；README 列出的测试加速器包括 NVIDIA B300、B200、H200、H800 与 PPU。此列表不意味着这些路径在其他设备上已验证。
- **许可提示：** 仓库声明项目自有贡献采用 Apache 2.0；模型组件和第三方依赖仍有各自适用条件。部署或再分发时应核对仓库与模型卡的当前许可说明。

## 对 wiki 的映射

- 主实体页：[wiki/entities/paper-groundanything.md](../../wiki/entities/paper-groundanything.md)
- 项目主页归档：[sources/sites/groundanything-project.md](../sites/groundanything-project.md)
- 论文摘要归档：[sources/papers/groundanything_arxiv_2609_39600.md](../papers/groundanything_arxiv_2609_39600.md)
