# Xing4.1-Preview 官方模型卡与权重

- **Hugging Face：** <https://huggingface.co/XingChen-AGI/Xing4.1-Preview>
- **ModelScope：** <https://www.modelscope.ai/models/XingChen-AGI/Xing4.1-Preview>
- **原生权重：** [MindSpore 模型](https://huggingface.co/XingChen-AGI/Xing4.1-Preview-MindSpore)
- **发布机构：** 中国电信人工智能（TeleAI）
- **许可证：** Hugging Face 发布卡标记 Apache-2.0，使用时以对应工件 LICENSE 和平台条款为准。
- **核对范围：** Hugging Face 模型卡及文件清单、公开配置；通过 ModelScope 仓库 API 核对 README 和文件清单。
- **通读：** 已完成
- **原文：** [Xing 4 技术报告 v1](../papers/telechat_xing4_arxiv_2611_22115.md)

Hugging Face 快照为 `6e54e416f8499a14d3d6f6729b0e1c917b20bc61`。两站列出的 46 个 PyTorch safetensors 分片逐个比对了平台 SHA-256 元数据，均一致（合计约 573 GB），另有索引、模板、tokenizer 和 Python 模型配置/实现。公开配置为 45 层、256 个路由专家、每 token 路由 8 个、最大位置数 1,048,576，声明模型架构 `DeepseekV32ForCausalLM`。这里只核对发布物，未下载整套权重或跑推理。

模型卡中的 Terminal-Bench 2.1 对照并非固定 harness：Preview 用 terminus-2，对比模型用 Claude Code。SWE-Bench Pro V2 的 74.5 是 Hard51 子集得分。不能据此直接判断其它 harness、任务或服务配置的结果。

**对应 wiki：** [Xing 4 技术报告](../../wiki/entities/paper-telechat-xing4.md)
