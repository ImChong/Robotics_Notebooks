# The Ultimate MoE Training Handbook / MoE 训练终极手册

- **类型：** 官方技术手册
- **作者：** Nouamane Tazi、Leandro von Werra、Thomas Wolf（Hugging Face）
- **发布日期：** 2026-11-05
- **核查日期：** 2026-11-06
- **原文：** <https://huggingface.co/spaces/huggingface/moe-blogpost>
- **图表合集：** <https://huggingface.co/spaces/huggingface/moe-gallery>
- **配套脚本：** <https://gist.github.com/nmntz/5de952d686f66d7bce5bce8cba9d7139>
- **训练实现：** <https://github.com/NVIDIA/Megatron-LM>，手册示例固定提交 `d6316eb45bf1bfbb6f1f848c0b56e55166564d80`。

## 已核实内容

手册面向希望从稠密模型转向混合专家模型训练的读者。目录涵盖硬件与网络检查、训练环境、数据预处理、稠密基线与首个 MoE，随后讨论路由、负载均衡、专家设计、训练系统及推理与强化学习。

首个训练示例使用 64 个专家、每 token 选择 8 个专家；手册给出的模型规模为 6.86B 总参数、约 1.22B 活跃参数。示例配置包含 expert parallelism、all-to-all token 分发及 grouped GEMM；对应稠密基线与 MoE 保持相同数据和主要训练设置，以便比较。

## 核验状态与后续工作

**草稿，尚未完成 ingest。** 官方网页可打开，但当前读取结果被截断，未完整阅读手册各章及附录，也未完整核查配套脚本和 Megatron 源码。尚不能确认全部实验条件、局限与代码开放范围。

- 通读手册全文与附录，核实实验设置、结论和适用边界。
- 阅读配套脚本和固定版本 Megatron 的实际运行入口。
- 核对现有知识节点后编写实体页，补充基于原文的 Mermaid 流程图和基于源码的运行时序图。
- 完成仓库要求的本地检查后再将 PR 转为可评审状态。
