# Xing4.1-Preview 官方部署资料与 vLLM 插件

- **官方仓库：** <https://github.com/XingChen-AGI/Xing4.1-Preview>
- **推理插件：** <https://github.com/XingChen-AGI/xing4-vllm-plugin>
- **机构：** 中国电信人工智能（TeleAI）
- **核对范围：** 发布仓库 `505f67f` 的中英文 README、vLLM 部署流程及 SGLang、昇腾入口；插件 `ba73ab0` 的注册入口、模型适配与解析器。
- **通读：** 已完成
- **模型：** [Xing4.1-Preview 模型卡](../models/xing4-1-preview.md)
- **论文：** [Xing 4 技术报告](../papers/telechat_xing4_arxiv_2611_22115.md)

发布仓库主要维护部署说明和若干昇腾运行脚本；vLLM 插件才提供模型类型兼容、MLA 识别、可选推测路径适配，以及可复用的思考/工具调用解析器。插件的 `xing4_1_preview` 注册入口以同名及 `xing4_0` 别名管理解析器，底层模型执行复用上游实现。报告中的 DSpark 和完整 TeleGym 后训练栈不属于这份部署插件的已核实内容。

自部署需核对 vLLM/插件版本、张量并行、上下文和并发配额，再测最短请求与工具返回格式。启用自定义模型代码前应固定版本并审查来源。未进行 GPU/NPU 运行验证。

**对应 wiki：** [Xing 4 技术报告](../../wiki/entities/paper-telechat-xing4.md)
