# Inkling 2 Mini：Hugging Face 模型卡与开放权重

- **模型卡 / 权重：** https://huggingface.co/thinkingmachines/Inkling-2-Mini
- **文件列表：** https://huggingface.co/thinkingmachines/Inkling-2-Mini/tree/main
- **官方技术博客：** https://thinkingmachines.ai/news/inkling-2-mini/
- **发布方：** Thinking Machines Lab
- **发布日期：** 2026-11-20
- **入库日期：** 2026-11-21
- **类型：** 模型卡与官方推理权重（非论文）
- **许可：** Apache-2.0；另附 Model Acceptable Use Policy
- **核对范围：** 模型卡第 1–8 节（概述、硬件/软件、模型属性、训练、评测、安全、限制、法律）；核对模型库文件列表、README 所列 Transformers / vLLM / SGLang 示例，及博客对架构的补充解释。
- **通读：** 已完成

## 权重身份与发布边界

- 模型 ID：thinkingmachines/Inkling-2-Mini；26.4B 总参数，3.4B 单遍活跃参数；循环计算后每 token 的有效激活量约 5.4B。
- 24 层 decoder-only MoE：前 6 层一次，中间 12 层共享权重重复 3 次，末 6 层一次，合计 48 次层应用；每 token 每层路由到 128 个专家中的 8 个。
- 原生图文多模态：输入文本 + 图像、输出文本；图像通过 ViT 编码后与文本进入共享表示；上下文窗口上限标称 1M tokens。
- 发布 checkpoint 约 20GB（模型库文件列表总量约 20.2GB）；混合 NVFP4（20 层）与 MXFP8（末 4 层）。这些数字不是低于 20GB 内存即可运行的保证。
- 官方支持 NVIDIA DGX Spark、Apple Silicon 内存 >=32GB；其他 GPU 建议 VRAM >=32GB。推理框架列出 Transformers、SGLang、vLLM、Ollama 等，实际兼容性随版本和平台变化。
- **开放状态：** 权重、配置、tokenizer/chat template 和模型卡公开；模型卡展示推理使用示例，未将完整预训练数据、训练脚本或官方模型内部 kernel 源码列为可复现发布物。不能把开放权重理解为完整训练方案开源。

## 模型卡直接给出的推理入口

- Transformers（模型卡示例）：AutoProcessor.from_pretrained 与 AutoModelForMultimodalLM.from_pretrained 加载模型；processor.apply_chat_template 编码消息；model.generate 生成；processor.decode 取出文本。
- vLLM（页面示例）：vllm serve "thinkingmachines/Inkling-2-Mini"，向 /v1/chat/completions 发起 OpenAI 兼容请求。
- SGLang（页面示例）：python3 -m sglang.launch_server --model-path "thinkingmachines/Inkling-2-Mini" --host 0.0.0.0 --port 30000。

## 模型卡评测与边界

| 指标 | Inkling 2 Mini | 口径 |
|---|---:|---|
| DeepSWE v1.1 | 48.2% | mini-swe-agent v2.4.6 |
| Vals Index v2.2 | 46.5% | 发布方汇总 |
| BrowseComp | 79.0% | with compaction |
| DeepSearchQA | 85.3% | with compaction |
| MCP Atlas Public | 82.7% | 发布方汇总 |
| OmniDocBench v1.6 | 92.8% | 发布方汇总 |
| AA-Omniscience v1.1 | -8.6 | 事实知识不足以凭循环增加 |
| ComplexConstraints v1.1 | 33.4% | 较强约束指令跟随仍有限 |

- 同表不同模型/任务使用的 harness、工具与配置可能不同，不能直接当作同条件独立复现排行榜。
- 模型卡披露多轮对话、中文表达、长时间代码调试与高风险工具调用的局限；建议 sandbox、人类审批不可逆操作。
- 模型卡不报告机器人真机闭环指标；将其用于具身机器人语义 Agent 是部署方向的讨论，**不是该模型已验证的机器人策略能力**。

## 对 wiki 的映射

- [Inkling 2 Mini 模型实体](../../wiki/entities/inkling-2-mini.md)
- [模型架构纵深路线](../../roadmap/depth-model-architecture.md)
