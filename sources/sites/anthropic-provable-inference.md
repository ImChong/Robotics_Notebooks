# Progress on provable inference for large language models（大语言模型可证明推理的进展）

> 官方研究博客来源归档

- **组织：** Anthropic Frontier Red Team 与合作者
- **日期：** 2026-11-02；入库：2026-11-04
- **原文：** https://www.anthropic.com/research/provable-inference
- **技术报告：** https://www.anthropic.com/document/flashprove-technical-report
- **代码：** https://github.com/anthropics/FlashProve
- **项目节点：** [FlashProve](../../wiki/entities/flashprove.md)

## 内容要点

FlashProve 将模型权重承诺与事后零知识证明结合，验证指定输入上的输出是否符合承诺的整数模型；权重及专家选择受隐藏，模型架构不隐藏。官方报告 Llama 4 Maverick 4-bit、1024 token 窗口及 24 请求批量下，H100 上约 110 输入 token/s、每请求约 18.78 MB；每个证明检查一个 token，而非自动证明整段生成。与 vLLM 的 FP8、8 H100 比较是 GPU 时间口径，约 70–95 倍。

Lean 协议性质基于随机预言机模型；快速 C++ 实现尚未形式化证明与 Lean 等价。整数模型的贪心解码、上下文长度、侧信道和事后延迟均限制实际使用。

## 阅读范围与待办

已阅读官方博客及仓库 README、GPU 快速启动与目标示例脚本；23 页技术报告尚未完成通读，协议实现和形式化假设仍待进一步核查，项目页暂列草稿。
