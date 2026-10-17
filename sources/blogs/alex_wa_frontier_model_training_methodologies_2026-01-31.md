# Frontier model training methodologies（Alex Wa）

> 来源归档

- **标题：** frontier model training methodologies
- **类型：** blog（Alex Wa's Blog）
- **作者：** Alex Wa
- **原始链接：** <https://djdumpling.github.io/2026/01/31/frontier_training.html>
- **发布日期：** 2026-01-31
- **入库日期：** 2026-10-17
- **一句话说明：** 汇总 SmolLM3、Intellect 3、Hermes 4、gpt-oss-120b、Kimi K2、DeepSeek-R1 与 Trinity 等开放权重前沿模型的训练方法，覆盖架构、稳定性、分词器、优化器、数据配比、中训、SFT、偏好优化与强化学习。

## 内容摘要（归纳，非全文）

文章以 Hugging Face 的 SmolLM3 报告为主要结构，横向对照多个模型报告与技术说明，重点讨论训练方法而非算力基础设施。

- **训练决策与实验：** 先确立产品目标和评测集，再从成熟基线出发；通过快速、可靠、单变量消融判断改动是否有帮助，并重视多随机种子一致性。
- **架构：** 对比 dense 与 MoE、MHA/GQA/MLA、RoPE/RNoPE 及长上下文扩展。文章将 dense + GQA + RoPE/RNoPE 作为资源受限或追求快速迭代时的稳健起点；MoE 的收益依赖负载均衡与路由管理。
- **数值稳定性：** 汇总 z-loss、logit softcapping、QK-norm、RMSNorm、权重衰减与梯度裁剪等方案，并指出高学习率、异常数据批次、专家负载失衡和存储/基础设施问题是常见故障来源。
- **数据与预训练：** 强调 tokenizer 应匹配目标语言和领域；去重、过滤、污染检查及数据质量度量需前置。多阶段数据配比与训练后段注入高质量、推理型数据会影响最终能力。
- **优化与规模：** AdamW 仍是默认选择；Muon 可能有帮助，但需要额外并行通信与缩放处理。缩放定律提供指导，最终架构也要兼顾推理成本与稀疏化收益。
- **后训练：** 将 mid-training、SFT、偏好优化、RL/RLVR 和蒸馏视为形成推理、工具使用等行为的重要阶段；按可验证奖励与目标能力选择后训练方法，并持续评估安全性。
- **训练运维：** 长跑前验证吞吐、数据加载、序列打包、检查点与张量并行随机种子；训练中监控 loss、评测和基础设施告警。

## 对机器人知识库的映射

| 主题 | 关联 |
|------|------|
| 机器人基础模型训练配方 | 可作为理解预训练数据、训练稳定性和阶段式训练的通用参考；迁移到机器人时仍需单独验证动作数据、闭环评测与具身部署约束 |
| 机器人后训练 | 与 [通用机器人后训练](../../wiki/concepts/universal-post-training-robotics.md) 的 LLM 配方类比背景有关，但本文讨论的是通用 LLM 训练，不提供机器人实验或机器人专用配方 |

本次仅归档原始博客摘要，暂不单独新增 wiki 页面：文章覆盖面广，但不是机器人实证研究；后续若整理机器人基础模型训练路径，可按具体议题引用。

## 参考链接

- [SmolLM3 技术报告](https://huggingface.co/blog/smollm3)
- [Intellect 3 技术报告](https://arxiv.org/abs/2512.04047)
- [Hermes 4](https://arxiv.org/abs/2508.18255)
- [gpt-oss-120b](https://arxiv.org/abs/2508.10925)
- [Kimi K2](https://arxiv.org/abs/2507.20534)
- [DeepSeek-R1](https://arxiv.org/abs/2501.12948)
- [Trinity 系列](https://github.com/Arcee-AI/Trinity)

> 注：以上链接为原文引用对象的延伸阅读入口；具体模型版本与报告内容以各自官方页面为准。
