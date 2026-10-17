# Intern-Decision 论文来源归档

## 元数据

- **题目：** Intern-Decision: Efficient Multimodal Decision Making with Autoregressive Masked Language Modeling
- **作者：** Tong Zhu, Zhe Xu, Zefeng He, Yunkai Chai, Xuyang Hu, Dehong Zheng, Yun Luo, Yafu Li, Qi Zhang, Yu Cheng
- **预印本：** <https://arxiv.org/abs/2610.22131>
- **PDF：** <https://arxiv.org/pdf/2610.22131>
- **提交日期：** 2026-10-15
- **代码：** <https://github.com/InternLM/Intern-Decision>
- **项目页：** <https://github.com/InternLM/Intern-Decision>
- **演示 / 权重：** <https://huggingface.co/spaces/internlm/intern-decision> / <https://huggingface.co/collections/internlm/intern-decision>
- **机构：** 上海人工智能实验室（Shanghai AI Lab；以项目官方仓库标识为准）

## 论文要解决的问题

交互式 agent 经常需要在有限候选项、离散动作或等级结果中选择。逐项生成自由文本会增加延迟，也让模型输出与环境接口更难约束。Intern-Decision 将多条结构化字段放入一个 JSON 骨架，以自回归掩码语言建模训练并直接返回候选概率。

## 方法与报告结果

- 模型系列包括 0.8B、2B 与 4B 多模态决策模型，基于 Qwen3.5 微调语言主干；视觉编码器和投影层冻结。支持 `choice`、`score` 与 `noul`（是/否）字段。
- 每个字段在完整 assistant JSON 输入骨架中以 `<decision>` 占位。训练标签使用普通 causal next-token 对齐；推理将 logits 限制到合法答案 token 后做 softmax，并映射回候选标签。多个字段共用一次因果前向计算。
- 七项测试套件上的 4B 模型未加权平均准确率为 90.02%，论文所列 Jev v1.13.0 为 88.74%。
- 固定 289 输入 token、三个字段、单张 RTX 4090 的平均本地请求延迟为 4B 44.16 ms、2B 33.28 ms、0.8B 33.98 ms；Jev 为 109.70 ms。测量使用 BF16、SDPA 与 Hugging Face 原生推理。
- 独立的 96 条已知分布校准 pilot 上，4B 温度校准后期望多类别 Brier / ECE 为 0.550 / 0.089；Jev 为 0.595 / 0.130。该 pilot 是小规模诊断测试，协议不同于七项准确率基准。
- Jevbench-Hard 的置信度交接实验把 42 条不确定样本交给 Atria-Dawn-Preview，准确率从 82/111（73.87%）升至 97/111（87.39%）；这是带外部大模型路由的系统结果，不是 Intern-Decision 单模型准确率。

## 对 wiki 的映射

- [paper-intern-decision.md](../../wiki/entities/paper-intern-decision.md)
- [AI Agent 评测](../../wiki/concepts/ai-agent-evaluation.md)

## 一手来源

- arXiv 摘要与论文：<https://arxiv.org/abs/2610.22131>
- 官方项目仓库：<https://github.com/InternLM/Intern-Decision>
- README 提供的模型集合：<https://huggingface.co/collections/internlm/intern-decision>
