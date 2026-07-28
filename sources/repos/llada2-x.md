# inclusionAI/LLaDA2.X

> 来源归档（ingest）

- **标题：** LLaDA2.X — Large Diffusion Language Models（LLaDA2.0 → 2.2）
- **类型：** repo
- **组织：** Inclusion AI / 蚂蚁集团（Ant Group）
- **代码：** <https://github.com/inclusionAI/LLaDA2.X>
- **权重（本 ingest 焦点）：** <https://huggingface.co/inclusionAI/LLaDA2.2-flash>
- **技术报告：** [`LLaDA2_2_tech_report.pdf`](https://github.com/inclusionAI/LLaDA2.X/blob/main/LLaDA2_2_tech_report.pdf)
- **License：** Apache License 2.0
- **入库日期：** 2026-07-28
- **一句话说明：** LLaDA2 系列官方入口仓：README（系列叙事 / 模型表 / 部署指针）、多版 tech report PDF、`figures/`；**不含**训练脚本与权重分片。LLaDA2.2 主打 **Levenshtein Editing** 的 agent-oriented dLLM。

## 开源核查（2026-07-28）

| 项 | 状态 |
|----|------|
| **模型权重** | **已开源（开放权重）** · HF / ModelScope `inclusionAI/LLaDA2.2-flash`（约 **192 GiB** / 32× safetensors；另有 2.0 / 2.1 / CAP 变体） |
| **技术报告** | **已发布** · `LLaDA2_2_tech_report.pdf`（11 页）；另有 `llada2_1_tech_report.pdf`、`tech_report.pdf`（2.0） |
| **本仓代码** | 仅 `LICENSE` + `README.md` + PDF + `figures/` — **无可运行训练 / 推理脚本** |
| **推理框架** | 系列推荐 [dInfer](https://github.com/inclusionAI/dInfer)（dLLM 推理）；HF 卡：长上下文 agent 建议 SGLang，但 **2.2 部署「coming soon」** |
| **微调框架** | [dFactory](https://github.com/inclusionAI/dFactory)（README 明示支持 LLaDA2.0 mini/flash；**2.2 是否已接入以仓内配置为准**） |
| **训练数据 / RL 全栈** | **未在本仓发布** |
| **许可证** | **Apache-2.0** |

## 入口速查（对齐 README）

| 路径 / 链接 | 作用 |
|-------------|------|
| `README.md` | 系列介绍、模型变体表、评测图、部署与 Citation |
| `LLaDA2_2_tech_report.pdf` | LLaDA2.2 技术报告（Levenshtein / L-EBPO / 128K / block routing） |
| `llada2_1_tech_report.pdf` / `tech_report.pdf` | 2.1 / 2.0 报告 |
| [HF collection LLaDA2.2](https://huggingface.co/collections/inclusionAI/llada22) | 2.2 权重集合 |
| [dInfer](https://github.com/inclusionAI/dInfer) | 高效 dLLM 推理（支持 LLaDA / LLaDA-MoE / LLaDA2） |
| [dFactory](https://github.com/inclusionAI/dFactory) | dLLM 微调 |

## Model Variants（README 表摘录）

| Model ID | 说明 |
|----------|------|
| `inclusionAI/LLaDA2.2-flash` | Agent-oriented MoE dLLM + Levenshtein Editing（本 ingest 焦点） |
| `inclusionAI/LLaDA2.1-mini` / `flash` | Token editing 加速版 |
| `inclusionAI/LLaDA2.0-mini` / `flash` | 首个 100B-scale MoE dLLM |
| `*-CAP` | Confidence-Aware Parallel 推理增强 |

## 与本仓库知识的关系

| 主题 | 关系 |
|------|------|
| [LLaDA2.2-flash](../../wiki/entities/llada2-2-flash.md) | 主实体：agentic dLLM 规格与开源部署 |
| [Diffusion Model](../../wiki/concepts/diffusion-model.md) | 离散文本 / block diffusion 交叉 |
| [真机策略 autoresearch 闭环](../../wiki/queries/real-robot-policy-autoresearch-harness.md) | 高吞吐 coding agent 后端选项 |
| [Kimi K3](../../wiki/entities/kimi-k3.md) | 开放权重 AR 旗舰对照 |

## 对 wiki 的映射

- 权重页：[`sources/sites/huggingface-inclusionai-llada2-2-flash.md`](../sites/huggingface-inclusionai-llada2-2-flash.md)
- ModelScope：[`sources/sites/modelscope-inclusionai-llada2-2-flash.md`](../sites/modelscope-inclusionai-llada2-2-flash.md)
- 技术报告：[`sources/papers/llada2_2_tech_report.md`](../papers/llada2_2_tech_report.md)
- 沉淀 **[`wiki/entities/llada2-2-flash.md`](../../wiki/entities/llada2-2-flash.md)**
