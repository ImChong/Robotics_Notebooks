# inclusionAI/LLaDA2.2-flash（Hugging Face）

> 来源归档（ingest）

- **标题：** LLaDA2.2-flash — Hugging Face 模型卡与开放权重
- **类型：** site / model card（Hugging Face Hub）
- **组织：** Inclusion AI（蚂蚁集团）
- **官方入口：** <https://huggingface.co/inclusionAI/LLaDA2.2-flash>
- **License：** Apache-2.0
- **关联仓 / 报告：**
  - <https://github.com/inclusionAI/LLaDA2.X>
  - <https://github.com/inclusionAI/LLaDA2.X/blob/main/LLaDA2_2_tech_report.pdf>
- **入库日期：** 2026-07-28
- **一句话说明：** LLaDA2.2-flash **主权重发布面**：`text-generation` MoE **dLLM**（`model_type: llada2_moe`），约 **192 GiB** BF16 safetensors（32 分片），附 transformers **custom code**（`trust_remote_code`）；面向长上下文 tool-use / 多轮 agent。

## 开源核查（2026-07-28）

| 项 | 状态 |
|----|------|
| **gated** | **否** |
| **权重** | **已开源** · `model-00000-of-00032.safetensors` … `00031`（合计约 **205.8 GB** / **191.7 GiB**） |
| **推理辅助代码** | `configuration_llada2_moe.py`、`modeling_llada2_moe.py`、`tokenization_llada2.py`、`tool_declaration_ts.py`、`chat_template.jinja` |
| **架构类名** | `LLaDA2MoeModelLM`（`AutoModelForCausalLM`） |
| **训练全栈** | **未随权重仓发布** |
| **镜像** | ModelScope `inclusionAI/LLaDA2.2-flash`（见 [modelscope-inclusionai-llada2-2-flash.md](./modelscope-inclusionai-llada2-2-flash.md)） |
| **SGLang** | 模型卡写明 **deployment support coming soon** |

## 模型卡要点（README 归纳）

- **定位：** agent-oriented diffusion LM；引入 **Levenshtein Editing**（`DELETE` / `INSERT` 控制 token；实现侧 `generate` 使用 `delete_token_id` / `split_token_id`）。
- **规格：** 非嵌入总参 **100B**；32 层；32 attn heads；RoPE；词表 **157,184**；上下文 **128K**（`max_position_embeddings: 131072`）。
- **MoE（config）：** `num_experts: 256`，`num_experts_per_tok: 8`，`num_shared_experts: 1`，`expert_capacity` / block pool **48**，`block_size: 32`。
- **评测默认：** `temperature=1.0`，`block_length=32`，`threshold=0.5`，`editing_threshold=0.0`，128K；SWE 用 Claude Code scaffold。
- **Best practices：** 稳定默认 `block_length=32`、`temperature=0.0`；按速度–质量调 `threshold` / `editing_threshold` / `max_post_steps`；长上下文 agent 推荐 SGLang（待正式支持）。

## `generate` 关键参数（custom code）

| 参数 | 默认 | 含义 |
|------|------|------|
| `block_length` | 32 | 块扩散块长 |
| `threshold` | 0.5 | M2T 解掩置信度阈值 |
| `editing_threshold` | 0.0 | T2T 改写阈值 |
| `max_post_steps` | 16 | 原 mask 消解后的精修步数 |
| `delete_token_id` / `split_token_id` | 156930 / 156931 | 删除 / 插入槽（SPLIT） |

## 对 wiki 的映射

- [LLaDA2.2-flash](../../wiki/entities/llada2-2-flash.md) — 「开源状态 / 工程实践 / 部署」
- [llada2-x.md（GitHub）](../repos/llada2-x.md)
- [llada2_2_tech_report.md](../papers/llada2_2_tech_report.md)
