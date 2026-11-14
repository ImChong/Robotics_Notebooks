# rasbt/LLMs-from-scratch

> 来源归档

- **标题：** Build a Large Language Model (From Scratch) — 官方代码仓库
- **类型：** repo
- **作者：** Sebastian Raschka（GitHub: [rasbt](https://github.com/rasbt)）
- **链接：** <https://github.com/rasbt/LLMs-from-scratch>
- **配套图书：** Manning, 2024 — ISBN 978-1633437166 — [Manning 书页](http://mng.bz/orYv)
- **入库日期：** 2026-07-11
- **复核日期：** 2026-11-14（本地克隆源码实读核查，HEAD `0311973e`，2026-11-09；`Apache-2.0`）
- **核对范围：** `README.md`（主线目录、Bonus、Companion Book）、`pkg/llms_from_scratch/README.md`、`ch02/README.md`、`ch03/README.md`、`appendix-E/README.md`，以及可运行主脚本 `ch04/01_main-chapter-code/gpt.py`、`ch05/01_main-chapter-code/gpt_train.py`、`ch05/01_main-chapter-code/gpt_generate.py`、`ch06/01_main-chapter-code/gpt_class_finetune.py`、`ch07/01_main-chapter-code/gpt_instruction_finetuning.py`、`ch07/01_main-chapter-code/ollama_evaluate.py`
- **通读：** 已完成
- **代码：** <https://github.com/rasbt/LLMs-from-scratch>（**已开源**；主线 Ch 2–7 + 附录 A–E 及 `llms-from-scratch` PyPI 包均可运行）
- **一句话说明：** 与同名书同步的 **PyTorch 从零实现 GPT 类 LLM** 教学仓库：文本分词 → 注意力 → GPT 架构 → 预训练 → 分类微调 → 指令微调；附 LoRA、KV cache、DPO、多架构 from-scratch bonus 与加载大模型权重示例。
- **沉淀到 wiki：** 是 → [`wiki/entities/llms-from-scratch-raschka.md`](../../wiki/entities/llms-from-scratch-raschka.md)

## 为什么值得保留

- 社区 **~105k stars**（2026-09-16）的 LLM 教育标杆之一，与 Karpathy *Zero to Hero*（micrograd / nanoGPT）形成 **视频驱动 vs 书+notebook 结构化** 互补。
- 主线 **不依赖 HuggingFace 等高层 LLM 库**，用纯 PyTorch 把 tokenizer、MHA、GPT block、预训练环、SFT 拆开讲清 —— 对理解 [VLA](../wiki/methods/vla.md) / [动作分词](../wiki/formalizations/vla-tokenization.md) 背后的 **序列建模与离散 token 接口** 有直接帮助。
- 笔记本可在普通笔记本 GPU 上跑通；bonus 含 Llama/Qwen/Gemma 等 **from-scratch 变体** 与 DPO 对齐示例，便于从教学小模型过渡到工业架构直觉。

## 仓库结构（主线章节）

| 章节 | 主题 | 主 notebook / 脚本 |
|------|------|-------------------|
| Ch 1 | 理解 LLM（概念，无代码） | — |
| Ch 2 | 文本数据与分词、dataloader | `ch02.ipynb` |
| Ch 3 | 注意力机制（含 MHA） | `ch03.ipynb` |
| Ch 4 | 从零实现 GPT | `ch04.ipynb`, `gpt.py` |
| Ch 5 | 无标注数据预训练与 GPT-2 加载推理 | `ch05.ipynb`, `gpt_train.py`, `gpt_generate.py` |
| Ch 6 | 文本分类微调（SMS Spam） | `ch06.ipynb`, `gpt_class_finetune.py` |
| Ch 7 | 指令微调（instruction SFT）与评分 | `ch07.ipynb`, `gpt_instruction_finetuning.py`, `ollama_evaluate.py` |
| PyPI 可选包 | 章节模块与 KV cache / Llama3 封装 | `pkg/llms_from_scratch/` |
| 附录 A | PyTorch 入门与 DDP 示例 | `appendix-A/`（`DDP-script.py` 等） |
| 附录 D | 训练环增强（LR schedule 等） | `appendix-D.ipynb` |
| 附录 E | LoRA 参数高效微调 | `appendix-E.ipynb` |

**Bonus（节选，2026-09 复核）：**

| 类别 | 代表材料 |
|------|----------|
| 分词 / 数据 | BPE from scratch、SimpleTokenizerV3、dataloader 直觉 |
| 注意力变体 | KV cache、GQA、MLA、滑动窗口、Gated DeltaNet、DeepSeek Sparse Attention (DSA)、Cross-Layer KV Sharing、MoE |
| 架构 from-scratch | GPT→Llama 转换；Llama 3.2、Qwen3/Qwen3.5、Gemma 3/4、Olmo 3、Tiny Aya 等 |
| 训练 / 对齐 | LR schedule、Gutenberg 预训练、DPO 偏好对齐、Ollama 评估、用户界面 demo |
| 续作交叉 | [`reasoning-from-scratch`](https://github.com/rasbt/reasoning-from-scratch) 推理 scaling、GRPO、MMLU/MATH 评测 bonus 链到本仓 README |

## 核心摘录

### 1) 教学哲学：与 ChatGPT 同源但可亲手跑通

- **要点：** 用小但完整的 GPT 复现 **预训练 → 任务微调 → 指令对齐** 全链路，方法论与大规模基础模型一致；读者改的是实现细节而非黑盒 API。
- **对 wiki 的映射：** [`wiki/entities/llms-from-scratch-raschka.md`](../../wiki/entities/llms-from-scratch-raschka.md)

### 2) Ch 3–4：注意力与 GPT 是 VLA 骨干的前置课

- **要点：** 缩放点积注意力、多头注意力、位置编码、残差+LayerNorm 均在纯代码中实现，与 [Transformer](../../wiki/concepts/transformer.md) 概念页一一对应。
- **对 wiki 的映射：** [`wiki/concepts/transformer.md`](../../wiki/concepts/transformer.md)

### 3) Ch 5–7 + 附录 E：预训练与微调代码实况（源码核查）

- **要点：**
  - `ch05/01_main-chapter-code/gpt_train.py` 用 `tiktoken("gpt2")` 在短篇文本 `the-verdict.txt` 上从零预训练缩减上下文的 `GPTModel`（`context_length=256`，10 epochs）并保存 `model.pth`；同目录 `gpt_generate.py` 则从 `openaipublic.blob.core.windows.net/gpt-2/models` 下载 OpenAI GPT-2 TensorFlow checkpoint 并映射到 `GPTModel` 做 temperature / top-k 采样。
  - `ch06/01_main-chapter-code/gpt_class_finetune.py` 加载 GPT-2 124M（或 `--test_mode` 小模型），冻结骨干参数后仅微调最后一个 `TransformerBlock`、`final_norm` 与替换后的 2 类 `out_head`（取末 token logits 计算交叉熵）。
  - `ch07/01_main-chapter-code/gpt_instruction_finetuning.py` 加载 `gpt2-medium (355M)`（或 `--test_mode` 小模型），以 Alpaca 模板格式化 `instruction-data.json`、`custom_collate_fn` 将首个 `<|endoftext|>` 之后的 padding target 置为 `ignore_index=-100`，微调后输出 `instruction-data-with-response-standalone.json`，再交由 `ollama_evaluate.py` 调用本地 Ollama（`llama3`）按 0–100 打分。
- **对 wiki 的映射：** [`wiki/entities/llms-from-scratch-raschka.md`](../../wiki/entities/llms-from-scratch-raschka.md)、[`wiki/methods/vla.md`](../../wiki/methods/vla.md)、[`roadmap/depth-vla.md`](../../roadmap/depth-vla.md) Stage 0 前置

### 4) Bonus 架构谱系（2025–2026 持续扩展）

- **要点：** Ch 4–5 bonus 已覆盖 GQA/DSA/MoE 等工业注意力与稀疏结构；Ch 5 架构族从 Llama 3.2 扩展到 **Qwen3.5、Gemma 4** 等「同书骨架、换骨干」练习，帮助读者从教学 GPT 过渡到当前开源 LLM 族谱。
- **对 wiki 的映射：** [`wiki/entities/llms-from-scratch-raschka.md`](../../wiki/entities/llms-from-scratch-raschka.md)、[`wiki/concepts/transformer.md`](../../wiki/concepts/transformer.md)

## 关联原始资料

- [`sources/courses/rasbt_llms_from_scratch_youtube.md`](../courses/rasbt_llms_from_scratch_youtube.md) — YouTube 七章配套视频（与书章对齐）

## 推荐继续阅读（外部）

- [Build A Reasoning Model (From Scratch)](https://github.com/rasbt/reasoning-from-scratch) — 续作：推理时 scaling、RLVR/GRPO 等
- [Karpathy Neural Networks: Zero to Hero](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ) — 更短、更视频驱动的从零 NN/LLM 路线
