# rasbt/LLMs-from-scratch

> 来源归档

- **标题：** Build a Large Language Model (From Scratch) — 官方代码仓库
- **类型：** repo
- **作者：** Sebastian Raschka（GitHub: [rasbt](https://github.com/rasbt)）
- **链接：** <https://github.com/rasbt/LLMs-from-scratch>
- **配套图书：** Manning, 2024 — ISBN 978-1633437166 — [Manning 书页](http://mng.bz/orYv)
- **入库日期：** 2026-07-11
- **一句话说明：** 与同名书同步的 **PyTorch 从零实现 GPT 类 LLM** 教学仓库：文本分词 → 注意力 → GPT 架构 → 预训练 → 分类微调 → 指令微调；附 LoRA、KV cache、多架构 bonus 与加载大模型权重示例。
- **沉淀到 wiki：** 是 → [`wiki/entities/llms-from-scratch-raschka.md`](../../wiki/entities/llms-from-scratch-raschka.md)

## 为什么值得保留

- 社区 **~99k stars** 的 LLM 教育标杆之一，与 Karpathy *Zero to Hero*（micrograd / nanoGPT）形成 **视频驱动 vs 书+notebook 结构化** 互补。
- 主线 **不依赖 HuggingFace 等高层 LLM 库**，用纯 PyTorch 把 tokenizer、MHA、GPT block、预训练环、SFT 拆开讲清 —— 对理解 [VLA](../wiki/methods/vla.md) / [动作分词](../wiki/formalizations/vla-tokenization.md) 背后的 **序列建模与离散 token 接口** 有直接帮助。
- 笔记本可在普通笔记本 GPU 上跑通；bonus 含 Llama/Qwen/Gemma 等 **from-scratch 变体** 与 DPO 对齐示例，便于从教学小模型过渡到工业架构直觉。

## 仓库结构（主线章节）

| 章节 | 主题 | 主 notebook / 脚本 |
|------|------|-------------------|
| Ch 1 | 理解 LLM（概念，无代码） | — |
| Ch 2 | 文本数据与分词、dataloader | `ch02.ipynb` |
| Ch 3 | 注意力机制（含 MHA） | `ch03.ipynb` |
| Ch 4 | 从零实现 GPT | `ch04.ipynb`, `gpt.py` |
| Ch 5 | 无标注数据预训练 | `ch05.ipynb`, `gpt_train.py` |
| Ch 6 | 文本分类微调 | `ch06.ipynb` |
| Ch 7 | 指令微调（instruction SFT） | `ch07.ipynb` |
| 附录 A | PyTorch 入门 | `appendix-A/` |
| 附录 D | 训练环增强（LR schedule 等） | `appendix-D.ipynb` |
| 附录 E | LoRA 参数高效微调 | `appendix-E.ipynb` |

**Bonus（节选）：** BPE from scratch、KV cache、GQA/MoE/滑动窗口注意力、GPT→Llama 转换、DPO、Qwen3/Llama3.2/Gemma 等 from-scratch 实现。

## 核心摘录

### 1) 教学哲学：与 ChatGPT 同源但可亲手跑通

- **要点：** 用小但完整的 GPT 复现 **预训练 → 任务微调 → 指令对齐** 全链路，方法论与大规模基础模型一致；读者改的是实现细节而非黑盒 API。
- **对 wiki 的映射：** [`wiki/entities/llms-from-scratch-raschka.md`](../../wiki/entities/llms-from-scratch-raschka.md)

### 2) Ch 3–4：注意力与 GPT 是 VLA 骨干的前置课

- **要点：** 缩放点积注意力、多头注意力、位置编码、残差+LayerNorm 均在纯代码中实现，与 [Transformer](../../wiki/concepts/transformer.md) 概念页一一对应。
- **对 wiki 的映射：** [`wiki/concepts/transformer.md`](../../wiki/concepts/transformer.md)

### 3) Ch 6–7 + 附录 E：微调谱系

- **要点：** 分类头微调 → 指令 SFT → LoRA / DPO bonus，对应机器人侧 **BC 微调、语言条件策略、参数高效适配** 的常见工程阶段。
- **对 wiki 的映射：** [`wiki/methods/vla.md`](../../wiki/methods/vla.md)、[`roadmap/depth-vla.md`](../../roadmap/depth-vla.md) Stage 0 前置

## 关联原始资料

- [`sources/courses/rasbt_llms_from_scratch_youtube.md`](../courses/rasbt_llms_from_scratch_youtube.md) — YouTube 七章配套视频（与书章对齐）

## 推荐继续阅读（外部）

- [Build A Reasoning Model (From Scratch)](https://github.com/rasbt/reasoning-from-scratch) — 续作：推理时 scaling、RLVR/GRPO 等
- [Karpathy Neural Networks: Zero to Hero](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ) — 更短、更视频驱动的从零 NN/LLM 路线
