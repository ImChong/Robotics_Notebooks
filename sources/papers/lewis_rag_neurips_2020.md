# Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks（NeurIPS 2020）

> 论文来源归档（ingest）

- **标题：** Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks
- **作者：** Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktächel, Sebastian Riedel, Douwe Kiela（Facebook AI Research / UCL / UW）
- **类型：** paper / nlp / retrieval / generation / knowledge-intensive
- **arXiv：** <https://arxiv.org/abs/2005.11401> · PDF：<https://arxiv.org/pdf/2005.11401.pdf>
- **会议：** NeurIPS 2020
- **代码：** DPR <https://github.com/facebookresearch/DPR>；HuggingFace `transformers` RAG 实现
- **入库日期：** 2026-09-20
- **一句话说明：** 提出 **RAG** 范式：把 **非参数化检索器**（Dense Passage Retrieval）与 **参数化 seq2seq 生成器**（BART）结合，在开放域 QA 等知识密集型任务上 **减少幻觉、可更新知识、可溯源**，成为后续 LLM 应用与 agent grounding 的奠基工作。

## 开源状态（步骤 2.5，2026-09-20）

- **已开源：** DPR 训练/索引/检索代码与 checkpoint 公开；RAG 模型权重经 HuggingFace Hub 分发。
- **结论：** 论文级可复现；工业部署通常换用更新 embedding 与向量库，但 **retrieve-then-generate** 骨架不变。

## 核心摘录（面向 wiki 编译）

### 1) 问题：参数化 LM 的知识边界

- **要点：** 预训练 LM 把世界知识 **压缩进权重**；更新知识需重训，且生成时易 **幻觉**、难 **归因** 到具体文档。
- **对 wiki 的映射：** [`wiki/concepts/retrieval-augmented-generation.md`](../../wiki/concepts/retrieval-augmented-generation.md) — 为什么重要

### 2) RAG 架构：检索器 + 生成器

- **要点：** **Retriever** $p_\eta(z|x)$ 从大规模语料（如 Wikipedia）取 top-$k$ 段落 $z$；**Generator** $p_\theta(y|x,z)$ 以查询 $x$ 与检索段 $z$ 为条件生成答案 $y$。检索与生成 **端到端可微**（通过检索段作为生成器额外输入）。
- **对 wiki 的映射：** [`wiki/concepts/retrieval-augmented-generation.md`](../../wiki/concepts/retrieval-augmented-generation.md) — 核心原理

### 3) RAG-Sequence vs RAG-Token

- **要点：** **RAG-Sequence** — 同一检索段生成整段输出；**RAG-Token** — 每个 token 可在不同检索段间 **边际化**，更细粒度融合多文档证据。
- **对 wiki 的映射：** [`wiki/concepts/retrieval-augmented-generation.md`](../../wiki/concepts/retrieval-augmented-generation.md) — 变体

### 4) DPR：双塔稠密检索

- **要点：** 查询编码器与段落编码器分别映射到向量空间，**内积相似度** 近似 BM25；比稀疏检索更适合语义匹配，是 modern RAG **向量检索** 的直接前传。
- **对 wiki 的映射：** [`wiki/concepts/retrieval-augmented-generation.md`](../../wiki/concepts/retrieval-augmented-generation.md) — 检索子系统

### 5) 实验结论（开放域 QA）

- **要点：** 在 Natural Questions、TriviaQA、WebQuestions 等上，RAG **优于** 纯参数化 BART 与 **retrieve-then-extract** 基线；生成更 **具体**、更少胡编；换检索语料即可 **更新知识** 而无需重训生成器全部参数。
- **对 wiki 的映射：** [`wiki/concepts/retrieval-augmented-generation.md`](../../wiki/concepts/retrieval-augmented-generation.md) — 工程意义

## 对 wiki 的映射（汇总）

| 目标页 | 关系 |
|--------|------|
| [`wiki/concepts/retrieval-augmented-generation.md`](../../wiki/concepts/retrieval-augmented-generation.md) | 概念主链：定义、架构、变体、局限 |
| [`wiki/references/llm-wiki-karpathy.md`](../../wiki/references/llm-wiki-karpathy.md) | LLM Wiki 模式 vs 查询时 RAG 的对照 |
| [`wiki/entities/paper-notebook-safehumanoid-vlm-rag-driven-control-of-upper-bod.md`](../../wiki/entities/paper-notebook-safehumanoid-vlm-rag-driven-control-of-upper-bod.md) | 机器人侧 **非文本** RAG：语义模板检索 |

## 当前提炼状态

- [x] 架构与 DPR 要点摘录
- [x] wiki 概念页映射
