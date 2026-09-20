# facebookresearch/DPR（Dense Passage Retrieval）

> 仓库来源归档（ingest）

- **名称：** Dense Passage Retrieval for Open-Domain Question Answering
- **机构：** Meta AI（Facebook AI Research）
- **类型：** repo / retrieval / open-domain-qa
- **GitHub：** <https://github.com/facebookresearch/DPR>
- **论文：** [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](../papers/lewis_rag_neurips_2020.md)（arXiv:2005.11401）；DPR 原文 Karpukhin et al. EMNLP 2020
- **入库日期：** 2026-09-20
- **一句话说明：** RAG 论文配套的 **双塔稠密检索** 实现：bi-encoder 训练、FAISS 索引、top-$k$ 段落检索，是现代向量 RAG 的 **参考实现** 之一。

## 开源状态（步骤 2.5，2026-09-20）

- **已开源：** 训练脚本、预训练 checkpoint、Wikipedia 索引构建流程；License MIT。
- **结论：** 可直接复现 Lewis et al. 检索链路；生产环境多迁移至 Milvus/Qdrant/pgvector + 新 embedding 模型。

## 核心能力

| 模块 | 说明 |
|------|------|
| `train_dense_encoder.py` | 查询/段落双塔对比学习 |
| `generate_dense_embeddings.py` | 语料向量化 |
| `dense_retriever.py` | FAISS 检索 top-$k$ |
| 预训练权重 | NQ/Trivia 等 checkpoint |

## 对 wiki 的映射

- [`wiki/concepts/retrieval-augmented-generation.md`](../../wiki/concepts/retrieval-augmented-generation.md) — DPR 检索子系统与复现路径
- [`sources/papers/lewis_rag_neurips_2020.md`](../papers/lewis_rag_neurips_2020.md) — 奠基论文

## 当前提炼状态

- [x] 开源边界与模块索引
