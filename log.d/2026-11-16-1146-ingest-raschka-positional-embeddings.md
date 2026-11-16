# 位置编码教程 ingest 记录

## [2026-11-16] ingest | sources/blogs/raschka_understanding_positional_embeddings_2026_11_15.md — 通读 Raschka 教程，归纳现代 LLM 位置编码

- 原文为 2026-11-15 发布的技术博客；全文 §1–§7 已核对，非独立论文或新项目。
- 新建位置编码概念页（GPT-2 absolute、sinusoidal、RoPE、Partial RoPE、NoPE、相对偏置），并为 Transformer 与模型架构纵深 Stage 1 建立交叉引用。
- 源码核对了原文所引用的 rasbt/LLMs-from-scratch Qwen3 notebook，绘制基于真实调用关系的运行时序图；未执行模型。
- 对长上下文泛化与行业趋势的结论保留文章采样边界，区分数学性质与模型表现。
