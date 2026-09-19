# deepgrounding/recursive-self-improvement

- **URL：** <https://github.com/deepgrounding/recursive-self-improvement>
- **类型：** 论文配套仓库 / 语料 + 复现脚本 + 手稿源
- **维护方：** DeepGrounding（Mingguang Chen 等）
- **关联论文：** [arXiv:2607.07663](../papers/rsi_survey_arxiv_2607_07663.md) — *Recursive Self-Improvement in AI: From Bounded Self-Refinement to Autonomous Research Loops*
- **收录日期：** 2026-09-19
- **Tags：** #recursive-self-improvement #survey #corpus #literature-review #reproducibility

## 一句话

arXiv:2607.07663 的 **canonical 开放发布**：1,250 篇语料 taxonomy 标注、分类/补充/图表/bib 生成脚本，以及与 arXiv v1 对齐的 `draft/main.md` 手稿 — **非** Agent 训练或 RSI 运行时。

## 为什么值得保留

- **可审计语料：** `artifacts/corpus_v2.csv` 为 Table 1 与 §2–§6 分类的 ground truth；含 category / subcategory 与显式 overrides 可追溯性。
- **复现链完整：** `draft/scripts/` 覆盖 reclassify、supplement harvest、build_bib、build_figures、build_latex — Data availability 承诺的可执行脚本。
- **链接更正：** 取代 arXiv v1 中不可用的 `bamboodrift/recursive_self_improvement` 私有仓引用。

## 核心结构

| 路径 | 内容 |
|------|------|
| `artifacts/corpus_v2.csv` | 1,250 篇 + taxonomy 标注（主语料） |
| `artifacts/self_improvement_corpus.csv` | 871 篇种子 + 早期 13-topic 聚类 |
| `artifacts/table1_category_stats.md` | Table 1 统计 |
| `draft/main.md` | 手稿 Markdown 源 |
| `draft/references.bib` | 自动生成参考文献（勿手改） |
| `draft/scripts/reclassify_corpus.py` | 分类规则 + OVERRIDES |
| `draft/scripts/supplement_harvest.py` | +379 定向补充 |
| `draft/scripts/build_*.py` | 图表 / bib / LaTeX |

**Taxonomy codes：** `deployment` · `training` · `evaluation` · `research` · `foundations`

## 开源边界（步骤 2.5）

| 已发布 | 不适用 |
|--------|--------|
| 语料 CSV、分类脚本、图表源、手稿 | 统一 RSI Agent 运行时 / 训练栈 |
| 可本地 regenerate bib 与 figures | 各被引论文的权重与代码（逐条跳转） |

## 对 wiki 的映射

- 论文实体：[paper-rsi-survey-2607-07663](../../wiki/entities/paper-rsi-survey-2607-07663.md)
- 概念：[recursive-self-improvement](../../wiki/concepts/recursive-self-improvement.md)
- 互补索引：[Awesome RSI](../../wiki/entities/awesome-rsi.md)

## 参考来源（原始）

- README：<https://github.com/deepgrounding/recursive-self-improvement>（2026-09-19 结构级摘录）
