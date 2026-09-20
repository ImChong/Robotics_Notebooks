# Retrieval-Augmented Generation for Large Language Models: A Survey（arXiv:2312.10997）

> 论文来源归档（ingest）

- **标题：** Retrieval-Augmented Generation for Large Language Models: A Survey
- **作者：** Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Liu, Jiawei Liu, Haofen Wang, Haofen Wang, Yuqiang Li, et al.（同济大学 / 复旦大学等）
- **类型：** paper / survey / rag / llm / retrieval
- **arXiv：** <https://arxiv.org/abs/2312.10997> · PDF：<https://arxiv.org/pdf/2312.10997.pdf>
- **发表：** 2023-12（v1）；后续多版修订
- **配套：** GitHub 综述仓库 <https://github.com/Tongji-KGLLM/RAG-Survey>（论文索引与分类）
- **入库日期：** 2026-09-20
- **一句话说明：** 系统梳理 **Naive RAG → Advanced RAG → Modular RAG** 三代演进，按 **检索（索引/查询/后处理）— 生成（融合/控制）— 增强流程（预/中/后检索）** 分解组件，并汇总评测、瓶颈与机器人/agent 外延，适合作为 RAG 工程选型的 **taxonomy 一手索引**。

## 开源状态（步骤 2.5，2026-09-20）

- **部分开源：** 综述配套 GitHub 维护论文列表与分类；无统一 benchmark 代码仓。
- **结论：** 以 **分类框架 + 文献地图** 为主；具体系统实现需跟链到各论文/框架（LangChain、LlamaIndex 等）。

## 核心摘录（面向 wiki 编译）

### 1) 三代 RAG 范式

| 代际 | 特征 | 典型增强 |
|------|------|----------|
| **Naive RAG** | Index → Retrieve → Generate 线性三段 | 固定 chunk、top-$k$、直接拼 prompt |
| **Advanced RAG** | 检索前/后处理 | query rewrite、HyDE、rerank、context compression |
| **Modular RAG** | 可编排模块 + 反馈环 | routing、agent 工具调用、GraphRAG、Self-RAG |

- **对 wiki 的映射：** [`wiki/concepts/retrieval-augmented-generation.md`](../../wiki/concepts/retrieval-augmented-generation.md) — 演进与选型

### 2) 检索子系统四阶段

1. **Indexing** — 解析、分块（chunk）、向量化/倒排、元数据
2. **Retrieval** — 稀疏（BM25）、稠密（bi-encoder）、混合、多向量
3. **Post-retrieval** — rerank、去重、压缩、重排序
4. **Generation** — 拼接 vs 交叉注意力融合；引用约束

- **对 wiki 的映射：** [`wiki/concepts/retrieval-augmented-generation.md`](../../wiki/concepts/retrieval-augmented-generation.md) — 工程实践

### 3) 常见失效模式（综述归纳）

- **检索失败：** chunk 切分丢上下文、embedding 域偏移、query-document 语义 gap
- **生成失败：** 检索段未进 context 窗口、模型忽略检索、错误段被「自信总结」
- **评估陷阱：** 只看生成流畅度不看 **faithfulness / attribution**

- **对 wiki 的映射：** [`wiki/concepts/retrieval-augmented-generation.md`](../../wiki/concepts/retrieval-augmented-generation.md) — 局限与风险

### 4) 与 agent / 机器人外延

- **要点：** Modular RAG 与 **tool-using agent** 收敛：检索库、API、仿真状态均可作「外部记忆」；机器人侧常见 **catalog 约束 VLM**（图书馆书架）、**安全模板库**（阻抗参数）、**技能/行为检索**（behavior retrieval）——结构同 RAG，只是 **检索对象非纯文本**。
- **对 wiki 的映射：**
  - [`wiki/entities/paper-scanford-robot-powered-data-flywheel.md`](../../wiki/entities/paper-scanford-robot-powered-data-flywheel.md)
  - [`wiki/entities/paper-notebook-safehumanoid-vlm-rag-driven-control-of-upper-bod.md`](../../wiki/entities/paper-notebook-safehumanoid-vlm-rag-driven-control-of-upper-bod.md)

### 5) 评测维度

- **检索：** Recall@k、MRR、nDCG
- **生成：** EM/F1（QA）、RAGAS（faithfulness、answer relevance、context precision）
- **端到端：** 任务成功率 + 人工 attribution 审计

- **对 wiki 的映射：** [`wiki/concepts/retrieval-augmented-generation.md`](../../wiki/concepts/retrieval-augmented-generation.md) — 调试指标

## 对 wiki 的映射（汇总）

| 目标页 | 关系 |
|--------|------|
| [`wiki/concepts/retrieval-augmented-generation.md`](../../wiki/concepts/retrieval-augmented-generation.md) | 主概念页：taxonomy、工程 checklist |
| [`wiki/concepts/ai-auto-research.md`](../../wiki/concepts/ai-auto-research.md) | S2 文献综合中的 RAG 方法族 |
| [`wiki/entities/painode-125-langchain.md`](../../wiki/entities/painode-125-langchain.md) | 工业 RAG 编排框架实例 |

## 当前提炼状态

- [x] 三代范式与检索四阶段摘录
- [x] 机器人/agent 外延映射
