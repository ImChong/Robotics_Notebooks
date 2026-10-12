# STAIR 论文来源归档

## 元数据

- **题目：** STAIR (STructure Aware Information Retriever): A novel dataset and LLM based retriever for document structure augmentation
- **作者：** Vineet Kumar, Meghanadh Pulivarthi, Vishwajeet Kumar, Jaydeep Sen, Riyaz Ahmad Bhat, Sachindra Joshi
- **单位：** IBM（HTML 页署名；第一作者工作时在 IBM，现 Amazon Books Science）
- **预印本：** https://arxiv.org/abs/2609.03874
- **HTML：** https://arxiv.org/html/2609.03874v1
- **版本：** v1，2026-09-03，cs.AI，CC BY 4.0
- **代码与数据：** 正文给出匿名仓库 https://anonymous.4open.science/r/s_331/README.md ；本次抓取仅返回页面标题“Anonymous Github”，无法核实内容，视为“匿名评审链接，未核实”。

## 论文要解决的问题

RAG 的检索器通常按固定长度切块，丢弃文档已有的全局层级结构；LLM 对长上下文中间信息又不敏感，所以检索精度很关键。论文用文档目录（ToC）作为全局结构，让 LLM 把语料知识存入参数并通过目录寻址，做生成式检索。

## 方法与训练信息

1. **输入输出：** 输入查询 + 整本书的完整 ToC，输出正确叶节点的标题；叶节点章节标题（如“12.9 Step seven: …”）充当文档标识符，而非 DSI 式数值 docid。
2. **训练数据：** 用 Mixtral 8x7B 对每个段落生成多条问题，构成合成 QA。
3. **训练：** 监督微调，把“ToC 增强的查询”映射到正确叶节点；骨干 Mistral-7B-Instruct-v0.2，LoRA（r=16，α=32），最多 200 epoch，dev Recall@1 早停（patience 20）；STAIR 输入上限 14k token（DSI 为 512），输出 64 token。
4. **推理：** 约束解码，输出限制在合法 ToC 叶节点内。
5. **SearchTome 基准：** 摘要与表 1 为 18 本书、6 个领域（教育、金融、法律、医学、自然科学、社会科学，每域 3 本），来自 opentextbooks 的开放教材，PyMuPDF 解析出 ToC 并对齐章节文本。引言与贡献处写作“1818 books / 66 domains”，与摘要及表 1 矛盾，此处按 18/6 记录并标明不一致。总查询数论文未直接给出；对表 1 各列求和约为测试 10.3 万、训练 6.2 万、开发 2.45 万（需假设列含义，未核实）。

## 论文报告的结果

指标 Recall@1、Recall@3、nDCG@3（BEIR），论文未报告 Recall@5。

| 模型 | 平均 R@1 | 平均 R@3 | 平均 nDCG@3 |
|---|---|---|---|
| Mistral（开箱，beam search） | 13.8 | 16.4 | 15.4 |
| BM25（Elasticsearch 8.11.2） | 59.5 | 77.6 | 70.1 |
| Claude Haiku 4.5 | 45.6 | 64.1 | 56.4 |
| DPR（NV-Embed-v2，开箱） | 68.7 | 85.4 | 78.6 |
| DSI（微调） | 76.9 | 85.3 | 81.9 |
| STAIR | 82.6 | 90.8 | 87.5 |

- STAIR 比 DSI 的 R@1 绝对高 5.7 个点（论文的“+7.4%”是相对增益）；论文称随机化检验 p<0.05（其领域级检验措辞引用“66 domains”，与 6 个领域不符）。Haiku 4.5 出现在表中，但基线小节未描述。
- **幻觉（生成非法标识符）：** 定义为生成非叶节点。错误率：Mistral 86.20%（其中非叶 26.81%、幻觉 23.86%），DSI 24.31%（非叶 3.25%），STAIR 18.67%（非叶 0.05%）。
- **少样本泛化：** STAIR 幻觉率与每叶训练样本数基本无关，DSI 随样本减少上升；R@1 差距在训练样本少的叶节点最大（图 3、4，无具体数字）。唯一消融是 DSI（无 ToC）对 STAIR（有 ToC）。

## 复现与解释边界

- 评测只覆盖有全局结构（ToC）的语料；论文把“对无结构基准人工诱导 ToC”“百万级 URL 企业语料”列为未来工作。
- 查询由 Mixtral 8x7B 合成，评测分布与真实用户查询的差异未验证。
- “幻觉 <0.05%”是约束解码下的非法标识符比例，不等同于答案事实性幻觉。
- 匿名仓库内容未能核实。

## Robotics_Notebooks 映射

- 详情节点：wiki/entities/paper-stair-toc-generative-retrieval.md
- 概念入口：wiki/concepts/retrieval-augmented-generation.md

## 一手来源

- <https://arxiv.org/abs/2609.03874>
- <https://arxiv.org/html/2609.03874v1>
