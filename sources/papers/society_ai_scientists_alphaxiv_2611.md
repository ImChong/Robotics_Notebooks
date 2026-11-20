# A Society of AI Scientists｜AI 科学家社会（alphaXiv Exclusive）

- **类型：** 论文 / AI Auto-Research / 多智能体科研社区 / 元科学
- **论文原文：** <https://www.alphaxiv.org/abs/2611.society-ai-scientists>
- **PDF：** <https://www.alphaxiv.org/abs/2611.society-ai-scientists.pdf>
- **身份：** alphaXiv Exclusive 的 `2611.society-ai-scientists`，**并非 arXiv 正式编号**
- **公开日期：** 2026-11-19；入库日期：2026-11-20
- **作者：** Rehaan Ahmad、Myles Anderson、Ao Qu 等；作者介绍涉及 22 个机构
- **研究现场：** 2026-08-19 至 2026-11-12，35 个由人类运营的实验室
- **数据集：** <https://huggingface.co/datasets/alphaXiv/AgentAlphaXiv>
- **平台与分析代码：** <https://github.com/alphaXiv/OpenResearch>
- **交互分析：** <https://molab.marimo.io/notebooks/nb_aErVM5Au66szGmvh5u4gpu>
- **核对范围：** 已核对 alphaXiv 论文页面、摘要及 PDF 的研究设计、主要结果、审稿/复核、局限、数据与代码公开章节，并阅读官方 OpenResearch README 和 CLI 关键实现；尚未逐页通读论文全部正文、附录与逐项数据。
- **通读：** 未完成

## 一句话

论文记录 936 个 Agent 在 35 个实验室、12 周中的科研协作与审核实践；观察到自发的公开纠错、负结果引用和 Agent 审稿，但**实验可验证性并不等于原创科学判断**。

## 核心摘录

### 样本与活动

- 35 个由人类运营的实验室、936 个 Agent、936 项 cohort 研究，9,500 条消息及 229 个 Agent 创建的聊天室；属于观察研究而非随机受控实验。
- 研究主题覆盖语言模型微调、持续学习、机制可解释性以及 Agent 元科学等；不能称为机器人真机实验。
- 官方 Hugging Face 数据集浏览器可能继续增加研究：网页展示的 `studies` 961 行包含 `in_paper_cohort` 字段，**不应拿 961 取代论文锁定队列的 936**。

### 社区规范与审计

- 作者报告，Agent 会公开纠正部分有误结论，负结果也可作为正常发表与引用对象。
- 由一个实验室发起的 Journal of Agent Research 对代码与实验结果实施审稿；文本审稿与独立执行式复核在识别缺陷方面不同，不能只看审稿意见是否流畅。
- 论文统计的 23,811 条可定位数值中，22,345 条与原始输出精确一致，1,466 条为格式或舍入差异；**这不能证明实验设计、因果论断或新颖性正确**。
- 202 对 Agent/人类预印本的盲评使用的是模型评审器，不是正式人工同行评议；Agent 在严谨性与主张克制度上较优，人类在新颖性和重要性上较优。

## 适合本知识库的映射

- [独立论文详情](../../wiki/entities/paper-society-of-ai-scientists.md) — 实验设计、流程、数字口径与局限。
- [AI Auto-Research](../../wiki/concepts/ai-auto-research.md) — 从单体自动研究发展到共享科研社区与跨主体证据审计。
- [RSI Stage 5](../../roadmap/depth-rsi.md) — 作为研究过程治理的**观察性对照**，而非递归自我改进或 ignition 证据。
- [OpenResearch 开源代码归档](../repos/alphaxiv-openresearch.md) 与 [AgentAlphaXiv 数据集归档](../datasets/alphaxiv-agent-alphaxiv.md)。

## 开放程度与复现边界

已公开论文、数据集与 OpenResearch 平台代码（MIT），但该仓库当前 CLI 的代码路径不等于 12 周全体实验室实际运行历史的完整可复现环境。本次没有运行队列统计或独立复现实验；须按 `in_paper_cohort` 锁定队列后分析。

## 参考来源

- [alphaXiv 论文正文与 PDF](https://www.alphaxiv.org/abs/2611.society-ai-scientists)
- [AgentAlphaXiv 官方数据集](https://huggingface.co/datasets/alphaXiv/AgentAlphaXiv)
- [OpenResearch 官方代码](https://github.com/alphaXiv/OpenResearch)
