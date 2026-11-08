# ResearchBoost: RSI starts with Infra — 作者技术原文

- **类型：** 官方技术博客／研究系统方法
- **英文标题：** ResearchBoost: RSI starts with Infra
- **中文标题：** ResearchBoost：从研究基础设施出发的有界 AI4AI 优化
- **作者：** Xu Huang（黄旭；原文署名网页）
- **原文发表：** 2026-10-19
- **原文：** https://ydove0324.github.io/hx_resume/blog/researchboost/
- **官方代码：** https://github.com/DAGroup-PKU/ResearchBoost-Agent
- **入库/复核日期：** 2026-11-08
- **核对范围：** 通读作者技术博客《ResearchBoost: RSI starts with Infra》全文，包括 §1 序言、§2 四个初始项目结果、§3 Fast/Faithful/Local/Dense 及数值验证、§4 先冻结评测/逐轮单假设/WORK_LOG、§5 未来计划和引用；对照现行仓库 README、docs/methodology.md 和 docs/judge.md，以仓库当前规则解释后续更新。
- **通读：** 已完成

## 原始论证与时间边界

1. 作者将 ResearchBoost 定位于 **AI4AI / 研究基础设施加速**：把“人规定目标和约束，Agent 分析热点、修改代码并获得及时反馈”落实为可验证的工程闭环。文章没有宣称 Agent 已经自主设计下一代模型。
2. 原文 2026-10-19 首发时展示的四项训练加速：RAE **5.32×**、Self-Forcing **2.08×**、DMD2 **1.72×**、LLaVA **2.39×**，均有各自固定配置；与 2026-11-08 GitHub 汇总中的 **20 个项目**不是同一时间点的样本。
3. 作者把评测反馈分为 **Fast / Faithful / Local / Dense**。先运行上游基线，记录训练轨迹及自然波动，再固定数值验证和性能计时口径；优化 Agent 不应为通过候选而放松 judge。
4. 一轮只优化 **一个热点或一类瓶颈**，记录 profile、候选开关、时间和验证结论在 `WORK_LOG.md`。失败候选仍保留证据。
5. 示例：一次 LLaVA 融合修改看似更快却切断 MLP 梯度；另一次 Triton 缓存引入全局 RNG 副作用，均由评测发现。可见“快”不等于“训练正确”。
6. 文章提出未来 Infra Optimization Mini-Bench，属于**规划**，不是已证明完整独立基准已发布。

## 阅读边界

- 此归档是**对作者全文的归纳与证据索引**，不是博客转载。
- 数值和任务表现均为项目作者/贡献者公开报告，本次未复跑 GPU 训练。
- 2026-11-08 的冻结判官细则以同时核对的仓库文档为准；技术博客的初始案例与最新案例不能相加为单一实验。

## 对 wiki 的映射

- [ResearchBoost-Agent：冻结判官驱动的训练代码自动优化](../../wiki/entities/researchboost-agent.md)
- [递归自改进 RSI：边界与分级](../../wiki/concepts/recursive-self-improvement.md)
