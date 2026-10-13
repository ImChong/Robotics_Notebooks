# Spec-Driven Development: From Code to Contract in the Age of AI Coding Assistants

> 来源归档（预印本摘要与方法框架；它提出的术语谱系是作者框架，不是行业标准）

- **标题：** Spec-Driven Development: From Code to Contract in the Age of AI Coding Assistants
- **作者：** Deepak Babu Piskala
- **类型：** arXiv preprint
- **链接：** https://arxiv.org/abs/2602.00180
- **版本：** v1，2026-01-30
- **入库日期：** 2026-10-13
- **一句话说明：** 将 AI 编码时代的 SDD 分为 spec-first、spec-anchored、spec-as-source 三种规格权威等级，并总结工作流与适用场景。

## 核心摘录

- **起点：** 把规格设为先于实现、并用于生成或验证代码的主要工件。论文指出，AI 编码助手让“规格而非代码是否应成为主要工件”重新成为实际工程问题。
- **三种严谨度：** spec-first 表示先写规格再编码，但代码可能很快成为事实依据；spec-anchored 表示规格和实现互相校验；spec-as-source 则把规格视为权威，代码由其生成或持续验证。本文把它们作为连续谱，而不是彼此互斥的标准类别。
- **工作流与工具：** 作者讨论从行为驱动框架到 GitHub Spec Kit 等工具的映射，并以 API、企业系统和嵌入式软件为例说明不同的规格落地方式。
- **适用判断：** 选择所需规格严谨度，应取决于风险、变更频率、可执行验证能力与维护成本；更重的流程并非总是更好。
- **阅读边界：** 该文是单作者预印本中的实践分类，不是标准组织或系统性实证研究；本库将分类作为分析坐标，不把它表述为全行业共识。

## 对 wiki 的映射

- [规格驱动开发（SDD）](../../wiki/concepts/specification-driven-development.md)
- [SDD / TDD / BDD / ATDD 对照](../../wiki/comparisons/sdd-tdd-bdd-development-methods.md)

## 参考来源（原始）

- [arXiv:2602.00180](https://arxiv.org/abs/2602.00180)
