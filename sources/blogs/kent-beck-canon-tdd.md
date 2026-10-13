# Canon TDD — Kent Beck

- **类型：** 作者原文 / 方法说明
- **链接：** https://newsletter.kentbeck.com/p/canon-tdd
- **作者：** Kent Beck
- **核查日期：** 2026-10-13
- **一句话说明：** TDD 提出者对其“canonical TDD”编程工作流做出的直接说明，澄清先写完整测试套件再一次性实现并非其所指的 TDD。

## 核心摘录

- TDD 是一种编程工作流：开发者维护一个短的 test list，每次选取一项，将其转成当前失败测试，然后修改生产代码让测试通过。
- 通过后可以选择重构；若出现新需求或边界情况，应加入测试清单并继续小步循环。
- 重点是极短的反馈循环和渐进式设计，不是先穷举全部测试、一次写完实现。
- Beck 将这种规范形态称为 canonical TDD；团队日常会有变体，因此讨论“是否在用 TDD”前应先说明所指实践。
- 本文是作者对概念的直接阐释，适合与其经典著作 *Test-Driven Development: By Example* 对读。

## 对 wiki 的映射

- [测试驱动开发（TDD）](../../wiki/concepts/test-driven-development.md)
- [SDD / TDD / BDD / ATDD 对照](../../wiki/comparisons/sdd-tdd-bdd-development-methods.md)

## 参考来源

- [Kent Beck, Canon TDD](https://newsletter.kentbeck.com/p/canon-tdd)
- [Test-Driven Development: By Example — Addison-Wesley / Pearson](https://www.pearson.com/en-us/subject-catalog/p/Beck-Test-Driven-Development-By-Example/P200000009421/9780321146533)
