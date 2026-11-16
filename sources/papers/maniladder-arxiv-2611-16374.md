# ManiLadder: Benchmarking Robot Manipulation Through a Categorized and Multi-Difficulty Level Task Ladder｜操作难度阶梯

- **类型：** 论文
- **原文：** https://arxiv.org/abs/2611.16374 （v1）
- **HTML：** https://arxiv.org/html/2611.16374
- **PDF：** https://arxiv.org/pdf/2611.16374
- **项目：** https://maniladder-benchmark.github.io/
- **核对范围：** arXiv:2611.16374v1 的基准构造、112 个任务与四难度等级、参考策略和预算校准、AL@50 规则、跨策略及八个对应真机任务评测、结论和局限。项目站点本次无法可靠读取，未核实代码仓。
- **通读：** 已完成
- **入库日期：** 2026-11-16
- **wiki：** [ManiLadder](../../wiki/entities/paper-maniladder-manipulation-benchmark.md)

## 原文摘录与映射

1. **结构：** 112 项任务按类型与形态组织为四级；等级由统一参考策略及数据、训练预算校准，非绝对难度。映射：核心原理。
2. **公平比较：** 统一示教、训练与评测协议，让策略的不同能力短板可按类型和等级呈现。映射：工程实践。
3. **AL@50：** 目标等级及以下每一级都须有至少 80% 任务达到不低于 50% 成功率。映射：指标公式与结果解释。
4. **真机：** 8 个相应任务各 50 次试验基本保留难度次序，但无法证明所有任务的 sim2real 排名恒定。映射：评测与局限。
5. **开源：** 论文提供项目 URL，但无法核实项目页 Code/Resources 与可运行源码入口。映射：站点归档和“不适用”源码时序。
