# LLMs are making formal software verification feasible at a much larger scale（Anthropic Research）

> 来源归档（官方研究文章；不等同于论文）

- **标题：** LLMs are making formal software verification feasible at a much larger scale
- **类型：** research / blog
- **作者 / 组织：** Keane Lucas、Aengus Lynch、Luna Tong、Luke Champine、Zygi Straznickas、Dino Distefano、Harshikaa Agrawal、Evyatar Ben Asher、Ava Howell、Sami Jawhar、Chris Rohlf、Nickolai Zeldovich、Nicholas Carlini、Rajashree Agrawal、Jason Gross；Anthropic Frontier Red Team，与 Theorem 合作
- **原始链接：** <https://www.anthropic.com/research/formal-verification>
- **发表日期：** 2026-10-22
- **入库日期：** 2026-10-26
- **代码与语义：** [OSS Verify proofs](../repos/theorem-labs-oss-verify-proofs.md)；[RISC-V/Linux process semantics](https://github.com/theorem-labs/rocq-linux-process-semantics)
- **一句话说明：** Anthropic 报告使用 Claude 为 OSS-Fuzz 测试入口对应的 RISC-V 可执行文件编写 Rocq 证明，并在证明迭代中发现、修复传统模糊测试难以触发的缺陷；证明结论严格受测试规格、二进制构建和机器语义模型范围约束。

## 核心摘录（归纳，非全文）

- **主要实验：** 将既有 fuzz target 的“测试不崩溃”要求转成针对编译后 RISC-V 指令的形式规格；Claude 生成证明，Rocq kernel 检查证明。文章报告覆盖 123 个 OSS-Fuzz 开源库、239 个入口，证明产物约 1.8 亿行 Rocq。
- **性质边界：** 文章所述证明覆盖小于 64 MB 的输入、建模范围内的 Linux 响应与内存布局，并假设程序是单线程静态二进制、运行于固定内存预算的 RISC-V/Linux 环境。它证明指定入口和规格满足性质，不代表整个库或任意实际部署都无缺陷。
- **漏洞发现：** 证明尝试暴露出数百个问题，包括大输入触发的越界访问、特定调用序列触发的错误，以及逻辑缺陷；Claude 可先修补，再对更新后的程序重新证明。
- **其他案例：** 文章还讨论从 RFC 1950 规格证明 Linux kernel zlib 实现、用 Verus 检查 Rust Python 子集解释器对 CPython 语义的符合性，以及对 PDF 阅读器和其他关键软件做形式化分析。
- **成本与复用：** 证明可能消耗数亿 token；在 36 个小型任务上，Opus 5.5 以约少 40% token 的成本证明一半任务；共享工作区让代理平均少用 17% token。上述数字均为文章报告的实验结果。

## 开源 / 项目页核查（2026-10-26）

| 项 | 结论 |
|----|------|
| 证明产物 | **已公开一部分**：Theorem 的 [oss-verify-proofs](../repos/theorem-labs-oss-verify-proofs.md) 发布 OSS-Fuzz 入口规格与机器检查的 Rocq 证明 |
| 可信语义 | **已公开**：[RISC-V/Linux process semantics](https://github.com/theorem-labs/rocq-linux-process-semantics) 提供证明使用的 Linux process 与 RISC-V 语义模型 |
| Claude skill | Anthropic 文章链接到 Theorem 的 [oss-verify add-project skill](https://github.com/theorem-labs/oss-verify/blob/main/.claude/skills/add-project/SKILL.md)，用于引导项目接入 |
| 覆盖差异 | Anthropic 文章报告总体实验规模为 123 个库、239 个入口；截至本次核查，公开证明仓库 README 列出 25 个库、60 个入口。二者口径不同，公开仓库是文章结果的子集 |

## 对机器人工程的映射

该案例可作为软件层规格验证与机器人安全论证的参考：先明确要保证的性质，再核对程序实际运行的硬件、操作系统、并发模型和调用边界是否落在形式模型内。二进制层证明本身不覆盖未建模的机器人动力学、传感器故障、控制时序或物理安全约束；这些仍需单独建模和验证。

## 推荐继续阅读

- [Anthropic 原文](https://www.anthropic.com/research/formal-verification)
- [OSS Verify proofs 项目归档](../repos/theorem-labs-oss-verify-proofs.md)
- [Theorem OSS Verify 项目页](https://theorem.dev/oss-verify/)
