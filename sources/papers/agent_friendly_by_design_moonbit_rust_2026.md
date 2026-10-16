# Agent-Friendly by Design：MoonBit 与 Rust 的 AI 编程对照实验

## 元数据

- **题目：** Agent-Friendly by Design: An Empirical Comparison of MoonBit and Rust in AI-Driven Software Construction
- **类型：** 预印本 / AI 编程实验
- **预印本 PDF：** <https://github.com/bobzhang/minisql-moonbit-vs-rust/blob/main/paper/moonbit-agent-friendly.pdf>
- **代码与数据：** <https://github.com/bobzhang/minisql-moonbit-vs-rust>
- **结果报告：** <https://github.com/bobzhang/minisql-moonbit-vs-rust/blob/main/results/report.md>
- **PDF 元数据日期：** 2026-10-02
- **入库日期：** 2026-10-16

## 研究问题

在相同的中大型软件任务上，AI coding agent 使用一种熟悉语言（Rust）或训练数据较少的新语言（MoonBit），其 token 消耗、API 价目估算成本、完成质量、代码规模与编译调试成本有何差别？实验以 SQLite 兼容 SQL 引擎为任务，分九个里程碑从核心查询管线推进到数据库文件读写。

## 实验设计

- **模型与执行：** Claude Opus 5.5、high effort，通过 headless Claude Code 运行；每个里程碑开启新会话。
- **样本：** 每种语言 3 个独立试验；试验按 Rust / MoonBit 配对排程。
- **任务对照：** 规格、约 450 个符合性测试、提示、模型、工具、harness 与标准库限制一致；仅语言专属 primer 和构建指令不同。
- **测试防泄漏：** 用案例名哈希将测试分成可见 / 隐藏两半；生成的期望输出来自 SQLite，并检查无序查询是否因排序随机而被误判。会话 transcript 也扫描网络与隐藏测试目录访问。
- **语言 primer：** MoonBit 获得约 4–6k token 的语法、工具链与常见错误说明；Rust 使用相同四段结构但更短。每个里程碑新会话须重新读取说明，因此熟悉度和上下文重读成本都进入测量。
- **实现限制：** 仅标准库；不允许第三方包、FFI、C、SQLite 链接或子进程。两边提供功能相同、不可由 agent 修改的 I/O 模块。

## 主要结果

| 指标（均值，n=3/语言） | Rust | MoonBit | MoonBit / Rust |
|------------------------|-----:|--------:|---------------:|
| API 价目估算成本 | $43.86 ± 13.25 | $41.09 ± 3.97 | 0.94（比值中位数 1.12） |
| 输出 token | 867,780 | 762,261 | 0.88 |
| 达到全部可见测试通过前的输入 token | 21.7M | 23.6M | 1.09 |
| 隐藏测试通过率 | 96.3% | 95.6% | 0.99 |
| 最终源码 token | 193,931 | 171,530 | 0.88 |
| 编译失败率 | 5.8% | 13.0% | 2.25 |
| 中位墙钟时间 | 2.19 h | 2.19 h | 1.00 |

报告指出，总成本差异处于试验间噪声范围；模型通过测试后的持续加固占总输入 token 的 47–82%。MoonBit 的最终源码 token 和输出 token 约低 12%，但编译失败率约为 Rust 的 2.25 倍，且 primer 重读产生了额外输入成本。

## 对 wiki 的映射

- 实体页：[Agent-Friendly by Design](../../wiki/entities/paper-agent-friendly-by-design.md)
- 关联概念：[Agentic Coding 时代的软件工程基础](../../wiki/concepts/agentic-coding-software-fundamentals.md)

## 解读边界

这是一个有价值的可复核对照实验，但样本仅 3 次/语言、仅使用一个模型与一个 SQL 任务；价格按 API 价目估算，不代表订阅用户的实际账单。结果衡量的是给定提示、primer、工具链与软件规格下的 Claude Opus 5.5，不足以推出 MoonBit 或 Rust 对所有模型、任务与开发者都更省成本。README 与结果报告可见结果数据、逐里程碑指标和运行资料，复现实验仍需 Claude Code 与模型访问权限。

## 原始来源

- 论文 PDF：<https://github.com/bobzhang/minisql-moonbit-vs-rust/blob/main/paper/moonbit-agent-friendly.pdf>
- 项目 README：<https://github.com/bobzhang/minisql-moonbit-vs-rust>
- 结果报告：<https://github.com/bobzhang/minisql-moonbit-vs-rust/blob/main/results/report.md>
- 原始摘要：<https://gist.github.com/bobzhang/cbbfc96423a23212d24847a485f1a86d>
