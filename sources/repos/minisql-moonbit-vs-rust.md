# bobzhang/minisql-moonbit-vs-rust（官方实验仓库）

> 来源归档（repo）

- **代码：** <https://github.com/bobzhang/minisql-moonbit-vs-rust>
- **预印本：** <https://github.com/bobzhang/minisql-moonbit-vs-rust/blob/main/paper/moonbit-agent-friendly.pdf>
- **结果报告：** <https://github.com/bobzhang/minisql-moonbit-vs-rust/blob/main/results/report.md>
- **类型：** AI 编程实验 / MoonBit 与 Rust 对照
- **核查日期：** 2026-10-16
- **入库日期：** 2026-10-16

## 一句话摘要

以同一份 SQLite 兼容 SQL 引擎规格、测试、提示和模型，比较 Claude Opus 5.5 使用 MoonBit 与 Rust 完成九个里程碑的 token、成本、代码规模、编译失败率与隐藏测试表现。

## 实验设计

- 项目为 minisql：目标约 20k 行，九个里程碑，从核心 SQL 管线扩展到读写 SQLite 数据库文件。
- 每种语言 3 次试验；每个里程碑开启新 headless Claude Code 会话，沿用同一试验代码。
- 约 450 个由 SQLite 生成期望值的符合性用例，按案例名哈希分成可见与隐藏两半；每个阶段评分会覆盖此前里程碑的测试。
- 两边共享语言无关规格、提示、工具、标准库限定、测试与 harness；语言差异保留各自构建命令、二进制路径及配套 primer。MoonBit primer 的成本会计入实验。
- 仓库公开实验 harness、指标与报告脚本、试验状态、测试、结果 CSV、最终实现和运行历史。

## 结果摘要（3 次试验均值）

| 指标 | Rust | MoonBit | 读法 |
|------|-----:|--------:|------|
| API 价目估算成本 | $43.86 ± 13.25 | $41.09 ± 3.97 | MoonBit/Rust = 0.94；中位数比值为 1.12，差异处于运行波动范围 |
| 输出 token | 868k | 762k | MoonBit 约少 12% |
| 最终源码 token | 194k | 172k | MoonBit 约少 12% |
| 隐藏测试通过率 | 96.3% | 95.6% | 接近 |
| 编译失败率 | 5.8% | 13.0% | MoonBit 约为 Rust 的 2.25 倍 |
| 中位墙钟时间 | 2.19 h | 2.19 h | 相同 |

完整指标见仓库 [report.md](https://github.com/bobzhang/minisql-moonbit-vs-rust/blob/main/results/report.md)、[trials.csv](https://github.com/bobzhang/minisql-moonbit-vs-rust/blob/main/results/trials.csv) 与 [milestones.csv](https://github.com/bobzhang/minisql-moonbit-vs-rust/blob/main/results/milestones.csv)。

## 开源与复现边界

- **源码和数据：** 已公开；仓库有实验 harness、提示、primers、试验状态、轨迹/历史、测试与结果。
- **复现条件：** 需要 Claude Code 登录及模型额度（Claude 订阅也可运行）、Rust 与 MoonBit 工具链、Python SQLite；重新运行模型会产生调用成本。
- **测量对象：** 是 Claude Opus 5.5（Claude Code headless，high effort）在此规格与 harness 中的表现，不是对编程语言普遍生产力或所有模型的比较。
- 仓库未提供独立项目主页；GitHub README 同时承担项目介绍与官方代码入口。根目录未见 LICENSE 文件，使用或再分发前应自行确认权利条款。

## 对 wiki 的映射

- 实体页：[Agent-Friendly by Design](../../wiki/entities/paper-agent-friendly-by-design.md)
- 概念页：[Agentic Coding 时代的软件工程基础](../../wiki/concepts/agentic-coding-software-fundamentals.md)
- 论文摘录：[来源归档](../papers/agent_friendly_by_design_moonbit_rust_2026.md)

## 原始来源

- 官方仓库 README：<https://github.com/bobzhang/minisql-moonbit-vs-rust>
- 官方结果报告：<https://github.com/bobzhang/minisql-moonbit-vs-rust/blob/main/results/report.md>
