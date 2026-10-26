# Theorem OSS Verify（证明与语义仓库）

- **URL：** <https://github.com/theorem-labs/oss-verify-proofs>
- **语义模型：** <https://github.com/theorem-labs/rocq-linux-process-semantics>
- **项目页：** <https://theorem.dev/oss-verify/>
- **关联研究文章：** [Anthropic: LLMs are making formal software verification feasible at a much larger scale](../sites/anthropic-formal-verification.md)
- **许可证：** 证明仓库标注 MIT；语义仓库许可应以其仓库文件为准
- **核查日期：** 2026-10-26

## 一句话说明

Theorem OSS Verify 发布针对部分 OSS-Fuzz fuzz target 编译后二进制的规格与 Rocq 机器检查证明，并提供形式化的 RISC-V/Linux 进程语义。

## 当前公开范围

截至核查日，证明仓库 README 列出 **25 个库、60 个入口**的证明，并说明证明由 **Rocq 9.2.0** 检查。该仓库 README 同时提醒，其 Linux process semantics（LPS）是 best-effort、非完备模型；证明结论取决于模型覆盖的行为与假设。

Anthropic 研究文章报告的是更广的实验总体（123 个库、239 个入口）。公开仓库目前展示的是其中一个子集，不能把公开仓库条目数误写成文章实验覆盖总数。

## 复现与阅读入口

- 浏览 [证明仓库 README](https://github.com/theorem-labs/oss-verify-proofs) 了解“证明了什么”、信任边界、二进制构建和 proof explorer。
- 查阅 [RISC-V/Linux process semantics](https://github.com/theorem-labs/rocq-linux-process-semantics) 了解形式模型及其假设。
- 文章所列项目接入指导：[oss-verify add-project skill](https://github.com/theorem-labs/oss-verify/blob/main/.claude/skills/add-project/SKILL.md)。

## 关联来源

- [Anthropic 研究文章归档](../sites/anthropic-formal-verification.md)
