# Harvard CS2680：Modern AI Systems — Agents and System Optimizations（Fall 2026）原始课程资料核读

> 核对日期：2026-11-15。对应 [Wiki 课程节点](../../wiki/entities/harvard-cs2680-modern-ai-systems.md)。

- **资料类型：** course
- **机构：** 哈佛大学（Harvard University）John A. Paulson School of Engineering and Applied Sciences
- **授课教师：** Juncheng Yang
- **原始官网：** <https://cs2680.seas.harvard.edu/>
- **官方作业总览：** <https://cs2680.seas.harvard.edu/assignments/>
- **官方阅读资料：** <https://cs2680.seas.harvard.edu/readings/>
- **课程评测与基准说明：** <https://cs2680.seas.harvard.edu/notes/agent-evaluation>
- **课程成本优化笔记：** <https://cs2680.seas.harvard.edu/notes/cost-effective-agents>
- **核对范围：** 课程官网 2026 Fall 的 Overview、公开日程、先修课和评分方式；官方 Assignments 总览及 Assignment 1–5 的任务说明与工程约束。下述「通读」仅指这些明确列出的官网公开课程/作业原始资料，不含全部 lecture slides、课外论文正文、课内私有 starter archives 或真实学生提交。
- **通读：** 已完成

## 定位与范围

官网将课程分成两部分：

1. **Introduction to LLMs and agents：** coding agent 使用、context/tool 设计、多智能体协调、memory/reliability、agent evaluation、安全、成本。
2. **Systems for LLMs and agents：** LLM 推理和 GPU 基础、batching、scheduling、paging、KV-cache/prefix reuse、routing、quantization、speculative decoding、服务部署。

课程采用贯穿式的同一 Agent 系统：先观察现成 coding agent，再独立实现 loop，然后在保持任务质量的约束下优化 API harness，最后控制模型的 serving 与全栈性能。不是 5 个互不相关的工具教学单元。

官网列出 5 次作业（共 70%）、final project 20%、课堂参与 10%；无期末考试。学校公布的授课时间为 2026 年秋季每周一、周三 11:15–12:30（美国东部当地时间），课程表注明可能变动。

## 官方五次作业

| 阶段 | 一手资料 | 官方核心任务 | 可验证结果及边界 |
|---|---|---|---|
| A1 · Mads-Lens | [Assignment 1](https://cs2680.seas.harvard.edu/assignments/assignment1) | 用 Claude Code 做自己的浏览器观察界面：实时 tool call/result、subagent 树、错误状态、session resume、成本及耗时 | 交付 Web UI、writeup、演示录像；练习 Claude Code headless stream-json，**不要求**自建 Agent Loop |
| A2 · Mads-Loop | [Assignment 2](https://cs2680.seas.harvard.edu/assignments/assignment2) | 围绕课程模型 API 自建 coding-agent harness：工具 schema、执行和反馈、上下文、重试、终止、日志 | 用 SWE-bench Pro 相关任务验证 patch；课程指明评测与容器约束，不能把 starter 设计等同唯一正确架构 |
| A3 · Mads-Opt | [Assignment 3](https://cs2680.seas.harvard.edu/assignments/assignment3) | 在保持模型为 API 黑盒时优化 harness：token、工具调用、上下文、并行调度、agent 控制流 | 将成功、成本、端到端延迟同时纳入对比；官方作业涉及 agent/task 容器和 dispatcher |
| A4 · Mads-Serve | [Assignment 4](https://cs2680.seas.harvard.edu/assignments/assignment4) | 将 A3 Agent 的模型 API 替换为自行运行的模型服务，调整推理引擎或 Agent | 官方指定一块 RTX PRO 4500（32 GB）GPU 的限制，使用 SGLang 基线；以任务求解和端到端耗时评估 |
| A5 · Mads-Stack | [Assignment 5](https://cs2680.seas.harvard.edu/assignments/assignment5) | 从 A4 继续改进 Agent、可选模型和 serving，在统一资源限制下优化完整栈 | Terminal-Bench 2.0 任务及官方 leaderboard；要求总结 A2–A5 的证据，而非只报告吞吐量 |

“Mads” 是官网对 Modern Agent Design and Systems 的缩写，也与 Harvard MaDSys 研究组呼应。上述阶段箭头表示教学内容的逐步接管，不代表某个具体运行时进程调用顺序。

## 课程原文支持的工程判断

- **端到端度量：** 高 token 吞吐不一定带来更低的任务总耗时；Agent 的模型调用、工具延迟、重试、运行验证和失败都需要观测。
- **质量—成本权衡：** A3 的优化不应只压 token 数而牺牲通过率；课程成本笔记还提出按完成任务计量总成本、p50/p95 延迟以及各任务得失。
- **自托管推理：** A4 才接管 serving；模型性能、KV-cache、batching、调度和 GPU 资源约束会影响 A5 的 Agent 端到端质量。
- **系统可靠性：** A1 要观察轨迹与分叉，A2 对工具异常、无效格式和终止做明确处理，A3–A5 强调在相同任务基准上比较实现。

## 可用资源与获取边界

- **公开可阅读：** 官网课程大纲、排课链接、Assignment 1–5 说明、阅读清单与部分专题笔记，以及课程网页上链接的公开讲义。
- **未确认公开完整源码：** 官网提供每次作业的 starter archive 下载入口，但本次没有取得并审阅完整的独立可运行参考 Agent / Serving 源码仓库，不能声称官方已公开一套可复现的端到端实现。
- **获取限制：** 课程模型 API、GPU/HPC、私有 GitHub 作业仓库的教学资源面向授课环境；公开读者不能据此默认获得同等计算配额或隐藏评测集。
- **未核读：** 本次未逐页通读全学期所有幻灯片、论文 reading list 与 2026 年 11 月 15 日之后的预定讲次内容。官网 Schedule 标明 tentative，未来条目按计划而非已发生事实理解。
- **安全边界：** A1 headless 运行含自动工具授权示例，仅应在隔离的测试目录使用；保管 API key / 会话记录，避免在公网暴露可执行工具的 Agent UI。

## 对 wiki 的映射

- [Harvard CS2680：现代 AI 系统与 Agent 优化](../../wiki/entities/harvard-cs2680-modern-ai-systems.md)
- [Stanford CS146S：现代软件开发者](../../wiki/entities/cs146s-modern-software-developer.md)
- [Agentic Coding 时代的软件工程基础](../../wiki/concepts/agentic-coding-software-fundamentals.md)
