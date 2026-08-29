# Andrew Ng — AI Engineering Skills Map: Software engineering fundamentals

> 原始资料归档（ingest）。正文为归纳，非全文转载。

- **标题：** AI Engineering Skills Map: Software engineering fundamentals
- **类型：** blog / X Article（同文亦见 LinkedIn Pulse 与 DeepLearning.AI *The Batch*）
- **作者：** Andrew Ng（吴恩达）
- **机构：** DeepLearning.AI / Stanford CS adjunct
- **日期：** 2026-08-28
- **入库日期：** 2026-08-29
- **触发 URL：** <https://x.com/andrewyng/status/2093388974194872781>
- **X Article：** <https://x.com/i/article/2093384274372419585>
- **稳定镜像：**
  - LinkedIn Pulse：<https://www.linkedin.com/pulse/ai-engineering-skills-map-software-fundamentals-andrew-ng-7lnac>
  - *The Batch*：<https://www.deeplearning.ai/the-batch/the-ai-engineering-skills-map-in-detail-software-engineering-fundamentals>
- **系列位置：** [AI Engineering Skills Map](https://www.linkedin.com/pulse/ai-engineering-skills-map-andrew-ng-m479c) 四支柱之二（Software engineering fundamentals）。前一篇展开「Building and deploying AI applications」；文末预告后续两篇将写 **Using coding agents** 与 **Shaping the build**。
- **一句话说明：** 即便 coding agent 写完全部代码，软件工程基础仍是 **转向（steer）取舍** 的前提；只会 vibe coding 的人往往不知道延迟 / 可用性 / 一致性 / 可靠性 / 可维护性 / 简洁 / 成本这些权衡存在，因而无法给 agent 正确上下文。
- **开源状态（步骤 2.5）：** 无项目页、无可运行代码；资料是技能地图论述，**不适用** 源码核查。

## 为什么值得保留

- 本库已有大量 **coding agent 工具/技能页**（[mattpocock/skills](../../wiki/entities/mattpocock-skills.md)、[Superpowers](../../wiki/entities/superpowers-obra.md)）与 **真机/研究闭环**（[ENPIRE](../../wiki/methods/enpire.md)、[autoresearch harness](../../wiki/queries/real-robot-policy-autoresearch-harness.md)），缺的是「agent 写代码之后，人还该懂什么」的判断框架。
- 把「语法记忆过时」和「取舍语言更值钱」拆开，避免把 vibe coding 当成工程能力本身。
- **数据架构** 被单列：难改、且是 AI 系统自己的输入上下文来源——与机器人数据飞轮、日志/回放缓冲同源。

## 核心主张（归纳）

1. **转向，不是代写。** Agent 可以写完全部代码；你仍要知道有哪些取舍可被转向，并用软件工程的精确语言给出上下文。
2. **AI 核包在更宽的软件应用里。** 构建 AI 应用时，模型只是内核；外围应用仍要人帮助建造或塑形。
3. **Vibe coding 的失败模式：** 能做出简单应用，但 agent 会在 latency / availability / consistency / reliability / maintainability / simplicity / cost 上做坏选择——因为开发者不知道这些维度存在。
4. **过时的是语法记忆，不是「软件如何工作」。** 懂原理的开发者远优于只会 vibe coding 的人。
5. **软件能做什么 / 不能做什么** 也是后续「用 coding agent」与「塑造构建」两篇的上下文。

## 五项软件工程技能

资料基于对 **1 万+ 职位、专家/招聘访谈与问卷** 的综合（见总图文）。本篇展开的五项：

| 技能 | 原文要点 |
|------|----------|
| **Building full-stack applications** | Agent 让专精角色（前端 / 移动）能做更宽的全栈；仍须理解 UI、缓存、渲染、API 设计、认证、会话/状态、异步、持久化、测试、安全、无障碍。 |
| **Managing data** | 数据是难改的地基（即便 agent 能帮迁移）。按访问模式决定存什么、存多久；选型（关系 / 文档 / KV / 图）影响速度、扩展、可用、可靠、成本。事务、并发、干净/一致/新鲜；隐私、治理、合规；生命周期。演进时同步演进数据架构。**AI 的输入上下文来自数据源**——架构选错，模型不知道自己不知道。面向 agent（而不只是人或传统软件）的数据基础设施仍在快速演化。 |
| **Designing system architectures** | 先理解软件要做什么（用户量、延迟、成本），再选平台、前后端边界、分解、状态放置、粒度（单体 vs 微服务）与技术栈；可用实验评估后再钉死。**架构随阶段移动**：原型 ≠ 首版生产 ≠ 规模化。 |
| **Making systems secure and reliable** | 测试策略（单测 / 集成、框架、覆盖度）；围绕失败设计（限流、优雅降级、缩小爆炸半径）。**Shift left**：安全左移到生命周期更早阶段；开发者同时是部分安全工程师。AI 可扫漏洞、供应链注入、云配置攻击面，但仍需安全知识才能用好。 |
| **Scaling and operating in production** | SDLC：部署环境、发布策略、CI/CD、IaaS。生产：可观测、告警、事故。扩展：真实负载、扩容、负载均衡、分片/索引/复制或改架构。日常：版本控制、代码评审、依赖维护、技术债。 |

## 对 wiki 的映射

- [`wiki/concepts/agentic-coding-software-fundamentals.md`](../../wiki/concepts/agentic-coding-software-fundamentals.md) — 升格概念页：五项技能 + 对机器人训练/部署栈的读法
- [`wiki/entities/mattpocock-skills.md`](../../wiki/entities/mattpocock-skills.md) — 对抗 vibe coding 的技能库（对齐 / TDD / 架构卫生）
- [`wiki/entities/superpowers-obra.md`](../../wiki/entities/superpowers-obra.md) — 编码代理交付流程（brainstorm → TDD → 评审）
- [`wiki/queries/real-robot-policy-autoresearch-harness.md`](../../wiki/queries/real-robot-policy-autoresearch-harness.md) — 真机闭环里「有 agent ≠ 可跳过工程判断」
- [`wiki/concepts/ai-auto-research.md`](../../wiki/concepts/ai-auto-research.md) — 研究自动化里 SWE 能力 ≠ 科研能力

## 当前提炼状态

- [x] X 推文 + X Article 正文要点
- [x] LinkedIn / *The Batch* 镜像核对（同文；Batch 标题作 *Why Software Fundamentals Remain Essential for AI Developers*）
- [x] 升格概念页并交叉索引
