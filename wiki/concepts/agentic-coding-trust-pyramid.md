---
type: concept
tags:
  - coding-agents
  - software-engineering
  - static-analysis
  - verification
  - agent-infrastructure
status: complete
updated: 2026-11-17
summary: "Lauren Tan 在 AI Engineer CODE 2026 提出的 Coding Agent 信任金字塔：先用 API 与架构防止错误、静态分析拒绝反模式，再用真实运行及形式化验证提供证据，最后才扩展可自主合并的多 Agent 软件工厂。"
related:
  - ../methods/coding-agent-verification-loops.md
  - ./agentic-coding-software-fundamentals.md
  - ./coding-agent-orchestration-and-cognitive-oversight.md
  - ./test-driven-development.md
  - ./ai-agent-evaluation.md
original:
  - ../../sources/videos/lauren-tan-all-code-is-slop-2026.md
sources:
  - ../../sources/videos/lauren-tan-all-code-is-slop-2026.md
---

# Coding Agent 信任金字塔 / Trust Pyramid for Agentic Coding

## 一句话定义

**Coding Agent 信任金字塔**是 Lauren Tan 在 2026 年演讲中给出的工程优先级：把高频纠错优先固化为**良好的代码架构和可执行约束**，再通过运行时 / 形式化验证建立信任，让软件开发吞吐提升时仍能检验质量，而不是只靠 prompt 或人工逐行评审。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
| --- | --- | --- |
| API | Application Programming Interface | 优先通过接口与架构设计消除无效操作 |
| CI | Continuous Integration | 为每次 PR 提供可重复的质量门禁 |
| PR | Pull Request | 演讲采用生产 PR 吞吐与回滚占比描述团队流程 |
| CDP | Chrome DevTools Protocol | 运行时操控桌面/Web 应用的调试能力 |
| E2E | End-to-End | 在真实运行环境中检查完整用户路径 |
| TLA+ | Temporal Logic of Actions | 为复杂状态流提供模型检查 |
| UI | User Interface | 需结合截图、视频、状态迁移检验用户行为 |

## 为什么重要

- **写得快不代表交付得好。** 当 agent 一次产出大量 PR，人工逐行评审的吞吐会成为瓶颈；重点是减少需要人反复判断的错误类型。
- **代码库同时是规范和示例。** Agent 很容易复制旧的 workaround、反模式和失效注释；积压的技术债会影响后续输出而不只是增加维护成本。
- **信任来自可执行证据。** 可重复的静态规则、真实应用交互和自动测试比在 AGENTS.md 追加模糊提醒更可核验。
- **与现有知识分工：** [Coding Agent 验证闭环](../methods/coding-agent-verification-loops.md) 讨论 E2E/属性/差分等验证手段；本页讨论更靠前的**防错架构、约束设计及外层自动化的先后顺序**。

## 核心原理

### 逆金字塔：把预算优先花在可重复的约束上

| 层级（按优先级） | 代表手段 | 主要解决的问题 |
| --- | --- | --- |
| ① 代码库与架构 | 设计良好 API、清理反模式、重构模块边界 | 让错误操作难以出现，减少 agent 模仿坏样板 |
| ② 静态分析 | Lint、类型系统、编译器、依赖图约束、测试 | 阻止可机械判定的错误进入 PR |
| ③ 运行时及形式化验证 | 真实应用操作、截图/视频、TLA+、Lean、属性测试 | 判断实际行为和系统不变量是否满足目标 |
| ④ 文字规则与人工判断 | Rules、Skills、AGENTS.md、风险审批 | 处理不能靠静态规则明确表达的意图和取舍 |

**不是禁止写规则。** 演讲的排序是：已知反复出现的问题应尽可能转成可执行约束；文字仍适合表达项目意图、上下文和判断边界。

### 流程总览：从一次纠错到可复用的高信任机制

```
flowchart TD
    A["用户目标和可检验验收条件"] --> B["Agent 阅读代码库并实现"]
    B --> C["架构/API 限制不良选择"]
    C --> D["Lint、类型和依赖图静态检查"]
    D -->|阻断| E["修正根因或改进接口与规则"]
    E --> B
    D -->|通过| F["运行真实应用与 E2E 验证"]
    F --> G["属性测试、模型检查及状态图证据"]
    G -->|失败或意图偏差| E
    G -->|证据通过| H["按风险评审与合并 PR"]
    H --> I["发布后监测、回滚、事故复盘"]
    I --> J["把反复错误固化成新的约束"]
    J --> C
    I --> K["基础稳定后扩展并行 Agent 与自动修复外环"]
```

此图根据演讲重构为工程流程，不是从 Grok Bot 或 Dune 公开源码提取的真实模块调用时序。

### 演讲的具体证据链

1. **Dune 的 React 限制：** 在该团队产品代码中限制容易误用的 useEffect，仅在部分框架 API 开放；目的是通过设计直接消除一类持续返工问题，而非断言 React Hook 本身有问题。
2. **Electron 线程边界：** 将 electron-main 与 electron-renderer 分目录，用依赖图及 lint 拒绝跨边界导入；防止主进程逻辑误入渲染线程引发卡顿。
3. **定时器的“杂草效应”：** 为规避声明式与命令式架构冲突而使用临时定时器，后续代码/注释被 agent 复制；团队倾向用能够报出修复方向的检查替代这种被合理化的绕路。
4. **从看 diff 到看证据：** 通过 feature map 提供用户视角的功能词汇，使用可复用 CLI / CDP 运行应用，审阅截图、视频与状态图；对复杂用户流探索 TLA+，对部分算法使用 Lean，UI 引入 Bombadil 属性测试。
5. **产能数字需谨慎读：** 演讲报告 Lauren 个人约 3,000 个生产 PR，团队上月约 11,000 个 PR、回滚比例约 0.5%、10 月约 227 位参与者。这些均为团队自述，不能将“较低回滚率”直接解释成无 bug 或经过对照研究验证。

## 工程实践

### 可应用到知识仓库、机器人仿真和真机工程的步骤

| 工程层面 | 建议动作 | 可检查的证据 |
| --- | --- | --- |
| 知识库 ingest | 把来源元数据、原文通读字段、重复节点和内链规则编入质量检查 | 原文归档、实际健康度、Wiki Lint、Mermaid 语法结果 |
| 机器人软件架构 | 让命令/状态接口显式区分控制模式、进程边界和实时任务 | 接口契约、依赖检查、限位约束、静态分析 |
| 仿真验证 | 让 Agent 在固定种子、版本与场景下运行测试，而非只生成策略代码 | rollout、失败回放、关节速度/力矩越限与性能记录 |
| 部署验收 | 按故障风险审查控制时序、通信超时、急停与回滚 | 真机实测、独立审核、版本/异常追溯 |
| 外层自动化 | 先稳定单次修改—验证闭环，再逐步增加并行与自动修复 | 错误率、回滚率、耗时、漏检、人工接管比例 |

以上机器人映射是本知识库的工程建议，**并非演讲报告的机器人实验结论**。

### 最小执行顺序

1. 对同一类重复 Review 意见，先判断能否通过 API 或目录边界彻底消除；否则增加有可读错误信息的 lint / test。
2. 清理旧反模式示例；否则 Agent 仍会从历史代码中学到坏做法。
3. 在每次 PR 里附真实运行证据和失败用例；避免仅输出“已完成”“已通过”之类不可核查的文字。
4. 以更广的性能、可靠性和回滚指标共同观察质量，再决定哪些低风险改动能自动合并。
5. **安全关键软件保留人工许可。** 验证工具不是急停、功能安全或现场风险评估的替代品。

**源码运行时序图：不适用。** 本次一手材料是大会演讲，未提供可供核对的 Dune/Grok Bot 完整可运行源码及入口；上图是演讲论证的概念流程。

## 局限与风险

- **团队自述不等于可推广基准。** 3,000、11,000、0.5% 等数字并非本仓库复测；回滚率还无法覆盖由后续补丁修复的缺陷。
- **禁止注释是局部做法。** 演讲提到其产品代码中约 99% 注释被禁止；对需要可读规范、审计记录或安全论证的工程不能照搬。
- **形式化验证不保证整机安全。** TLA+ 模型检查与 Lean 证明的有效范围取决于模型、假设和待证明性质；仿真与物理设备还需独立验证。
- **自动合并依赖仓库及风险等级。** UI 产品里的自动修复 PR 流程不可直接外推到真机运动控制、急停和安全回路。
- **因果关系不宜过度归因。** 质量改善与吞吐提高同时出现，不能仅由演讲推断某一项静态规则单独导致性能提升。

## 关联页面

- [Coding Agent 验证闭环](../methods/coding-agent-verification-loops.md) — 端到端、属性测试与差分验证如何组织为可复核交付门禁。
- [Agentic Coding 时代的软件工程基础](./agentic-coding-software-fundamentals.md) — 软件架构、系统取舍与工程能力基础。
- [多 Agent 编码管理与认知把关](./coding-agent-orchestration-and-cognitive-oversight.md) — 并行代理与人类责任边界。
- [测试驱动开发](./test-driven-development.md) — 将反复纠错转成可执行回归。
- [AI Agent 评测](./ai-agent-evaluation.md) — 验证器、指标与评测本身也需校准。

## 参考来源

- [Lauren Tan 演讲完整核对归档](../../sources/videos/lauren-tan-all-code-is-slop-2026.md)
- [AI Engineer CODE 2026：官方完整字幕](https://ai.engineer/code/2026/talks/lauren-tan)

## 推荐继续阅读

- [All Code Is Slop (Except Mine) — AI Engineer 官方视频](https://www.youtube.com/watch?v=O_zcc6Xl12g)
- [官方演讲页面、字幕与图表](https://ai.engineer/talks/O_zcc6Xl12g-all-code-is-slop-except-mine)
- [pstack 插件源码](https://github.com/cursor/plugins/tree/main/pstack) — 演讲中提到的 statechart skill 入口。
