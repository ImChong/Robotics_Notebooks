---
type: entity
project_id: paperspine
tags: [ai-auto-research, academic-writing, research-workflow, llm-agents, evidence-grounding]
status: complete
updated: 2026-11-11
code: https://github.com/WUBING2023/PaperSpine
project: https://wubing2023.github.io/PaperSpine/v5/
original:
  - ../../sources/repos/paperspine.md
sources:
  - ../../sources/repos/paperspine.md
related:
  - ../concepts/ai-auto-research.md
  - ./supervisor-skills.md
  - ./nature-polishing.md
  - ./aris-auto-research-in-sleep.md
summary: "PaperSpine5 是基于作者材料和可核查证据的论文全流程 Skill；先约束核心贡献与 Motivation，后组织写作、引文、配图、评审与可编辑格式交付。"
---

# PaperSpine5（贡献优先的学术论文生产）

## 一句话定义

**PaperSpine5** 是本地优先、证据约束的论文构建及审阅工作流；核心原则是先确认研究能辩护的 **Contribution 与 Claim Boundary**，再让 Motivation、各章节、图表和引用为这个主张提供支撑，而不是先把文字“润色漂亮”。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|---|---|---|
| AI | Artificial Intelligence | 配合研究者实施论文工作流的 Agent |
| UI | User Interface | 保存论文任务配置、材料与选择的本地 Web 界面 |
| PDF | Portable Document Format | 支持的最终只读交付格式 |
| CLI | Command-Line Interface | `paperspine5_web.py` 等脚本的命令行入口 |

## 为什么重要

导师反馈“贡献不清楚”常是论证与证据组织问题，而不是语法问题。官方 Contribution-First 指南将 **一条主要贡献 + 证据要求 + 缺证据 + 主张边界** 分开核对，防止写作掩盖方法与实验的缺口。

## 核心原理

- **Contribution-First：** 区分一条可检验主贡献、贡献类型、审稿人获得的价值；列出研究缺口、已知先例、设计回应和需要/已有/缺失的证据。
- **Motivation 受贡献约束：** 解释为什么研究重要，却不能替代“本论文究竟证明了什么”；随后通过 Introduction–Results–Discussion 形成可回指的逻辑链。
- **同一任务持续修订：** 当前 V5 `src/skill/SKILL.md` 以任务 ID、材料绑定、Web 保存的配置与人类选择为边界，保留同任务的修订历史，不因 launcher 失败擅自换任务。
- **交付：** 官方 README 声明可生成图文稿、核验引用并输出 Word/LaTeX/PDF；这些是仓库功能描述，不是已经独立完成的质量验收。

## 流程总览

```mermaid
flowchart LR
  Material["用户材料与研究问题"] --> Config["Web 选择配置和材料"]
  Config --> Contribution["贡献声明和证据边界"]
  Contribution --> Motivation["确认 Motivation 主线"]
  Motivation --> Plan["章节和证据映射"]
  Plan --> Draft["正文及图表草稿"]
  Draft --> Audit["引用核验与论文审阅"]
  Audit --> Deliver["可编辑文稿与 PDF 导出"]
  Audit -.->|修订建议| Plan
```

图据官方 README、`src/skill/SKILL.md` 及 Contribution/Motivation 指南。当前 V5 文档允许复用已保存的有效选择，**不等于每次都要重新生成旧版 `confirmed_contribution.md` 文件**。

## 源码运行时序图

核对主 `SKILL.md`、`paper-spine-production-protocol.md` 17 步全文，以及 Web launcher、public MCP client、Pandoc exporter 的核心入口源码。下图是 **PaperSpine5 当前宿主 Agent + 公共 Web + 导出器** 的已确认调用路径，非用户实际论文的测试日志。

```mermaid
sequenceDiagram
  autonumber
  actor User as 用户
  participant Agent as 宿主 Agent
  participant Skill as src/skill/SKILL.md
  participant Web as paperspine5_web.py
  participant Client as paperspine_public_client.py
  participant RPC as 公共 MCP 任务服务
  participant Export as manuscript_export.py
  participant Pandoc as Pandoc
  User->>Agent: 提供材料和论文任务
  Agent->>Skill: 读取 17 步协议和任务权限
  Agent->>Web: launch --no-open
  Web->>Client: web_launch
  Client-->>Web: 本地 Web 地址
  Web-->>Agent: 返回相同 profile 的服务地址
  Agent->>Web: host tools / snapshot
  Web->>Client: call
  Client->>RPC: paperspine_get_task
  RPC-->>Client: task_id、配置和版本
  Client-->>Agent: 已保存任务快照
  User->>RPC: 在 Web 确认配置、贡献与选图
  Agent->>Web: host wait / snapshot
  Web->>Client: call
  Client->>RPC: paperspine_wait_for_task_change
  RPC-->>Agent: 更新后的任务选择
  Agent->>Agent: 基于实际证据写作与独立评审
  opt 用户已授权导出且依赖可用
    Agent->>Export: 传入语义 Markdown 文稿
    Export->>Pandoc: markdown 转 JSON AST
    Pandoc-->>Export: 文稿 AST
    Export->>Pandoc: JSON AST 转 DOCX 或 LaTeX
    Pandoc-->>Export: 可编辑产物
    Export-->>Agent: 文件及核验报告
  end
```

上图明确区分 **host Agent 的科研工作**、**公共 MCP 的状态/权限保存** 与 **Pandoc 的格式导出**。已读完官方 17 步生产协议与关键运行源码；完整实机执行及最终页面视觉审查仍待另行验证。

## 工程实践

- 已通读主 Skill 与 17 步生产协议；使用上游非 Fork 仓库 `WUBING2023/PaperSpine`，固定核查版本 `42312b8d9c76`，当前 README 是 **PaperSpine5 Alpha**。
- 将作者材料、真实实验指标、可信参考文献放入论文任务；严格区分证据已具备与尚待实验。
- 优先确认 `Main contribution statement`、`Evidence required/available/missing`，再定动机与章节；把任何改变科学含义的候选文字交还作者确认。
- 导出涉及 `Pandoc >=3`，生成 PDF 还需 LaTeX 引擎，需在实际环境检查。

## 局限与风险

项目仍为 Alpha；虽然官方主 Skill 与 17 步协议已通读，但没有在本次执行环境运行整套任务、遍历所有插件或验证最终可下载 Word/PDF 的视觉质量。自动修改和 Web 操作依赖用户确认的任务绑定及授权；语言流畅不等于研究贡献成立。**文档通读完成不代表端到端功能测评完成。**

## 关联页面

- [Supervisor-Skills](supervisor-skills.md)：侧重科研判断清单与导师方法论。
- [nature-polishing](nature-polishing.md)：已有稿件的语言与主张边界修订。
- [ARIS](aris-auto-research-in-sleep.md)：实际实验迭代和交叉评审。
- [AI Auto-Research](../concepts/ai-auto-research.md)：研究自动化生命周期。

## 参考来源

- [PaperSpine5 原仓库归档](../../sources/repos/paperspine.md)
- [WUBING2023/PaperSpine（上游）](https://github.com/WUBING2023/PaperSpine)

## 推荐继续阅读

- [完整 17 步原始生产协议](https://github.com/WUBING2023/PaperSpine/blob/42312b8d9c76/src/skill/references/paper-spine-production-protocol.md)
- [Contribution-First 规则](https://github.com/WUBING2023/PaperSpine/blob/42312b8d9c76/src/skill/references/contribution.md)
- [PaperSpine5 官方产品页](https://wubing2023.github.io/PaperSpine/v5/)
