---
type: entity
project_id: agnes-harness
code: https://github.com/AgnesAI-Labs/agnes-harness
code_scope: "全仓 FDE agent runtime 本体（agnes.mjs、packages/{cli,sdk,host,loop-default,ai} 与 AgnesHub/MHS 插件体系）"
project_distinction: "Agnes Harness 软件运行时项目主实体；独立系统技术报告（账本语义与受治理 Playbook RSI 研究评估）见 paper-agnes-harness-rsi-technical-report。"
tags: [llm-agents, agent-infrastructure, open-source, device-integration, mcp, hardware]
status: complete
date: 2026-10-11
updated: 2026-11-22
related:
  - ../concepts/model-hardware-standard.md
  - ../concepts/llm-robotics-control-interfaces.md
  - ../concepts/safety-filter.md
  - ./muse-gadgets.md
  - ./agent-lightning.md
  - ./paper-agnes-harness-rsi-technical-report.md
  - ./paper-agnes-sys1-decision-model-technical-report.md
original:
  - ../../sources/repos/agnes-harness.md
sources:
  - ../../sources/repos/agnes-harness.md
  - ../../sources/papers/agnes_harness_technical_report_nov_2026.md
  - ../../sources/papers/agnes_sys1_decision_model_technical_report_2026.md
summary: "Agnes Harness（AGH）是 AgnesAI-Labs 开源的 FDE agent runtime：Web/CLI/SDK 共用会话，Agent Loop 负责工具执行、状态、记录与恢复，并通过插件、MCP、Skills 扩展；可选 AgnesHub/MHS 接入设备，但仍处 pre-alpha，且不替代设备侧实时控制与安全联锁。"
---

# Agnes Harness（AGH）

## 一句话定义

**Agnes Harness（AGH）** 是面向 Forward Deployed Engineering（FDE）的可插拔 agent runtime：把模型调用、工具、任务状态、跨端会话、人工审批与执行记录放进一个可复用的部署底座，并提供可选的物理设备接入路径。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| AGH | Agnes Harness | 本项目；管理 agent 运行、扩展与部署记录 |
| FDE | Forward Deployed Engineering | 在客户现场集成系统并交付持续迭代软件的工程方式 |
| MCP | Model Context Protocol | AGH 支持的外部工具/服务连接方式之一 |
| MHS | Model Hardware Standard | AGH 自有设备协议；同名但不等同于 Anthropic / HHMI Janelia 的 MHS |
| MOS | Model Observation Standard | AGH 设备侧观测数据约定，如图像、地图与地点 |
| SDK | Software Development Kit | AGH 提供的共享后端客户端入口之一 |

## 为什么重要

AGH 的机器人相关性在 **系统集成与高层设备编排**，不在运动控制算法：它展示了如何把 Web/终端工作台、模型、工具、工作流状态和设备能力置于同一任务记录下。部署到不同客户或实验室时，差异可通过插件和 Skills 承载，避免把每个连接器都写进 agent 核心。

把设备工具纳入 agent 任务流也会放大软件执行错误的物理影响，因此它适合与 [LLM 机器人控制接口](../concepts/llm-robotics-control-interfaces.md) 和 [Safety Filter](../concepts/safety-filter.md) 一起阅读：agent 保持在任务级编排，实时运动闭环和安全停机仍归设备控制器。

## 核心结构

| 层 | 作用 | 当前状态 / 入口 |
|----|------|------------------|
| **Web / CLI / SDK** | 创建任务、查看结果、跨端续接同一会话 | README 标为开发预览可用；客户端共享后端 session |
| **App Server 与 Agent Loop** | 会话、任务提交、事件分发、审批路由；执行模型与工具调用、状态恢复 | 默认 loop 是可替换的 @agnes/loop-default 插件；Sys1 需另行启用 judgment 插件，Jev 为探索性集成 |
| **插件与 Skills** | backend tools/services、Web panels、hooks、MCP 与可复用任务方法 | 示例包括工具、前端面板和 full-stack plugin |
| **审批与 Trace** | 在支持的命令执行路径上等待用户批准，并记录模型调用、工具调用和审批过程 | 提供审计线索；不构成任意插件代码的隔离边界 |
| **AgnesHub / MHS** | 将设备状态、工具和观测接入任务流 | 可选插件；设备以 WebSocket 注册能力，MOS 表达观测数据 |

## 流程总览

下图概括 README 与架构指南描述的高层请求路径；具体模块边界以官方架构文档和当前源码为准。

```mermaid
flowchart TB
  client["Web / CLI / SDK"] --> server["App Server<br/>共享会话与审批路由"]
  server --> loop["Agent Loop<br/>Core、任务状态与恢复"]
  loop --> model["AI Provider<br/>模型请求"]
  loop --> tools["Plugins / Skills / MCP"]
  tools --> devices["AgnesHub / MHS<br/>可选设备能力"]
  loop --> trace["事件与 Trace 记录"]
```

AGH 是任务级协调层。即使任务目标由语言模型生成，设备驱动仍应在设备侧维护实时控制、互锁和急停；agent 任务取消不保证硬件已停止。

## 源码运行入口

当前官方 README 提供以下**源码构建**路径，需 Node.js 24.10+、pnpm 10.34.5，未在本次环境执行：

```bash
mise trust
mise install
mise exec -- node tools/toolchain/run.mjs install --frozen-lockfile
mise exec -- node tools/toolchain/run.mjs --filter @agnes/cli build:local
mise exec -- node agnes.mjs serve
```

根 `agnes.mjs` 将参数转给打包 CLI（`packages/cli/dist/local/agnes.mjs`）。这是不迁移旧 home 的新预览版本；不得按旧版安装入口推断兼容性。


## 设备协议：同名 MHS 的消歧

AGH 使用 **Agnes MHS（Model Hardware Standard）** 指代自己的设备协议：设备通过 WebSocket 连接 AgnesHub，注册状态、工具和数据源；MOS 表达设备观测。官方说明该协议独立于 MCP，也独立于厂商 SDK。

这与 [Anthropic / HHMI Janelia 的 MHS 概念页](../concepts/model-hardware-standard.md) **不是同一协议**。它们恰好共用 “Model Hardware Standard” 全称与 MHS 缩写，组织、接口描述和发布状态均不同；不要据此推断协议兼容、代码复用或标准关系。

## 2026-11 新技术报告与版本边界

- [AGH 系统及 Playbook RSI 技术报告](./paper-agnes-harness-rsi-technical-report.md)：账本/一次性许可/未知结果对账、快慢系统与仅配置数据的受治理自改进。仿真及 model+harness 成绩属于作者报告，**未证实现场机器人效果**。
- [Agnes Sys1 1.0 决策模型技术报告](./paper-agnes-sys1-decision-model-technical-report.md)：基于 Qwen3.5-9B 的请求内候选概率、分块双向 attention 与拟议 RMSNorm TTT；作者能力表 AA Composite 71.2，但**TTT 无实测增益**。
- AGH 系统技术报告以 Jev 为 DM 案例；当前 README 中 Sys1 为默认建议的 side model，`@agnes/judgment` 属可选插件且检测默认关闭，不能混写成同一组 benchmark 配置。
- 两份技术报告均已核对原始 PDF：AGH 系统 21 页和 Sys1 1.0 五页，并与官方 `REPORT.md` 对照；两篇来源归档均有原文通读记录，**不把任何一篇的团队指标称为独立复现**。

## 工程实践与安全边界

- **源码可用，产品仍在预览期：** 仓库公开且项目代码采用 Apache-2.0，但发布形态仍是 source build / pre-alpha；没有公开包发行、安装器或生产支持承诺。
- **插件是可信进程内代码：** 安装插件前检查来源、版本、完整性摘要、能力 hash 和许可证。命令审批及 sandbox 约束适用于指定执行路径，不隔离普通 backend plugin。
- **物理安全必须落在设备层：** MHS 合规测试不等同于设备安全认证；保留设备控制器的实时闭环、互锁和急停，并针对断连、超时和 agent 取消单独验证。
- **平台支持要按目标机验证：** README 记录 macOS / Node 24 检查，并明确 Linux 与 Windows 需要独立验收。
- **接口可能变化：** pre-alpha API、配置和插件接口仍在演进，不宜按稳定依赖或生产升级路径处理。
- **外部 PR 受限：** README 当前称代码和文档 PR 限受邀内部开发者；公开 Issues 仍可提交问题和用例建议。

## 局限与风险

- AGH 是 FDE agent harness 与设备集成平台，不是 ROS 2 驱动集合、VLA、运动策略、实时总线或安全认证。
- 模型连接和工具插件能扩展软件能力，但不能自动带来物理接触推理、确定性延迟或硬件安全保证。
- 项目所称 MHS 是其自有接口，不要与 Anthropic/HHMI 同名 MHS 或 MCP 混为一谈。

## 源码运行时序图

下图取自当前官方 `agnes.mjs`、`docs/develop/architecture.md` 与 `docs/develop/source-map.md` 确认的源模块层次，不冒充两份技术报告中的实验脚本。

```mermaid
sequenceDiagram
    autonumber
    actor U as 用户
    participant CLI as agnes.mjs / Web
    participant SDK as packages/sdk
    participant D as daemon / worker
    participant H as packages/host
    participant L as packages/loop-default
    participant K as Core / core-ledger
    participant M as packages/ai
    participant T as 受控工具执行
    U->>CLI: 启动 serve 并输入任务
    CLI->>SDK: 创建或恢复会话
    SDK->>D: 请求与事件
    D->>H: 装配插件和能力
    H->>L: 默认 Loop 插件
    L->>K: 推进会话状态
    K->>M: 推理调用
    M-->>K: 工具调用意图
    K->>T: 授权后执行
    T-->>K: 结果或未知副作用
    K->>K: 账本记录和恢复
    K-->>D: 会话与 Trace
    D-->>SDK: 更新状态
    SDK-->>U: 结果或审批要求
```

未运行源码构建、GPU 评测或设备真机测试。此图仅陈述可复核的请求路径，细粒度插件实现仍以具体 commit 为准。


## 关联页面

- [AGH 2026-11 技术报告](./paper-agnes-harness-rsi-technical-report.md)
- [Sys1 1.0 决策模型报告](./paper-agnes-sys1-decision-model-technical-report.md)

- [Model Hardware Standard（Anthropic / HHMI Janelia）](../concepts/model-hardware-standard.md) — 与 AGH 同名但彼此独立的协议概念
- [LLM 机器人控制接口](../concepts/llm-robotics-control-interfaces.md) — agent 的抽象层级及控制边界
- [Safety Filter](../concepts/safety-filter.md) — 物理动作的约束与安全层
- [Muse Gadgets](./muse-gadgets.md) — 语言助手经边缘 SDK 访问设备的开源案例
- [Agent Lightning](./agent-lightning.md) — 真实 agent harness 的轨迹采集与 RL 训练设施

- [智能体规划与反馈执行](../methods/agent-planning-feedback.md)：了解技能被调用时的任务流程与反馈。
- [智能体技能封装](../methods/agent-skill-packaging.md)：了解任务指引、可复用资源与边界。

## 参考来源

- [AGH 技术报告摘要来源](../../sources/papers/agnes_harness_technical_report_nov_2026.md)
- [Sys1 模型技术报告摘要来源](../../sources/papers/agnes_sys1_decision_model_technical_report_2026.md)

- [Agnes Harness 仓库与 README 摘录](../../sources/repos/agnes-harness.md)
- 源码核对（2026-11-02）：AgnesAI-Labs/agnes-harness@2bfde24（默认分支 HEAD），读过 agnes.mjs、docs/develop/source-map.md、packages/cli/src/boot/local.ts（局部）、packages/sdk/src/submit.ts（局部）、packages/protocol/src/methods.ts（局部）、packages/daemon-rpc/src/local/methods/agnes.ts（局部）、packages/daemon-rpc/src/local/follow-up-runner.ts（局部）、packages/core/src/step/session.ts（局部）、packages/core/src/execution/turn/inference.ts（局部）、packages/core/src/execution/model-execution.ts（局部）；时序图依据以上文件，未运行代码。

## 推荐继续阅读

- [Agnes Harness 官方仓库](https://github.com/AgnesAI-Labs/agnes-harness)
- [架构指南](https://github.com/AgnesAI-Labs/agnes-harness/blob/main/docs/develop/architecture.md)
- [安全与信任指南](https://github.com/AgnesAI-Labs/agnes-harness/blob/main/docs/guide/security.md)
- [MHS 与设备指南](https://github.com/AgnesAI-Labs/agnes-harness/blob/main/docs/guide/mhs.md)
