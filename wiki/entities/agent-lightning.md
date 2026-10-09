---
type: entity
tags:
  - microsoft
  - llm-agents
  - coding-agents
  - agent-infrastructure
  - reinforcement-learning
  - agentic-rl
  - open-source
status: complete
date: 2026-09-19
updated: 2026-10-09
related:
  - ./deepseek-harness.md
  - ./hermes-agent.md
  - ./paper-metarsi-v1.md
  - ./karpathy-autoresearch.md
  - ../concepts/ai-auto-research.md
  - ../methods/reinforcement-learning.md
  - ../references/llm-wiki-karpathy.md
sources:
  - ../../sources/repos/agent_lightning.md
  - ../../sources/sites/agent-lightning-microsoft-research.md
  - ../../sources/sites/agent-lightning-v1-0-release-blog.md
  - ../../sources/papers/agent_lightning_v1_technical_report.md
summary: "Agent Lightning（microsoft/agent-lightning，MIT，v1.0.1）是微软开源的 Harnessed Agentic RL 栈：真实 agent harness 经 API Gateway 接入，Controller 在本地或 Kubernetes 执行 rollout，Trainer 用 verl + vLLM 汇总轨迹并更新策略；v1.0 还支持共置异步 rollout 与训练。Coding Agent 示例以约 6K 样本将 SWE-bench Verified Pass@1 从 41.8% 提升到 56.4%。"
---

# Agent Lightning（Microsoft）

**Agent Lightning**（[microsoft/agent-lightning](https://github.com/microsoft/agent-lightning)）是微软研究院开源的 **agentic 强化学习基础设施**（PyPI `agentlightning` **1.0.1**，MIT）。v1.0 用约 **3,500 行 Python** 把训练拆成 **Trainer / API Gateway / Rollout Controller** 三组件：agent 仍走原有 harness（工具、上下文、控制流、环境不变），仅把模型调用改经 **OpenAI 兼容代理** 以便采集轨迹并对接 **verl + vLLM** 做 GRPO 等策略更新。

## 一句话定义

用 **Gateway 代理 + 真实 harness rollout + verl 训练环**，把「任意 LLM agent 的交互轨迹」变成可优化的 RL 样本，而 **不必重写 agent 代码或沙箱编排**。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RL | Reinforcement Learning | 从交互回报中学习策略；本栈面向 LLM agent 轨迹级优化 |
| GRPO | Group Relative Policy Optimization | verl 侧常用的组相对策略优化变体（示例训练栈） |
| K8s | Kubernetes | v1.0 Controller 可将 agent rollout 直接调度为 Job |
| API | Application Programming Interface | Gateway 提供 OpenAI 兼容代理面，harness 零改动接入 |
| SWE | Software Engineering | Coding Agent 示例基准域（如 SWE-bench Verified） |
| vLLM | — | 高吞吐 LLM 推理服务；Trainer 侧与 verl 联用 |

## 核心信息

| 字段 | 内容 |
|------|------|
| 机构 | 微软（Microsoft / Microsoft Research） |
| 类型 | Agentic RL 训练框架（Gateway + Controller + Trainer） |
| 版本 | 1.0.1（v1.0 完全重构；v0.x 见独立分支） |
| 代码 | <https://github.com/microsoft/agent-lightning> |
| 文档 | <https://microsoft.github.io/agent-lightning/stable/> |
| 许可 | MIT |
| 开源结论 | **已开源**（完整包、示例、文档与 verl 安装脚本）；**不自带** 基座权重，GPU 栈需自行准备 |

## 为什么重要（对本知识库读者）

- **Harness 与训练解耦：** 本库已收录 [DeepSeek Harness](./deepseek-harness.md)、[Hermes Agent](./hermes-agent.md)、[RSI-Harness](paper-metarsi-v1.md) 等 **agent 运行时 / 实验组织** 层；Agent Lightning 解决 **下一层** — 如何把这些 harness 里真实的工具调用轨迹 **变成 RL 更新**，而不是另写一套「简化版 agent 环境」。
- **对接 autoresearch 思维：** [Karpathy autoresearch](./karpathy-autoresearch.md) 与 [AI Auto-Research](../concepts/ai-auto-research.md) 强调 **Explore→Execute→Verify**；Agent Lightning 的 Coding Agent 管线（数据清洗、reward hacking 防护、仓库测试奖励）是 **S3 代码与实验** 侧可复用的 **RL 飞轮** 参照，与机器人 sim RL 正交但共享「轨迹→策略」结构。
- **工程可落地：** 官方给出 **单卡 A100 Calc-X Quick Start**、**K8s Job** 模式，以及 Search R1 / LLM-in-Sandbox / SWE 等多域示例；对需要 **在线 agent rollout + 异步采集** 的研究栈（对照 [reinforcement-learning](../methods/reinforcement-learning.md) 中 verl 生态条目）是直接入口。

## 核心原理

### 三组件分工（v1.0）

| 组件 | 职责 |
|------|------|
| **Trainer** | 启动 Ray、`verl`、vLLM；构建训练 batch；执行策略更新 |
| **API Gateway（`agl-server`）** | 代理 LLM 请求/响应，记录 token 级轨迹供训练聚合 |
| **Rollout Controller（`agl-controller`）** | `runner_type=local` 或 K8s reconciler 启动 agent 进程 / Job |

Trainer 创建 rollout 任务 → Controller 拉起带真实 harness 的 agent → agent 经 Gateway 调模型 → Gateway 把交互转为训练事件 → Trainer 聚合轨迹并更新权重。

### 流程总览

```mermaid
flowchart TB
  subgraph trainSide [Trainer 侧]
    T[Trainer\nverl + vLLM + Ray]
    DS[(轨迹 / parquet 样本)]
  end
  subgraph rolloutSide [Rollout 侧]
    C[Rollout Controller\nlocal 或 K8s Job]
    A[Agent + 真实 harness\n工具 · 环境 · 控制流]
  end
  G[API Gateway\nOpenAI 兼容代理]
  T -->|下发 rollout| C
  C -->|启动| A
  A -->|模型请求| G
  G -->|推理 + 记录| T
  G --> A
  T --> DS
  DS --> T
```

### 从真实 harness 轨迹到训练样本

真实 harness 自己管理工具调用、上下文压缩和环境循环，训练端看到的是一串模型请求 / 响应，而不一定是单条连续 token 序列。官方 v1.0 文档把正确聚合视为训练算法的一部分：

- **Trajectory 模式（默认）：** 仅当后续请求的 token 前缀与前一轮 prompt + response 精确连续时才合并；轮间新增的工具观察可作为上下文保留并从 policy loss 中 mask。长度超限可能导致样本丢弃或截断。
- **Transition 模式：** 每次模型调用单独成为一个训练样本，不跨轮合并。
- **Rollout 级 advantage 与 per-rollout mean loss：** 一个 rollout 可被切成数量不等的样本；按 rollout 计算优势、按 rollout 归一化损失，可避免交互较多的轨迹仅因样本更多而获得额外权重。
- **调度：** rollout 的完成时间与最终样本数事先不确定，trainer 需要把可变 rollout 工作负载映射到固定的 GPU / 并行配置。

### 共置异步训练（Collocated Async RL）

当 rollout 耗时不均时，同步训练要等最慢一组；完全分离式异步通常又需要独立的 rollout GPU 池。Agent Lightning 的共置异步模式让 rollout 推理与策略更新共用 GPU，并以已完成的 prompt group 形成更新批次：

1. Trainer 维持最多 `async_train_batch_size` 个活动 prompt groups，Controller 在本地进程或 K8s Job 启动 agent。
2. 达到 `data.train_batch_size` 个完成组后开始更新；尚未完成的组结转到下一轮，GRPO/RLOO 同一组中的兄弟 rollout 不拆开。
3. 更新前 Gateway 暂停新模型请求并等待在途请求完成；模型更新结束后恢复推理。Agent 使用可重试的 OpenAI / HTTP 客户端处理暂停窗口。
4. 异步旧策略 rollout 可能发生 policy staleness；官方文档建议配置 verl token-level importance sampling correction，并给出 clipping threshold 2 的起点。

文档建议 `async_train_batch_size > train_batch_size`，可先从约 2 倍开始，再依据 agent rollout 时长、CPU / 内存压力和 carry-over 指标调整。微软 2026-10-07 发布说明在其测试中报告相较同步 RL 约 **2 倍端到端加速**，且 GPU 数少于传统独立 GPU 池的异步方案；此为特定实验报告，不是普遍性能承诺。

```mermaid
sequenceDiagram
  autonumber
  participant Trainer as Trainer（verl）
  participant Controller as Rollout Controller
  participant Agent as 真实 Agent harness
  participant Gateway as API Gateway
  participant GPUs as 共用 GPU 池
  Trainer->>Controller: 创建 prompt groups
  Controller->>Agent: 本地进程或 K8s Job rollout
  Agent->>Gateway: OpenAI 兼容模型请求
  Gateway-->>Agent: 推理响应
  Gateway->>Trainer: 请求轨迹与 rollout 事件
  Note over Trainer,Controller: 完成组进入更新批次；未完成组结转
  Trainer->>Gateway: 暂停新请求
  Gateway->>Gateway: 排空在途请求
  Trainer->>GPUs: 更新模型策略
  Trainer->>Gateway: 恢复推理请求
```

这一路径说明了「异步」并非让模型更新与在途推理无协调地并发：Gateway 的 pause / drain 边界用于管理共享 GPU 上的权重更新。

### 源码运行时序图

典型 **本地 Calc-X** 路径（`examples/calc_x/run_local.sh`）：

```mermaid
sequenceDiagram
  autonumber
  participant User as 维护者
  participant Run as run_local.sh
  participant Ray as Ray / verl + vLLM
  participant Srv as agl-server :8181
  participant Ctrl as agl-controller
  participant Agent as Calc-X agent\nAutoGen + MCP
  participant GW as API Gateway 代理
  participant Tr as Trainer 更新环

  User->>Run: 启动本地训练
  Run->>Ray: 启动推理后端
  Run->>Srv: 启动 Gateway
  Run->>Ctrl: runner_type=local
  Ctrl->>Agent: 调度 rollout 任务
  loop 每个 rollout
    Agent->>GW: OpenAI 兼容 chat/completions
    GW->>Ray: 转发 vLLM 推理
    Ray-->>GW: token 响应
    GW-->>Agent: 返回 + 记录轨迹
    Agent->>Agent: 工具 / MCP 调用
  end
  GW->>Tr: 轨迹聚合
  Tr->>Ray: GRPO / 策略更新
```

图下说明：agent 侧仅改 **模型 base URL** 指向 Gateway；**MCP 计算器、AutoGen 控制流** 等保持原 harness。完整 SWE / K8s 路径见官方 [Controller Configuration](https://microsoft.github.io/agent-lightning/stable/30-controller-configuration/)。

## 工程实践

| 步骤 | 要点 |
|------|------|
| **环境** | Python 3.12+；`uv sync` + `scripts/setup_verl.sh`（CUDA / verl 版本以文档为准） |
| **最小验证** | 下载 Calc-X 数据 → `examples/calc_x/run_local.sh`；日志默认 `/tmp/` |
| **生产 rollout** | `k8s_reconciler.py` 将 agent 跑为 **Kubernetes Job**，无需外部 sandbox 服务 |
| **Coding Agent** | `examples/swe_smith` + 公开数据清洗与 anti–reward-hacking 脚本；Qwen3.5-9B SWE-bench Verified **+14.6 pp** |
| **观测** | Quick Start 示例可接 W&B；Gateway / Controller 配置见文档 20–30 章 |

v1.0 文档的关键入口： [Basics](https://microsoft.github.io/agent-lightning/stable/05-basics/)（组件与 rollout）、[Trainer Configuration](https://microsoft.github.io/agent-lightning/stable/20-trainer-configuration/)（样本聚合与优化配置）、[Asynchronous Training](https://microsoft.github.io/agent-lightning/stable/35-asynchronous-training/)（共置异步）、[Coding Agent](https://microsoft.github.io/agent-lightning/stable/75-example-coding-agent/)（SWE-smith 端到端训练）。

## 局限与风险

- **GPU 依赖重：** 轻量的是 **编排代码**，策略推理与 GRPO 仍依赖 **verl + vLLM + GPU**；非「CPU 即可训 agent」。
- **Windows 本地限制：** `runner_type=local` **不支持原生 Windows**；需 WSL / Linux / K8s。
- **v1 破坏性迁移：** v0.x 与 v1.0 架构不兼容；读旧帖 / 社区项目时需核对分支（如 `contrib/youtu-agent-lightning`）。
- **与具身 RL 的距离：** 官方示例集中在 **搜索、沙箱代码、SWE、文本环境**；接入真机 / 仿真需自建 harness 与奖励，框架提供 **轨迹采集与训练环**，不替代机器人环境栈。
- **Retokenization：** 项目强调 OpenAI 兼容 API 返回 **token id** 的重要性（见 vLLM 合作文）；错误代理层可能导致 RL 信号漂移。

## 关联页面

- [DeepSeek Harness](./deepseek-harness.md) — 通用 **agent OS**；可经 Gateway 代理接 RL 训练环
- [Hermes Agent](./hermes-agent.md) — 常驻 agent 运行时 + 轨迹导出；与「外置 RL Trainer」互补
- [RSI-Harness](paper-metarsi-v1.md) — **实验组织 Genome**；Agent Lightning 偏 **策略权重 RL**
- [Karpathy autoresearch](./karpathy-autoresearch.md) — 固定验证指标的 **代理 ablation 环**
- [AI Auto-Research](../concepts/ai-auto-research.md) — S3 实验自动化生命周期坐标
- [Reinforcement Learning](../methods/reinforcement-learning.md) — RL 方法栈；verl 生态交叉引用

## 参考来源

- [Agent Lightning 仓库源归档（本站）](../../sources/repos/agent_lightning.md)
- [Microsoft Research 项目页源归档（本站）](../../sources/sites/agent-lightning-microsoft-research.md)
- [Agent Lightning v1.0 发布说明源归档（本站）](../../sources/sites/agent-lightning-v1-0-release-blog.md)
- [v1.0 技术报告源归档（本站）](../../sources/papers/agent_lightning_v1_technical_report.md)
- [microsoft/agent-lightning（GitHub）](https://github.com/microsoft/agent-lightning)

## 推荐继续阅读

- [Agent Lightning v1.0 文档 — Quick Start](https://microsoft.github.io/agent-lightning/stable/01-quick-start/) — 单卡 A100 Calc-X 端到端
- [Basics](https://microsoft.github.io/agent-lightning/stable/05-basics/) — Gateway / Controller / Trainer 与 rollout 状态
- [Trainer Configuration](https://microsoft.github.io/agent-lightning/stable/20-trainer-configuration/) — 数据处理、样本合并和 rollout 级优化
- [Asynchronous Training](https://microsoft.github.io/agent-lightning/stable/35-asynchronous-training/) — 共置异步批次、carry-over 与 staleness correction
- [Coding Agent 示例](https://microsoft.github.io/agent-lightning/stable/75-example-coding-agent/) — SWE-smith 编码 agent 训练
- [Agent Lightning v1.0: Towards Harnessed Agentic RL（arXiv:2608.17528）](https://arxiv.org/abs/2608.17528) — 架构与 benchmark 细节
- [No More Retokenization Drift（vLLM 博客）](https://blog.vllm.ai/2025/10/22/agent-lightning.html) — Gateway 返回 token id 的工程动机
