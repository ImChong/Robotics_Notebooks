---
type: entity
project_id: agentrobocritic
arxiv: "2611.26613"
project: https://adu2021.github.io/blog/AgentRoboCritic/
code: https://github.com/cxcscmu/AgentRoboCritic
tags: [paper, robot-evaluation, manipulation, failure-diagnosis, cmu, uiuc, bosch]
status: complete
updated: 2026-11-24
related:
  - ./robocasa.md
  - ../overview/hub-embodied-eval-benchmark.md
  - ../../roadmap/depth-embodied-eval.md
  - ../methods/agentic-manipulation-policies.md
  - ../tasks/manipulation.md
sources:
  - ../../sources/papers/agentrobocritic_arxiv_2611_26613.md
  - ../../sources/sites/agentrobocritic.md
  - ../../sources/repos/agentrobocritic.md
original:
  - ../../sources/papers/agentrobocritic_arxiv_2611_26613.md
summary: "在直接数值动作控制中加入局部意图，用动作执行前后证据诊断失败机制，补足回合级成功率盲区。"
---

# AgentRoboCritic：面向机器人直接控制的动作级批评器（Understanding the Capability of Frontier Agents on Direct Robot Control）

## 一句话定义

在直接数值动作控制中加入局部意图，用动作执行前后证据诊断失败机制，补足回合级成功率盲区。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|---|---|---|
| VLM | Vision-Language Model | 基于视觉证据的评判模型 |
| EFE | Embodied Frontier Evaluation | 意图与动作证据采集环境 |
| SR | Success Rate | 回合成功率 |

## 为什么重要

从回合成功率深入局部意图、动作失败和恢复诊断。将其作为可比较的独立研究或工程入口，而不是仅记录热度。

## 核心信息

| 字段 | 内容 |
|---|---|
| 机构 | 卡内基梅隆大学、伊利诺伊大学厄巴纳-香槟分校、博世研究院 / 博世人工智能中心 |
| 论文 | [arXiv:2611.26613](https://arxiv.org/abs/2611.26613) |
| 日期 | 2026-11-19 |
| 开放状态 | MIT；公开评测 harness、EFE 数据采集转换、训练数据与微调 checkpoint；不能据此推断各闭源 actor 权重开放。 |

## 核心原理与流程总览

actor 在提交 action chunk 时同时声明局部意图；评判器接收意图、命令和执行证据，输出意图是否成功、失败机制、抓取子类及原因。将动作级标签汇总成各失败类型与恢复统计。评测 Parquet 包含 id、messages、images、human_label；推理不读取人工标签。

### 方法与训练

- 直接控制指令是末端位姿、关节配置等数值目标，局部意图与动作在同一工具调用内提交。评判器读取全局任务、局部意图、动作前后观察和执行反馈，不读取最终回合结果；区分实现、失败、部分或未确认、无法核实及不适用。七类失败为抓取、对齐、干涉、任务效果、未完成运动、机制未决和错误局部选择；Choice 是相对局部意图，而非全局规划错误。
- 训练集 12844 条动作，400 条人工评测动作来自训练外的回合。评分中失败机制分母为 264 个真实失败动作；联合指标要求意图与机制同时匹配，不把解释和抓取子类纳入主分数。

以下为根据原文整理的概念流程。

```mermaid
flowchart TD
    A["任务与观察"]
    B["Actor 输出局部意图和动作"]
    C["环境执行与视觉证据"]
    D["Critic 诊断"]
    E["失败类型和恢复分析"]
    F["人工标注基准核验"]
    A --> B
    B --> C
    C --> D
    D --> E
    E --> F
```

## 源码运行时序图

`src/agentrobocritic/cli.py` 及 README 的 `arc verify`、`arc evaluate`、`arc score`；已核对 CLI、runner.run 与 scoring.score 的评测链。

```mermaid
sequenceDiagram
    autonumber
    participant U as 用户
    participant C as agentrobocritic.cli.main
    participant D as data.rows
    participant R as runner.run
    participant S as scoring.score
    U->>C: arc evaluate --profile muse --out runs/muse
    C->>D: 读取指定数据与 revision
    C->>R: 执行评判请求
    R->>D: rows / unpack
    R->>R: provider request / HTTP / SQLite 缓存
    R-->>U: predictions.jsonl
    U->>C: arc score --predictions runs/muse/predictions.jsonl
    C->>S: 对齐人工标签及预测 ID
    S->>S: 校验标签 / outcome / failure_mode / overall
    S-->>U: metrics.json
```

## 工程实践

MIT；公开评测 harness、EFE 数据采集转换、训练数据与微调 checkpoint；不能据此推断各闭源 actor 权重开放。 复现入口与版本依赖见参考来源。本次未执行训练、推理或仿真复现。

## 实验与评测

作者在 12 个操作基准分析动作，约半数成功回合仍出现失败动作。400 条人工标注动作中有 264 个失败；微调模型失败模式准确率由 32.58%→74.24%（196/264），意图成功判断 76.25%（305/400），联合正确率 67.75%（271/400）。训练数据 12,844 条标注动作。

### 对照、消融与边界

- 12 个任务套件中，49.8% 成功回合仍包含失败动作；1989 回合共 12163 个失败动作，10 动作以内恢复 38.9%，回合结束前恢复 57.7%。RoboDojo 占 9923 个失败动作，因此不能将样本加权统计误作套件均值。对齐恢复率 35.7%，低于抓取 63.7%。
- 400 条人工标注中 Astra 意图/机制/联合准确率 79.25%/80.68%/71.25%；去除局部意图后为 25.25%/25%/24.5%。微调 Muse 的对应值为 76.25%/74.24%/67.75%；400 条评测成本约 0.79 美元，对照 Astra 26.06 美元。附录 H：74 个匹配任务的决策延迟中位数 31.9 s 对 π0.5 的 0.11 s，直接控制尚不适合等同高频闭环。

## 与其他工作对比

- **对比回合级成功率（SR）评测：** 传统操作基准（如 [RoboCasa](./robocasa.md) 等回合级 SR 评测）只记录最终成败，会掩盖成功回合中约 49.8% 仍含失败动作以及对齐恢复率仅 35.7%（低于抓取 63.7%）等细粒度缺陷；AgentRoboCritic 将诊断粒度下沉到单次数值动作前后的局部意图与七类失败机制。
- **对比闭源前沿裁判与无意图基线：** 400 条人工标注动作上，微调开源模型 Muse 联合准确率 67.75%（失败机制 74.24%）接近闭源 Astra（71.25% / 80.68%），而单次评测成本由 26.06 美元降至约 0.79 美元；若去掉 actor 声明的局部意图，Astra 联合准确率会由 71.25% 骤降至 24.5%，说明局部意图声明本身是可诊断性的前提。
- **对比高频端到端 VLA 策略：** 74 个匹配任务中前沿 agent 直接控制的决策延迟中位数为 31.9 s，远高于 π0.5 的 0.11 s；因此该工作定位为[具身评测闭环](../overview/hub-embodied-eval-benchmark.md)的细粒度诊断层，而非高频低层闭环控制器的直接替代。

## 结论

在直接数值动作控制中加入局部意图，用动作执行前后证据诊断失败机制，补足回合级成功率盲区。 工程选型时应同时验证指标口径及使用边界。

- 同时报告回合 SR 和动作失败/恢复统计。
- 区分 400 条意图标签与 264 条失败模式标签的分母。
- 优先从高频、低恢复率的失败模式着手改策略。

## 局限与风险

critic 的标签误差、模型自偏好与局部意图的真实性会影响结论；动作级诊断并不自动修复策略或保证安全。抓取子类和解释不计入仓库所述主评分。

**核对状态：** 已完成所列原文核对（论文正文与附录、批评器 codebook；CLI、runner.py 与 scoring.py 评测调用链。）；未执行训练或实机复现。

## 关联页面

- [robocasa](./robocasa.md)
- [hub-embodied-eval-benchmark](../overview/hub-embodied-eval-benchmark.md)
- [相关纵深路线](../../roadmap/depth-embodied-eval.md)

- [智能体操作策略](../methods/agentic-manipulation-policies.md) — 动作及局部意图的生成端。
- [机器人操作任务](../tasks/manipulation.md) — 按任务阶段统计失败与恢复。

## 参考来源

- [papers：agentrobocritic](../../sources/papers/agentrobocritic_arxiv_2611_26613.md)
- [sites：agentrobocritic](../../sources/sites/agentrobocritic.md)
- [repos：agentrobocritic](../../sources/repos/agentrobocritic.md)

## 推荐继续阅读

- [论文 HTML](https://arxiv.org/html/2611.26613)
- [arXiv](https://arxiv.org/abs/2611.26613)
- [官方项目页](https://adu2021.github.io/blog/AgentRoboCritic/)
- [官方源码](https://github.com/cxcscmu/AgentRoboCritic)
- [人工标注基准](https://huggingface.co/datasets/cx-cmu/AgentRoboCritic-Bench)
- [微调评判器](https://huggingface.co/cx-cmu/AgentRoboCritic-Muse-Glimmer-30B)
- [训练数据](https://huggingface.co/datasets/cx-cmu/AgentRoboCritic-Dataset)
