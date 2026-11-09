---
type: entity
project_id: selfimprove-bench
arxiv: "2611.02797"
project: https://theseus-labs-rsi.github.io/SelfImprove-Bench/
code: https://github.com/theseus-labs-rsi/SelfImprove-Bench
tags: [benchmark, rsi, llm-agents, evaluation]
status: draft
updated: 2026-11-09
summary: "测量智能体利用反馈修改自身后，能力能否迁移到未见任务，而非只记住训练关卡。"
related:
  - ../concepts/ai-agent-evaluation.md
  - ../entities/rle-bench.md
  - paper-deliverbench.md
sources:
  - ../../sources/papers/selfimprove-bench_arxiv_2611_02797.md
  - ../../sources/sites/selfimprove-bench.md
  - ../../sources/repos/selfimprove-bench.md
original:
  - ../../sources/papers/selfimprove-bench_arxiv_2611_02797.md
---

# SelfImprove-Bench: Evaluating Generalization in Self-Improving Agents｜自我改进智能体的泛化评测

## 一句话定义

SelfImprove-Bench 是面向测量智能体利用反馈修改自身后，能力能否迁移到未见任务，而非只记住训练关卡。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|---|---|---|
| AI | Artificial Intelligence | 生成/执行被评内容的人工智能 |
| LLM | Large Language Model | 基准中的语言智能体或评分模型 |
| API | Application Programming Interface | 运行模型或裁判的程序接口 |
| RSI | Recursive Self-Improvement | 通过经验反馈修改自身的方法 |
| HCI | Headroom-Closure Index | 对比初始与最终性能的改善指标 |

## 为什么重要

- **五级泛化阶梯**：G0 同一实例重放；G1 同难度新实例；G2 同任务的新难度；G3 同一能力类别的其他任务；G4 跨能力类别任务。难度随层级增加，但不同游戏分数不可无条件横比。

对 [AI Agent 评测](../concepts/ai-agent-evaluation.md) 的价值是将笼统的“成功”拆为可审计且有复现边界的评价对象，而不把模型声明成功当作真实结果。

## 流程总览

```mermaid
flowchart TD
  A["配置难度、游戏和训练日程"] --> B["生成训练任务与保留集"]
  B --> C["Agent 运行与模拟器反馈"]
  C --> D["RSI 方法修订经验、harness 或方法"]
  D --> E["保存版本并可回放旧任务"]
  E --> C
  E --> F["冻结改进后状态"]
  F --> G["G0-G4 无进一步修订的评测"]
  G --> H["与初始基线比较并分析 HCI"]
```

## 核心原理

任务生成器生成 train/replay/holdout 池；同一 RSI 方法连续执行任务并从反馈中优化自身。保留改进快照后，在不同 G0–G4 的未见环境中冻结改进进行评测，比较初始与最终能力。

- **五级泛化阶梯**：G0 同一实例重放；G1 同难度新实例；G2 同任务的新难度；G3 同一能力类别的其他任务；G4 跨能力类别任务。难度随层级增加，但不同游戏分数不可无条件横比。
- 覆盖 **6 个游戏**：Farm、Combat、Sweepmine、Hero's Journey、Minecraft、Astray，组织在探索、规划和推理等能力类别下。
- 对比 **ACE**（经验层）、**AHE**（harness 层）与 **HyperAgents**（元层）等 RSI 方法，也提供 Codex、Claude Code 等固定系统基线。
- 训练阶段采用任务序列和中途反馈；holdout 使用独立任务且**不继续自我修改**，防止将测试学习当作泛化。
- 通过同配置初始框架与改进后框架的对照、HCI 与轨迹归因分析，检查改进是不是稳定和可迁移。

## 源码运行时序图

基于已核对的公开 README 和下列模块入口静态整理；未执行端到端实验。图描述已检查的调用职责，不是对每条内部调用的动态跟踪。

```mermaid
sequenceDiagram
  autonumber
  participant U as "python -m benchmark"
  participant M as "benchmark.__main__"
  participant R as "benchmark.runner"
  participant C as "runner.AgentController"
  participant G as "benchmark.game_adapters"
  participant S as "rsi_session"
  U->>M: 读取 JSON 配置
  M->>R: run_benchmark(config)
  R->>C: begin(训练任务)
  C->>G: 通过 request 执行 evaluate_task
  G-->>C: 动作回执和模拟器分数
  C->>S: refine(反馈与候选)
  S-->>C: 已选择的候选版本
  R->>C: begin(holdout / 冻结状态)
  C->>G: evaluate_task(不修订)
  G-->>R: holdout 指标与轨迹
```

对应源码位置参见 [官方仓库归档](../../sources/repos/selfimprove-bench.md)。

## 评测与结果

- G0–G4 五层分别检验重放、新实例、难度迁移、任务迁移和跨能力迁移。
- 公开 README 提供 18 套六游戏×三 RSI 方法的主要实验配置；训练、holdout 与初始框架比较需要保持种子、任务范围和模型后端一致。
- 没有在本次环境跑实测任务，因此不将论文结果冒充本机复现。

## 结论

**任务生成器生成 train/replay/holdout 池，评价边界与复现设置同样重要。**

1. 至少报告 G1/G2/G3/G4，单看 G0 重放容易高估方法泛化。
2. holdout 必须禁止继续 refine，并与训练池隔离；对比初始与改进后的相同任务。
3. 区分经验、harness 与元级修改，追踪修改作用范围及计算成本。
4. GameWorld 与第三方算法许可证各异，商业复现不能只依据主仓 MIT。

## 与其他工作对比

DeliverBench 关心项目最终多文件交付质量，SelfImprove-Bench 关心**反馈后自修改能力的跨任务迁移**；AmbiHTMLBench 关心视觉到 HTML 的重建与裁判一致性。

共同评测原则见 [AI Agent 评测](../concepts/ai-agent-evaluation.md)；机器人工作流的代码与实验验收可参照 [RLE-Bench](rle-bench.md)，但不能直接移植这三项榜单的数字。

## 工程实践

1. 参照主仓 README 使用 Python 3.13.12 建环境；AHE 还需要 Linux bubblewrap，GameWorld 需 Playwright，HyperAgents 需 Docker。
2. 训练 ACE Farm 示例：`python -m benchmark configs/main/farm_ace.json`。
3. 用同一配置运行 `python -m benchmark configs/main/farm_ace.json --initial`，固定 holdout 种子比较初始/改进状态。
4. 使用 `python -m benchmark.hci --trained outputs/main/farm/ace --initial outputs/main/farm/ace_initial --output analysis/farm_ace_hci.json`，并按游戏/难度报告而非仅给全局平均值。

## 局限与风险

- 这是一套以游戏环境为主的 RSI 评测，不能直接推出机器人真机泛化；HyperAgents 内嵌代码有非商业许可，主仓 MIT 不覆盖全部三方依赖。
- 论文全文及附录尚未逐节完成核对；**本节点保留 draft**，健康度不应写为满分。
- 本次仅静态阅读公开仓库和入口，尚未安装完整环境或独立复现实验成绩。

## 关联页面

- [AI Agent 评测](../concepts/ai-agent-evaluation.md) — agent 的评测任务、grader、隔离与判据原则。
- [RLE-Bench](rle-bench.md) — 与机器人学习工程交付相关的 Agent 评测。
- [DeliverBench](paper-deliverbench.md) — 区分被评对象与评价指标。
- [AmbiHTMLBench](paper-ambihtmlbench.md) — 区分被评对象与评价指标。

## 参考来源

- [论文一手归档](../../sources/papers/selfimprove-bench_arxiv_2611_02797.md)
- [官方项目页归档](../../sources/sites/selfimprove-bench.md)
- [官方源码归档](../../sources/repos/selfimprove-bench.md)

## 推荐继续阅读

- [作者项目主页](https://theseus-labs-rsi.github.io/SelfImprove-Bench/)
- [原始论文](https://arxiv.org/abs/2611.02797)
- [官方 GitHub 源码](https://github.com/theseus-labs-rsi/SelfImprove-Bench)
