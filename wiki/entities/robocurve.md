---
type: entity
tags: [company, evaluation, benchmark, physical-ai, open-source, robocurve, y-combinator, public-benefit-corporation]
title: Robocurve
status: complete
updated: 2026-09-06
related:
  - ./inspect-robots.md
  - ../overview/hub-embodied-eval-benchmark.md
  - ../concepts/sim-vs-real-eval-gap.md
  - ../concepts/simulation-evaluation-infrastructure.md
  - ./xpolicylab.md
  - ./isaac-lab-arena.md
  - ./lerobot.md
sources:
  - ../../sources/sites/robocurve-org.md
  - ../../sources/repos/robocurve_inspect_robots.md
summary: "Robocurve 是 YC 支持的 Public Benefit Corporation，独立测量并向公众报告 physical AI 真机能力；旗舰开源工具 Inspect Robots（MIT）提供可审计评测与 Rerun 可视化。"
---

# Robocurve

| 字段 | 内容 |
|------|------|
| **机构** | Robocurve（Public Benefit Corporation） |
| **类型** | 独立 physical AI / 机器人能力评测机构 |
| **公开锚点** | [robocurve.org](https://robocurve.org/) |
| **旗舰开源** | [Inspect Robots](./inspect-robots.md)（MIT） |
| **背书** | Y Combinator（站点 2026-09-06） |
| **开源** | **框架已开源**；被测模型权重/API 由第三方提供 |

## 一句话定义

**Robocurve**：以公益公司形态运作的 **独立 robotics 能力计量机构**——在 cherry-picked demo 泛滥的背景下，用 **真机优先、可复现日志** 的评测向公众报告 frontier 进展，并开源 [Inspect Robots](./inspect-robots.md) 降低社区复现门槛。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| PBC | Public Benefit Corporation | 美国公益公司形态；依法兼顾股东与公共利益 |
| VLA | Vision-Language-Action | 站点评测对象之一（经 Inspect Robots 接入） |
| WAM | World-Action Model | 世界–动作模型；与 VLA 并列的可测策略族 |
| CaP | Code-as-Policy | 代码即策略 agent（Cap-X 插件） |
| API | Application Programming Interface | 报告中的 test-time scaling 以 API cost 为横轴 |

## 为什么重要

- **填补「无人知道 frontier 在哪」的空洞：** 站点主张互联网充斥精选 demo，缺少 **连续、标准化** 的 robotics 评测；Robocurve 自定位为 **agenda-neutral** 的独立测量者。
- **公益治理信号：** PBC 结构把「服务公众理解 physical AI 风险与收益」写入组织约束，区别于纯商业 demo 营销。
- **开源工具 + 公开报告双轨：** 不只发图表——[Inspect Robots](./inspect-robots.md) 让第三方能在 **同一套 EvalLog / Rerun** 口径下复跑或扩展 benchmark。
- **与 METR 叙事对齐：** 首页引用 METR「task-completion time horizon」曲线，强调 **任务长度 @ 50% 成功率** 随模型迭代拉长——Robocurve 把同类 **能力计量** 思维带到 **物理执行** 层。

## 公开能力报告（站点 2026-09-06）

| 主题 | 观测 |
|------|------|
| **Test-time scaling** | LLM **thinking effort / API cost** 与 robotics score 正相关（Opus / Sol 等曲线） |
| **Inference speed** | 多模型推理速度随时间指数改善（Haiku → Fable 5 等） |
| **长程任务** | Tower of Hanoi、block stacking、做 toast 等案例 |
| **Voice-guided** | 语音引导指令跟随 |

报告细节以站点原文为准；复现应走 Inspect Robots 日志 + 报告附录，而非只看 GIF。

## 工程实践

| 场景 | 建议 |
|------|------|
| **引用 frontier 能力** | 优先链 Robocurve 报告 + EvalLog，而非单独 demo 视频 |
| **自建 benchmark** | 用 [Inspect Robots](./inspect-robots.md) 定义 `Task`/`Scene`/`Scorer`；任务库见 [WorldEvals](https://github.com/robocurve/worldevals) |
| **对接现有策略栈** | XPolicyLab / OpenPI / LeRobot 等经 embodiment 插件接入；见 inspect-robots 实体 |
| **仿真 vs 真机** | 框架 **真机优先**；仿真通过 Isaac Lab 等 **opt-in capability**，勿把 sim 分数当真机结论 |

## 局限与风险

- **alpha 框架：** Inspect Robots 仍标 early development；API 可能变，生产依赖需 pin 版本。
- **权重不在 org 内：** 开源的是 **评测管线**，不是 Claude / π0 等被测权重。
- **独立性的 operational 定义：** testimonial 来自多家 lab 专家，但 **资金与评测协议** 仍需读者自行核对。
- **报告样本：** 早期报告可能集中在少数 rig / 任务；外推至全产业需谨慎。

## 关联页面

- [Inspect Robots（评测框架）](./inspect-robots.md)
- [具身评测基准选型闭环](../overview/hub-embodied-eval-benchmark.md)
- [XPolicyLab](./xpolicylab.md) — `--policy xpolicylab` 插件
- [Isaac Lab-Arena](./isaac-lab-arena.md) — `--embodiment isaacsim` 仿真路径
- [仿真 vs 真机评测 gap](../concepts/sim-vs-real-eval-gap.md)

## 参考来源

- [Robocurve 站点归档](../../sources/sites/robocurve-org.md)
- [inspect-robots 源码归档](../../sources/repos/robocurve_inspect_robots.md)

## 推荐继续阅读

- 公司首页：<https://robocurve.org/>
- Inspect Robots 文档：<https://docs.inspectrobots.org/>
- GitHub：<https://github.com/robocurve/inspect-robots>
