---
type: query
tags: [manipulation, il, rl, data-collection, bc, dagger]
status: complete
summary: "面向机器人操作任务的选型指南：何时优先模仿学习、何时需要 RL，以及演示数据该如何收集和扩展。"
related:
  - ../tasks/manipulation.md
  - ../methods/imitation-learning.md
  - ../methods/behavior-cloning.md
  - ../methods/dagger.md
  - ../methods/reinforcement-learning.md
sources:
  - ../../sources/papers/imitation_learning.md
  - ../../sources/papers/diffusion_and_gen.md
  - ../../sources/papers/survey_papers.md
---

> **Query 产物**：本页由以下问题触发：「做机器人操作用模仿学习还是 RL？怎么收集数据？」
> 综合来源：[Manipulation](../tasks/manipulation.md)、[Imitation Learning](../methods/imitation-learning.md)、[Behavior Cloning](../methods/behavior-cloning.md)、[DAgger](../methods/dagger.md)、[Reinforcement Learning](../methods/reinforcement-learning.md)

# Query：机器人操作任务该优先用 IL 还是 RL？

## TL;DR 决策树

```text
你能稳定拿到专家演示吗？
├── 能
│   ├── 任务主要是桌面操作 / 工具使用 / 接触技能复现
│   │   └→ 先 IL（BC / Diffusion / ACT）
│   └── 演示能覆盖失败恢复吗？
│       ├── 能 → IL 可直接成为主路线
│       └── 不能 → IL 初始化 + RL / DAgger 补恢复能力
└── 不能
    ├── Reward 易定义，仿真可信，失败成本低
    │   └→ RL 或 RL + curriculum
    └── Reward 难定义
        └→ 先做 teleop / scripted demo 收集，再回到 IL 路线
```

## 快速结论

- **操作任务大多数时候先做 IL，而不是从零 RL。**
- **纯 BC 足够做 baseline，但长时序和接触任务容易 compounding error。**
- **如果演示里缺少恢复动作，优先考虑 DAgger、回放补标或 IL+RL 微调。**
- **RL 更适合在仿真中补“恢复、鲁棒性、超越专家”的那部分能力。**

## 选型对比表

| 路线 | 什么时候用 | 优点 | 风险 |
|------|------------|------|------|
| BC / ACT | 已有高质量 teleop 演示；任务相对稳定 | 快、简单、真机友好 | 分布漂移、长时序误差 |
| DAgger | 策略会偏离演示分布；专家可在线纠偏 | 比纯 BC 更能处理 covariate shift | 真机标注和安全成本高 |
| Diffusion / Flow | 多模态、长时序、精细接触 | 动作质量高、对操作更强 | 训练和推理更重 |
| RL | 奖励好写、仿真可信、需探索恢复动作 | 可超越专家、可学鲁棒性 | 样本效率低、sim2real 难 |
| IL + RL | 既想快起步，又想补恢复能力 | 最实用的折中 | 工程链路更复杂 |

## 数据该怎么收集

### 1. 先定义数据粒度

先想清楚你要收的是：
- **单步动作监督**：适合简单 BC
- **action chunk / 子技能序列**：适合长时序操作
- **恢复动作与失败样本**：决定策略能否真机闭环

### 2. 优先保证覆盖而不是“演示漂亮”

最有价值的数据通常不是完美轨迹，而是：
- 对齐误差后的纠正动作
- 接触建立时的细微调整
- 快失败时如何 recover

### 3. 常见来源

- 6D 鼠标 / VR / 力反馈遥操作
- kinesthetic teaching（可拖动机械臂）
- scripted expert / MPC / 人类教师混合标注
- 真实 demo + 仿真扩增

## 推荐 pipeline

1. **先做 20~100 条高质量真实演示**，跑通 BC baseline
2. 如果任务是长 horizon 或多模态，升级到 **ACT / Diffusion / Flow**
3. 发现策略在闭环里会偏离，就补 **DAgger / offline relabel / hard-negative demo**
4. 若还需要恢复能力或超越专家，再做 **RL fine-tune**

## 什么时候该考虑 RL

下面这些信号出现时，RL 价值会快速上升：
- 演示很难覆盖所有环境扰动
- 任务存在明确成功信号，但过程难示教
- 需要学会“撞到之后怎么办”“快滑落时怎么救”
- 你已经有可用 IL policy，想进一步提高成功率和鲁棒性

## 常见坑

- **只收成功 demo，不收恢复和失败边缘状态** → 真实部署脆弱
- **动作和观测不同步** → 再大的模型也学不稳
- **过度依赖仿真数据** → 接触和相机分布一换就崩
- **过早上纯 RL** → 调 reward 花的时间比收 demo 还多

## 一句话记忆

> 做机器人操作，默认先 IL；当你发现“会做，但不会救、不会泛化、不会恢复”时，再引入 DAgger 或 RL。

## 参考来源

- [sources/papers/imitation_learning.md](../../sources/papers/imitation_learning.md) — BC / DAgger / ACT / Diffusion 的来源整理
- [sources/papers/diffusion_and_gen.md](../../sources/papers/diffusion_and_gen.md) — 生成式 IL 在操作上的升级路线
- [sources/papers/survey_papers.md](../../sources/papers/survey_papers.md) — 机器人学习大图景与方法选型补充

## 关联页面

- [Manipulation](../tasks/manipulation.md)
- [Imitation Learning](../methods/imitation-learning.md)
- [Behavior Cloning](../methods/behavior-cloning.md)
- [DAgger](../methods/dagger.md)
- [Reinforcement Learning](../methods/reinforcement-learning.md)
- [Contact-Rich Manipulation](../concepts/contact-rich-manipulation.md) — 模仿学习在接触丰富型操作任务中的关键应用场景
