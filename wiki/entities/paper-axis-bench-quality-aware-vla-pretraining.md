---
type: entity
project_id: axis-bench-quality-aware-vla
tags: [paper, benchmark, vla, imitation-learning, data-quality]
status: complete
updated: 2026-10-14
arxiv: "2610.18869"
project: https://axis-bench.github.io/
summary: "AXIS-Bench 在固定训练预算下比较 VLA 示范质量的加权、过滤与条件化；发现标签粒度的平均偏好随方法而变，代码和数据目前仍待发布。"
related:
  - ../methods/vla.md
  - ../methods/imitation-learning.md
  - ../tasks/manipulation.md
  - ../overview/hub-embodied-eval-benchmark.md
  - ./paper-robodrop-vla-post-training.md
  - ./paper-splat2act-personalized-vla.md
sources:
  - ../../sources/papers/axis_bench_arxiv_2610_18869.md
  - ../../sources/sites/axis-bench-project.md
---

# AXIS-Bench：质量感知 VLA 预训练实证研究

**AXIS-Bench** 是一套研究 VLA 预训练中示范质量如何使用的基准与受控比较：它对齐数据量和训练预算，比较加权、过滤和条件化，并用可回放轨迹支持重新计算质量标签。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLA | Vision-Language-Action | 基准比较的机器人视觉语言动作策略 |
| BC | Behavioral Cloning | 用示范动作监督策略的基线训练方式 |
| DROID | Distributed Robot Interaction Dataset | 项目页真实验证 recipe 兼容的机器人数据/工作空间生态 |
| SR | Success Rate | 目标任务成功率，须结合任务和评测设置阅读 |

## 为什么重要

机器人示范往往不是全成功的。过滤低质量片段可能丢失有价值的状态覆盖；直接模仿又可能让策略学到错误动作。研究质量标签究竟应该改变样本权重、过滤数据还是作为条件输入，需要控制数据规模和训练预算，否则难以区分策略收益与资源差异。

## 方法

AXIS-Bench 使用 5,000 条自然次优人类示范和 24 个目标任务，在三个 VLA 上进行预算匹配的比较：

| 处理方式 | 训练中的作用 | 论文观察 |
|----------|--------------|----------|
| 加权（weighting） | 按质量标签改变示范损失贡献 | 效果依赖奖励定义、标签粒度和模型 |
| 过滤（filtering） | 依据质量阈值选择训练轨迹 | 平均更偏好 chunk 级质量标签 |
| 条件化（conditioning） | 将质量标签作为策略条件 | approach-aware episode 条件化平均优于纯 BC；平均更偏好 episode 级标签 |

基准基于众包仿真示范，记录物体和接触状态，允许对共享轨迹重新计算质量标签；轨迹回放支持工作区对齐渲染以及用户定义真实目标任务。

## 实验与评测

- **对比规模：** 5,000 条自然次优示范、24 个目标任务、三个 VLA；每个 VLA 内匹配示范数量与训练预算。
- **平均趋势：** approach-aware episode conditioning 优于 plain BC；conditioning 平均更适合 episode 标签，filtering 平均更适合 chunk 标签。
- **微调：** 条件预训练后以固定标签微调可提高平均成功率；推理端是否输入质量标签都报告了收益。
- **真实验证：** 仿真预训练将真实任务成功率从 9.6% 提高到 31.7%。
- **反转现象：** 标签粒度偏好存在模型相关反转，不能把总体均值直接视为跨模型默认超参。

项目页截至 2026-10-14 将代码与数据集标为 Coming Soon，因此当前无法从仓库复跑受控对比。

## 结论

**质量标签的价值取决于它如何进入训练，以及模型看到的标签粒度；没有一种质量处理方式对所有 VLA 都占优。**

1. **先固定数据预算再比方法。** 不匹配样本量或训练步数会把算力和数据规模误当成质量建模收益。
2. **conditioning 和 filtering 的粒度选择不同。** 平均规律分别偏向 episode 与 chunk 标签，但需要在目标模型上复核。
3. **条件化可改善预训练到微调的衔接。** 固定质量标签微调收益提示质量条件不应只在预训练阶段使用。
4. **保留次优数据可能比全过滤更有信息。** 与权重/条件化对照，避免把“质量差”等同于“无用”。
5. **等待数据与代码发布后再做复现结论。** 当前结果来自论文报告，无法独立核验完整基准实现。

## 与其他工作对比

| 对照 | 差异 |
|------|------|
| [RoboDrop VLA 后训练筛选](./paper-robodrop-vla-post-training.md) | RoboDrop 从梯度兼容性筛选后训练轨迹；AXIS-Bench系统比较示范质量的加权、过滤和条件化，并研究标签粒度 |
| [Splat2Act](./paper-splat2act-personalized-vla.md) | Splat2Act 自动生成目标场景成功演示用于个性化；AXIS-Bench研究混合质量演示应如何进入预训练 |
| [VLA 方法](../methods/vla.md) | AXIS-Bench提供数据质量维度的训练选择证据，不是新的 VLA 主干架构 |

## 工程实践

- **建立可重放记录：** 保存观测、动作、物体与接触状态，才能在统一轨迹上重算质量标签。
- **记录粒度：** 同时保留 episode 与动作 chunk 标签，按具体策略的 loss 入口分别测试。
- **公平对照：** 固定模型、轨迹数、数据增强和训练步数；单独报告预训练、微调与推理是否使用标签。
- **开放状态：** [AXIS-Bench 项目页](../../sources/sites/axis-bench-project.md)仍标代码和数据 Coming Soon，没有可画源码运行时序图的仓库入口。

## 局限与风险

- 论文报告总体均值及明确的模型相关反转，不能将平均粒度偏好直接变成部署策略。
- 质量标签依赖奖励定义和标注粒度；若标签与最终目标错位，条件化可能教会模型追逐错误信号。
- 公开代码、数据尚未发布，当前无法独立审计基准切分、标签计算和真实任务设置。

## 关联页面

- [VLA 方法](../methods/vla.md)
- [模仿学习](../methods/imitation-learning.md)
- [具身评测基准选型闭环](../overview/hub-embodied-eval-benchmark.md)
- [Manipulation](../tasks/manipulation.md)
- [RoboDrop](./paper-robodrop-vla-post-training.md)

## 参考来源

- [AXIS-Bench 论文摘录](../../sources/papers/axis_bench_arxiv_2610_18869.md)
- [AXIS-Bench 项目页核查](../../sources/sites/axis-bench-project.md)
- [arXiv:2610.18869](https://arxiv.org/abs/2610.18869)

## 推荐继续阅读

- [AXIS-Bench 项目页](https://axis-bench.github.io/)
- [arXiv:2610.18869](https://arxiv.org/abs/2610.18869)
