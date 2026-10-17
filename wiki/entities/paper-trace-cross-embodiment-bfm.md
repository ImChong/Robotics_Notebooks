---
type: entity
tags: [paper, humanoid, hkust-gz, hkust, behavior-foundation-model, cross-embodiment, whole-body-tracking, sonic, reinforcement-learning]
status: complete
updated: 2026-10-17
project_id: trace-cross-embodiment-humanoid-bfm
arxiv: "2610.17999"
project: https://bfm-trace.github.io/
related:
  - ../queries/cross-embodiment-transfer-strategy.md
  - ../concepts/whole-body-tracking-pipeline.md
  - ../methods/sonic-motion-tracking.md
  - ./paper-sonic-transfer.md
  - ./paper-any2any-cross-embodiment-wbt.md
sources:
  - ../../sources/papers/trace_arxiv_2610_17999.md
  - ../../sources/sites/trace-bfm-project.md
summary: "TRACE 将 Unitree G1 上的 SONIC 迁移到 H2，通过运动学与动力学预对齐、精简动作集、编码器蒸馏和低秩强化学习进行目标机适配；论文报告约 8 小时训练与 96.08% 跟踪成功率，官方代码开放状态待核实。"
---

# TRACE：跨具身人形控制迁移

TRACE（*Efficient Transfer Learning for Neural Whole-Body Control Across Humanoid Embodiments*）将 Unitree G1 上的 SONIC 全身控制器迁移到 Unitree H2，先对齐运动学和关节动力学，再以紧凑动作集上的表征蒸馏与低秩强化学习适配目标机。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|---|---|---|
| TRACE | Transfer learning ACross humanoid Embodiments | 本文的跨人形具身迁移框架 |
| BFM | Behavior Foundation Model | 预训练多种全身行为的控制模型 |
| WBT | Whole-Body Tracking | 控制器跟踪参考动作的任务 |
| LoRA | Low-Rank Adaptation | 以低秩参数更新适配控制策略 |
| VLA | Vision-Language-Action | 论文迁移演示数据后训练的高层策略 |

## 为什么重要

人形控制器换机身时，关节空间、连杆尺度、质量与执行器响应都会变化。只重排输入输出维度不能保证闭环控制稳定；从头为每个机型训练大模型又很昂贵。TRACE 给出一条中间路径：尽量保留预训练的动作表征，把微调预算集中在目标机所需的对齐和反馈控制适配上。

本方法的实际意义是把已有全身控制先验从 Unitree G1 迁移到全尺寸 Unitree H2，并将迁移后的控制器用于真机动作跟踪和全身遥操作数据采集。

## 核心原理

### 方法栈

| 阶段 | 机制 | 目的 |
|---|---|---|
| 运动学预对齐 | 向输入、输出投影层注入目标机运动学先验 | 缓和关节布局与身体尺度差异 |
| 指令与动力学对齐 | 对齐动作尺度和关节名义动态响应 | 降低初始闭环失配 |
| 动作集构造 | 从质量筛选动作中平衡类别，并扩大命令特征覆盖 | 用较小迁移集覆盖有代表性的参考动作 |
| 表征锚定 | 编码器蒸馏目标机表征 | 保留预训练运动潜空间 |
| 控制微调 | 对低秩参数进行强化学习更新 | 适应目标机器人动力学 |
| 两阶段课程 | 先稳定跟踪，再优化困难动作 | 避免一开始直接追逐高难参考导致不稳定 |

### 流程总览

```mermaid
flowchart LR
  A["G1 上的预训练 SONIC"] --> B["运动学与关节响应预对齐"]
  C["质量筛选与类别平衡动作集"] --> D["编码器蒸馏 + 低秩强化学习"]
  B --> D
  D --> E["Unitree H2 目标机控制器"]
  E --> F["全身动作跟踪与遥操作"]
  F --> G["下游 VLA 后训练演示"]
```

### 实验与评测

论文将 SONIC 从 Unitree G1 迁移到 Unitree H2，使用 1000 万动作帧，在单张 RTX PRO 6000 上训练约 8 小时。其报告的跟踪成功率为 **96.08%**，评测集包含 **67,953 段动作**，成功判据按身高缩放的跟踪标准定义。

算力比较需要按边界解读：论文以 SONIC 从头预训练报告的 21,000 GPU 小时作参照，称适配成本低于其 0.1%。该数值不是硬件归一化的速度比较，也不包含源 SONIC 模型训练成本。

## 与其他工作对比

| 工作 | 迁移方式 | 目标 | 区别 |
|---|---|---|---|
| TRACE | 运动学与关节响应预对齐、编码器蒸馏、低秩 RL | G1 → H2 | 约 8 小时适配，强调表征与反馈控制协同适配 |
| SONIC-Transfer | 解析关节 codec + decoder LoRA | G1 → AgiBot X2 Ultra | 冻结边界更强，面向近亲关节对应关系 |
| Any2Any | 可学习跨具身对齐与高效后训练 | 多种人形形态 | 覆盖不同关节数与形态，重点处理更广的形态差异 |
| 从头训练 | 目标机独立训练完整策略 | 单一目标平台 | 适合作为目标机基线，计算预算较高 |

TRACE 与 SONIC-Transfer 是不同论文和迁移设置；两者都复用 SONIC 运动跟踪先验，但机器人目标和对齐/微调设计各异。

## 工程实践

- 先核对源机与目标机的关节名称、关节顺序、活动范围和身体尺度，再做投影层映射。
- 对齐动作尺度后还要验证目标机上的关节响应；相同位置指令在不同执行器和质量分布下并不产生相同闭环效果。
- 迁移动作集要覆盖动作类别和命令空间，并用困难动作检查课程后段是否改善接触与跟踪。
- 评测报告需同时写清成功判据、动作数量、目标硬件及计算统计边界。
- **源码运行时序图：不适用（原因：本次核查未找到可确认的官方可运行代码入口；官方项目页目前无法访问，开放状态待复核）。**

## 结论

**TRACE 的主要价值是把跨机适配拆成表征保留与目标机反馈控制专门化，并在 G1→H2 案例中以小于完整预训练成本的预算完成迁移。**

1. **先处理输入/输出和关节响应的语义差异**，再启动策略微调；只改向量维度不足以保证闭环稳定。
2. **同时保持运动表征并适配目标动力学**，分别由编码器蒸馏与低秩强化学习承担。
3. **以两阶段课程安排训练难度**：先让广泛动作稳定跟踪，再集中改进高难片段。
4. **把 96.08% 和 8 小时连同评测条件一起引用**；论文使用高度缩放成功标准和 67,953 段动作，GPU 小时没有跨硬件归一化。
5. **迁移不是免验证部署**：目标机仍需真机安全检查，论文演示不能替代对不同硬件和动作分布的独立评测。
6. **官方代码状态尚待核实**，不要将论文提供的项目页入口等同于训练管线已公开。

## 局限与风险

- 论文展示的是 G1→H2 一组迁移方向，不能据此推断任意厂商、任意关节拓扑都能达到同样的成功率。
- 96.08% 是指定高度缩放跟踪标准下的基准评测结果，不能直接等同于真实任务成功率或所有动作的安全保证。
- 小于 0.1% 的算力表述对比论文所引 SONIC 预训练开销；硬件未归一化，且源模型训练成本排除在迁移成本外。
- **开源状态待核实**：已验证的 arXiv 页面指向官方项目页，但本次无法打开项目页，未发现可确认的官方代码仓库或数据集入口。

## 关联页面

- [跨具身策略迁移选型指南](../queries/cross-embodiment-transfer-strategy.md) — 对比从头训练、专门后训练和多具身联合路线。
- [Whole-Body Tracking Pipeline](../concepts/whole-body-tracking-pipeline.md) — 将跨具身迁移放在全身动作控制流水线中理解。
- [SONIC motion tracking](../methods/sonic-motion-tracking.md) — TRACE 复用的源控制器范式。
- [SONIC-Transfer](./paper-sonic-transfer.md) — 另一篇 G1 源模型跨机迁移工作。
- [Any2Any](./paper-any2any-cross-embodiment-wbt.md) — 跨较宽人形形态差异的迁移对照。

## 参考来源

- [TRACE 论文摘录](../../sources/papers/trace_arxiv_2610_17999.md)
- [TRACE 官方项目页归档与开源状态](../../sources/sites/trace-bfm-project.md)

## 推荐继续阅读

- [TRACE arXiv HTML](https://arxiv.org/html/2610.17999v1)
- [SONIC](../methods/sonic-motion-tracking.md)
- [Any2Any](./paper-any2any-cross-embodiment-wbt.md)
