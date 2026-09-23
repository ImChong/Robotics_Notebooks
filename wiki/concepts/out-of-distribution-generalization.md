---
type: concept
tags: [ood, generalization, evaluation, deployment, sim2real, robustness, embodied-ai]
status: complete
updated: 2026-09-23
summary: "OOD（分布外）在具身智能里同时是三件事：评测里的一档测试集、部署时的一个监控量、数据配方里的一个目标；本页把这三义分开，给出「OOD 相对谁」的判据、常见的伪 OOD 陷阱与读 OOD 指标的口径。"
sources:
  - ../../sources/papers/freedof_sim2real_37_rapt-sim2real-ood-detection.md
  - ../../sources/blogs/wechat_freedof_sim2real_four_routes_dr_to_residual_2026-08-23.md
related:
  - ./sim-vs-real-eval-gap.md
  - ./sim2real.md
  - ./domain-randomization.md
  - ../queries/embodied-eval-benchmark-selection-loop.md
  - ../entities/paper-rapt-sim2real-ood-detection.md
  - ../entities/paper-huro.md
---

# 概念：分布外（OOD）泛化与 OOD 指标怎么读

> **一句话**：**OOD 不是一个属性，而是一个二元关系**——某个样本只能「相对某个具体分布」是分布外的；一篇论文报的 `OOD 成功率`、一套部署系统报的 `OOD 检测率`、一份数据配方说的「提升 OOD」，锚定的往往是**三个不同的分布**，混着读就会把互不可比的数字排成一个榜。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| OOD | Out-of-Distribution | 分布外；相对某个参考分布而言的样本/状态 |
| ID | In-Distribution | 分布内；与参考分布同源的样本 |
| DR | Domain Randomization | 域随机化，把部分变化提前纳入训练分布 |
| sim2real | Simulation-to-Real | 仿真到真机迁移，最常见的一类分布位移 |
| SR | Success Rate | 成功率；OOD 条件下常单独分档报告 |

## 为什么单独立这一页

站内多个实体页都在报 **OOD** 相关的数——[HuRo](../entities/paper-huro.md) 报「空间与视觉 shift 下完成率 **34.9% → 72.2%**」，[RAPT](../entities/paper-rapt-sim2real-ood-detection.md) 报的是「部署期能不能**检出**自己已经出了分布」，域随机化类工作报的则是「把哪些变化**提前塞进**训练分布」。这三类数字共用一个缩写，但**参考分布、观测量与成败判据都不同**。把「OOD 相对谁」这一问单独沉淀，是为了让读者在跨页比较前先做一次对齐，而不是把三张表并成一张。

## 三种 OOD，三个不同的参考分布

| 用法 | 参考分布是谁 | 观测量 | 典型页面 |
|------|--------------|--------|----------|
| **① 评测档位**：报 `OOD 成功率` | **训练集**（或演示数据分布） | 任务成功率 / 完成率，按 ID / OOD 分档 | [HuRo](../entities/paper-huro.md) 的空间与视觉 shift 档 |
| **② 部署监控**：做 OOD 检测 | **仿真里学到的标称执行流形** | 预测偏差、残差、似然/GMM 分数 | [RAPT](../entities/paper-rapt-sim2real-ood-detection.md) |
| **③ 数据/训练目标**：提升 OOD | **当前数据配方覆盖的范围** | 覆盖面本身（物体、视角、场景、本体） | [域随机化](./domain-randomization.md)、跨本体预训练类工作 |

**判据**：看到 OOD 三个字母，先问「**相对谁**」。① 的对照组是自己的训练集，换数据配方就换了定义；② 的对照组是仿真标称流形，它的失败是 **漏报/误报** 而不是任务失败；③ 根本不是一个测出来的数，而是一个设计目标。

## 读 OOD 指标的口径

- **ID/OOD 划分由作者定义，不是客观事实。** 「6 对视觉相似物体」「未见房间布局」「未见本体」都能叫 OOD，难度相差一个量级。**不同论文的 OOD 档之间默认不可比**，除非划分规则逐条对齐。
- **OOD 增益与 ID 基线要一起看。** ID 成功率低时 OOD 提升几十个百分点往往只是从「全崩」回到「勉强能跑」；只报 OOD 相对增益会放大这类改进。
- **OOD 成功率 ≠ 已经覆盖长尾。** 分档评测仍是**均值**，一个 OOD 档内部照样可能集中崩在某类物体或某个初值上；长尾问题要靠按失败模式分层，见 [sim↔real 评测 gap](./sim-vs-real-eval-gap.md)。
- **把变化塞进训练分布，OOD 就不再是 OOD。** 这是 [域随机化](./domain-randomization.md) 的基本逻辑，也意味着「强 DR + 报 OOD 高分」需要说明随机化范围，否则等于自己给自己出题。
- **部署侧的 OOD 检测是另一条指标线。** 它的代价函数是漏报（没检出就继续执行危险动作）与误报（频繁误停机）之间的权衡，与任务成功率不共享单位。

## 常见误区

1. **把 OOD 当成模型属性**：说「这个模型 OOD 好」而不说相对什么分布——同一模型换个训练集，ID/OOD 的划分立刻反转。
2. **用 sim2real gap 代替 OOD**：sim→real 只是分布位移的一种；同为真机，换光照、换操作者、换季节同样是分布外，与仿真无关。
3. **跨论文横比 OOD 档**：见上；这是站内 OOD 数字最常见的误用。
4. **把 OOD 检测的高准确率读成任务更稳**：检得出不等于处理得了；检测只负责触发降级策略，恢复能力是另一件事。

## 关联页面

- [仿真评测可复现性 ↔ 真实代表性取舍（sim↔real 评测 gap）](./sim-vs-real-eval-gap.md) — OOD 分档评测为何仍不足以代表长尾
- [Sim2Real](./sim2real.md) — 最常被与 OOD 混为一谈的那一类分布位移
- [Domain Randomization](./domain-randomization.md) — 把变化提前纳入训练分布，从而缩小 OOD 定义域
- [具身大模型评测基准选型闭环](../queries/embodied-eval-benchmark-selection-loop.md) — OOD 档在四层评测里的位置（③ 成功率层的分档 + ④ 校准层的外推）
- [RAPT（部署期 OOD 检测与失配诊断）](../entities/paper-rapt-sim2real-ood-detection.md) — ② 部署监控一义的代表页
- [HuRo](../entities/paper-huro.md) — ① 评测档位一义的代表页（空间与视觉 shift 完成率 34.9%→72.2%）

## 参考来源

- [freedof_sim2real_37_rapt-sim2real-ood-detection.md](../../sources/papers/freedof_sim2real_37_rapt-sim2real-ood-detection.md) — 部署期 OOD 检测与失配诊断的来源归档
- [wechat_freedof_sim2real_four_routes_dr_to_residual_2026-08-23.md](../../sources/blogs/wechat_freedof_sim2real_four_routes_dr_to_residual_2026-08-23.md) — Sim2Real 四条路线中「监控与评测」一支的上下文
- 本页其余口径由站内既有实体页与评测闭环 Query 汇编而成，未引入新外部资料

## 推荐继续阅读

- 站内：[具身大模型评测基准选型闭环](../queries/embodied-eval-benchmark-selection-loop.md) 的 ③④ 层，理解 OOD 分档在整条评测链里的位置
- 外部：Hendrycks & Gimpel, *A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks* — <https://arxiv.org/abs/1610.02136>
