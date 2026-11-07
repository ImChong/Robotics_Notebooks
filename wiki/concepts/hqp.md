---
type: concept
tags: [wbc, tsid, optimization, qp, humanoid, control]
status: complete
updated: 2026-11-07
summary: "HQP 用严格层级的二次规划（词典序最小化）表达任务优先级：Kanoun 2011 的 QP 级联与 Escande 2014 的单一有效集求解器，面向速度级逆运动学，并可扩展到 TSID 的加速度/力层。"
sources:
  - ../../sources/papers/escande_hqp_ijrr_2014.md
  - ../../sources/papers/kanoun_task_priority_inequality_tro_2011.md
  - ../../sources/papers/saab_dynamic_whole_body_motion_tro_2013.md
  - ../../sources/papers/crocoddyl_mastalli_icra_2020.md
  - ../../sources/repos/tsid.md
  - ../../sources/papers/whole_body_control.md
  - ../../sources/papers/null_space_control.md
original:
  - ../../sources/papers/escande_hqp_ijrr_2014.md
  - ../../sources/papers/kanoun_task_priority_inequality_tro_2011.md
  - ../../sources/papers/saab_dynamic_whole_body_motion_tro_2013.md
  - ../../sources/papers/crocoddyl_mastalli_icra_2020.md
  - ../../sources/repos/tsid.md
---

# HQP（Hierarchical QP）

**分层二次规划（Hierarchical Quadratic Programming，HQP）**：全身控制（WBC）中处理多任务优先级冲突的优化框架，通过将任务按优先级分层求解，确保高优先级任务精确满足，低优先级任务在不影响高优先级的前提下尽量完成。

## 一句话定义

> HQP 回答的是："当机器人同时有多个目标，而且这些目标互相冲突时，应该优先满足哪个？" — 答案是分层，先保命（平衡），再走路，再优雅（手臂姿态）。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| HQP | Hierarchical Quadratic Programming | 按优先级分层堆叠任务与约束的二次规划 |
| WBC | Whole-Body Control | 协调全身关节满足多任务/约束的控制基础设施 |
| TSID | Task-Space Inverse Dynamics | 任务空间逆动力学求解关节力矩的 WBC 实现 |
| QP | Quadratic Programming | 将 WBC/控制问题写成二次规划的标准求解形式 |
| DoF | Degrees of Freedom | 自由度，人形通常 20–50+ 关节 |
| MPC | Model Predictive Control | 滚动时域内优化控制序列的预测控制 |
| LQR | Linear Quadratic Regulator | 线性系统二次型代价下的最优反馈控制器 |
| iLQR | iterative Linear Quadratic Regulator | 对非线性系统迭代线性化求解的轨迹优化方法 |

## 为什么重要

人形机器人全身控制的核心挑战是**任务冲突**：

- 躯干稳定 vs. 摆腿速度
- 精确轨迹跟踪 vs. 关节力矩限制
- 保持接触 vs. 全身动作幅度

单层加权 QP（把所有任务加权相加）无法保证高优先级任务被**精确满足**——权重调整只能做软性平衡。HQP 提供了**硬性优先级**的解决方案，是全身控制框架中处理任务冲突的一类主流求解机制。

## 核心原理：问题形式化

> **溯源（对照原文）**：严格层级 HQP 的两个源头都是**速度级逆运动学**。Kanoun 等 2011 提出"QP 序列"：每一级只在上一级最优集合内求解，不等式用松弛变量 $w\ge0$ 表示；Escande 等 2014 指出该级联会反复增删有效集，给出**单一专用求解器**（分层完全正交分解 HCOD + 统一有效集搜索），纯等式层级时退化为 Siciliano–Slotine 的零空间投影且快至 10 倍，40 变量带不等式"certified 1 ms"。把它用到动力学层（加速度、力矩、接触力）是 Saab 等 2013 的工作。

### 单层加权 QP（对比用）

$$\min_z \sum_i w_i \| A_i z - b_i \|^2 \quad \text{s.t.} \quad C z \leq d$$

问题：权重比值再大，也不能保证任务 $i$ 精确满足。

### HQP：分层求解

设有 $L$ 个优先级层次，层 $l$ 的任务为 $A_l z = b_l$：

**层 1（最高优先级）**：

$$z_1^* = \arg\min_z \| A_1 z - b_1 \|^2 \quad \text{s.t.} \quad C z \leq d$$

**层 2**（在不影响层 1 的前提下）：

$$z_2^* = \arg\min_z \| A_2 z - b_2 \|^2 \quad \text{s.t.} \quad A_1 z = A_1 z_1^*, \; C z \leq d$$

即：层 2 的解在层 1 的**零空间（null space）**中求最优。

**层 $l$**（一般形式）：

$$z_l^* = \arg\min_z \| A_l z - b_l \|^2 \quad \text{s.t.} \quad A_k z = A_k z_{l-1}^* \; \forall k < l, \; C z \leq d$$

## 人形控制中的典型优先级结构

```text
Priority 1（最高）：动力学约束 + 接触约束
    → M(q)q̈ + h = Sᵀτ + Jcᵀf
    → Jc q̈ + J̇c q̇ = 0

Priority 2：CoM 跟踪 + 躯干姿态稳定
    → 保持平衡，防止摔倒

Priority 3：摆动腿末端轨迹跟踪
    → 精确控制落脚点

Priority 4：手臂/上身动作
    → 在不影响平衡和步态的前提下做动作

Priority 5（最低）：关节正则化 / 零空间姿态
    → 避免奇异、关节限位保护
```

**一句话记忆**：先保命，再走路，再优雅。

## HQP vs. 加权 QP

| 维度 | 加权 QP | HQP |
|------|---------|-----|
| 优先级表达 | 软性（权重） | 硬性（分层，词典序） |
| 高优先级保证 | ❌（只是权重大） | ✅（低优先级永不破坏高优先级的最优值，Escande §2.6） |
| 计算复杂度 | 低（单个 QP） | 高（级联 QP，或专用求解器） |
| 实现难度 | 简单 | 较复杂 |
| 适用场景 | 任务不严格冲突 | 严格优先级需求 |

## 工程实践

### 实时求解

Escande 2014 给出的数据点（HRP-2，速度级逆运动学，约 40 变量）：纯等式层级约 0.2 ms；带不等式"certified 1 ms"；实验中 97–98% 的控制周期有效集搜索无需额外迭代，平均约 1.1 ms/周期，结论称普通 PC 上 200 Hz 实时（摘要称有望到 1 kHz 量级）。对比用的 De Lasa 式级联在实验中单周期迭代最多 30 次、均值 4.5 次。Saab 2013 把 HQP 用到动力学层（变量 $(\tau,\ddot q,f)$）时，运动是**离线生成**的，原文称"接近但尚未实时"。因此"人形 500–1000 Hz 跑严格多层 HQP"并不是这些原文的结论（推测：需要更快的求解器与更少的层）。

**常见优化方案**：

1. **热启动 / 复用有效集**：Escande 的单一有效集求解器正是用上一周期的有效集做 warm start，并可设迭代上限换取确定耗时（代价是少数周期次优，实验中行为无可感知变化）。
2. **减少层数**：层数越少越便宜。`stack-of-tasks/tsid` 的默认 eiquadprog 求解器只支持 **2 层**（读源码：超过 2 层直接报错）；"3–5 层足够"这类说法没有原文依据（推测，未核实）。
3. **近似 HQP**：用加权方案做一层 QP，权重设计成近似层次效果；Kim 等 2019（MIT WBIC）就明确选择"松弛浮动基座动力学 + 两项加权 QP"，而不是用 HQP 严格层级。
4. **论文里的数值**：HCOD 用 Givens 变换做增量更新，细节见 Escande Part 2（本页未逐式核对）。

### 和 TSID 的关系

TSID（Task Space Inverse Dynamics）与 HQP 是两条相关但不同的线：

- Del Prete 2014 的 TSID 原论文用**伪逆 + 零空间投影**实现层级，全驱动、无不等式，**没有用 QP**；
- 浮动基座 + 接触力 + 不等式时，Saab 2013 把动力学方程、接触条件和任务写成 HQP：(21) ≺ (22) ≺ (28) ≺ (14)；
- `stack-of-tasks/tsid` 库把任务放进"优先级 0 硬约束层 + 带权重的任务层"，多层严格 HQP 并非其默认用法。

见：[TSID](./tsid.md)

### 典型开源实现

| 库 | 说明（已核对） |
|----|------|
| [TSID](https://github.com/stack-of-tasks/tsid) | 基于 Pinocchio 的加速度-力 QP 实现；硬约束层 + 加权任务层，默认求解器至多 2 层（BSD-2-Clause） |
| [Crocoddyl](https://github.com/loco-3d/crocoddyl) | 多接触**最优控制**（FDDP，DDP 型）求解器；不是 HQP 型优先级求解器（Mastalli 2020） |
| mc_rtc | 本页旧版称其"内置 HQP"，本次未核对官方文档（推测，勿当结论引用） |

## 局限与风险

- **源头是速度级**：Kanoun / Escande 的实验是逆运动学；用在力矩层需要 Saab 式扩展，并且接触力非负那一层可能不可行（强接触不稳定），需要切换接触模式。
- **词典序的代价**：严格层级意味着低优先级任务可能被完全牺牲；任务间没有"折中"，权重式方案则相反。
- **有效集迭代次数不定**：最坏情形迭代数多，需 warm start 与迭代上限；上限会带来次优解。
- **正则化**：任务冲突时 Kanoun 实验需要把正则参数调到约 0.01 量级以防不稳定。

## 关联页面

- [TSID](./tsid.md) — HQP 与 TSID 的关系见上文（原论文用零空间投影；Saab 2013 起用 HQP）
- [零空间控制](./null-space-control.md) — 显式 $N=I-J^+J$ 是 HQP「下层在上层核里优化」的连续时间形态
- [Whole-Body Control](./whole-body-control.md) — HQP 是 WBC 任务冲突处理的主流方案
- [MPC 与 WBC 集成](./mpc-wbc-integration.md) — 典型控制架构中 WBC 层用 HQP 求解
- [LQR / iLQR](../formalizations/lqr.md) — LQR 是单任务最优控制解析解；HQP 是多任务优先级控制的数值求解
- [Crocoddyl](../entities/crocoddyl.md) — 相关最优控制求解框架（DDP 型，不是 HQP）

## 参考来源

- [Escande et al. 2014 — HQP 通读归档](../../sources/papers/escande_hqp_ijrr_2014.md)（IJRR 33(7)，HAL hal-00751924）
- [Kanoun et al. 2011 — 任务优先级推广到不等式任务通读归档](../../sources/papers/kanoun_task_priority_inequality_tro_2011.md)（IEEE T-RO 27(4)）
- [Saab et al. 2013 — 刚性接触下动态全身运动生成通读归档](../../sources/papers/saab_dynamic_whole_body_motion_tro_2013.md)
- [Mastalli et al. 2020 — Crocoddyl 通读归档](../../sources/papers/crocoddyl_mastalli_icra_2020.md)
- [TSID 仓库归档](../../sources/repos/tsid.md) — 层数限制等源码核对
- [sources/papers/whole_body_control.md](../../sources/papers/whole_body_control.md) — HQP 相关来源整理（二手）
- [零空间控制论文簇](../../sources/papers/null_space_control.md) — Nakamura 任务优先级 → 显式投影；HQP 是其不等式推广
- Del Prete et al. (2014), *Task Space Inverse Dynamics*（见 [TSID 论文归档](../../sources/papers/delprete_tsid_arxiv_1410_3863.md)）

## 推荐继续阅读

- [TSID](./tsid.md)
- [Whole-Body Control](./whole-body-control.md)
- Escande et al., *Hierarchical quadratic programming: Fast online humanoid-robot motion generation*：<https://hal.science/hal-00751924/document>
- Kanoun et al., *Kinematic control of redundant manipulators: generalizing the task priority framework to inequality tasks*：<https://hal.science/hal-00486755/document>
- TSID library: <https://github.com/stack-of-tasks/tsid>

## 一句话记忆

> HQP 是全身控制的"优先级调度器"——确保关键约束（平衡、接触）被精确满足，再用剩余的自由度去完成次要任务，是人形机器人多任务冲突处理的主流框架。
