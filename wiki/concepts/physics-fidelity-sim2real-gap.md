---
type: concept
tags: [simulation, sim2real, physics, dynamics, contact, friction, fidelity]
status: complete
updated: 2026-06-25
related:
  - ../queries/simulation-physics-fidelity.md
  - ./sim2real.md
  - ./contact-dynamics.md
  - ./joint-friction-models.md
  - ./friction-compensation.md
  - ./differentiable-simulation.md
  - ./urdf-robot-description.md
  - ./floating-base-dynamics.md
  - ./centroidal-dynamics.md
  - ../formalizations/articulated-body-algorithms.md
sources:
  - ../../sources/courses/quadruped_control_simulation_rl_curriculum.md
  - ../../sources/repos/sage-sim2real-actuator-gap.md
  - ../../sources/papers/bam_extended_friction_servos_arxiv_2410_08650.md
  - ../../sources/papers/contact_dynamics.md
summary: "物理保真度 ↔ sim2real gap 的因果概念页：把仿真保真度拆成几何/URDF → 刚体动力学 → 接触/摩擦 → 执行器四层，明示每层简化如何转化为可观测的 sim2real gap，并与域随机化/系统辨识互补定位。"
---

# Physics Fidelity ↔ Sim2Real Gap（物理保真度与仿真到现实差距）

**物理保真度（Physics Fidelity）**：仿真器对真实物理过程的逼近程度。它不是一个标量，而是沿 **几何/URDF 精度 → 刚体动力学算法 → 接触/摩擦模型 → 执行器模型** 四层分别成立的维度组合——每一层的简化都会在真机上转化为一种**可观测**的 sim2real gap。

## 一句话定义

> sim2real gap 不是一团模糊的「现实更难」，而是四层物理近似各自漏掉的真实成分之和；提高某一层的保真度，只能补上**那一层**对应的 gap，且往往以仿真速度 / 可微性 / 吞吐为代价。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Sim2Real | Simulation to Real | 把仿真中学到的策略迁移落地真机的工程主线 |
| DR | Domain Randomization | 域随机化，用分布覆盖未建模残差 |
| SysID | System Identification | 系统辨识，用真机数据反推物理参数 |
| URDF | Unified Robot Description Format | 描述连杆/关节几何与惯量的机器人模型格式 |
| ABA | Articulated Body Algorithm | O(n) 正向动力学递归算法 |
| RNEA | Recursive Newton-Euler Algorithm | O(n) 逆动力学递归算法 |
| LCP | Linear Complementarity Problem | 硬接触互补问题，求导困难 |
| SEA | Series Elastic Actuator | 串联弹性执行器，力矩传递含柔性 |
| BLDC | Brushless DC Motor | 无刷直流电机，力矩-电流-温度非线性 |

## 为什么重要

把 sim2real gap 当成单一黑盒，调参时就只能盲目加大域随机化范围，既拖慢训练又掩盖真正的建模缺口。把它**分层归因**后，每一种真机失败现象都能定位到某一层的近似：

- 站立时脚底「打滑 / 抖动」→ 接触/摩擦层（库仑摩擦系数、接触刚度）。
- 迈步落地冲击与仿真对不上 → 接触层（硬 LCP vs 软接触）叠加几何层（脚底碰撞体简化）。
- 力矩跟踪在高速段塌陷 → 执行器层（理想力矩源假设忽略了力矩-转速曲线与温度降额）。
- 整机姿态长期漂移 → 几何/惯量层（URDF 质量/质心/惯量标定误差）经动力学层放大。

分层让 **DR / SysID / 补偿** 各自落到该补的那一层，而不是用一个超大随机化范围掩盖一切。

## 四层保真度分解

```
真实机器人
   ↑  ← 执行器层：力矩源理想化 → 力矩 gap、延迟、温度降额
   │      （SEA 柔性、BLDC 力矩-电流-温度曲线、传动摩擦/背隙）
   ↑  ← 接触/摩擦层：刚体硬接触 / 库仑摩擦简化 → 打滑、抖动、冲击失配
   │      （摩擦锥、接触刚度/阻尼、软接触 vs LCP）
   ↑  ← 刚体动力学层：ABA/RNEA 数值积分 → 能量漂移、长视界发散
   │      （积分步长、刚度/约束求解器精度）
   ↑  ← 几何/URDF 层：连杆几何/质量/质心/惯量标定误差 → 被上层逐级放大
仿真模型
```

### ① 几何 / URDF 精度

URDF 的连杆尺寸、碰撞体形状、质量、质心与惯量张量是动力学的输入。任何标定误差都会被上面三层**逐级放大**：质心偏移直接改变接触力分布，惯量误差让同一力矩产生不同加速度。这是「便宜但易被忽视」的一层——通常靠真机称重 / CAD 复核 / SysID 修正，而非靠 DR 覆盖。

### ② 刚体动力学算法（ABA / RNEA）

给定模型，正/逆动力学由 [ABA/RNEA](../formalizations/articulated-body-algorithms.md) 这类 O(n) 递归算法在固定步长下数值积分。保真度损失来自**积分步长与约束求解器精度**：步长过大或求解器迭代不足会引入能量漂移，在长视界 rollout 上累积成发散。这一层的取舍是 **精度 vs 吞吐**——更小步长/更多迭代换更高保真，但直接拖慢并行采样。

### ③ 接触 / 摩擦模型

腿式与操作任务里，[接触动力学](./contact-dynamics.md)与[关节摩擦模型](./joint-friction-models.md)是 gap 的主战场。硬 LCP 接触物理最真但**不连续、求导困难**；软接触 / XPBD 平滑可微但会引入穿透与虚假阻尼。库仑+黏滞摩擦的简化会低估静摩擦区，导致仿真里「站得住」的策略在真机上打滑。这一层的核心矛盾是 **接触保真度 ↑ 与可微性 / 吞吐冲突**。

### ④ 执行器模型

最顶层、也最常被理想化的一层。把关节当成「无限带宽的力矩源」会忽略 [SEA](./joint-friction-models.md) 柔性、BLDC 的力矩-电流-温度曲线、传动背隙与摩擦、控制延迟。这条 [actuator gap](../../sources/repos/sage-sim2real-actuator-gap.md) 在高速、大力矩或长时运行时最显著，常用执行器网络（actuator net）或 [BAM 扩展摩擦模型](../../sources/papers/bam_extended_friction_servos_arxiv_2410_08650.md) 在仿真侧补回。

## 与域随机化 / 系统辨识的互补关系

物理保真度与 DR/SysID 不是替代而是**互补**：

| 手段 | 作用层 | 何时用 |
|------|--------|--------|
| 提高保真度 | 把已知物理建进模型 | 该层有明确机理、可标定（几何、执行器曲线） |
| 系统辨识 | 反推该层参数 | 机理已知但参数未知（摩擦系数、惯量） |
| 域随机化 | 覆盖**残差**分布 | 机理无法精确建模或在线变化（地面材质、磨损） |

经验法则：**先用保真度 + SysID 把能建的建准，再用 DR 覆盖剩下的残差**。一上来就靠超大 DR 范围掩盖一切，会同时牺牲训练效率与策略性能。

## 常见误区

- **误区：「保真度越高越好」。** 接触/积分保真度越高，仿真越慢、越难可微；在吞吐受限的并行 RL 里，过高保真反而拖垮采样效率。保真度是**按层选择性投资**，不是一味拉满。
- **误区：「DR 能替代保真度」。** DR 覆盖的是残差分布；若某层机理性偏差（如执行器力矩 gap）系统性存在，再大的随机化也只是把策略训得保守，而非消除 gap。

## 关联页面

- 端到端决策：[仿真物理保真度链路选型](../queries/simulation-physics-fidelity.md)
- 工程主线：[Sim2Real](./sim2real.md)
- 几何层：[URDF 机器人描述](./urdf-robot-description.md)
- 动力学层：[ABA/RNEA 铰接体算法](../formalizations/articulated-body-algorithms.md)、[Floating Base Dynamics](./floating-base-dynamics.md)、[Centroidal Dynamics](./centroidal-dynamics.md)
- 接触/摩擦层：[Contact Dynamics](./contact-dynamics.md)、[Joint Friction Models](./joint-friction-models.md)、[Friction Compensation](./friction-compensation.md)
- 可微性取舍：[Differentiable Simulation](./differentiable-simulation.md)

## 参考来源

- [sources/courses/quadruped_control_simulation_rl_curriculum.md](../../sources/courses/quadruped_control_simulation_rl_curriculum.md) — 几何/URDF、刚体动力学与摩擦建模的课程链路
- [sources/repos/sage-sim2real-actuator-gap.md](../../sources/repos/sage-sim2real-actuator-gap.md) — 执行器层 sim2real gap 与补偿
- [sources/papers/bam_extended_friction_servos_arxiv_2410_08650.md](../../sources/papers/bam_extended_friction_servos_arxiv_2410_08650.md) — 舵机扩展摩擦模型（BAM）
- [sources/papers/contact_dynamics.md](../../sources/papers/contact_dynamics.md) — 接触动力学一手资料
