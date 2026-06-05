---
title: 人形接触与角色化控制实践指南
type: query
status: complete
created: 2026-05-21
updated: 2026-05-21
summary: 接触丰富人形控制与角色化/娱乐机器人路线的工程选型：触觉 dreaming、柔顺跟踪、角色平台与相关学习框架。
sources:
  - ../../sources/papers/motion_control_projects.md
---

> **Query 产物**：本页由以下问题触发：「做人形接触交互或角色化表演控制，该用哪条方法线？」
> 综合来源：[GentleHumanoid](../methods/gentlehumanoid-motion-tracking.md)、[Humanoid Transformer with Touch Dreaming](../methods/humanoid-transformer-touch-dreaming.md)、[Teleoperation](../tasks/teleoperation.md)

# 人形接触与角色化控制实践指南

## TL;DR

| 目标 | 优先方法 | 备注 |
|------|----------|------|
| 跟踪参考且力控柔顺 | [GentleHumanoid](../methods/gentlehumanoid-motion-tracking.md) | 与通用 tracker 互补 |
| 触觉遥操作 → 接触策略 | [HTTD](../methods/humanoid-transformer-touch-dreaming.md) | 需触觉演示数据 |
| 全身协调 + 语言/接触 | [CLAW](../methods/claw.md) | 见 [操作 VLA 架构选型](./manipulation-vla-architecture-selection.md) |
| 角色化/娱乐硬件路线 | [Being-H07](../methods/being-h07.md)、[Disney OLAF](../methods/disney-olaf-character-robot.md) | 偏平台与表演 |
| 特定论文/系统实现 | [HiPAN](../methods/hipan.md)、[Zest](../methods/zest.md)、[EFGCL](../methods/efgcl.md) | 按论文假设选型 |

---

## 接触丰富控制：两条主线

### 1. 柔顺运动跟踪

[GentleHumanoid](../methods/gentlehumanoid-motion-tracking.md) 把力/柔顺目标写进跟踪层，适合**推、扶、携带**等需要限制接触力的任务。通常在上游已有参考（MoCap / 遥操作）的前提下使用，而不是单独解决「无参考探索」。

### 2. 触觉条件策略学习

[Humanoid Transformer with Touch Dreaming](../methods/humanoid-transformer-touch-dreaming.md) 强调从**触觉遥操作数据**学习接触感知策略，与 [Tactile Impedance Control](../methods/tactile-impedance-control.md) 的阻抗控制视角形成「学习 vs 显式控制」对照。

**工程检查清单**：

- 触觉采样率与控制环是否对齐（见 [实时控制中间件指南](./real-time-control-middleware-guide.md)）
- 接触标签与视觉/本体是否时间同步
- Sim 中接触模型是否与真机一致（见 [Sim2Real 部署清单](./sim2real-deployment-checklist.md)）

---

## 角色化与娱乐向平台

| 方法/平台 | 定位 | 选型提示 |
|-----------|------|----------|
| [Being-H07](../methods/being-h07.md) | 特定人形硬件/系统路线 | 评估驱动器、散热与全身 DoF |
| [Disney OLAF](../methods/disney-olaf-character-robot.md) | 角色化表演机器人 | 强调外观与表演稳定性，而非通用 loco-manip |
| [HiPAN](../methods/hipan.md) | 论文级全身交互方案 | 复现前核对观测与动作空间 |
| [Zest](../methods/zest.md) | 特定学习/控制框架 | 与任务 reward 假设绑定 |
| [EFGCL](../methods/efgcl.md) | 图/对比学习向接触表示 | 适合作为表征模块而非完整栈 |

---

## 与其他层的组合

```text
遥操作/动捕 → GentleHumanoid（柔顺跟踪）→ 下游操作或导航
触觉 demo → HTTD → 与 VLA/IL 栈并联或级联
角色平台 → Being-H07 / Disney OLAF → 表演轨迹库 + 安全监控
```

运动跟踪通用选型见 [人形运动跟踪方法选型](./humanoid-motion-tracking-method-selection.md)。

---

## 常见误区

1. **把角色机器人当研究通用平台**：表演稳定性 ≠ loco-manip 泛化。
2. **触觉策略无阻抗兜底**：真机建议保留 [Tactile Impedance Control](../methods/tactile-impedance-control.md) 或 WBC 力矩上限。
3. **接触 rich 仍用纯位置 tracking**：应切换到 GentleHumanoid 或力控层。

---

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| DR | Domain Randomization | 训练时随机化仿真参数以提升跨域鲁棒迁移 |
| VLA | Vision-Language-Action | 视觉-语言-动作多模态基础策略方向 |
| MoCap | Motion Capture | 动作捕捉，参考动作与演示数据的主要来源 |
| Sim2Real | Simulation to Real | 把仿真中学到的策略迁移落地真机的工程主线 |
| DoF | Degrees of Freedom | 自由度，人形通常 20–50+ 关节 |
| WBC | Whole-Body Control | 协调全身关节满足多任务/约束的控制基础设施 |

## 参考来源

- [运动控制开源项目笔记](../../sources/papers/motion_control_projects.md)

## 关联页面

- [GentleHumanoid](../methods/gentlehumanoid-motion-tracking.md)
- [Humanoid Transformer with Touch Dreaming](../methods/humanoid-transformer-touch-dreaming.md)
- [Tactile Impedance Control](../methods/tactile-impedance-control.md)
- [Being-H07](../methods/being-h07.md)、[Disney OLAF](../methods/disney-olaf-character-robot.md)
- [HiPAN](../methods/hipan.md)、[Zest](../methods/zest.md)、[EFGCL](../methods/efgcl.md)
- [CLAW](../methods/claw.md)
- [接触丰富操作指南](./contact-rich-manipulation-guide.md)
- [人形运动跟踪方法选型](./humanoid-motion-tracking-method-selection.md)

## 一句话记忆

> **柔顺跟踪用 GentleHumanoid，触觉策略用 HTTD，角色平台单独评估；阻抗与 sim2real 是接触任务的保险丝。**
