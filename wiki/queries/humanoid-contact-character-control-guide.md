---
title: 人形接触与角色化控制实践指南
type: query
status: complete
created: 2026-05-21
updated: 2026-11-19
summary: 接触丰富人形控制与角色化/娱乐机器人路线工程选型：区分柔顺运动跟踪、触觉模仿学习、语言标注全身动作数据生成和角色表演平台。
sources:
  - ../../sources/papers/motion_control_projects.md
---

> **Query 产物**：本页由以下问题触发：「做人形接触交互或角色化表演控制，该用哪条方法线？」
> 综合来源：[GentleHumanoid](../entities/paper-gentlehumanoid.md)、[Humanoid Transformer with Touch Dreaming](../methods/humanoid-transformer-touch-dreaming.md)、[Teleoperation](../tasks/teleoperation.md)

# 人形接触与角色化控制实践指南

## TL;DR

| 目标 | 优先方法 | 备注 |
|------|----------|------|
| 跟踪参考且力控柔顺 | [GentleHumanoid](../entities/paper-gentlehumanoid.md) | 与通用 tracker 互补 |
| 触觉遥操作 → 接触策略 | [HTD](../methods/humanoid-transformer-touch-dreaming.md) | 需含视觉、本体、手部力和触觉的同步示范；Touch Dreaming 为训练期辅助目标 |
| 语言标注的全身动作数据生成 | [CLAW](../methods/claw.md) | 属于 MuJoCo 轨迹数据生成与标注管线，**不是**运行时接触控制策略 |
| 角色化/娱乐硬件路线 | [Being-H07](../methods/being-h07.md)、[Disney OLAF](../methods/disney-olaf-character-robot.md) | 偏平台与表演 |
| 特定论文/系统实现 | [HiPAN](../methods/hipan.md)、[Zest](../methods/zest.md)、[EFGCL](../methods/efgcl.md) | 按论文假设选型 |

---

## 接触丰富控制：两条主线

### 1. 柔顺运动跟踪

[GentleHumanoid](../entities/paper-gentlehumanoid.md) 把力/柔顺目标写进跟踪层，适合**推、扶、携带**等需要限制接触力的任务。通常在上游已有参考（MoCap / 遥操作）的前提下使用，而不是单独解决「无参考探索」。

### 2. 触觉条件策略学习

[Humanoid Transformer with Touch Dreaming](../methods/humanoid-transformer-touch-dreaming.md) 强调从**触觉遥操作数据**学习接触感知策略，与 [Tactile Impedance Control](../methods/tactile-impedance-control.md) 的阻抗控制视角形成「学习 vs 显式控制」对照。HTD 中，策略在训练时预测未来手部关节力与触觉潜变量；Touch Dreaming 的预测分支并非部署时的力控闭环。[下肢稳定控制器](../entities/paper-humanoid-touch-dream.md) 与上肢触觉策略也承担不同职责。

**工程检查清单**：

- 触觉采样率与控制环是否对齐（见 [实时控制中间件指南](./real-time-control-middleware-guide.md)）
- 接触标签与视觉/本体是否时间同步
- Sim 中接触模型是否与真机一致（见 [Sim2Real 工程 Checklist](./sim2real-checklist.md#快速部署检查)）

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
触觉 demo → HTD（训练时触觉预测辅助监督）→ 推理时动作头 + 独立下肢控制
角色平台 → Being-H07 / Disney OLAF → 表演轨迹库 + 安全监控
```

运动跟踪通用选型见 [人形运动跟踪方法选型](./humanoid-motion-tracking-method-selection.md)。

---

## 常见误区

1. **把角色机器人当研究通用平台**：表演稳定性 ≠ loco-manip 泛化。
2. **触觉策略无阻抗兜底**：真机建议保留 [Tactile Impedance Control](../methods/tactile-impedance-control.md) 或 WBC 力矩上限。
3. **接触丰富任务只用纯位置 tracking**：需要按任务评估柔顺跟踪、阻抗或可证实的力控策略，不能假设所有任务都可直接切换至 GentleHumanoid。
4. **把 CLAW 当在线力控器**：其目标是可组合、语言标注的全身运动**数据生成**，不替代真正的接触反馈与控制。

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
- [GentleHumanoid 论文实体](../entities/paper-gentlehumanoid.md) — 上身柔顺参考动力学、安全力阈值与真机交互边界（仓库核对更新于 2026-11-16）。
- [HTD 技术路线](../methods/humanoid-transformer-touch-dreaming.md) — 训练期 Touch Dreaming、部署期动作头及 LBC 分工（仓库核对更新于 2026-11-15）。
- [CLAW 数据生成管线](../methods/claw.md) — 区分语言标签轨迹数据生成和在线控制。

## 关联页面

- [GentleHumanoid](../entities/paper-gentlehumanoid.md)
- [Humanoid Transformer with Touch Dreaming](../methods/humanoid-transformer-touch-dreaming.md)
- [Tactile Impedance Control](../methods/tactile-impedance-control.md)
- [Being-H07](../methods/being-h07.md)、[Disney OLAF](../methods/disney-olaf-character-robot.md)
- [HiPAN](../methods/hipan.md)、[Zest](../methods/zest.md)、[EFGCL](../methods/efgcl.md)
- [CLAW](../methods/claw.md)
- [接触丰富操作指南](./contact-rich-manipulation-guide.md)
- [人形运动跟踪方法选型](./humanoid-motion-tracking-method-selection.md)

## 一句话记忆

> **柔顺跟踪参考 GentleHumanoid；触觉策略参考 HTD；语言标注运动数据生成参考 CLAW；角色平台单独评估。接触安全仍需按任务验证力阈值、阻抗与 Sim2Real 差异。**

> **2026-11-19 复核记录：** 对照仓库当前 GentleHumanoid（2026-11-16）、HTD（2026-11-15）及 CLAW（2026-10-21）技术页，修正旧版 HTTD 缩写与 CLAW 的“语言/接触控制”误分类；本指南为方法选型综述，未声称重新复现实验或验证真机安全。
