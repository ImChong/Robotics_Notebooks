---
type: query
tags: [humanoid, hardware, comparison, actuator]
related:
  - ../entities/humanoid-robot.md
  - ../concepts/whole-body-control.md
  - ../tasks/locomotion.md
  - ../comparisons/wbc-vs-rl.md
sources:
  - ../../sources/papers/humanoid_hardware.md
summary: "主流人形机器人硬件对比"
updated: 2026-04-25
---

# 主流人形机器人硬件对比

> **Query 产物**：本页由问题「主流人形机器人硬件平台横向对比」触发。
> Query 类型：对比分析
> 生成日期：2026-04-15
> 问题：主流人形机器人平台在硬件能力上有何差异？如何根据任务选择平台？

---

## 平台速览

| 平台 | 组织 | DOF | 重量 | 执行器 | 状态 |
|------|------|-----|------|--------|------|
| Atlas | Boston Dynamics | 28 | ~80kg | 液压 | 研究平台 |
| Unitree H1 | Unitree | 20 | ~47kg | 刚性电机 | 商业可购 |
| Unitree G1 | Unitree | 23 | ~35kg | 刚性电机 | 商业可购 |
| Agility Digit | Agility Robotics | 20 | ~65kg | SEA | 商业部署中 |
| Fourier GR-1 | Fourier Intelligence | 44 | ~55kg | 力控电机 | 商业可购 |
| Apptronik Apollo | Apptronik | 22 | ~73kg | 电机 | 预量产 |
| Figure 02 | Figure AI | 未公开 | ~60kg | 电机 | 研究中 |

---

## 执行器类型与任务适配

### 液压驱动（Atlas）
- **优势**：高功率密度，适合高动态运动（跑步、跳跃、空翻）
- **劣势**：重、能效低（~50%）、维护复杂、不适合产品化
- **适合**：高动态 locomotion 研究、极限运动展示

### 刚性电机（H1/G1）
- **优势**：轻量、低成本（H1 ~$90k）、易部署、适合 RL 研究
- **劣势**：无内置力矩传感，力控精度依赖电流估计；接触任务精度有限
- **适合**：locomotion RL 研究、工业搬运、低成本部署

### 系列弹性执行器 SEA（Digit）
- **优势**：内置弹性元件提供力矩测量和柔顺性；适合人机交互
- **劣势**：带宽受限（~20Hz）；高频控制困难
- **适合**：物流配送、人机协作、接触丰富任务

### 准直驱 QDD（MIT Cheetah 风格）
- **优势**：高带宽、高透明度、力矩可测
- **劣势**：峰值力矩受限；热管理要求高
- **适合**：高频力控、精密接触任务

---

## 按任务选择平台

| 任务目标 | 推荐平台 | 理由 |
|---------|---------|------|
| locomotion RL 研究 | Unitree H1/G1 | 低成本、轻量、RL 社区基础好 |
| 全身操作（loco-manipulation） | Fourier GR-1 / Digit | 高 DOF + 力控 |
| 高动态运动（跑跳） | Atlas | 液压高功率密度 |
| 工业搬运部署 | Digit / Apollo | SEA 柔顺性 + 商业就绪 |
| 科研算法验证（低预算） | Unitree G1 | 最低成本全尺寸人形 |

---

## Sim2Real 难度对比

| 执行器 | 主要 sim2real gap | 难度 |
|--------|-----------------|------|
| 液压 | 液压动态建模误差大 | ★★★★★ |
| 刚性电机 | 摩擦模型、关节刚度 | ★★★ |
| SEA | 弹性动态、共振 | ★★★★ |
| QDD | 热效应、非线性摩擦 | ★★★ |

---

## 关联页面
- [人形机器人总览](../entities/humanoid-robot.md)
- [全身运动控制](../concepts/whole-body-control.md)
- [Locomotion 任务](../tasks/locomotion.md)
- [WBC vs RL 对比](../comparisons/wbc-vs-rl.md)

## 参考来源
- [humanoid_hardware.md](../../sources/papers/humanoid_hardware.md)
