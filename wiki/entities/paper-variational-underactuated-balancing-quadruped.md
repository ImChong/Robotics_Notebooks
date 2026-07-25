---
type: entity
tags: [paper, balance, optimal-control, quadruped, qp, notre-dame, mit]
status: complete
updated: 2026-07-25
venue: "IEEE Access 2020"
related:
  - ./mit-mini-cheetah.md
  - ../concepts/whole-body-control.md
  - ../concepts/optimal-control.md
  - ./paper-wbic-mpc-mini-cheetah.md
  - ../methods/model-predictive-control.md
sources:
  - ../../sources/papers/variational_underactuated_balancing_quadruped_ieee_access_2020.md
  - ../../sources/blogs/robot_daycare_mini_cheetah_2019.md
summary: "Chignoli & Wensing IEEE Access 2020：变分线性化 + 约束最优控制的欠驱动（两点足）四足平衡，凸 QP 近似；Mini Cheetah 扰动恢复。"
---

# Variational-Based Optimal Control of Underactuated Balancing

## 一句话定义

**Chignoli & Wensing（[IEEE Access 2020](https://doi.org/10.1109/ACCESS.2020.2980446)）** 提出面向四足**欠驱动平衡**（如两点足支撑）的控制框架：结合约束最优控制与**变分线性化**，用**凸 QP** 近似摩擦约束下的最优策略；在 **Mini Cheetah** 上演示角动量利用与 CoM 出支撑域恢复。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| QP | Quadratic Programming | 在线求解的凸二次规划 |
| CoM | Center of Mass | 质心；支撑域关系是平衡核心 |
| MPC | Model Predictive Control | 文中对照的更重计算基线 |
| DoF | Degree of Freedom | 欠驱动指可控自由度不足 |
| VBL | Variational-Based Linearization | 变分线性化思路 |

## 为什么重要

- 常规 QP 平衡控制器常假设充分驱动接触；两点足等工况直接失效。
- 给出比完整 MPC 更紧凑、仍能处理极端摩擦限制的平衡律。
- 对人形双足平衡亦有方法迁移价值（文中明示）。

## 核心信息

| 项 | 内容 |
|----|------|
| **机构** | 圣母大学（University of Notre Dame）等 |
| **平台** | MIT Mini Cheetah |
| **形式** | 凸 QP（由无约束最优控制解构造） |
| **开源** | **未开源**独立官方仓；OA PDF 可获取 |

## 核心原理

- 挑战：摩擦锥 + 构型流形使 cart-pendulum/acrobot 教科书解不能直接搬。
- 方法：变分线性化简化动力学 → 无约束最优控制提供方向 → QP 投影到摩擦可行集。
- 能力：两点支撑扰动恢复；CoM 离开支撑多边形后的恢复。

```mermaid
flowchart LR
  x["状态"] --> vbl["变分线性化模型"]
  vbl --> uuc["无约束最优控制"]
  uuc --> qp["摩擦约束凸 QP"]
  qp --> tau["接触力 / 姿态命令"]
```

## 源码运行时序图

**不适用**（截至入库日无官方可运行仓库）。

## 工程实践

| 项 | 建议 |
|----|------|
| 场景 | 先验于「双足化」或对角支撑的平衡实验 |
| 对照 | 与全身 MPC 比 CPU 时间与可恢复扰动幅度 |
| 安全 | 真机测试需保护架；欠驱动恢复失败成本高 |

## 评测

| 维度 | 要点 |
|------|------|
| 仿真 + 真机 | Mini Cheetah |
| 能力 | 两点支撑角动量恢复；CoM 出支撑域恢复 |
| 计算 | 相对 MPC 更紧凑 |

## 结论

**总判：** 这是 Mini Cheetah 上少见的**欠驱动平衡专用**论文，补齐「不只跑得快，还能在少接触下稳住」的控制拼图。

- 真影响：欠驱动接触下的凸 QP 平衡可行。
- 次要代价：依赖简化模型；开源缺失。
- 部署：作为平衡模块研究参考，勿与 trot MPC 混为一谈。

## 与其他工作对比

| 对照对象 | 差异要点 |
|----------|----------|
| 常规充分驱动 QP 平衡控制器 | 常规 QP 假设充分驱动接触，两点足等欠驱动工况直接失效；本文用变分线性化 + 凸 QP 处理 |
| 完整 [MPC](../methods/model-predictive-control.md) | 相对全身 MPC 更紧凑（CPU 时间更省），仍能处理极端摩擦约束下的平衡恢复 |
| trot 步态 MPC | 本文是欠驱动平衡专用模块，勿与 trot loco 混为一谈；对人形双足平衡有迁移价值 |

## 局限与风险

- 无代码仓提高复现成本。
- 极端地形/软地面摩擦模型失配风险。

## 关联页面

- [MIT Mini Cheetah](./mit-mini-cheetah.md)
- [Whole-Body Control](../concepts/whole-body-control.md)
- [WBIC+MPC](./paper-wbic-mpc-mini-cheetah.md)
- [Optimal Control](../concepts/optimal-control.md)

## 参考来源

- [论文归档](../../sources/papers/variational_underactuated_balancing_quadruped_ieee_access_2020.md)
- [博文清单](../../sources/blogs/robot_daycare_mini_cheetah_2019.md)

## 推荐继续阅读

- DOI：<https://doi.org/10.1109/ACCESS.2020.2980446>
