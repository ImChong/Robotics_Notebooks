---
type: overview
tags: [humanoid, actuator, tesla-optimus, unitree, agility-digit, category-hub]
status: complete
updated: 2026-06-02
summary: "Actuator 102 · 07 — No Free Lunch 评分卡；三大物种（工厂/快递/家庭）；关节指标 >15Nm/kg、反向驱动 <1Nm、峰值持续 3:1、智能关节集成。"
related:
  - ./humanoid-actuator-102-technology-map.md
  - ./humanoid-actuator-102-compliance-sensing.md
  - ./humanoid-actuator-102-industrial-actuator-trap.md
  - ../queries/humanoid-hardware-selection.md
sources:
  - ../../sources/blogs/wechat_human_five_humanoid_actuator_102.md
  - ../../sources/raw/wechat_humanoid_actuator_102_2026-06-02.md
---

# Actuator 102 · 07：决策矩阵与三大物种

> **图谱分类节点**：**X 总决策矩阵** + **XI 关节执行器设计要求**。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| SEA | Series Elastic Actuator | 串联弹性执行器，提供柔顺与力控 |
| QDD | Quasi-Direct Drive | 准直驱，低减速比、高背驱动性的作动方案 |
| G1 | Unitree G1 Humanoid | 宇树入门级教育科研人形平台 |
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略的范式 |

## 三大物种（文内策展）

| 物种 | 任务 | 执行器倾向 | 案例 |
|------|------|------------|------|
| **A 工厂工人** | 搬重箱、8h+、精准放置 | 滚柱直线 + 谐波旋转 | Tesla Optimus、Figure 02、Apptronik Apollo |
| **B 敏捷快递** | 快移、抗摔、续航 | SEA 或 QDD | Agility Digit、Unitree G1/H1 |
| **C 家庭助手** | 静音、安全、可负担 | 低减速谐波或 QDD | 1X Neo、Ubtech Walker |

短期 **不会** 趋同为单一「通用」架构：高减速↔反射惯量、弹簧↔带宽等 **物理权衡** 难用软件消除。

## 仿真性维度（X）

- 刚性（谐波、滚柱）：**易仿真** → 加速 RL（文内 Tesla 隐性理由）。
- SEA/QDD、液压：仿真到现实更难。

## 关节硬指标（XI 摘要）

| 指标 | 文内门槛 |
|------|----------|
| 比力矩 | **>15 Nm/kg** 峰值（竞争 >25） |
| 反向驱动 | 输出端外力反驱 **<1 Nm** |
| 峰值:持续 | **≥3:1** |
| 力矩带宽 | **50–100 Hz**；延迟 **<1 ms** |
| 反射惯量 | 输出端 **<0.1 kg·m²** |
| 冲击 | 线接触传动；**>1000 万次** 循环量级（文内表述） |

膝关节样本参数见原文图 31；髋/踝/肩/肘/腕按负荷缩放。上述峰值:持续与比力矩指标应在供应商 [TN 曲线](../concepts/motor-torque-speed-curve.md) 与 [TI 曲线](../concepts/motor-torque-current-curve.md) 上逐项核对，而非仅看单点标称值。

## 关联页面

- [电机转矩-转速曲线（TN 曲线）](../concepts/motor-torque-speed-curve.md)
- [人形硬件选型 Query](../queries/humanoid-hardware-selection.md)
- [分离架构](./humanoid-actuator-102-split-architecture.md)
- [产业与成本](./humanoid-hardware-101-supply-chain-economics.md)

## 参考来源

- [wechat_human_five_humanoid_actuator_102.md](../../sources/blogs/wechat_human_five_humanoid_actuator_102.md)
- [wechat_humanoid_actuator_102_2026-06-02.md](../../sources/raw/wechat_humanoid_actuator_102_2026-06-02.md)
