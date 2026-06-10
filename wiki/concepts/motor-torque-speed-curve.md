---
type: concept
tags: [motor, actuator, hardware, humanoid, torque, datasheet]
status: complete
updated: 2026-06-10
related:
  - ./motor-torque-current-curve.md
  - ../comparisons/motor-em-simulation-software.md
  - ../overview/humanoid-actuator-102-thermal-and-control.md
  - ../overview/humanoid-actuator-102-decision-species.md
  - ../overview/humanoid-hardware-101-integrated-actuators.md
  - ./field-oriented-control.md
sources:
  - ../../sources/personal/motor_curves_and_em_simulation_faq.md
summary: "TN 曲线（转矩-转速曲线）描述电机在各转速下可输出力矩；恒转矩区与恒功率区的分界（基速）决定人形关节能否兼顾爆发与高速摆腿。"
---

# 电机转矩-转速曲线（TN 曲线）

**TN 曲线**（Torque-Speed Curve，转矩-转速曲线）以转速为横轴、输出转矩为纵轴，刻画电机/关节模组在全速域内的力矩能力边界；评估人形或腿足执行器时，它通常是 datasheet 与测试报告里**第一张要看的图**。

## 一句话定义

TN 曲线回答：「这个关节在某一转速下最多能出多大力矩」，并区分低速 **恒转矩** 与高速 **恒功率** 两段工作区。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| TN | Torque-Speed (curve) | 转矩-转速特性曲线 |
| PMSM | Permanent Magnet Synchronous Motor | 永磁同步电机，腿足关节常见类型 |
| BLDC | Brushless DC Motor | 无刷直流电机 |
| FOC | Field-Oriented Control | 无刷电机的磁场定向控制 |
| MTPA | Maximum Torque Per Ampere | 单位电流最大转矩控制策略 |

## 为什么重要

- **硬件选型**：峰值转矩决定起跳、蹲起、倒地起身与抗冲击能力；基速决定最高关节转速与摆腿速度；连续转矩决定行走、站立等**长时间**任务是否过热降额。
- **与 Actuator 102 指标对齐**：[决策与物种](../overview/humanoid-actuator-102-decision-species.md) 要求峰值:持续力矩 **≥3:1**；[热学与力矩控制](../overview/humanoid-actuator-102-thermal-and-control.md) 强调行走脉冲负载下持续区比宣传峰值更关键。
- **功率误判**：高力矩低速电机的 **机械功率** 未必高于中等力矩中速电机；跑步、快速摆腿等动态动作更依赖 **峰值功率** \(P \approx T\omega\)，不能只看峰值转矩标称值。

## 核心结构/机制

### 坐标与典型分区

| 分区 | 物理含义 | 转矩行为 | 功率行为 |
|------|----------|----------|----------|
| **恒转矩区** | 电流达上限、磁链未饱和 | 近似恒定 | \(P = T\omega\) 随转速上升 |
| **恒功率区** | 超过基速，母线电压与弱磁限制 | 随转速反比下降 | 近似恒定 |

基速（Base Speed）是恒转矩区与恒功率区的分界转速。

### 功率换算（工程常用）

- 角速度形式：\(P = T\omega\)（\(P\)：W，\(T\)：Nm，\(\omega\)：rad/s）
- 转速 rpm 形式：\(P_{\mathrm{kW}} = \dfrac{T \cdot n}{9550}\)（\(n\)：rpm）

**例**：120 Nm @ 100 rpm → 约 1.26 kW；50 Nm @ 3000 rpm → 约 15.7 kW。前者力矩更大，后者输出功率远高于前者。

### 测试报告中的配套曲线

完整电机/关节评估常同时给出：

| 曲线 | 作用 |
|------|------|
| **TN 曲线** | 力矩-转速能力边界（本页） |
| **效率地图** | 各工作点效率，指导热与能耗 |
| **电流-转速** | 与驱动器限流、铜损相关 |
| **功率-转速** | 峰值/连续功率包络 |

电磁仿真侧如何生成 TN 曲线，见 [电机电磁仿真软件选型](../comparisons/motor-em-simulation-software.md)。

### 人形/腿足选型读图清单

| 读图项 | 工程含义 |
|--------|----------|
| **峰值转矩** | 爆发动作上限；常仅可持续 1–3 s |
| **连续转矩** | 行走、站立的真实工作区 |
| **基速** | 恒转矩结束点 → 最高可持续摆腿速度量级 |
| **峰值:持续比** | 与散热设计、谐波/减速器效率强相关 |

## 常见误区

| 误区 | 实际情况 |
|------|----------|
| 只看峰值转矩 | 连续转矩与温升决定能否「走完全程」 |
| 力矩大就等于动力强 | 动态任务看功率 \(T \times n\)，低速大扭矩可能功率不足 |
| TN 曲线可外推到任意负载 | 关节模组 TN 含减速器、热限与驱动器电流限幅，需看整机测试条件 |
| 恒功率区可长期满载 | 热极限与绝缘等级仍约束连续工作点 |

## 关联页面

- [电机转矩-电流曲线（TI 曲线）](./motor-torque-current-curve.md) — 从电流侧理解峰值/连续力矩与发热
- [Humanoid 执行器 102 · 热学与力矩控制](../overview/humanoid-actuator-102-thermal-and-control.md)
- [Humanoid 执行器 102 · 决策与物种](../overview/humanoid-actuator-102-decision-species.md)
- [Humanoid Hardware 101 · 集成执行器](../overview/humanoid-hardware-101-integrated-actuators.md)
- [磁场定向控制（FOC）](./field-oriented-control.md)

## 参考来源

- [motor_curves_and_em_simulation_faq.md](../../sources/personal/motor_curves_and_em_simulation_faq.md)

## 推荐继续阅读

- [Ansys Motor-CAD 产品页](https://www.ansys.com/products/electronics/ansys-motor-cad) — 快速生成 TN 曲线与效率地图
- [human five · Humanoid 执行器 入门 102](https://mp.weixin.qq.com/s/zinp6ulTorzfqmCR_HaI5A) — 腿足执行器峰值/持续与热学语境
