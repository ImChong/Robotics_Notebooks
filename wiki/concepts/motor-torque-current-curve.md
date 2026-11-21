---
type: concept
tags: [motor, actuator, hardware, humanoid, torque, current, efficiency]
status: complete
updated: 2026-11-21
original:
  - ../../sources/papers/modern_robotics_ch8_actuators.md
  - ../../sources/sites/simplefoc_documentation.md
  - ../../sources/repos/odrive_based_electric_motor_dynamometer.md
related:
  - joint-motor-rated-current-overload.md
  - ./motor-torque-speed-curve.md
  - ./field-oriented-control.md
  - ../formalizations/field-oriented-control-derivation.md
  - ../overview/humanoid-actuator-102-thermal-and-control.md
  - ../overview/humanoid-actuator-102-decision-species.md
  - ../comparisons/motor-em-simulation-software.md
  - ../queries/actuator-drive-chain-selection-loop.md
sources:
  - ../../sources/personal/motor_curves_and_em_simulation_faq.md
summary: "TI 曲线（转矩-电流曲线）给出转矩与相电流的线性关系及饱和区；力矩常数 Kt（Nm/A）直接决定同样转矩下的铜损与温升压力。"
---

# 电机转矩-电流曲线（TI 曲线）

**TI 曲线**（Torque-Current Curve，转矩-电流曲线）描述电机输出转矩与驱动电流之间的关系。对永磁同步电机（PMSM）和无刷电机（BLDC），它在理想区近似直线，斜率即 **力矩常数** \(K_t\)。

## 一句话定义

TI 曲线回答：「要输出这么多力矩，需要多大电流」——从而判断效率、发热与驱动器选型是否合理。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| TI | Torque-Current (curve) | 转矩-电流特性曲线 |
| PMSM | Permanent Magnet Synchronous Motor | 永磁同步电机 |
| BLDC | Brushless DC Motor | 无刷直流电机 |
| FOC | Field-Oriented Control | 无刷电机的磁场定向控制 |
| MTPA | Maximum Torque Per Ampere | 单位电流最大转矩控制策略 |

## 为什么重要

- **发热与连续能力**：铜损 \(P_{\mathrm{loss}} = I^2 R\)；电流翻倍 → 损耗约 **4 倍**。很多腿足平台「跑不快」并非力矩不够，而是电流过大导致过热降额。
- **与 TN 曲线互补**：[TN 曲线](./motor-torque-speed-curve.md) 给速度维度的能力包络；TI 曲线解释这些力矩是「用多少安培换来的」。
- **系统辨识**：从实测 TI 可反推 \(K_t\)、连续/峰值转矩，比单独相信 datasheet 标称峰值更可靠。

## 核心原理与结构

### 理想线性区

在 [FOC](./field-oriented-control.md) 控制下，\(q\) 轴电流主要产生力矩：

\[
T = K_t \, I_q
\]

| 符号 | 含义 |
|------|------|
| \(T\) | 输出转矩（Nm） |
| \(I_q\) | \(q\) 轴电流（A） |
| \(K_t\) | 力矩常数（Nm/A） |

**例**：\(K_t = 0.5\) Nm/A → 40 A 对应 20 Nm；若另一电机 \(K_t = 0.8\) Nm/A，则 25 A 即可 20 Nm，铜损显著更低。

### 高电流饱和区

实测曲线在高电流段常变平，可能原因：

| 因素 | 表现 |
|------|------|
| 磁饱和 | 继续加电流，转矩增益下降 |
| 温升降额 | 热保护限制有效电流 |
| 驱动器限流 | 硬件或软件电流天花板 |
| MTPA/弱磁策略变化 | 高转速区 \(i_d, i_q\) 分配改变 |

### 从 TI 反推工作点

若 datasheet 或测试给出：

- 连续电流 \(I_{\mathrm{cont}}\)、峰值电流 \(I_{\mathrm{peak}}\)
- 力矩常数 \(K_t\)

则近似：

\[
T_{\mathrm{cont}} \approx K_t \cdot I_{\mathrm{cont}}, \quad T_{\mathrm{peak}} \approx K_t \cdot I_{\mathrm{peak}}
\]

这比单独标注「峰值 48 Nm」更能看出电流代价，并与 [热学与力矩控制](../overview/humanoid-actuator-102-thermal-and-control.md) 中的峰值:持续目标对照。

## 工程实践

- **静态力臂测 $K_t$ 与温漂修正**：利用测功机或固定力臂称重传感器标定不同 $i_q$ 阶跃下的实测 $\tau$ 拟合 $K_t$（例如 [ODrive 测功机](../../sources/repos/odrive_based_electric_motor_dynamometer.md)）；钕铁硼剩磁温漂约 $-0.11\%/^\circ\mathrm{C}$，无位置/力传感器纯电流力控在 60 K 温升下可带来 ~6%–7% 力矩误差。
- **开环估算的 $R(T)$ 与 $K_v$ 校核**：仅靠电阻+反电势估算电流（无实测电流闭环）时，铜阻按 $R(T) = R_{20}[1 + 0.004(T-20^\circ\mathrm{C})]$ 随温度漂移，SimpleFOC 文档建议实配置 $K_v$ 可在手册值基础上留 10%–20% 修正系数，高精度动态场景仍须使用实测相电流的 FOC 闭环。
- **以 $I_{\mathrm{rms}}$ 校核连续发热**：依据完整负载谱计算 $I_{\mathrm{rms}} = \sqrt{\sum I_i^2 t_i / \sum t_i}$，密封关节按自由空气额定电流的 0.65–0.75 打折，防止高电流区铜损平方放大与磁钢高温不可逆退磁。

## 常见误区或局限

| 误区 | 实际情况 |
|------|----------|
| 峰值电流 × \(K_t\) 可任意维持 | 峰值电流对应短时热极限，连续区由 \(I_{\mathrm{cont}}\) 决定 |
| \(K_t\) 越大越好 | 过大常伴随低转速高力矩设计，峰值功率未必最优 |
| TI 直线段可无限外推 | 饱和与温升使高电流区斜率下降 |
| 仿真 TI 等于整机 TI | 关节模组含减速器摩擦、温升与驱动器限流，需台架验证 |

## 关联页面
- [关节模组：额定电流小于负载电流的后果](joint-motor-rated-current-overload.md)

- [电机转矩-转速曲线（TN 曲线）](./motor-torque-speed-curve.md)
- [电机测功机（Dynamometer）](./motor-dynamometer.md) — 静态扭矩与 \(K_t\) 的台架测法
- [磁场定向控制（FOC）](./field-oriented-control.md)
- [Humanoid 执行器 102 · 热学与力矩控制](../overview/humanoid-actuator-102-thermal-and-control.md)
- [电机电磁仿真软件选型](../comparisons/motor-em-simulation-software.md)
- [执行器驱动链选型闭环知识链](../queries/actuator-drive-chain-selection-loop.md) — 力矩–电流曲线是②层 FOC 力矩标定要面对的标称非线性背景

## 参考来源

- [motor_curves_and_em_simulation_faq.md](../../sources/personal/motor_curves_and_em_simulation_faq.md)
- [Modern Robotics Ch. 8.9 Actuators](../../sources/papers/modern_robotics_ch8_actuators.md) — 线性 $\tau = k_t I$ 与连续电流热限区
- [SimpleFOC 官方文档](../../sources/sites/simplefoc_documentation.md) — `voltage_torque_control.md` 中参数敏感性与真实电流闭环对比
- [ODrive 开源电力测功机](../../sources/repos/odrive_based_electric_motor_dynamometer.md) — load-cell 静态 $K_t$ 扫测脚本
- [motor_dynamometer_primary_refs.md](../../sources/sites/motor_dynamometer_primary_refs.md)

## 推荐继续阅读

- [SimpleFOC 文档](https://docs.simplefoc.com/) — 开源 FOC 栈中 \(K_t\) 与电流环实践
- [电机驱动器底软通信协议总览](../overview/motor-drive-firmware-bus-protocols.md) — 上层力矩指令如何落到电流环
