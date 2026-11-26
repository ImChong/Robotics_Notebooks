---
type: entity
tags: [hardware, actuator, qdd, open-source, cycloidal, bldc, foc]
status: complete
updated: 2026-11-18
related:
  - ../comparisons/open-source-qdd-actuator-projects.md
  - ./berkeley-humanoid-lite.md
  - ./internal-cycloidal-actuator.md
  - ./moteus.md
  - ./ironless-qdd-actuator.md
  - ./odri-solo-and-bolt.md
  - ../../roadmap/depth-torque-motor-design.md
  - ../queries/actuator-drive-chain-selection-loop.md
sources:
  - ../../sources/repos/quasi_direct_drive_actuator.md
  - ../../sources/personal/open_source_qdd_actuator_learning_curator.md
summary: "JeongSeoJin 开源双摆线 QDD：大直径 pancake BLDC（36N42P）+ 10:1 双盘 180° 相位摆线 + moteus-c1；峰值约 8.8 N·m / 22 rad/s @24V；Onshape CAD，适合与 BHL 摆线关节对照学平衡与紧凑集成。"
---

# Cycloidal Quasi-Direct Drive Actuator（Jeong 双摆线 QDD）

## 一句话定义

**Cycloidal Quasi-Direct Drive Actuator**（[JeongSeoJin/quasi-direct-drive-actuator](https://github.com/JeongSeoJin/quasi-direct-drive-actuator)）是面向动态腿足的开源 **QDD** 执行器：大直径 pancake 无刷电机 + **10:1** 内嵌双摆线盘（**180°** 相位差）+ [moteus-c1](./moteus.md) FOC，强调扭矩密度、紧凑性与可反驱透明性。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| QDD | Quasi-Direct Drive | 准直驱，低减速比、高背驱动性的作动方案 |
| FOC | Field-Oriented Control | 无刷电机的磁场定向控制 |
| BLDC | Brushless DC Motor | 无刷直流电机 |
| CAD | Computer-Aided Design | 计算机辅助设计；本项目主用 Onshape |
| HRI | Human–Robot Interaction | 人机交互；低阻抗关节利于接触安全 |

## 为什么重要

- 明确用 **双摆线盘 180° 相位** 对消偏心轴引起的径向力与振动——学摆线 QDD 机械平衡的好样本。
- 与 [Berkeley Humanoid Lite](./berkeley-humanoid-lite.md)（打印摆线 + 成品电机）和 [Internal Cycloidal](./internal-cycloidal-actuator.md)（自制外转子 + 内嵌摆线）形成三角对照。
- 从作者早期小航模电机方案演进到大直径定子，直接演示「气隙半径平方律」对 QDD 扭矩密度的工程含义。

## 核心信息

| 项 | 内容（README 摘录） |
|----|---------------------|
| 减速 | **10:1** 双盘摆线，两盘相差 **180°** |
| 电机 | 自制/改装 frameless；**8110** 大直径定子；**36N42P**；N52 磁钢；气隙约 **0.5 mm** |
| 结构 | Onshape CAD；原型齿轮/轴/转子为 **CNC 铝**（目标仍指向可打印低成本版） |
| 驱动 | **moteus-c1**（10–51 V，峰值相电流约 20 A） |
| 性能 | 最大力矩约 **8.8 N·m**；最大速度约 **22 rad/s**；标称约 **12.6 rad/s @ 24 V** |
| 开源 | **已开源**（CAD / 文档 / 资产）；GitHub API 未标 SPDX，使用前核验 |

## 核心原理

```mermaid
flowchart TB
  stator["大直径 8110 定子\n36N42P 手绕"]
  rotor["转子 + N52\n气隙 ~0.5mm"]
  cyc["双摆线盘\n180° 相位 / 10:1"]
  out["输出端"]
  moteus["moteus-c1 FOC"]
  stator --- rotor
  rotor --> cyc --> out
  moteus --> stator
```

- **为何 QDD：** 高减速带来高机械阻抗与反射惯量，不利冲击与力透明；3:1–10:1 区间保留反驱与本体感知力矩（电流估计）。
- **为何大直径电机：** \(\tau \propto r_g^2\)，增大气隙半径比加长叠厚更划算；定子中空便于内嵌减速器。
- **为何摆线：** 相对谐波抗冲击更好；相对 3D 打印行星，载荷由多齿瓣分担，齿根应力更友好。

## 源码运行时序图

以下时序图依据仓库 `JeongSeoJin/quasi-direct-drive-actuator`（`main` 分支）中核对到的 ESP32 + DRV8302 + AS5600 SimpleFOC 控制固件源码绘制，核对路径：`README.md`、`esp-and-arduino-control/C-QDD_impedance_control.ino`、`esp-and-arduino-control/C-QDD_angle_control.ino`、`esp-and-arduino-control/C-QDD_open_loop_control.ino`（注：主原型硬件集成的商业 `moteus-c1` 控制器固件位于上游 mjbots 闭源/外部生态，本仓可运行控制代码为 `esp-and-arduino-control/` 下的三套 `.ino` 示例）。

```mermaid
sequenceDiagram
  autonumber
  participant Serial as "串口终端 / Commander"
  participant Sketch as "C-QDD_impedance_control.ino"
  participant Enc as "MagneticSensorI2C (AS5600)"
  participant CS as "LowsideCurrentSense (DRV8302)"
  participant Motor as "BLDCMotor(21) / SimpleFOC"
  participant Driver as "BLDCDriver3PWM (ESP32 PWM)"

  Sketch->>Enc: setup(): encoder.init() -> motor.linkSensor(&encoder)
  Sketch->>Driver: 配置 M_OC/M_PWM/OC_ADJ/OC_GAIN 引脚并 driver.init()
  Sketch->>CS: cs.linkDriver(&driver), cs.init(), gain_a/b/c *= -1
  Sketch->>Motor: 设 TorqueControlType::foc_current 与 MotionControlType::torque
  Sketch->>Motor: motor.init() -> motor.initFOC() (d/q 轴对齐)
  Sketch->>Serial: command.add('R'/'K'/'C', doRefAngle/doKp/doCd)

  loop 控制主循环 loop()
    Sketch->>Motor: motor.loopFOC()
    Motor->>Enc: 读取转子机械角度与角速度
    Motor->>CS: 采样三相低侧相电流并变换至 d/q 轴
    Motor->>Driver: 内部 i_q/i_d PI 闭环更新 SpaceVector PWM
    Sketch->>Sketch: 读取 shaft_angle / shaft_velocity 计算 Iq_ref = Kp*(theta_desired - theta) - Cd*dtheta
    Sketch->>Sketch: 计算输出力矩估计 output_torque = 10 * Kt * motor.current.q
    Sketch->>Motor: 设置 motor.target = Iq_ref 并调用 motor.move()
    Sketch->>Serial: motor.monitor() 与 command.run() 解析新指令
  end
```

## 工程实践

| 学习点 | 做法 |
|--------|------|
| 双盘平衡 | 对照 Onshape：两盘 180°、偏心输入与输出机构 |
| 与 BHL | BHL ~15:1 打印摆线进整机；本项目单关节 10:1 + 自制电机 |
| 与 Internal Cycloidal | 同为 36N42P + 内嵌摆线叙事；本项目驱动用 moteus，彼用 ODrive S1 |
| 验证 | README 用 load-cell 做 1–10 N·m 力矩估计试验 |

## 局限与风险（作者自述 + 策展）

- **无转子背铁 / 非 Halbach：** 漏磁，峰值力矩偏弱，尚不足高动态整机。
- **手绕铜填充率低：** 限流与扭矩密度。
- 原型铝件验证设计；完全打印版与下一轮磁路改进仍在规划。
- 个人原型，缺工业级寿命/热循环公开数据。

## 关联页面

- [Joint Actuator Parameter Identification（关节执行器参数辨识）](../methods/joint-actuator-parameter-identification.md) — 负载传感器力矩试验可与反射惯量、摩擦等关节表征一起核对。
- [MIT 关节阻抗模式参数整定](../methods/mit-joint-impedance-mode-tuning.md) — 反驱关节的增益和前馈应按输出端响应整定，驱动协议与力矩估计误差另行确认。

- [开源 QDD 执行器项目对比](../comparisons/open-source-qdd-actuator-projects.md)
- [Berkeley Humanoid Lite](./berkeley-humanoid-lite.md)
- [Internal Cycloidal Actuator](./internal-cycloidal-actuator.md)
- [Ironless QDD Actuator](./ironless-qdd-actuator.md)
- [moteus](./moteus.md)
- [执行器驱动链选型闭环](../queries/actuator-drive-chain-selection-loop.md)

## 参考来源

- [sources/repos/quasi_direct_drive_actuator.md](../../sources/repos/quasi_direct_drive_actuator.md)
- [开源 QDD 执行器学习策展](../../sources/personal/open_source_qdd_actuator_learning_curator.md)
- 上游 README：<https://github.com/JeongSeoJin/quasi-direct-drive-actuator>

## 推荐继续阅读

- 仓库 CAD / `cad-designs.md` / `off-the-shelf-list.md`
- Sensinger 摆线优化文献（README References [6][8]）
