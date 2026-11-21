---
type: overview
tags: [humanoid, actuator, thermal, torque-control, foc, category-hub]
status: complete
updated: 2026-11-21
summary: "Actuator 102 · 04 — 谐波低效率致热堆积；峰值与持续力矩比目标 3:1+；FOC 电流环 >20kHz、力矩带宽 50–100Hz、指令延迟 <1ms。"
related:
  - ../concepts/joint-module-shaft-thermal-expansion.md
  - ./humanoid-actuator-102-technology-map.md
  - ./humanoid-actuator-102-compliance-sensing.md
  - ./humanoid-hardware-101-power-compute-electronics.md
  - ../concepts/motor-torque-speed-curve.md
  - ../concepts/motor-torque-current-curve.md
sources:
  - ../../sources/blogs/wechat_human_five_humanoid_actuator_102.md
  - ../../sources/raw/wechat_humanoid_actuator_102_2026-06-02.md
original:
  - ../../sources/raw/wechat_humanoid_actuator_102_2026-06-02.md
---

# Actuator 102 · 04：热学与力矩控制

> **分类页**：**V 腿部的热学现实** + **VI 从 PWM 到力矩控制**。

## 一句话定义

人形腿部执行器的热学与控制体系要求在克服高阻力损耗热积聚（实现 $\ge 3:1$ 峰值/持续转矩比与受控温升）的同时，将底层驱动架构由开环 PWM 速度控制升级为高母线电压、低电感电机与 `>20 kHz` FOC 电流环及 `50–100 Hz` 力矩带宽的实时总线闭环。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| FOC | Field-Oriented Control | 无刷电机的磁场定向控制，解耦 $d/q$ 轴电流以实现快速线性力矩输出 |
| PWM | Pulse-Width Modulation | 脉宽调制，控制逆变器开关占空比的驱动技术 |
| EtherCAT | Ethernet for Control Automation Technology | 高实时性工业以太网现场总线，支撑微秒级多轴同步 |
| CAN | Controller Area Network | 关节模组常用的差分布线现场总线协议 |
| CAN-FD | CAN with Flexible Data-rate | CAN 的高波特率、大数据帧扩展版本 |

## 为什么重要

在人形机器人站立、下蹲、持物或高频率行走时，持续功率往往远低于瞬间冲击峰值，但**铜损发热（$I^2 R$）与齿轮啮合摩擦生热**会在封闭外壳内缓慢累加。一旦热边界失控触发退磁或驱动限流降额，策略将失去预期力矩导致摔倒；而在控制侧，如果不能将指令到实际力矩输出的延迟压入 `<1 ms` 窗口，强化学习或基于模型的落地阻抗控制根本来不及响应毫秒级触地碰撞。

## 核心原理与机理

### 腿部的热学现实（第 V 章）

- **谐波减速器与高阻损耗源**：谐波依赖**金属柔轮连续弹性变形摩擦**，加上极小的风冷散流面，导致连续重载工况下热积累速度远超工业外露电机。
- **脉冲负载 vs 静态蹲持**：平地行走时支撑相承担瞬态峰值、摆动相载荷近似为零；但屈膝半蹲、拎重物或慢速爬楼时会进入持续高电流铜损区间。
- **峰持比设计门槛**：高可用腿部模组需以**峰值转矩 : 持续转矩 $\ge 3:1$** 为设计目标（一般风冷粗放设计常仅 `~2:1`，要么瞬间峰值推不动、要么长时巡航过热降额）。评估手册时须对照 [TN 曲线](../concepts/motor-torque-speed-curve.md) 分清**宣传极限值**与**连续热稳态区**。
- **主动液冷边界**：采用液冷循流带走定子铜损与减速器热耗后，峰值与持续力矩比可进一步压低至 **2:1 甚至 1.5:1**（连续出力大幅提升），代价是管路、密封与泵重。
- 电流侧发热可通过 [TI 曲线](../concepts/motor-torque-current-curve.md) 与力矩常数 $K_t$ 直接估算：转矩常数下降或铁芯饱和后，所需电流非线性陡增，铜损 $I^2 R$ 呈平方级恶化。

### 从 PWM 到力矩控制（第 VI 章）

- 控制范式必须从传统电机速度/占空比控制，迁移到 **FOC $q$ 轴电流闭环 + 关节级力矩控制**。
- 文内 70 kg 级人形膝踝参考带宽目标：
  - **FOC 电流环**：**> 20 kHz**（保证电气响应速度紧跟脉冲参考信号）
  - **有效力矩环带宽**：**> 50–100 Hz**（$-3\,\mathrm{dB}$）
  - **指令到输出延迟**：**< 1 ms**
- 物理兑现要素：设计**低相电感绕组**（加速电流 di/dt 建立）、采用 **48–100 V 高压直流母线**（推高高速反电势余量）、最小化传动背隙与柔性延迟，并部署 **EtherCAT 或 CAN-FD**（保证多轴高帧率总线同步周期）。

## 工程实践

- **温度采样布局必须双点以上**：至少在**定子绕组最热点（绕组端部埋线 NTC/PT100）**与 **MOSFET 功率管基板**同步采样，防止绕组先烧毁或磁钢不可逆退磁。
- **结合机构公差校核热膨胀**：高温下轴系与壳体膨胀速率差异会挤压轴承预紧与齿轮啮合间隙，需同步进行 [关节模组轴的径向热膨胀校核](../concepts/joint-module-shaft-thermal-expansion.md)。
- **在总线拓扑中显式分配带宽预算**：确保多轴（28 DoF+）在 1–2 kHz 全身控制周期内拥有确定性的 PDO 收发抖动，而非被廉价传统 CAN（1 Mbps，单帧 8 字节）挤爆帧周期。

## 常见误区或局限

- **误区：只要电机标称效率 90%+ 关节模组就不怎么发热** — 低速高转矩（蹲姿或静态支撑）恰恰落在电机最低效率区，主要能量损耗全变成了绕组 $I^2 R$ 焦耳热；高比减速器本身的啮合损耗也会随负载同步转化为热量。
- **误区：电流环 20 kHz 就等于关节拥有几百赫兹的力矩带宽** — 编码器延迟、母线滤波电容、通信周期（CAN-FD/EtherCAT 延迟）以及传动刚度/柔轮扭振会把真实整机力矩带宽急剧拉低至 `50–100 Hz` 甚至更低。

## 关联页面

- [Actuator 102 技术地图](./humanoid-actuator-102-technology-map.md)
- [关节模组轴的径向热膨胀校核](../concepts/joint-module-shaft-thermal-expansion.md)
- [Actuator 102 · 05：柔顺与感知反馈](./humanoid-actuator-102-compliance-sensing.md)
- [Hardware 101 · 电力电子与算力](./humanoid-hardware-101-power-compute-electronics.md)
- [电机转矩-转速曲线（TN 曲线）](../concepts/motor-torque-speed-curve.md)
- [电机转矩-电流曲线（TI 曲线）](../concepts/motor-torque-current-curve.md)
- [旋转-直线分离架构](./humanoid-actuator-102-split-architecture.md)
- [决策与物种](./humanoid-actuator-102-decision-species.md)

## 推荐继续阅读

- [FOC 磁场定向控制推导](../formalizations/field-oriented-control-derivation.md) — Clarke/Park 变换与 $d/q$ 轴电流环推导
- [电机持续电流与过载倍率](../concepts/joint-motor-rated-current-overload.md) — 从热稳态视角拆解 $3:1$ 过载逻辑

## 参考来源

- [wechat_human_five_humanoid_actuator_102.md](../../sources/blogs/wechat_human_five_humanoid_actuator_102.md)
- [wechat_humanoid_actuator_102_2026-06-02.md](../../sources/raw/wechat_humanoid_actuator_102_2026-06-02.md)（第五、六章）
