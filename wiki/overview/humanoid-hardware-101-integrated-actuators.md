---
type: overview
tags: [humanoid, hardware, actuator, electric, hydraulic, category-hub]
status: complete
updated: 2026-11-21
summary: "Humanoid Hardware 101 · 04 集成执行器 — 电动主导（线束+软件）；腿膝踝偏直线高力、肩等大转角用旋转；Atlas 由液压转电、部分灵巧手仍液压。"
related:
  - ./humanoid-actuator-102-technology-map.md
  - ./humanoid-actuator-102-split-architecture.md
  - ./humanoid-hardware-101-technology-map.md
  - ./humanoid-hardware-101-actuation-sensing-chain.md
  - ./humanoid-hardware-101-linear-transmission-bearings.md
  - ../entities/open-source-humanoid-hardware.md
  - ../concepts/motor-torque-speed-curve.md
  - ../concepts/joint-module-self-development-workflow.md
  - ../comparisons/motor-em-simulation-software.md
sources:
  - ../../sources/blogs/wechat_human_five_humanoid_hardware_101.md
  - ../../sources/raw/wechat_humanoid_hardware_101_2026-06-01.md
original:
  - ../../sources/raw/wechat_humanoid_hardware_101_2026-06-01.md
---

# Humanoid Hardware 101 · 04：集成执行器

> **分类页**：**执行器** 章 — 电机+减速+传感+驱动的一体化 **关节模块**（如擎天柱文内 ~28 个执行器量级）。

## 一句话定义

集成执行器（Integrated Actuator / Joint Module）是将无刷电机、机械减速机构、位置/力矩传感器与功率驱动 PCB 高度封装为一体的机电执行单元，在以约 28 个主关节为核心的人形机器人硬件平台中扮演将电能高效、可控地转化为关节机械能的基础部件。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| DoF | Degree of Freedom | 自由度，整机或局部机构独立的关节运动轴数量 |
| PCB | Printed Circuit Board | 印刷电路板，关节级常内置驱动器与总线通信接口 |
| TN | Torque–Speed Curve | 转矩-转速包络曲线，界定电机峰值与连续工作域 |
| FOC | Field-Oriented Control | 磁场定向控制，电动执行器实现精准力矩控制的主流电流闭环 |

## 为什么重要

执行器决定了人形机器人的上限力量、运动带宽、电池续航以及整机 BOM 成本重心（通用人形 BOM 中执行器占比常达 40%–50%+）。相比松散分立电机与工业电柜，将动力、传感与通信直接集成到关节内部，才能大幅削减穿关节线束复杂度，支撑全身高频力矩闭环控制，并成为电动架构全面取代早期液压人形的关键工业拐点。

## 核心原理与动力源对比

- **电动主导趋势**：全电驱布线简洁、无需漏油与高压油泵维护，且电流与电磁力矩具备天然线性映射，与现代控制/RL 软件栈完美契合；**液压**仍保留在极少数超高紧凑力量场景（部分科研手或旧代特种作业机），**气动**则局限在低精度开关或柔性仿生手指等细分领域。
- **形态分化**：腿部**膝踝**等需高推力与强抗冲击部位常采用滚珠/滚柱丝杠直线执行器；肩、髋外展、腰部等**大转角**关节则普遍采用行星或谐波旋转执行器。

### 三大动力源横向对比（原文核心结论）

| 评估维度 | 电动执行器 | 液压执行器 | 气动执行器 |
|----------|------------|------------|------------|
| 功率密度 | 中–高 | 最高（极紧凑高出力） | 低–中 |
| 可控性与精度 | 优（电流环线形响应） | 良（受油液可压缩性与阀滞后影响） | 差（气路高度非线性） |
| 能量转换效率 | **~90%**（峰值电机效率） | **30–60%**（溢流与泵耗大） | 极低（压气与节流损耗） |
| 系统维护与复杂度 | 低（双绞总线+直流母线） | 高（高压泵站、阀块、密封漏油风险） | 中（需外部气源或微型压缩机） |

### 演进与工程选型注记

- **Atlas 动力转向里程碑**：Boston Dynamics 液压版 Atlas 验证了极高动态敏捷极限，但新一代商业化 Atlas 彻底转向全电动集成旋转关节，印证量产可维护性与商业效率对液压路线的替代；Sanctuary 手部、Clone 肌肉仿生等仍保留微液压/气动探索特殊灵巧生态位。
- 选型评估需跨越多目标权衡：功率密度、力矩可控性、反向驱动能力、热效率、动态响应速度与**全系统维护复杂度**。
- 在评估具体电动模组器件时，[TN 曲线](../concepts/motor-torque-speed-curve.md) 是校验数据手册边界的第一视角；预研阶段的电磁与热场仿真工作流见 [电机电磁仿真软件选型](../comparisons/motor-em-simulation-software.md)。
- 自主立项研发一体化旋转模组（需求瀑布定义→传动构型→五件套集成→台架测试）见 [自研关节模组开发流程](../concepts/joint-module-self-development-workflow.md)；按关节负载分派电机拓扑见 [关节电机拓扑选型](../queries/humanoid-joint-motor-topology-selection.md)。

## 工程实践

- **以整机线束拓扑反推模组集成度**：将驱动板与双编码器整合入模组后端后，穿过肩髋大转角中空轴的只需高柔电源总线与通信差分对，大幅提升整机万次往复弯折可靠性。
- **旋转与直线模组按关节载荷分布**：避免“一种构型打天下”，需根据冲击峰力、摆角行程与远端惯量预算精确配置旋转行星 QDD、旋转谐波及直线滚柱丝杠。
- **台架实测校准手册水分**：采购或自研集成关节时，必须对照实际温升后的连续转矩降额曲线与反向驱动阻尼，而非只听信峰流短时宣传数值。

## 常见误区或局限

- **误区：液压功率密度最高所以大负载人形最终还得回液压** — 忽略了系统层总效率（30–60% 导致电池重、续航骤降）、高频高压管路泄漏维护灾难以及噪声问题；在量产商业场景下，高母线电动 + 高比减速的综合收益远胜液压。
- **误区：把驱动板放到关节外面能更好散热所以干脆都做中央电柜** — 中央集控方案虽缓解了关节端散热，但会将数十根粗三相电力线和敏感编码器反馈线穿过精密运动旋转关节，在机械可靠性与电磁抗扰（EMI）上付出沉重代价。

## 关联页面

- [小鹏 IRON 54 项专利技术地图](./xpeng-iron-54-patents-technology-map.md) — 肩/膝/手/腰/步态等 CN 专利独立节点
- [Humanoid 执行器 102 技术地图](./humanoid-actuator-102-technology-map.md) — 为何工业方案在腿上失效与三大物种分化
- [电机转矩-转速曲线（TN 曲线）](../concepts/motor-torque-speed-curve.md) — 峰值/连续/基速读图
- [电机电磁仿真软件选型](../comparisons/motor-em-simulation-software.md)
- [自研关节模组开发流程](../concepts/joint-module-self-development-workflow.md)
- [Humanoid Hardware 101 技术地图](./humanoid-hardware-101-technology-map.md)
- [传动与感知链](./humanoid-hardware-101-actuation-sensing-chain.md)
- [传感与末端执行器](./humanoid-hardware-101-sensing-end-effectors.md)
- [产业与成本地缘](./humanoid-hardware-101-supply-chain-economics.md)

## 推荐继续阅读

- [Actuator 102 · 02：旋转-直线分离架构](./humanoid-actuator-102-split-architecture.md) — 深入拆解电动架构内部直线与旋转的分流机制
- [Hardware 101 · 电力电子与算力](./humanoid-hardware-101-power-compute-electronics.md) — 关节驱动母线电压与整机能源流

## 参考来源

- [wechat_human_five_humanoid_hardware_101.md](../../sources/blogs/wechat_human_five_humanoid_hardware_101.md)
- [wechat_humanoid_hardware_101_2026-06-01.md](../../sources/raw/wechat_humanoid_hardware_101_2026-06-01.md)（第四章·执行器）
