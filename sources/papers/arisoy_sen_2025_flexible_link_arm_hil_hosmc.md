# A Hardware-in-the-Loop Simulation Case Study of High-Order Sliding Mode Control for a Flexible-Link Robotic Arm

> 来源归档

- **标题：** A Hardware-in-the-Loop Simulation Case Study of High-Order Sliding Mode Control for a Flexible-Link Robotic Arm
- **类型：** paper（开放获取期刊）
- **作者：** Aydemir Arisoy, Deniz Kavala Sen
- **机构：** Istanbul Yeni Yuzyıl University（电气）；Bursa Technical University（机械）
- **期刊：** Applied Sciences（MDPI），2025, 15(19), 10484
- **发表日期：** 2025-09-28
- **DOI：** https://doi.org/10.3390/app151910484
- **文章页：** https://www.mdpi.com/2076-3417/15/19/10484
- **入库日期：** 2026-09-06
- **一句话说明：** **1-DOF 柔性连杆直驱臂** 的 HIL 案例：真实执行器/传感器 + **MATLAB/Simulink** 实时植物模型，对比 **HOSMC** 与经典 SMC 的轨迹跟踪、振动抑制与力矩平滑性。
- **开源状态：** **不适用** — 开放获取论文；未列官方代码仓。

---

## 核心贡献（摘录）

1. **HIL 平台：** 物理直驱执行器与传感器接入回路，植物动力学在 Simulink 实时环境仿真，无需全尺寸样机即可在 realistic 条件下测控算法。
2. **控制对象：** 柔性连杆臂轻量高效但 **非线性 + 易振**，传统 SMC 有抖振；**HOSMC** 在有限时间内驱动到降阶滑模面，执行器指令更平滑、机械应力更低。
3. **实验结论：** HOSMC 相对 SMC 有更快收敛、更小残余振动、更平滑控制信号；高频轨迹下因控制策略阻尼效应可能出现小稳态误差。
4. **方法论价值：** HIL 作为 **理论控制设计 ↔ 工程实现** 之间的桥梁，对先进机器人控制系统具有可复用范式。

## 对 wiki 的映射

- 概念页：[Hardware-in-the-Loop](../../wiki/concepts/hardware-in-the-loop.md)
