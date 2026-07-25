# 开源 QDD / 力矩关节执行器学习路线（策展整理）

> 来源归档

- **标题：** 真正值得参考的开源关节执行器项目（完整关节 vs 电机本体开源）
- **类型：** personal（维护者策展 / 学习路线整理）
- **入库日期：** 2026-07-25
- **一句话说明：** 把开源腿足/人形相关执行器分成「成品电机 + 开源减速器/结构/驱动」与「电机本体也开源」两类，给出优先学习清单与驱动器配套路线。
- **沉淀到 wiki：** [wiki/comparisons/open-source-qdd-actuator-projects.md](../../wiki/comparisons/open-source-qdd-actuator-projects.md)

---

## 两类项目

1. **完整开源关节执行器**：电机通常采购现成无刷外转子；减速器、结构、驱动与控制开源。
2. **电机本体也开源**：含定转子、绕组或转子设计，更适合从零学力矩电机，但成熟度通常较低。

现实路线：先用成品定子/外转子电机，自己设计转子、减速器、驱动器与关节结构；不要第一步就从硅钢片模具开始。真正同时公开电磁设计、加工图纸、驱动 PCB、控制固件与可靠性测试的项目非常少。

## 优先项目清单（策展摘录）

| 优先级 | 项目 | 入口 | 类别 | 学习重点 |
|--------|------|------|------|----------|
| 1 | Open Robot Actuator Hardware（ODRI） | https://github.com/open-dynamic-robot-initiative/open_robot_actuator_hardware | 完整关节 | 结构/行星或皮带减速、驱动 PCB、双编码器、电流/力矩环、CAN/以太网、装配与测试 |
| 2 | Berkeley Humanoid Lite | https://github.com/HybridRobotics/Berkeley-Humanoid-Lite | 完整关节 + 整机 | ~15:1 摆线、外转子电机参数→关节映射、电流/位置/速度环、Isaac Lab→实机 |
| 3 | Internal Cycloidal Actuator | https://github.com/aaedmusa/Internal-Cycloidal-Actuator | 电机本体+减速器一体 | 外转子电磁、绕组、中空定子内嵌双摆线、气隙半径与力矩密度 |
| 4 | OpenTorque Actuator | https://github.com/G-Levine/OpenTorque-Actuator | 完整关节（原型） | 大尺寸外转子航模电机 + 低减速同步带 + VESC，快速做 QDD 样机 |
| 5 | Stanford Doggo | https://github.com/Nate711/StanfordDoggoProject | 完整关节（四足） | QDD 同步带、ODrive 电流控制、KV/峰值电流/减速比权衡、跳跃冲击 |
| 6 | 3D Printed Open-Source Actuators | https://arxiv.org/abs/2202.12395 | 教材型论文 | 热限制、连续/峰值力矩、效率、背隙、42 万步态循环后性能 |
| 7 | Cycloidal QDD Actuator | https://github.com/JeongSeoJin/quasi-direct-drive-actuator | 减速器侧重 | 双摆线盘 180° 相位、低背隙、与 BHL 对照 |
| 8 | Ironless Rotor Cycloidal Planetary | https://github.com/CKraft11/Ironless-QDD-Actuator · https://cadenkraft.com/ironless-cycloidal-planetary-actuator/ | 低成本原型 | BOM&lt;$75、静态保持 ~30 N·m；须区分静态保持≠连续动态力矩 |

## 驱动器配套

| 项目 | 入口 | 适合学什么 |
|------|------|------------|
| Moteus | https://github.com/mjbots/moteus | 驱动 PCB、固件、FOC、编码器、CAN-FD、位置/速度/力矩 |
| Tinymovr | https://github.com/tinymovr/Tinymovr（现 `motionlayer/Tinymovr`） | 小型驱动原理图/PCB/固件、绝对编码器、CAN、Python 上位机 |
| VESC | https://github.com/vedderb/bldc · https://github.com/vedderb/bldc-hardware | 大电流功率级与 FOC；非专为高频关节设计 |
| SimpleFOC | https://github.com/simplefoc/Arduino-FOC | Clarke/Park、dq 电流、编码器对齐教学；低功率原型 |

## 建议学习阶段

| 阶段 | 项目 | 目标 |
|------|------|------|
| 1 | SimpleFOC | 理解 FOC 与编码器 |
| 2 | Moteus / Tinymovr | 关节驱动 PCB 与固件 |
| 3 | OpenTorque | 做出第一个低减速比关节 |
| 4 | ODRI open_robot_actuator_hardware | 成熟力控执行器体系 |
| 5 | Berkeley Humanoid Lite | 执行器如何装进人形并接 RL |
| 6 | Internal Cycloidal Actuator | 外转子电机 + 减速器一体 |
| 7 | 自研 | 电磁、机械、驱动、热联合设计 |

最值得优先下载的三个：**ODRI 执行器硬件**、**Berkeley Humanoid Lite**、**Internal Cycloidal Actuator**。

## 关键局限（策展强调）

- ODRI：电机本体一般采购现成外转子，不含完整电磁设计。
- BHL：官方指出 3D 打印摆线在高性能运动中偏脆弱，后续或改成品关节；适合学习验证，不宜原样用于重型人形。
- Internal Cycloidal：个人原型，缺大量冲击/寿命/热循环工业验证。
- Ironless：~30 N·m 是**静态保持**，不能直接当作连续动态、额定或冲击力矩。
- 论文 2202.12395：散热结构可使热限制下可用力矩接近提升一倍——提醒「峰值力矩」之外必须看热、效率、寿命与背隙。

## 对 wiki 的映射

- 对比主页：[open-source-qdd-actuator-projects](../../wiki/comparisons/open-source-qdd-actuator-projects.md)
- 电磁设计完整度（互补）：[open-source-torque-motor-em-design](../../wiki/comparisons/open-source-torque-motor-em-design.md)
- 纵深路线：[depth-torque-motor-design](../../roadmap/depth-torque-motor-design.md)
- 既有实体：[odri-solo-and-bolt](../../wiki/entities/odri-solo-and-bolt.md)、[berkeley-humanoid-lite](../../wiki/entities/berkeley-humanoid-lite.md)、[stanford-doggo-and-pupper](../../wiki/entities/stanford-doggo-and-pupper.md)、[simplefoc](../../wiki/entities/simplefoc.md)
