# Development of Variable Chain Motor with Shape and Speed-Torque Characteristics Variability and Its Application to a Humanoid（IROS 2025）

- **标题：** Development of Variable Chain Motor with Shape and Speed-Torque Characteristics Variability and Its Application to a Humanoid
- **类型：** paper（会议）
- **会议：** IEEE/RSJ IROS 2025，Hangzhou，2025-10-19 — 2025-10-25
- **DOI：** <https://doi.org/10.1109/iros60139.2025.11246199>
- **页码：** pp. 21102–21109（researchr 索引为 article 21102）
- **机构：** 东京大学（The University of Tokyo）— Hiromi Tada, Jin Hirai, Takuma Hiraoka, Masanori Konishi, Tomoya Himeno, Kunio Kojima, Kei Okada
- **入库日期：** 2026-09-25
- **一句话说明：** 提出 **Variable Chain Motor（VC motor）**：兼具 **形状可变形** 与 **速度–扭矩特性可切换** 的电动作动器；专用电路切换绕组串并联以扩展输出包络，并在人形肘关节演示高速高负载能力。

## 摘要级要点

- **动机：** 人形等细长构型里，作动器与传动占用空间小，传统变传动增包络方案在 **重量与体积** 上不利。
- **形状可变性（shape variability）：** 作动器可在运动中改变外形，例如跨相邻连杆布置并随关节旋转变形，在框架约束下实现 **更密集电机排布** → 更高输出扭矩。
- **速度–扭矩可变性：** 通过 **专用电路** 在运行中切换绕组连接（串 / 并），在不显著增尺寸重量的情况下 **扩大速度与扭矩范围**。
- **验证：** 测输出扭矩与效率；应用于人形 **肘关节**，展示高速与高负载操作。

## 核心论文摘录（MVP）

### 1) VC motor 机电结构

- **摘录要点：** 「链式」多单元电机 +  deformable 布置；电气模式切换改变等效电机常数，从而改变 speed–torque 曲线。
- **对 wiki 的映射：**
  - [可变链电机（VC motor）](../../wiki/entities/variable-chain-motor.md) — 硬件定义与模式切换机制。

### 2) 人形肘关节应用

- **摘录要点：** 在空间受限关节上同时利用形状可变与模式切换，支撑动态臂运动。
- **对 wiki 的映射：**
  - [IROS 2026 动态甩臂轨迹优化](../../wiki/entities/paper-iros26-vc-motor-dynamic-arm-swing.md) — 后续工作在 TO 中显式利用模式切换。

## 参考

- DOI：<https://doi.org/10.1109/iros60139.2025.11246199>
- 更早 Robomec 2025 相关：**Chain Motor**（形状可变，尚无 speed–torque 切换）— <https://doi.org/10.1299/jsmermd.2025.1p1-r04>
