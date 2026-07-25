# Ironless Axial Flux Motor（Caden Kraft Part 1）

> 来源归档

- **标题：** Designing a Coreless Axial Flux Motor (Part 1) / Ironless Axial Flux Motor
- **类型：** blog
- **作者：** Caden Kraft（CKraft11）
- **链接：** https://cadenkraft.com/designing-a-coreless-axial-flux-motor-part-1/
- **入库日期：** 2026-07-25
- **一句话说明：** DIY 无铁芯轴向磁通电机：Halbach 转子替代铁芯聚磁；按 Batzel 等文献估算匝数（12 极 / 18 线圈、约 14 匝/线圈）；3D 打印结构 + LCR 相电阻校验；**文内未列专用 CAD/代码仓**。
- **开源状态：** **未开源**（截至入库日项目页/博文未列 GitHub CAD 或仿真仓；仅博文叙述、公式与装配照片）
- **项目页归档：** [cadenkraft_coreless_axial_flux_motor_part1.md](../sites/cadenkraft_coreless_axial_flux_motor_part1.md)
- **沉淀到 wiki：** [cadenkraft-ironless-axial-flux-motor](../../wiki/entities/cadenkraft-ironless-axial-flux-motor.md)

---

## 文中要点（归纳，非全文）

### 为何轴向 / 无铁芯

- **轴向气隙近似二维**：可用垫片调气隙，不必每次重做转子径向配合。
- **磁钢**：矩形条形钕铁硼便宜可得，避免径向弯曲定制磁钢。
- **无铁芯线圈**：公寓级制造无法做硅钢/铁氧体叠片；用 **Halbach**（磁极每 90° 转向）替代铁芯塑形主磁通；名义 12 极实际用 **24** 块磁钢。

### 规格与匝数估算（作者表）

| 参数 | 值 |
|------|-----|
| 母线电压 | 24 V |
| 基速 | 700 rpm |
| 额定电流 | 7 A (rms) |
| 线径 | 0.65 mm |
| 极数 / 线圈 | 12 / 18（Y 接；高绕组因数） |
| \(r_o\) / \(r_i\) | 65 mm / 35 mm（\(\alpha \approx 0.5\)） |
| 目标气隙磁密量级 | 文中由 0.5 T 推到均值约 0.785 T |
| 每相匝数 → 每线圈 | \(N_{ph}\approx 82\) → 约 **14 匝/线圈**（6 线圈/相） |

公式链路：转子磁钢覆盖面积 → \(\alpha=r_i/r_o\) → 气隙磁密 → \(e_{ph}=N_{ph}A_{coil}\omega_e B_m\) → 反解匝数；\(\omega_e\) 由 700 rpm 与 6 对极得到。

### 机械与装配

- 主轴承 **62×40×12 mm** 滚子轴承（相对早期 608 轴承更刚）。
- 打印：PLA（一般件）+ PETG（定子基底，耐温）；磁钢 **36×6×6 mm**；手机磁力计辨北极以排 Halbach。
- LCR（相间）：电阻约 **699–708 mΩ**，电感约 **56–58 μH**。
- 已首次通电旋转；**无测功机转矩数据**（作者明确留作后续）。

### 参考文献（文内）

- T. Batzel, A. Skraba, R. D. Massi, “Design and Test of an Ironless Axial Flux Permanent Magnet Machine using Halbach Array,” IAJC-ISAM, 2014.

### 谱系位置

- 前作：作者「Creating a 3D Printed Brushless Motor」。
- 后续：双摆线执行器（同站 2024-02）在此电机上加摆线减速；再后续 **Ironless QDD**（径向 10010 + Halbach + 摆线—行星）把 Halbach/无铁芯经验迁到更完整开源仓。

## 对 wiki 的映射

- [Caden Kraft Ironless Axial Flux Motor](../../wiki/entities/cadenkraft-ironless-axial-flux-motor.md)
- [Ironless QDD Actuator](../../wiki/entities/ironless-qdd-actuator.md)（后续完整开源关节）
- [PCB Motor](../../wiki/entities/pcb-motor.md) · [axfluxmdo](../../wiki/entities/axfluxmdo.md)
- [开源力矩电机电磁设计完整度对比](../../wiki/comparisons/open-source-torque-motor-em-design.md)
