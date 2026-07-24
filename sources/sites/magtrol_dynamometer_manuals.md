# Magtrol 测功机手册与电机测试软件（官方资料）

> 来源归档（ingest）

- **标题：** Magtrol Hysteresis / Eddy-current / Powder Dynamometers + M-TEST
- **类型：** site / manual（厂商官方用户手册与软件说明）
- **手册索引：** <https://www.magtrol.com/manuals/>
- **关键 PDF：**
  - HD/ED 磁滞测功机：<https://www.magtrol.com/wp-content/uploads/hdmanual.pdf>
  - WB32 高速涡流：<https://www.magtrol.com/wp-content/uploads/wb32manual.pdf>
  - WB/PB 涡流/磁粉：见 manuals 页 WB/PB 条目
  - M-TEST 7：<https://www.magtrol.com/wp-content/uploads/mtest7.pdf>
- **入库日期：** 2026-07-24
- **一句话说明：** 工业测功机的经典一手手册：三类吸收制动原理、扭矩–转速–功率选型边界，以及用 M-TEST 扫出电机性能曲线的软件能力。
- **代码：** 不适用（商业封闭硬件/软件；手册与规格公开）
- **沉淀到 wiki：** [motor-dynamometer](../../wiki/concepts/motor-dynamometer.md)

## 为什么值得保留

- 「测功机」一词在实验室常被泛化；Magtrol 手册把 **磁滞 / 涡流 / 磁粉** 的扭矩–转速特性与散热功率限制写清楚，是选型第一手依据。
- 手册中的功率公式与「连续 vs ≤5 min 间歇」额定曲线，直接解释为何峰值扭矩台架测得了、连续区却烧制动器。

## 三类吸收制动（HD 手册归纳）

| 类型 | 系列 | 扭矩–转速特征 | 典型用途边界 |
|------|------|---------------|--------------|
| 磁滞 Hysteresis | HD / ED | 扭矩与转速近似无关；可到 **堵转** | 低–中功率（手册称间歇最高约 14 kW 量级）；全斜坡 TN |
| 涡流 Eddy-current | WB | 扭矩随转速上升，额定转速附近达峰值 | 高速；水冷可高连续功率 |
| 磁粉 Powder | PB | **零速** 即可达额定扭矩 | 低速大扭矩 |

选型顺序（手册明确）：**最大扭矩（含堵转/失步）→ 最大机械功率（散热）→ 最高安全转速（轻载）**。最高转速额定 **不等于** 可在该转速施加全扭矩。

## 功率与热

SI 机械功率（手册）：

\[
P[\mathrm{W}] = T[\mathrm{N\cdot m}] \cdot n[\mathrm{min^{-1}}] \cdot 1.047\times 10^{-1}
\]

测功机制动系统吸收的功率全部变为热；额定分 **连续** 与 **短时（常 ≤5 min）** 两条包络。磁滞机冷却可为自然对流、压缩空气或专用风机。

## 磁滞制动原理（摘要）

- 极结构（pole structure）与特种钢转子/拖杯 **无机械接触**；励磁线圈在气隙建立磁通后，转子受磁滞约束 → **无摩擦** 加载。
- 因此可做从空载到 locked rotor 的完整斜坡，适合扫 TN、堵转力矩与低速特性。

## M-TEST 7（软件公开说明摘要）

与可编程测功机控制器联用，可做：

| 模式 | 作用 |
|------|------|
| Ramp | 斜坡加载；可含惯量修正、空载/堵转外推 |
| Curve | 多工作点曲线（速/扭矩/电流/输入输出功率等） |
| Manual | 面板或屏幕手动控负载，PC 采集 |
| Pass/Fail | 与用户阈值比对 |
| Coast / Overload-to-trip | 断电滑行角度；过载至热保护跳闸 |

曲线数据可表格/图形导出，是工业侧生成 **TN / 效率相关曲线** 的常见路径。

## 开源 / 复现状态

- **确认未开源**：硬件与 M-TEST 为商业产品；公开的是用户手册与规格 PDF。
- 实验室低成本四象限替代见 [`../repos/odrive_based_electric_motor_dynamometer.md`](../repos/odrive_based_electric_motor_dynamometer.md)。

## 对 wiki 的映射

- [电机测功机（Dynamometer）](../../wiki/concepts/motor-dynamometer.md)
- [电机转矩-转速曲线（TN）](../../wiki/concepts/motor-torque-speed-curve.md)

## 相关一手索引

- [motor_dynamometer_primary_refs.md](motor_dynamometer_primary_refs.md)
