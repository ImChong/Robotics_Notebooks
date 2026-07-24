# IEC 60034-2-1:2024 旋转电机损耗与效率试验方法

> 来源归档（ingest）

- **标题：** IEC 60034-2-1:2024 — Rotating electrical machines — Part 2-1: Standard methods for determining losses and efficiency from tests (excluding machines for traction vehicles)
- **类型：** standard（国际标准）
- **版本：** 第 3 版（2024），取代 2014 第 2 版
- **官方入口：** [IEC Webstore publication 67756](https://webstore.iec.ch/en/publication/67756)
- **入库日期：** 2026-07-24
- **一句话说明：** 规定旋转电机 **损耗与效率** 的标准试验方法；直接法依赖扭矩–转速机械测功，间接法做损耗分离，并对拖（back-to-back）给出总损耗测量路径。
- **沉淀到 wiki：** [motor-dynamometer](../../wiki/concepts/motor-dynamometer.md)

## 为什么值得保留

- 测功机在国际电机试验语境中的「法定用法」之一是 **直接测效率**：\(\eta = P_{\mathrm{mech}} / P_{\mathrm{elec}}\)，其中 \(P_{\mathrm{mech}} = T\omega\)。
- 人形关节厂常同时要看 **效率地图** 与 TN；IEC 方法说明「有扭矩传感器/测功机」与「无可靠机械测功」时该走哪条路径，避免把间接法结果当成直接法精度。

## 核心公开要点（Webstore / 标准样本摘要）

| 概念 | 含义 |
|------|------|
| Direct efficiency determination | 直接测输入电功率与输出机械功率 |
| Indirect / summation of losses | 分项损耗求和再算效率 |
| Dual-supply back-to-back | 两台同型机机械耦合，由电功率差求两机总损耗 |
| Single-supply back-to-back | 两机接同一电源系统的对拖变体 |
| No-load / locked-rotor 等 | 空载与堵转等经典试验工况 |

适用范围：IEC 60034-1 覆盖的直流、同步、感应电机（电网运行额定）；牵引车辆电机排除在本部分之外。第 3 版与 IEC 60034-2-2 / 2-3 在版式与要求上做了协调。

> 全文为 IEC 版权作品；本仓库不转载规范性条款，仅保留公开元数据与映射。

## 与测功机的工程对应

- **直接法** → 吸收式测功机或联轴扭矩传感器 + 转速测量 + 功率分析仪。
- **对拖法** → 与机器人实验室「电力测功机 / 伺服对拖」同构：一台速度环负载、一台力矩环被测（或反之）。
- 扭矩测量精度不足时，标准路径应切到间接法并如实声明方法与不确定度。

## 对 wiki 的映射

- [电机测功机（Dynamometer）](../../wiki/concepts/motor-dynamometer.md)
- [电机转矩-转速曲线（TN）](../../wiki/concepts/motor-torque-speed-curve.md)

## 相关一手索引

- [motor_dynamometer_primary_refs.md](motor_dynamometer_primary_refs.md)
