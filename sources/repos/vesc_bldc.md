# VESC（bldc + bldc-hardware）

> 来源归档

- **标题：** VESC motor control firmware & hardware
- **类型：** repo（双仓）
- **作者：** vedderb（Benjamin Vedder）
- **固件：** https://github.com/vedderb/bldc
- **硬件：** https://github.com/vedderb/bldc-hardware
- **星标（截至 2026-07-25）：** 固件 ~3293 · 硬件 ~1339
- **入库日期：** 2026-07-25
- **一句话说明：** 完整 FOC 固件与驱动硬件设计，适合学大电流功率级；最初非专为高频机器人关节控制设计。
- **开源状态：** **已开源**（许可需按仓库文件确认；API 对 license 字段常为 null）
- **沉淀到 wiki：** [vesc](../../wiki/entities/vesc.md)、[opentorque-actuator](../../wiki/entities/opentorque-actuator.md)、[open-source-qdd-actuator-projects](../../wiki/comparisons/open-source-qdd-actuator-projects.md)

---

## 定位

- 强项：功率级、FOC、社区工具链。
- 弱项（相对关节驱动）：控制周期与协议生态未必按「人形 1 kHz 力矩环 + CAN-FD」优化；常见于航模/滑板/OpenTorque 类 DIY 关节。

## 对 wiki 的映射

- [VESC](../../wiki/entities/vesc.md)