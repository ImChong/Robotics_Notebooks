# bgkatz/motorcontrol

> 来源归档

- **标题：** motorcontrol — motor controller firmware
- **类型：** repo
- **作者：** Benjamin G. Katz（bgkatz）
- **链接：** https://github.com/bgkatz/motorcontrol
- **许可：** MIT
- **星标（截至 2026-07-25）：** ~306
- **硬件目标：** https://github.com/bgkatz/3phase_integrated
- **工具链：** STM32CubeIDE（STM32F446）
- **入库日期：** 2026-07-25
- **一句话说明：** 面向 `3phase_integrated` 板的非 mbed 电机控制固件；论文附录原列 mbed 固件，本仓为后续可移植主线。
- **开源状态：** **已开源**
- **关联论文：** [low_cost_modular_actuator_katz_mit_2018](../papers/low_cost_modular_actuator_katz_mit_2018.md)
- **沉淀到 wiki：** [paper-low-cost-modular-actuator-katz](../../wiki/entities/paper-low-cost-modular-actuator-katz.md)

---

## 覆盖范围

| 模块 | 内容 |
|------|------|
| 固件 | FOC / 电流环等关节驱动逻辑（`Core/`，配置见 `hw_config.h`） |
| 可移植性 | README 称易于移植到相近硬件；目标兼容旧 mbed 行为 |

## 对 wiki 的映射

- [paper-low-cost-modular-actuator-katz](../../wiki/entities/paper-low-cost-modular-actuator-katz.md)
- [bgkatz_3phase_integrated](./bgkatz_3phase_integrated.md)
