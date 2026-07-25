# bgkatz/3phase_integrated

> 来源归档

- **标题：** 3phase_integrated — 3-phase motor controller with integrated position sensor
- **类型：** repo
- **作者：** Benjamin G. Katz（bgkatz）
- **链接：** https://github.com/bgkatz/3phase_integrated
- **许可：** MIT
- **星标（截至 2026-07-25）：** ~638
- **配套固件：**
  - mbed（旧）：https://os.mbed.com/users/benkatz/code/Hobbyking_Cheetah_Compact/
  - mbed + DRV8323（Mini Cheetah 向）：https://os.mbed.com/users/benkatz/code/HKC_MiniCheetah/
  - 非 mbed / STM32CubeIDE：https://github.com/bgkatz/motorcontrol（见 [bgkatz_motorcontrol](./bgkatz_motorcontrol.md)）
- **使用文档：** https://docs.google.com/document/d/1dzNVzblz6mqB3eZVEMyi2MtSngALHdgpTaDJIW_BpS4/edit
- **入库日期：** 2026-07-25
- **一句话说明：** Katz 模块化执行器的集成三相驱动 PCB（含磁编），对应 MIT 2018 硕士论文附录 A 的电机控制器硬件入口。
- **开源状态：** **已开源**（硬件设计；固件分仓）
- **关联论文：** [low_cost_modular_actuator_katz_mit_2018](../papers/low_cost_modular_actuator_katz_mit_2018.md)
- **沉淀到 wiki：** [paper-low-cost-modular-actuator-katz](../../wiki/entities/paper-low-cost-modular-actuator-katz.md)、[mit-mini-cheetah](../../wiki/entities/mit-mini-cheetah.md)

---

## 覆盖范围

| 模块 | 内容 |
|------|------|
| 硬件 | 三相逆变器 + 集成位置传感的电机驱动 PCB |
| 固件 | 见 mbed / [motorcontrol](./bgkatz_motorcontrol.md) |
| 总线 | CAN（与论文中菊花链执行器一致） |

论文中设计目标：约 24 V 标称、40 A 峰值相电流；FET 顶侧散热到铝壳；配合 AS5047P。

## 对 wiki 的映射

- [paper-low-cost-modular-actuator-katz](../../wiki/entities/paper-low-cost-modular-actuator-katz.md)
- [开源 QDD 执行器项目对比](../../wiki/comparisons/open-source-qdd-actuator-projects.md)
- [执行器驱动链选型闭环](../../wiki/queries/actuator-drive-chain-selection-loop.md)
