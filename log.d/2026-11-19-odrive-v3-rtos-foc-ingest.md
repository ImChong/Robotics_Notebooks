## [2026-11-19] ingest | ODrive v3 固件 0.5.6 RTOS/双轴 FOC 实时控制

- 从比特Z 2026-09-24 技术文章提炼 RTOS 与硬实时控制的两层架构，文章与官方版本锁定源码分别归档。
- 官方 `fw-v0.5.6` 核对 TIM8 sampling、ControlLoop ISR、Axis 任务与 8 kHz/125 μs 控制周期、超期双轴卸力、单目标梯形轨迹。
- 新建 `wiki/entities/odrive-v3-firmware-rtos-foc.md`，包含 Mermaid 控制流与源码运行时序图。
- 与 FOC 概念页及力矩电机设计 Stage 3 交叉链接；不将该实现外推至 ODrive 新一代产品，也不把静态源码分析误写为 WCET 实测。
