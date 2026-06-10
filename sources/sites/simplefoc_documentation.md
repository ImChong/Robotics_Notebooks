# SimpleFOC 官方文档门户

> 来源归档（documentation site）

- **标题：** docs.simplefoc.com — Arduino SimpleFOC 文档站
- **类型：** site / course（结构化教程 + API）
- **链接：** <https://docs.simplefoc.com/>
- **镜像仓库：** <https://github.com/simplefoc/simplefoc.github.io>
- **入库日期：** 2026-05-26
- **一句话说明：** SimpleFOC 库的官方文档：安装、硬件接线、FOC 理论、运动/扭矩控制模式、PID 调参、电流采样对齐与多 MCU 移植说明。
- **关联 repo 来源：** [simplefoc_arduino_foc.md](../repos/simplefoc_arduino_foc.md)

---

## 为什么值得保留

- 官网 [simplefoc.com](https://simplefoc.com/) 偏营销与硬件商店入口；**技术细节以 docs 为准**。
- 文档与代码版本绑定（如 v2.4.0 release notes），避免仅依赖过时社区视频。

## 文档结构（一级导航）

| 区块 | 典型内容 |
|------|----------|
| Getting started | Arduino IDE / PlatformIO 安装、第一个 `loopFOC` 例程 |
| Hardware | 电机、驱动、传感器、MCU 支持列表与接线 |
| Theory corner | FOC 坐标变换、PID、低通滤波、对齐流程 |
| Motion / Torque control | `move()` vs `loopFOC()`、开环/闭环、电流模式 |
| Examples | 按板卡与传感器分类的 sketch |
| SimpleFOCBoards | Shield/Mini 原理图、制板、BOM |
| Alternative projects | Odrive、VESC 等对比表 |

## 核心摘录

### 控制双环约定

- `motor.loopFOC()`：扭矩环（FOC），典型 **1–10 kHz**。
- `motor.move()`：运动环（位置/速度/力矩目标），默认同频，可用 `motion_downsampling` 降频。
- 可通过 `motor.loopfoc_time_us` / `motor.move_time_us` 实测周期。

### v2.4 亮点（文档 release 摘要）

- 运动模式与扭矩模式任意组合（如 `velocity_openloop` + `foc_current`）。
- `estimated_current` 扭矩模式；STM32 多电机 ADC 电流采样；ESP32 定时/ADC 对齐修复。

## 对 wiki 的映射

- [wiki/entities/simplefoc.md](../../wiki/entities/simplefoc.md)
- [wiki/concepts/field-oriented-control.md](../../wiki/concepts/field-oriented-control.md)
- [wiki/formalizations/field-oriented-control-derivation.md](../../wiki/formalizations/field-oriented-control-derivation.md)

## 推荐继续阅读（外部）

- [community.simplefoc.com](https://community.simplefoc.com/) — 论坛与项目展示
