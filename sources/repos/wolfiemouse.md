# kbumsik/WolfieMouse

> 来源归档

- **标题：** WolfieMouse
- **类型：** repo（竞赛级 Micromouse 全栈：固件 / 算法 / PCB / 调试工具）
- **作者：** Bumsik Kim（[kbumsik](https://github.com/kbumsik)）
- **链接：** https://github.com/kbumsik/WolfieMouse
- **Topics：** `ieee` · `kicad` · `maze` · `micromouse` · `pcb` · `robotics` · `stm32`
- **语言：** C（主）· C++ · Python · ARM 汇编
- **星标（截至 2026-07-27）：** ~74
- **许可：** `firmware` / `simulation` 标注为 GPLv2.1 相关条款；`firmware/lib` 内 FreeRTOS / CMSIS / STM32F4 HAL 遵循各自许可；仓根未见统一 SPDX
- **入库日期：** 2026-07-27
- **一句话说明：** IEEE Region 1 Micromouse 参赛鼠：迷宫求解算法、STM32 底层驱动、KiCad PCB、Python 传感器抓取/绘图与桌面仿真。
- **开源状态：** **已开源** — 固件、仿真、硬件设计与工具目录均公开；硬件灵感来自 Green Ye / Project Futura。
- **获奖：** 2018 IEEE Region 1 Special Mention；2019 IEEE Region 1 第 3 名
- **沉淀到 wiki：** [WolfieMouse](../../wiki/entities/wolfiemouse.md)、[Micromouse](../../wiki/concepts/micromouse.md)

---

## 为什么值得保留

- **竞赛级全栈参考**：算法（C++）+ 驱动（C/ASM）+ KiCad PCB + 传感器 log 工具，覆盖从迷宫搜索到高速跑常见工程面。
- **可复现工具链**：GNU Arm Embedded Toolchain、OpenOCD、Makefile；另提供 Vagrant 环境降低入门摩擦。
- **与教学平台对照**：相对 [UKMARSBOT](ukmarsbot.md)（Arduino 入门）更偏 STM32F4 / FreeRTOS 竞赛栈。

## 目录结构（仓内）

| 目录 | 内容 |
|------|------|
| `firmware/` | 机载程序：算法 + 硬件驱动 + 第三方库 |
| `simulation/` | 桌面迷宫算法仿真（终端可视化：M/D/S） |
| `hardware/` | KiCad 原理图 / PCB |
| `tools/` | 传感器数据捕获与 Vagrant 脚本 |
| `doc/` | What-is-Micromouse、Get-started、设计文档 |

## 开源核查（2026-07-27）

| 项 | 结论 |
|----|------|
| 主仓 | **已开源**（公开 clone；固件/仿真有许可说明） |
| PCB | **已开源**（KiCad；灵感自 Project Futura） |
| 项目页 | 无独立 `*.github.io`；文档在仓内 `doc/` |
| 活跃度 | 最近推送约 2021-11；仍可作为竞赛架构教材 |

## 对 wiki 的映射

- [WolfieMouse](../../wiki/entities/wolfiemouse.md)
- [Micromouse](../../wiki/concepts/micromouse.md)
- [UKMARSBOT](../../wiki/entities/ukmarsbot.md)
- [KiCad](../../wiki/entities/kicad.md)
- [A*](../../wiki/methods/a-star.md)
- [PID Control](../../wiki/methods/pid-control.md)
