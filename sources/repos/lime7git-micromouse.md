# lime7git/micromouse

> 来源归档

- **标题：** lime7git/micromouse
- **类型：** repo（硕士论文：STM32 鼠 + EasyEDA PCB + Qt 迷宫仿真器）
- **作者：** [lime7git](https://github.com/lime7git)
- **链接：** https://github.com/lime7git/micromouse
- **Topics：** `micromouse` · `stm32` · `maze-algorithms` · `maze-simulator` · `bluetooth` · `hardware`
- **语言：** C++（主）· C
- **星标（截至 2026-07-27）：** ~60
- **许可：** 仓根未见 SPDX LICENSE 文件（入库日）
- **入库日期：** 2026-07-27
- **一句话说明：** Kielce 理工大学硕士课题：STM32 固件、EasyEDA 以 PCB 为底盘、自建小迷宫，以及 Qt 6 迷宫仿真器用于对比搜索算法。
- **开源状态：** **已开源（代码与硬件目录公开）** — 含 Android 伴侣应用、文档、硬件、仿真器、机载软件；许可文件缺失需二次确认使用条款。
- **沉淀到 wiki：** [Micromouse](../../wiki/concepts/micromouse.md)

---

## 为什么值得保留

- **仿真器先行**：Qt Maze Simulator 可在无真机前对比迷宫搜索算法——与 Webots / WolfieMouse 桌面仿真形成三角对照。
- **硬件栈差异**：EasyEDA + Keil uVision5 + STM32，相对 WolfieMouse（KiCad + Makefile/OpenOCD）是另一条常见创客路径。
- **多媒体演示**：README 挂多条仿真与真机 YouTube，便于快速建立直觉。

## 目录结构（仓内）

| 目录 | 内容 |
|------|------|
| `micromouse-software/` | STM32 机载软件（C / Keil） |
| `micromouse-hardware/` | EasyEDA PCB / 硬件设计 |
| `micromouse-maze-simulator/` | Qt 6 迷宫仿真器 |
| `micromouse-android-app/` | Android 伴侣应用 |
| `micromouse-docs/` | 文档 |

## 开源核查（2026-07-27）

| 项 | 结论 |
|----|------|
| 源码 / 硬件目录 | **公开可克隆** |
| LICENSE | **未发现根级 SPDX** — wiki 中写明「使用前自行确认许可」 |
| 项目页 | 无独立站点；演示视频在 README |

## 对 wiki 的映射

- [Micromouse](../../wiki/concepts/micromouse.md)
- [A*](../../wiki/methods/a-star.md)
- [WolfieMouse](../../wiki/entities/wolfiemouse.md)
- [KiCad](../../wiki/entities/kicad.md)（EDA 对照：本项目用 EasyEDA）
