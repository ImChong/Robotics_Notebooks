# ukmars/ukmarsbot

> 来源归档

- **标题：** UKMARSBOT
- **类型：** repo（英国 Micromouse 协会入门多用途机器人平台）
- **组织：** UK Micromouse and Robotics Society（[ukmars](https://github.com/ukmars)）
- **链接：** https://github.com/ukmars/ukmarsbot
- **协会站点：** https://ukmars.org/（[站点归档](../sites/ukmars-org.md)）
- **官方从零教程视频：** https://www.youtube.com/watch?v=kLJjyr8uiFg（[课程归档](../courses/ukmarsbot_getting_started_youtube.md)）
- **许可：** MIT
- **星标（截至 2026-07-27）：** ~130
- **入库日期：** 2026-07-27
- **一句话说明：** 低成本、可外购件的入门多用途机器人：循线 / 沿墙 / 直线加速 / 迷你相扑；默认 Arduino Nano，模块化传感器与 Gerber/BOM 齐全。
- **开源状态：** **已开源** — 硬件（Gerber/原理图 PDF/BOM）、机械 STL、文档齐全；迷宫求解软件在姊妹仓。
- **项目页归档：** [ukmars-org.md](../sites/ukmars-org.md)
- **沉淀到 wiki：** [UKMARSBOT](../../wiki/entities/ukmarsbot.md)、[Micromouse](../../wiki/concepts/micromouse.md)

---

## 为什么值得保留

- **Micromouse 学习主入口**：教程与竞赛生态完整，协会站点 + GitHub + 月度线上会。
- **多赛种一台机**：同一底盘可做循线、沿墙、drag race、迷你相扑，再升级迷宫求解。
- **后续平台**：GEMINI（Pico / MicroPython，三层板）延续 UKMARSBOT 设计原则。

## 配套仓库（同组织，截至 2026-07-27）

| 仓库 | 角色 |
|------|------|
| [ukmars/ukmarsbot](https://github.com/ukmars/ukmarsbot) | 硬件主仓：ECAD / Gerber / 机械 / 文档 |
| [ukmars/ukmarsbot-examples](https://github.com/ukmars/ukmarsbot-examples) | 入门示例 |
| [ukmars/mazerunner-core](https://github.com/ukmars/mazerunner-core) | 经典迷宫探索与回起点核心固件（PlatformIO / Arduino） |
| [ukmars/ukmarsbot-mazerunner](https://github.com/ukmars/ukmarsbot-mazerunner) | 旧版迷宫代码（deprecated） |
| [ukmars/gemini](https://github.com/ukmars/gemini) | GEMINI：Pico + MicroPython 新平台 |
| [ukmars/maze-building](https://github.com/ukmars/maze-building) | 自建迷宫说明 |
| [ukmars/turn-tuner](https://github.com/ukmars/turn-tuner) | 转弯调参可视化 |

## 硬件要点

| 项 | 说明 |
|----|------|
| 处理器 | 默认 **Arduino Nano**；插座可换 STM32 / Pico 等变体 |
| 底盘约束 | 核心约 **小于 100×100 mm**（加传感器后增大） |
| 驱动 | 差速轮椅式；常用 N20 减速电机 + 编码器 |
| 通信 | HC-05/06 蓝牙串口；V1.1+ 可边连蓝牙边烧录 |
| ECAD | 早期 Eagle 7；后续倾向 KiCad；对外交付 Gerber zip + PDF + BOM |
| 电池 | 常见 8.4/9 V PP3 可充；LiPo 需保护板 |

## 开源核查（2026-07-27）

| 项 | 结论 |
|----|------|
| 硬件主仓 | **已开源**（MIT） |
| 软件示例 / 迷宫核心 | **已开源**（姊妹仓） |
| 协会站点 | [ukmars.org](https://ukmars.org/) 与 GitHub 互指 |
| GEMINI | **已开源**（[ukmars/gemini](https://github.com/ukmars/gemini)） |

## 对 wiki 的映射

- [UKMARSBOT](../../wiki/entities/ukmarsbot.md)
- [Micromouse](../../wiki/concepts/micromouse.md)
- [WolfieMouse](../../wiki/entities/wolfiemouse.md)
- [MuSHR](../../wiki/entities/mushr.md)（同属低成本教育移动平台对照）
- [PID Control](../../wiki/methods/pid-control.md)
