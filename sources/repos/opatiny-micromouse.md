# opatiny/micromouse（Algernon）

> 来源归档

- **标题：** Algernon — A custom micromouse robot
- **类型：** repo（Bachelor Thesis：PCB / 机械 / 文档；固件分仓）
- **作者：** Olivier Patiny（[opatiny](https://github.com/opatiny)）
- **链接：** https://github.com/opatiny/micromouse
- **项目演示视频：** https://youtu.be/nz4QlaSIkbY（[课程归档](../courses/opatiny_algernon_remote_debug_youtube.md)）
- **Topics：** `micromouse` · `mobile-robot` · `robot-hardware`
- **许可：** CERN-OHL-P-2.0（开源硬件许可）
- **星标（截至 2026-07-27）：** ~13
- **入库日期：** 2026-07-27
- **一句话说明：** 学士论文级自研 Micromouse：KiCad PCB、ESP32-S3、5×ToF + IMU、编码器差速、Wi‑Fi 远程调试网页；物理样机已组装测试。
- **开源状态：** **已开源（硬件主仓 + 固件/网页分仓）** — 主仓以电子/机械/文档为主；MCU 软件与调试网页另仓。
- **沉淀到 wiki：** [Micromouse](../../wiki/concepts/micromouse.md)

---

## 为什么值得保留

- **完整本科毕业设计路径**：电气 + 机械 + 制造 + 传感 bring-up + 里程计/轮速环 + 远程调试，适合对照竞赛鼠「缺什么才能上场」。
- **Wi‑Fi 远程调试**：TypeScript 调试页烧入 MCU，对嵌入式调试工具链有参考价值。
- **开源硬件许可明确**：CERN-OHL-P-2.0。

## 配套仓库

| 仓库 | 角色 |
|------|------|
| [opatiny/micromouse](https://github.com/opatiny/micromouse) | 主仓：KiCad、机械、迷宫、论文 PDF、Matlab |
| [opatiny/ms-software](https://github.com/opatiny/ms-software) | ESP32-S3 固件 |
| [opatiny/ms-webpage](https://github.com/opatiny/ms-webpage) | 远程调试网页（TypeScript → 构建上 MCU） |

## 硬件要点（论文摘要）

| 项 | 选型 |
|----|------|
| MCU | ESP32-S3 |
| 测距 | 5× ToF |
| 惯性 | 加速度计 |
| 驱动 | 2× 有刷 DC + 编码器 |
| 供电 | 2P1S LiPo |
| 结构 | PCB 为主要结构件；PETG 脚轮、柔性灯丝保险杠 |
| 软件进展 | 传感独立验证、基础里程计、轮速调节；迷宫高速求解非本项目重点 |

## 开源核查（2026-07-27）

| 项 | 结论 |
|----|------|
| 硬件 / 文档 | **已开源**（CERN-OHL-P-2.0） |
| 固件 / 调试页 | **已开源**（分仓 `ms-software` / `ms-webpage`） |
| 项目页 | 无独立站点；演示以 YouTube 为准 |
| 竞赛完备度 | 论文明确：传感与基础运动环已通；完整 Flood Fill 高速跑非交付重点 |

## 对 wiki 的映射

- [Micromouse](../../wiki/concepts/micromouse.md)
- [WolfieMouse](../../wiki/entities/wolfiemouse.md)（竞赛对照）
- [UKMARSBOT](../../wiki/entities/ukmarsbot.md)（入门对照）
- [KiCad](../../wiki/entities/kicad.md)
