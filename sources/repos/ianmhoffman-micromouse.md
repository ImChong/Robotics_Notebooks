# IanMHoffman/Micromouse

> 来源归档

- **标题：** Ian Hoffman Micromouse
- **类型：** repo + 设计日志站点（硬件选型到软件笔记）
- **作者：** Ian M. Hoffman（[IanMHoffman](https://github.com/IanMHoffman)）
- **链接：** https://github.com/IanMHoffman/Micromouse
- **项目页 / 设计记录：** https://ianmhoffman.github.io/Micromouse/（[站点归档](../sites/ian-hoffman-micromouse-github-io.md)）
- **星标（截至 2026-07-27）：** ~4
- **许可：** 仓根未见 SPDX（入库日）
- **入库日期：** 2026-07-27
- **一句话说明：** 个人国际竞赛向 Micromouse 设计日志：经典 / 半尺寸规则、平滑转弯与对角线、STM32F405、VL6180X ToF、MPU-6500、自研 PCB 思路。
- **开源状态：** **部分 / 设计文档为主** — GitHub Pages 设计笔记公开；完整可复现 BOM/Gerber/固件以站点进展为准，不宜当作成品鼠仓库。
- **沉淀到 wiki：** [Micromouse](../../wiki/concepts/micromouse.md)

---

## 为什么值得保留

- **规则与目标特性写清楚**：经典 16×16（180 mm 格）vs 半尺寸 32×32（90 mm）；平滑转弯、已知格加速、对角线走 Z 形。
- **硬件选型决策过程**：MCU / ToF / IMU / 电机驱动取舍，适合作为竞赛鼠设计 checklist。
- **与 Green Ye 等经典技巧互指**：pivot vs smooth curve、已知格加速视频等。

## 开源核查（2026-07-27）

| 项 | 结论 |
|----|------|
| 设计站点 | **已公开** |
| 完整固件 / Gerber 交付 | **未作为成品套件宣称** — 站点定位为进行中设计记录 |
| 仓 | 公开但体量小（~4★）；以笔记价值为主 |

## 对 wiki 的映射

- [Micromouse](../../wiki/concepts/micromouse.md)
- [WolfieMouse](../../wiki/entities/wolfiemouse.md)
- [UKMARSBOT](../../wiki/entities/ukmarsbot.md)
