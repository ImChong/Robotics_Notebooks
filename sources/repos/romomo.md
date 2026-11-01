# ROMOMO（RMM 关节模组公开资料）

> 来源归档

- **标题：** ROMOMO Open Materials — Public documentation, specifications, and test reports for RoboParty / ROMOMO joint modules
- **类型：** repo（仅文档 / 规格 / 测试报告）
- **机构：** 萝卜派对（RoboParty）
- **链接：** https://github.com/Roboparty/ROMOMO
- **入库日期：** 2026-10-27（核查快照：`master` @ `a9140df`，2026-10-14 "compress gif"）
- **一句话说明：** RMM6562 / RMM8463 / RMM10062 三款 48 V 关节模组的规格图纸、实验室测试报告、CAN-FD 协议手册、硬件介绍与上位机手册（PDF）。
- **沉淀到 wiki：** 是（[`wiki/entities/romomo.md`](../../wiki/entities/romomo.md)）
- **上级目录：** [RP1 总览仓库归档](roboparty_rp1.md)

## 开源状态

- **仅文档开放**：README 明确 "documentation and test materials only"，不含软件 / 固件 / 源码；代码另在各自仓库以 GPL-3.0-only 发布（README 声明，本次未逐仓核对）。上位机软件 RoboDrive 需去 roboparty.com 下载中心获取（上位机手册 §1）。
- **许可：** 文档、图片、测试资料 CC BY 4.0（`LICENSE`）；不授予商标、专利权。
- RP1 总览 README 将 ROMOMO 描述为 "Joint modules, firmware and host tools"，与仓库实际内容（仅 PDF + GIF，共 14 个文件）不一致，以仓库内容为准。

## 关键文件清单

| 路径 | 内容 |
|------|------|
| `10062/RMM10062-2EV1.0.pdf`、`10062/RMM10062-2E-关节模组测试报告-1013.pdf` | 大功率模组图纸 + 测试报告 |
| `6562/RMM6562-2EV1.0.pdf`、`6562/RMM6562-2E-关节模组测试报告-1013.pdf` | 小功率模组图纸 + 测试报告 |
| `8463/RMM8463-2EV1.0.pdf` | 中功率模组图纸（无测试报告） |
| `CANFD协议手册.pdf` | CAN FD 自定义协议 V0.4，46 页 |
| `RMM关节模组硬件介绍-V261013.pdf` | 驱动板硬件介绍 V1.0（2026.10） |
| `上位机使用手册.pdf` | RoboDrive 上位机手册 V1.0 |
| `关节模组Pre-DV_测试内容_V1.0.pdf`、`萝博派对关节模组测试用例v1.0.pdf` | Pre-DV 测试用例（P0/P1 分级，预期值与实测值待填） |
| `motor-exploded-cropped.gif` | 关节模组爆炸图 |

## 对 wiki 的映射

- [wiki/entities/romomo.md](../../wiki/entities/romomo.md)
- 关联：[roboparty-rp1](../../wiki/entities/roboparty-rp1.md)、[rp1-hardware](../../wiki/entities/rp1-hardware.md)、[roboto-origin](../../wiki/entities/roboto-origin.md)
