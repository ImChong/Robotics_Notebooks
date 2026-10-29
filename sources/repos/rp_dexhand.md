# rp_dexhand（RP Hand 五指灵巧手产品与开发资料）

> 来源归档

- **标题：** RP Hand — A compact five-finger dexterous hand for humanoid robots
- **类型：** repo（产品文档 + 资源清单；SDK 在另一仓库）
- **机构：** 萝卜派对（RoboParty）
- **链接：** https://github.com/Roboparty/rp_dexhand
- **入库日期：** 2026-10-27（核查快照：`main` @ `338b53f`，2026-10-19 PR #2 `docs/developer-experience-20261019`）
- **一句话说明：** RP Hand（11 机械自由度 / 6 主动轴 / 蜗轮蜗杆 / CAN-FD）的规格、接线、SDK 接入、ROS 2 Jazzy 说明与 Release 资源（上位机、手势演示、URDF/MJCF、CAD、协议表）。
- **沉淀到 wiki：** 是（[`wiki/entities/rp-hand.md`](../../wiki/entities/rp-hand.md)）
- **上级目录：** [RP1 总览仓库归档](roboparty_rp1.md)

## 开源状态

- **文档开放；SDK 与 ROS 包在 `roboparty_dexhand`**（本次未克隆该仓库，SDK 细节来自本仓文档对基线 `eb923ac` 的转述）。
- **许可未统一**：`LICENSES.md` 明确本仓原创文档无独立许可证；SDK 基线无独立许可证文件；`rphand_ros` 的 `package.xml` 声明 `Proprietary`；厂商应用 / 协议表 / 模型 / CAD 沿用各自条款。
- 大文件（上位机 205 MB、手势软件约 300 MB、CAD 约 45 MB）放 Release `resources-2026.10.05`，本仓只存 `manifest.json` / `SHA256SUMS.txt` / `validation.json`；本次未下载。

## 关键文件清单

| 路径 | 内容 |
|------|------|
| `docs/en/{hardware,sdk,quickstart,ros2,software,models,downloads,compatibility,troubleshooting}.md` | 英文指南（中文镜像在 `docs/*.md`） |
| `docs/maintenance.md` | 产品手册 revision 13 等来源版本表 |
| `resources/resources-2026.10.05/{manifest,validation}.json`、`SHA256SUMS.txt` | Release 资源清单与校验 |
| `resources/model-inspection-2026.10.19.json` | DH116S URDF/MJCF 静态结构检查记录 |
| `assets/{dimensions,pinout}.png`、`rp-hand.gif` | 尺寸图、引脚图、RP1 快换安装动图 |
| `tools/check_docs.py`、`tests/test_docs.py`、`.github/workflows/docs.yml` | 文档检查脚本与 CI |

## 对 wiki 的映射

- [wiki/entities/rp-hand.md](../../wiki/entities/rp-hand.md)
- 关联：[roboparty-rp1](../../wiki/entities/roboparty-rp1.md)、[fourier_dexhand_sdk 归档](fourier_dexhand_sdk.md)（同类对照）
