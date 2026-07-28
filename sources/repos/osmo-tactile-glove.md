# OSMO Tactile Glove（jessicayin/osmo_tactile_glove）

> 来源归档

- **标题：** OSMO Tactile Glove
- **类型：** repo / hardware
- **来源：** Meta FAIR、密歇根大学、宾夕法尼亚大学
- **链接：** <https://github.com/jessicayin/osmo_tactile_glove>
- **项目页：** <https://www.jessicayin.com/osmo_tactile_glove/>
- **论文：** <https://arxiv.org/abs/2512.08920>
- **许可：** 仓库未提供顶层 LICENSE（截至 2026-07-28）
- **入库日期：** 2026-07-28
- **一句话说明：** 可构建 OSMO 手套并运行触觉采集、手姿后处理、重定向、扩散策略训练与部署的官方仓库。
- **开源状态：** **源码与设计文件已公开，许可未明确**；示例数据下载脚本仍标 TODO。
- **沉淀到 wiki：** [`wiki/entities/paper-notebook-osmo-open-source-tactile-glove-for-human-to-robo.md`](../../wiki/entities/paper-notebook-osmo-open-source-tactile-glove-for-human-to-robo.md)

## 仓库概况（2026-07-28）

| 字段 | 值 |
|------|-----|
| 硬件 | `firmware/`、`hardware/`、`pcb/`；装配资料在 `website` 分支 |
| 环境 | `conda/osmo.yml`、`conda/osmo_kinematics.yml` |
| 数据处理 | HaMeR 关键点提取 → `construct_retarget_dataset.py` |
| 策略 | `glovedp/train.py train --output_name im_state_touch_dp` |
| 部署 | `glovedp/README.md`；策略训练与部署说明 |

## 对 wiki 的映射

- 项目页：[`osmo-tactile-glove.md`](../sites/osmo-tactile-glove.md)
- 论文来源：[`humanoid_pnb_osmo-open-source-tactile-glove-for-human-to-robo.md`](../papers/humanoid_pnb_osmo-open-source-tactile-glove-for-human-to-robo.md)
