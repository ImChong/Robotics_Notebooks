# google-deepmind/barkour_robot — Barkour 开源四足（硬件 / 固件 / 文档）

> 仓库来源归档（ingest）

- **类型：** repo / hardware / firmware / quadruped
- **URL：** <https://github.com/google-deepmind/barkour_robot>
- **许可：** 软件 **Apache-2.0**；其余材料 **CC BY-NC 4.0**（以仓库 LICENSE 声明为准）
- **入库日期：** 2026-05-18
- **一句话说明：** 公开 **Barkour 系列敏捷四足** 的 **CAD、PCBA、装配文档、EtherCAT 电机固件（Pigweed 基线）与主机侧示例**，README 同时给出 **OnShape CAD** 与 **MuJoCo Menagerie** 仿真入口；当前主推硬件代号为 **vB**，并保留论文期 **v0** 资产外链。

## 维护者整理的结构化入口（摘自 README）

| 主题 | 入口 |
|------|------|
| 上手指南 | `docs/getting_started.md` |
| 固件 | `docs/firmware.md`，代码在 `actuator/firmware/`（Pigweed） |
| EtherCAT | `docs/ethercat_config.md`、`docs/motor_control.md` |
| 硬件总览 | `docs/hardware_overview.md` |
| 整机装配 | `docs/full_barkour_robot_assembly.md` |
| 执行器装配 | `docs/actuator_assembly_and_setup.md` |
| BOM | `hardware/barkour_robot_bill_of_materials.csv` |

## CAD 与仿真（官方外链）

- **当前代（vB）CAD（OnShape / gdm）：** <https://gdm.onshape.com/documents/2587dbf423d784b45437b14a/v/b07bb08c3dc8b02bae21b866/e/bdcd5797385cee9e4f78dfef?aa=true>（固定头与可动头两版）
- **v0（论文期）CAD：** <https://gdm.onshape.com/documents/8bcc0544056aa5de830b6353/w/9f4df6916bccef9b9b882a52/e/9296b0bcf28e5f27569d4cdb>
- **v0 URDF 简化 CAD：** <https://deepmind.onshape.com/documents/bd3aaf26c384d7d058cee090/w/9bd0468bf4dae717e9b02f17/e/6151d1e161dfa46066201d62>
- **MuJoCo Menagerie 当前仿真模型：** `google_barkour_vb` — <https://github.com/google-deepmind/mujoco_menagerie/tree/main/google_barkour_vb>
- **Menagerie v0（障碍课与论文期机体）：** <https://github.com/google-deepmind/mujoco_menagerie/tree/main/google_barkour_v0>
- **障碍课评分脚本（Brax 实验路径）：** <https://github.com/google/brax/blob/main/brax/experimental/barkour/score_barkour.py>
- **MJX 教程：** <https://github.com/google-deepmind/mujoco/blob/main/mjx/tutorial.ipynb>

> 注：用户提供的泛域名 `https://cad.onshape.com/` 为 **OnShape 产品首页**；本仓库实际托管链接为 **gdm.onshape.com / deepmind.onshape.com** 文档 URL，见上表。

## 对 wiki 的映射

- [`wiki/entities/paper-barkour-quadruped-agility-benchmark.md`](../../wiki/entities/paper-barkour-quadruped-agility-benchmark.md)
- [`wiki/entities/mujoco.md`](../../wiki/entities/mujoco.md)

## 当前提炼状态

- [x] README 级入口梳理与版本（v0 / vB）脚注
