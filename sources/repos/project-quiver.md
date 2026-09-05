# Project Quiver（Arrow-air/project-quiver）

> 来源归档

- **标题：** Project Quiver
- **类型：** repo（开源多用途四旋翼机架 / 电子 / CAD / 装配文档）
- **链接：** https://github.com/Arrow-air/project-quiver
- **Stars / Forks：** 372 / 96（2026-09-05）
- **许可：** CERN-OHL-S-2.0（CERN Open Hardware Licence Version 2 — Strongly Reciprocal）
- **入库日期：** 2026-09-05
- **一句话说明：** Arrow Air 的 **25 kg MTOW** 开源模块化四旋翼：三接口热插拔载荷、四块可单独更换的定制 PCB、build123d 参数化 CAD + KiCad + 装配文档；飞控跑 **ArduPilot**（Pix32 V6），地面站走 QGroundControl / Mission Planner。
- **项目页：** [arrowair.com/quiver](https://arrowair.com/quiver)（归档：[arrowair-quiver.md](../sites/arrowair-quiver.md)）
- **代码：** https://github.com/Arrow-air/project-quiver
- **沉淀到 wiki：** [project-quiver](../../wiki/entities/project-quiver.md)、[multirotor-simulation-planning-control-stack](../../wiki/overview/multirotor-simulation-planning-control-stack.md)

---

## 开源状态（步骤 2.5）

**已开源（硬件全栈）** — 截至 2026-09-05 项目页与 GitHub 互指：

| 资产 | 位置 | 开放程度 |
|------|------|----------|
| 参数化 CAD | `src/quiver/`（[build123d](https://github.com/gumyr/build123d) Python 包） | 已开源；`python -m quiver.assembly` 导出整机 STEP |
| KiCad PCB | `src/pcb/`：Battery / Main / FC / Attachment Interface 四板 | 已开源；含 production gerber |
| 制造图纸 | `src/manufacturing/`（DXF / 切割表） | 已开源 |
| 3D 打印配置 | `src/printing/` | 已开源 |
| 装配与工程报告 | `docs/`（站点镜像 `arrowair.com/docs/quiver`） | 已开源 |
| 试飞日志 | `flight-test/` | 已开源（历史日志） |
| 附件赏金规格 | `task-grant-bounty/` | 已开源 |
| 飞控固件 | **不上本仓** — 使用上游 [ArduPilot](https://ardupilot.org/) 刷 Pix32 V6 | 固件开源在上游；本仓提供机架配置与文档 |

**不是** PX4 / Betaflight 固件仓，也 **不是** 仿真环境。本仓交付的是 **可制造机架 + 分布式电子 + 文档**。

---

## 核心定位

相对本库已有的 [PX4](px4_autopilot.md) / [Crazyflie](crazyflie_firmware.md) / [Betaflight](betaflight.md)：

- **Crazyflie** ≈ 27 g 室内微四轴（CRTP）
- **典型研究四旋翼 + PX4** ≈ 1–3 kg 级实验室机体
- **Quiver** = **25 kg MTOW / 5–8 kg 载荷** 的户外作业机架，跑 **ArduPilot**，目标是「可借出的 dev-kit + 社区附件生态」，而不是再写一套飞控。

Arrow 路线图把 Quiver 当作无人机起点：后续 Cargo UAV（Spearhead）、载人 eVTOL（Feather）、空中出租车（Volley）不在本仓范围内。

---

## 仓库结构

```
project-quiver/
├── docs/                    # 文档（站点托管）
├── src/
│   ├── quiver/              # CAD 装配 — build123d
│   ├── pcb/                 # 四块 KiCad 板
│   ├── printing/            # 打印配置
│   └── manufacturing/       # CNC / 激光 DXF
├── bom/                     # BOM
├── flight-test/             # 历史试飞日志
└── task-grant-bounty/       # 附件 / 赏金规格
```

### 四块定制板（`src/pcb/`）

| 目录 | BOM | KiCad 工程 | 职责 |
|------|-----|------------|------|
| `battery_pcb/` | 3311 | `Front_PCB` | 电源切换与保护、SOC / 温度、kill switch |
| `main_pcb/` | 3321 | `Quiver_PT3_Main_PCB` | 电源与数据分配枢纽 |
| `fc_pcb/` | 3331 | `Quiver_PT3_FC_PCB` | Pix32 V6 适配板 |
| `attach_pcb/` | 3341 | `QuiverAttachPCB` | 附件接口板 ×3（底 / 左 / 右）：稳压 + CAN + Ethernet |

### CAD 装配（`src/quiver/`）

BOM 分层：`1000` 机架结构（板 / 梁 / 起落架 / 电机臂）→ `2000` 支撑结构（附件接口 / 电池滑轨 / 座舱）→ `3000` 设备（推进 / 外设 / PCB / 电池）。`python -m quiver.assembly` 合成整机 STEP；可 `--show` 进 ocp-vscode。

---

## 平台规格（项目页 + README）

| 项 | 值 |
|----|-----|
| MTOW | 25 kg |
| 载荷 | 5–8 kg |
| 悬停续航 | 25–31 min（Tattu 14S 30 Ah + 四套 Hobbywing XRotor X6 Plus） |
| 定位 | 双 GNSS RTK（Holybro DroneCAN H-RTK F9P 或 Here4）+ Mateksys M9N-G4-3100 备份 |
| 高度 | Nanoradar NRA15 雷达高度计 |
| 避障（当前 dev-kit） | 360° LiDAR + 前视 / 下视雷达 |
| 飞控 | Pix32 V6 + ArduPilot |
| 伴机 | 可选 Raspberry Pi（Ethernet + CAN） |
| 结构 | 激光切割铝板 + 碳管电机臂 / 起落架（可折叠装箱） |
| 当前量产设计 | PT3 / Dev-kit；美 8 + 德 4 架 DAO 资助 |

原型迭代：PT1（Pixhawk 6X、单板、无防水）→ PT2（Matek H743、定制主板）→ PT3（四板、三接口）→ Dev-kit（防水、可拆起落架、360 LiDAR）。

---

## 对 wiki 的映射

- 机架实体：[project-quiver](../../wiki/entities/project-quiver.md)
- 多旋翼栈总览（补「ArduPilot 机架」层）：[multirotor-simulation-planning-control-stack](../../wiki/overview/multirotor-simulation-planning-control-stack.md)
- 飞控对照：[px4_autopilot.md](px4_autopilot.md)（PX4，非本仓固件）
- 伴机 API：[mavsdk.md](mavsdk.md)（MAVLink，兼容 ArduPilot）
- 尺度对照：[crazyflie_firmware.md](crazyflie_firmware.md)
- DroneCAN 外设：[cia_dronecan_uavcan.md](../sites/cia_dronecan_uavcan.md)
