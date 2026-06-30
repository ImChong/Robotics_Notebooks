# EN02-OP

> 来源归档

- **标题：** EN02-OP — Westwood Robotics 开源三指末端执行器
- **类型：** repo
- **来源：** Westwood Robotics
- **链接：** <https://github.com/Westwood-Robotics/EN02-OP>
- **官网：** <https://www.westwoodrobotics.io/>
- **License：** GPL-3.0
- **Stars / Forks：** ~55 / 5（2026-06-30 快照）
- **入库日期：** 2026-06-30
- **一句话说明：** 面向通用机械臂/人形末端的 **开源三指夹爪**：7 DoF、Dynamixel XL/XC-330 舵机、3D 打印 + 单块 EN02-PWR 电源/通信板；官方宣称 DIY 成本可低至约 **200 USD**（全 XL-330 配置）。
- **沉淀到 wiki：** 是 → [`wiki/entities/en02-op.md`](../../wiki/entities/en02-op.md)

---

## 为什么值得保留

- **低成本可复刻末端**：与全尺寸仿人五指手（Allegro、RUKA-v2 等）不同，EN02-OP 走 **三指 + 7 DoF** 的「够用且便宜」路线，适合桌面臂、移动操作与人形 **非高灵巧** 抓取实验。
- **制造资料完整**：公开 STEP 总成、分件 3D 打印目录、机械 BOM、装配 PDF、电源板 Gerber/BOM/原理图，并强调 **掌座需按法兰改孔** 再打印。
- **电源与总线友好**：EN02-PWR 将 **20–36 V** 机器人母线降压到 **5 V/10A** 供舵机，TTL 级 Dynamixel 菊花链；可选 BOM 改 **12 V** 输出以适配其他 XC330 型号。
- **整机语境**：Westwood 将其作为全尺寸人形 **THEMIS**（Westwood 产品线，勿与 Caltech **Themis** 混淆）的末端执行器开源。

## 核心设计（README 摘录）

| 维度 | 说明 |
|------|------|
| **机构** | 三指；**7 DoF**；强调结构简单、姿态可调以适应多类物体 |
| **执行器** | 默认 **Dynamixel XC-330-M288-T**（约 $98.89/个，5 V）；轻载可用 **XL-330**（约 $26.29/个）降本 |
| **制造** | 多数零件可 **3D 打印**；`/Design/Parts_to_Print/CF_Reinforced/` 内件建议 **碳纤维/玻纤增强**（如 Bambu PAHT-CF） |
| **掌座** | `EN02-PALM_BASE` 底部圆截面安装面 **故意无预置螺孔**，需按目标法兰改 STEP 后再打印 |
| **电子** | 单 PCBA **EN02-PWR**：降压 + 手指菊花链；可购预装板约 $35/ea |
| **控制** | **无专用上位机**；按 [Robotis XC330 手册](https://emanual.robotis.com/docs/en/dxl/x/xc330-m288/) 用 Dynamixel SDK 直接控关节 |
| **成本** | 官方估算：全 XL-330 + 通用 PLA/PAHT-CF + 自装 EN02-PWR，**约 $200**（不含工具） |

## 仓库目录（main 分支）

| 路径 | 内容 |
|------|------|
| `Design/EN02-OP.STEP` | 整机 STEP（不含 EN02-PWR 板体） |
| `Design/EN02-OP-BOM.xls` | 机械 BOM |
| `Design/Parts_to_Print/` | 分件打印 STL/STEP（General / CF_Reinforced / Modify_Before_Printing） |
| `Electronics/` | EN02OP-PWR Gerber、BOM（含 5 V / 12 V 表）、原理图 PDF |
| `EN02_Assembly_Guide.pdf` | 装配说明 |
| `Pics/` | 产品图与安装示意 |

## 生态与教程

- **组装视频**：[Build EN02-OP with Kevin](https://www.youtube.com/watch?v=WY0bp_8Il1o)（Kevin Wood / kevinwoodrobotics）
- **商业授权**：GPL-3.0 适合学术/DIY；商用需联系 Westwood 获取独立许可

## 对 wiki 的映射

- 新建 [`wiki/entities/en02-op.md`](../../wiki/entities/en02-op.md)：硬件规格、制造链、与仿人五指/平行夹爪的对照。
- 交叉更新 [`wiki/overview/topic-grasp.md`](../../wiki/overview/topic-grasp.md)、[`wiki/tasks/manipulation.md`](../../wiki/tasks/manipulation.md) 的低成本开源末端入口。
