# Arrow Project Quiver 项目页

> 来源归档

- **标题：** Project Quiver — Arrow
- **类型：** site（项目页 / 平台能力与路线图）
- **来源：** Arrow Air（箭头航空）
- **链接：** https://arrowair.com/quiver
- **机构站：** https://arrowair.com/
- **入库日期：** 2026-09-05
- **一句话说明：** Quiver 官方产品页：25 kg 模块化四旋翼的能力表、PT1–Dev-kit 迭代对照、DAO 资助的 12 架 dev-kit、附件生态激励，以及 CAD / KiCad / ArduPilot 配置「全在 GitHub」。
- **代码：** https://github.com/Arrow-air/project-quiver
- **沉淀到 wiki：** [project-quiver](../../wiki/entities/project-quiver.md)

---

## 开源状态（步骤 2.5）

项目页正文写明：**All CAD files (Fusion 360 / STEP), KiCAD PCB designs, ArduPilot configurations, and assembly documentation are available in the GitHub repository.**

- **已开源：** 机架 / PCB / 文档 / 装配指南 — 见 [project-quiver.md](../repos/project-quiver.md)
- **飞控固件：** 使用上游 ArduPilot，不在本项目页另发闭源二进制
- **数据集 / 权重：** 不适用（硬件平台，无学习权重）
- **dev-kit 硬件：** DAO 资助美 8 + 德 4 架，可借出 / 出售 / 赠送；**不是**「代码已开但真机买不到」的占位页

交叉链接：本页 ↔ [project-quiver.md](../repos/project-quiver.md)

---

## 站点要点

### 定位

开源、多用途四旋翼 **开发套件**：可直接用现有附件库作业，也可当定制附件 / 功能的底座。强调可靠性、模块化、野外可维护。

### 关键能力（页面对齐 README）

- 三接口快拆载荷（底 / 左 / 右），每口一块 Attachment Interface PCB（稳压 + CAN + Ethernet）
- 双 GNSS RTK + 备份模块
- Nanoradar NRA15 雷达高度计
- 防水座舱（雨罩、滴水结构、反向坡面）
- 四块可单独更换的定制板
- 可选 Raspberry Pi 伴机
- 25–31 min 悬停

### 迭代表（项目页）

| | PT1 | PT2 | PT3 | Dev-kit（当前） |
|---|-----|-----|-----|-----------------|
| 飞控 | Pixhawk 6X | Mateksys H743 | Pix32 V6 | Pix32 V6 |
| PCB | 复用单板 | 定制主板 | 四块定制板 | 更新连接器与母排 |
| 载荷口 | 1 | 1 | 3 | 3 |
| 避障 | 仅下视雷达 | 仅下视雷达 | 仅下视雷达 | 360 LiDAR + 前 / 下雷达 |
| 伴机 | 无 | 可选 Pi | 可选 Pi + Ethernet | 同左 |
| 电源管理 | Arduino 接触器 | SSR 预充 | Battery PCB（SOC / 温 / kill） | 同左 |
| 防水 | 否 | 否 | 否 | 是 |

### 当前状态与路线

- PT3 为当前生产设计；美 / 德两地常规试飞，稳定到「开发者与企业可在其上构建」。
- 平台侧转向维护；主攻 **社区附件生态**（贷款机、竞赛、赏金）。
- 平台剩余工作：社区制造商扩产、伴机自主与避障增强。
- 协调渠道：GitHub、Arrow Discord、DAO 论坛（[dao.arrowair.com](https://dao.arrowair.com)）。

### 机构语境（arrowair.com 首页）

Arrow 自称「全球开源航空社区」：从无人机做到空中出租车。Quiver（2025–）是第一级；后续 Spearhead（货运 VTOL，2026）、Feather（载人尺度，2027）、Volley（空中出租车，2028+）**不是**本 ingest 对象。

---

## 与本库关系

| 资料 | 关系 |
|------|------|
| [project-quiver.md](../repos/project-quiver.md) | 源码 / CAD / PCB 仓 |
| [px4_autopilot.md](../repos/px4_autopilot.md) | **PX4** 飞控对照（Quiver 跑 ArduPilot） |
| [mavsdk.md](../repos/mavsdk.md) | 伴机 MAVLink API，兼容 ArduPilot |
| [cia_dronecan_uavcan.md](cia_dronecan_uavcan.md) | 双 GNSS 走 DroneCAN |
| [multirotor_uav_stack_catalog.md](../repos/multirotor_uav_stack_catalog.md) | 多旋翼栈索引 |
