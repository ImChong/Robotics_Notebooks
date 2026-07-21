# reBot-DevArm

> 来源归档

- **标题：** reBot-DevArm（reBot Arm B601）
- **类型：** repo（开源桌面六轴机械臂 · 硬件 + 软件栈）
- **机构：** 矽递科技（Seeed Studio） / Seeed-Projects
- **链接：** https://github.com/Seeed-Projects/reBot-DevArm
- **文档站：** https://wiki.seeedstudio.com/robotics_page/
- **OSHWA：** https://certification.oshwa.org/cn000024.html（CN000024）
- **Stars：** ~3889（2026-07）
- **Forks：** ~380（2026-07）
- **入库日期：** 2026-07-21
- **许可证：**
  - **硬件：** CERN-OHL-W-2.0（2026-05-11 起由 CC BY-SA NC 切换；允许商业使用，弱互惠）
  - **软件 / SDK：** Apache-2.0
- **代码 / 开源状态：** **已开源（全栈）** — GitHub 仓公开 STEP/钣金/3D 打印件、BOM（细到螺丝采购链）、组装步骤与性能测试；软件侧对接 Motorbridge Python SDK、ROS 1/2、Pinocchio、LeRobot；RS 版 Isaac Sim 演示仓 [Seeed-Projects/reBot-Isaacsim](https://github.com/Seeed-Projects/reBot-Isaacsim)；Seeed Wiki 提供组装 / ROS2 / LeRobot / 视觉抓取等教程
- **一句话说明：** Seeed 面向开发者的桌面级开源六轴臂（B601-DM 达妙 / B601-RS 灵足），强调「真开源」：结构图纸 + BOM + 电机 SDK + ROS/LeRobot/Pinocchio 全链路，服务具身智能学习与桌面操作实验。
- **沉淀到 wiki：** 是 → [`wiki/entities/rebot-devarm.md`](../../wiki/entities/rebot-devarm.md)
- **交叉归档：** [rebot-devarm-seeed-wiki.md](../sites/rebot-devarm-seeed-wiki.md)

---

## 核心定位

- **问题：** 低成本开源臂（如 SO-ARM100/101）与工业臂之间，缺少 **结构全开源 + 准桌面负载 + 主流学习栈已适配** 的中间档开发臂。
- **产品线（同外观双电机）：**
  - **B601-DM**：达妙（Damiao）43 系电机，DC **24V**，负载约 **1.5 kg**，自重约 **4.5 kg**
  - **B601-RS**：灵足（Robstride）电机，DC **48V**，负载约 **2.5 kg**，自重约 **6.7 kg**
  - 二者均为 **6 DoF + 1 夹爪**，重复定位精度宣称 **< 0.2 mm**（以官方规格表为准）
- **仓内结构：** `hardware/reBot_B601_DM/`、`hardware/reBot_B601_RS/`（3D 打印件 / 金属件 / 外购件 STEP、组装步骤、性能测试）、`media/`、`community/`；根目录多语言 README。
- **软件生态（README 路线图表，2026-07 快照）：**
  | 生态 | DM | RS |
  |------|----|----|
  | Motorbridge / Python SDK | ✅ | ✅ |
  | ROS 2（运动学 / 轨迹 / 重力补偿；RS 含 MoveIt2） | ✅ | ✅ |
  | Pinocchio + MeshCat | ✅ | ✅ |
  | LeRobot | ✅ | ✅ |
  | 深度相机视觉抓取 Demo | ✅ | ✅ |
  | Isaac Sim | 🚧 进行中 | ✅（[reBot-Isaacsim](https://github.com/Seeed-Projects/reBot-Isaacsim)） |
- **遥操作：** 兼容 [Star Arm 102](https://github.com/servodevelop/Star-Arm-102) 作 Leader；套件可选成品臂或散件。
- **灵感来源（README 致谢）：** SO-ARM100、Mobile ALOHA、Dummy-Robot、OpenArm、I2RT、TRLC-DK1。

---

## 对 wiki 的映射

- 实体页：[reBot-DevArm](../../wiki/entities/rebot-devarm.md)
- 任务交叉：[Manipulation](../../wiki/tasks/manipulation.md)、[Teleoperation](../../wiki/tasks/teleoperation.md)
- 学习栈：[LeRobot](../../wiki/entities/lerobot.md)、[NVIDIA SO-101 Sim2Real 课](../../wiki/entities/nvidia-so101-sim2real-lab-workflow.md)、[Pinocchio 快速上手](../../wiki/queries/pinocchio-quick-start.md)
- 同档硬件对照：[PAROL6](../../wiki/entities/parol6-source-robotics.md)、[ALOHA](../../wiki/entities/aloha.md)、[StackForce](../../wiki/entities/stackforce.md)
