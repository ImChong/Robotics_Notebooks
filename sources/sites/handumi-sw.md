# HandUMI Software 文档站

> 来源归档

- **标题：** HandUMI — Software Documentation
- **类型：** site（项目文档）
- **链接：** https://robonet-ai.github.io/handumi-sw/
- **代码：** https://github.com/robonet-ai/handumi-sw
- **硬件：** https://github.com/BrikHMP18/HandUMI
- **机构：** RoboNet AI
- **入库日期：** 2026-07-19
- **一句话说明：** HandUMI 软件官方文档：无机器人双臂示范采集、内置校准与转换前 QA、仿真/真机回放、LeRobot 兼容导出，以及向 PiPER / OpenArm / TRLC-DK1 / YAM 等平行夹爪双臂的重定向集成指南。

---

## 项目页核查（步骤 2.5）

| 核查项 | 结论 |
|--------|------|
| **GitHub 代码** | [robonet-ai/handumi-sw](https://github.com/robonet-ai/handumi-sw) — **已开源**，Apache-2.0 |
| **硬件设计** | [BrikHMP18/HandUMI](https://github.com/BrikHMP18/HandUMI) — 独立仓库 |
| **Quest 应用** | [robonet-ai/handumi-quest-app](https://github.com/robonet-ai/handumi-quest-app) |
| **文档入口** | 本站 + README Quick Start |
| **数据集/权重** | 文档未列独立 Hub 权重；强调用户自采 LeRobot 兼容数据 |

---

## 文档站核心信息（首页摘录）

- **定位：** Collect robot-free bimanual demonstrations once with HandUMI, then validate, retarget, and reuse them across different bimanual arms with parallel grippers.
- **价值主张：** 双臂 + 平行夹爪是现实世界中开始创造价值的正确具身形态；HandUMI 帮助初创公司加速部署、帮助研究人员做更多实验。
- **开发扩展：** [Add a new robot embodiment](https://robonet-ai.github.io/handumi-sw/development/new_embodiment.html) — 贡献新双臂集成

---

## 支持的双臂 embodiment（文档 / README 一致）

| 双臂平台 | 上游仓库 |
|----------|----------|
| AgileX PiPER | [agilexrobotics/piper_ros](https://github.com/agilexrobotics/piper_ros) |
| OpenArm | [enactic/openarm](https://github.com/enactic/openarm) |
| TRLC-DK1 | [robot-learning-co/trlc-dk1](https://github.com/robot-learning-co/trlc-dk1) |
| I2RT YAM | [i2rt-robotics/i2rt](https://github.com/i2rt-robotics/i2rt) |

---

## 对 wiki 的映射

- [handumi](../../wiki/entities/handumi.md)
- [handumi-sw 仓库归档](../repos/handumi-sw.md)
