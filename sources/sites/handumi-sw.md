# HandUMI Software 文档站

> 来源归档

- **标题：** HandUMI — Software Documentation
- **类型：** site（项目文档）
- **链接：** https://robonet-ai.github.io/handumi-sw/
- **代码：** https://github.com/robonet-ai/handumi-sw
- **硬件：** https://github.com/robonet-ai/handumi-hw
- **Quest 应用：** https://github.com/robonet-ai/handumi-quest-app
- **机构：** RoboNet AI
- **入库日期：** 2026-07-19
- **刷新日期：** 2026-07-27
- **一句话说明：** HandUMI 软件官方文档：无机器人双臂示范采集、内置校准与转换前 QA、仿真/真机回放、LeRobot 兼容导出，以及向 PiPER / OpenArm / TRLC-DK1 / YAM 等平行夹爪双臂的重定向集成指南。

---

## 项目页核查（步骤 2.5，2026-07-27）

| 核查项 | 结论 |
|--------|------|
| **GitHub 代码** | [robonet-ai/handumi-sw](https://github.com/robonet-ai/handumi-sw) — **已开源**，Apache-2.0 |
| **硬件设计** | [robonet-ai/handumi-hw](https://github.com/robonet-ai/handumi-hw) — **已开源**，Apache-2.0（旧仓 BrikHMP18/HandUMI 301） |
| **Quest 应用** | [robonet-ai/handumi-quest-app](https://github.com/robonet-ai/handumi-quest-app) — 公开仓 + Releases APK |
| **文档入口** | 本站（Sphinx）+ README Quick Start |
| **数据集/权重** | 文档未列独立 Hub 预训练策略；强调用户自采 LeRobot 兼容数据；可选 `--push-to-hub` |

---

## 文档站核心信息

- **定位：** Collect robot-free bimanual demonstrations once with HandUMI, then validate, retarget, and reuse them across different bimanual arms with parallel grippers.
- **价值主张：** 双臂 + 平行夹爪是现实世界中开始创造价值的正确具身形态；HandUMI 帮助初创公司加速部署、帮助研究人员做更多实验。
- **能力清单（与用户宣称对齐）：**
  - 无需机器人遥操即可采集
  - 模块化：更换臂上使用的夹爪 tip 即可开录
  - LeRobot v3 兼容数据集格式
  - 内置校准 + 转换前 QA（`handumi validate` → `meta/handumi_quality.json`）
  - 仿真回放与真实遥操作（真机 backend 当前以 PiPER / OpenArm 为主）
  - Apache-2.0 完全开源（软件；硬件仓同牌照）

### 文档目录（2026-07 快照）

| 分区 | 页面 |
|------|------|
| Getting Started | Installation、Setup and Calibration |
| Core Workflows | Teleoperation、Record、Replay in Sim、Quality Assurance（datasets） |
| Physical Robots | Piper Setup、OpenArm v1 Setup |
| Help / Dev | Troubleshooting、Add a New Robot Embodiment |

---

## 支持的双臂 embodiment（README 主表）

| 双臂平台 | 上游仓库 |
|----------|----------|
| AgileX PiPER | [agilexrobotics/piper_ros](https://github.com/agilexrobotics/piper_ros) |
| OpenArm | [enactic/openarm](https://github.com/enactic/openarm) |
| TRLC-DK1 | [robot-learning-co/trlc-dk1](https://github.com/robot-learning-co/trlc-dk1) |
| I2RT YAM | [i2rt-robotics/i2rt](https://github.com/i2rt-robotics/i2rt) |

仓内另有 Axol / R1-Lite 等仿真配置；贡献新臂见 [Add a new robot embodiment](https://robonet-ai.github.io/handumi-sw/development/new_embodiment.html)。

---

## 对 wiki 的映射

- [handumi](../../wiki/entities/handumi.md)
- [handumi-sw](../repos/handumi-sw.md) · [handumi-hw](../repos/handumi-hw.md) · [handumi-quest-app](../repos/handumi-quest-app.md)
