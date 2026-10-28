---
type: entity
project_id: roboparty-rp1
project: https://roboparty.com/products/RP1
code: https://github.com/Roboparty/RP1
tags: [humanoid, hardware, open-source, ros2, roboparty]
status: complete
updated: 2026-10-27
related:
  - ./roboto-origin.md
  - ./roboparty.md
  - ./humanoid-robot.md
  - ./open-source-humanoid-hardware.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/sites/roboparty_rp1_official.md
  - ../../sources/repos/roboparty_rp1.md
summary: "RP1（ROBOTO 01）是 RoboParty 面向研究与开发的人形机器人平台，官方公开硬件、机器人描述、固件、部署和训练等模块入口；各模块开放状态与许可证需按仓库核验。"
---

# RoboParty RP1（ROBOTO 01）

## 一句话定义

RP1（ROBOTO 01）是 RoboParty 发布的人形机器人开发平台，目标是把本体设计、仿真模型、底层系统、部署和学习工作流放在可检查、可扩展的开放体系中。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RP1 | RoboParty 1 / ROBOTO 01 | RoboParty 全栈人形机器人平台名称 |
| DoF | Degrees of Freedom | 产品页列出 RP1 24 DoF 与 RP1 Pro 30 DoF 配置 |
| CAD | Computer-Aided Design | 机械零件与整机设计源文件 |
| PCB | Printed Circuit Board | 电子控制板设计与生产文件 |
| BOM | Bill of Materials | 硬件装配所需物料清单 |
| URDF | Unified Robot Description Format | 机器人运动学与仿真描述格式 |
| MJCF | MuJoCo XML Format | MuJoCo 场景和模型描述格式 |
| ROS 2 | Robot Operating System 2 | 部署文档采用的机器人软件中间件 |

## 为什么重要

开源机器人硬件常见的断点是 CAD、驱动、仿真模型和策略部署分别散落，导致开发者很难确认一份策略对应哪台本体、哪组关节和哪套 I/O。RP1 的官方仓库总览把这些模块放在同一项目入口，并配套产品配置、仿真模型与初次部署文档，便于按模块检查和扩展。

## 核心结构

RP1 是一组相互衔接的仓库和产品配置，而不是单一代码仓。总览仓库（`Roboparty/RP1`）本身只有 README、许可证和图片（14 个文件，2026-10-27 核查），不含任何工程代码；每个子项目在本库有独立详情节点：

### 硬件与工具

| 子项目 | 详情节点 | 范围 |
|--------|----------|------|
| `rp1_hardware` | [RP1 Hardware](./rp1-hardware.md) | 机械源文件、电子、测试与装配 SOP |
| `ROMOMO` | [ROMOMO](./romomo.md) | 关节模组规格、图纸、文档与测试资料 |
| `rp_dexhand` | [RP Hand](./rp-hand.md) | 灵巧手产品文档、接线、配套应用、CAD 与模型，含 SDK 入口 |
| `rp1_test` | [RP1 Test](./rp1-test.md) | RP1 整机测试 |
| `roboparty_joystick` | [RoboParty Joystick](./roboparty-joystick.md) | 手柄与 App |

### 软件与系统

| 子项目 | 详情节点 | 范围 |
|--------|----------|------|
| `roboparty_deploy` | [roboparty_deploy](./roboparty-deploy.md) | RPO / RP1 部署框架与 imu / motor / inference / camera SDK |
| `RP1_description` | [RP1 Description](./rp1-description.md) | URDF / MJCF、网格与机器人配置 |
| `roboparty_train` | [roboparty_train](./roboparty-train.md) | Isaac Lab 训练工作区（RSL-RL、重定向） |
| `RP1_firmware` | [RP1 Firmware](./rp1-firmware.md) | Linux BSP、USB-CAN bridge、底层 SDK |

### Demo 与相关论文

| 子项目 | 详情节点 | 范围 |
|--------|----------|------|
| AMP | [robolab AMP](./roboparty-robolab-amp.md) | 对抗运动先验 locomotion 任务（`robolab` 仓内模块） |
| BeyondMimic | [robolab BeyondMimic](./roboparty-robolab-beyondmimic.md) | 全身动作跟踪任务（`robolab` 仓内模块） |
| Parkour | [robolab Parkour](./roboparty-robolab-parkour.md) | 感知式越障 locomotion（RP1 支持待发布） |
| MimicLite | [MimicLite](./mimiclite.md) | 通用人形运动跟踪训练 / 部署基础设施 |
| UFO | [UFO](./roboparty-ufo.md) | 无监督 RL（FB / TeCH）通用人形控制框架 |
| PBFM | [P-BFM](./paper-p-bfm.md) | 感知式行为基础模型 |
| IBFM | [I-BFM](./paper-i-bfm.md) | 奖励条件化的人形交互（搬运 / 推 / 踢箱） |
| INTACT-JEPA | [INTACT](./paper-intact.md) | JEPA 意图到动作映射（RP1 支持待发布） |
| 相关论文 | [PredActor](./paper-predactor.md)、[From Hitter to Player](./paper-from-hitter-to-player-humanoid-table-tennis.md)、[VIStream](./vistream.md) | README「Related papers」其余条目 |

### 生态入口

[Party OS](./party-os.md)、[Roboto Origin（RPO）](./roboto-origin.md)、[human-humanoid-tools](./human-humanoid-tools.md) 与 [ROMOMO](./romomo.md) 在 README 中作为 RoboParty 生态仓库展示。

## 官方规格与配置

RoboParty 产品页目前列出 1.35 m 高、24 DoF 的 RP1 与 30 DoF 的 RP1 Pro；重量约 39 kg / 41 kg（含电池和末端），10 Ah 电池标称约 3 小时运行时间，100 mm 关节模组峰值扭矩标注 160 N·m。页面也说明实际运动性能和续航受负载、地形、电池状态及软件配置影响。这些数字应视为厂商规格，不能代替统一负载和工况下的实测。

## 工程实践

官方快速开始把流程串成“模型 → 训练 → 导出 → Sim2Sim → 真机”：

1. **先核对版本和 DoF**：选择与实体机器人一致的 URDF/MJCF、执行器映射和控制配置。官方文档示例包含 24 轴真机路径，而产品页还列出 30 DoF 配置。
2. **先做仿真检查**：按文档使用 MuJoCo 打开模型，确认关节、执行器和场景，再运行训练/验证工作流。
3. **检查部署仓库状态**：[roboparty_deploy](./roboparty-deploy.md) 主 README 仍把 RP1 专用部署指南标为 Coming soon，但仓库 `docs/operation.md` 已有 `rp1_24dof` 的启动命令，推理子仓 `robots/rp1_24dof/` 带 6 份策略配置与 ONNX；28 轴 `rp1` 只有一份 `default.yaml`、无 `models/`，不能开箱运行（据 2026-10-27 仓库核查）。实施前按当前代码、配置和固件版本确认可用路径。
4. **真机分阶段验证**：首次归零和推理可能突然输出力矩。官方文档要求使用承重保护架、清空工作区并保证实体急停可触及；Sim2Sim 通过不能视为真机安全或稳定的保证。
5. **逐仓检查许可**：产品的“全栈开放”定位不意味着所有子仓、第三方代码、模型或资产共享同一许可。

官方产品页标注 24/30 DoF；GitHub 总览另外列出部分支持配置。购买、组装或移植策略前，应以目标 Release 的 BOM、模型与驱动配置为准。

## 局限与风险

- **与 RPO 区分**：[Roboto Origin](./roboto-origin.md) 是 RoboParty 的另一套开源本体项目。不能把 RPO 参数、模型或策略训练结果直接当作 RP1 的属性。
- **训练支持需要按配置核实**：训练仓总览仍以 RPO locomotion 为定位，但其 `robolab` 子模块已注册 `RP1-Flat`、`RP1-Rough`、`RP1-AMP`、`RP1-BeyondMimic`、`RP1-Interrupt` 等任务（仅见 `RP1_24DOF_CFG`，即 24 DoF）；RP1 Parkour 未注册，与 README 的 “coming soon” 一致。详见 [roboparty_train](./roboparty-train.md)。30 / 42 DoF 的模型文件在 [RP1 Description](./rp1-description.md) 中未找到，支持范围以实际文件为准。
- **官方资料同步存在差异**：RP1 文档已给出 24 轴真机快速开始，而部署仓库 README 仍把 RP1 部署文档标为 Coming soon。引用时注明所依据的文档版本。
- **规格不是独立测评**：峰值扭矩、速度、爬坡和续航为厂商页面指标，工况与持续时间会影响结果。
- **许可是多仓分立的**：RP1 总览仓库与硬件仓库的许可不同，第三方文件亦保留自己的条款。

## 关联页面

- [Roboto Origin（RPO）](./roboto-origin.md)
- [RoboParty（公司）](./roboparty.md)
- [人形机器人](./humanoid-robot.md)
- [开源人形机器人硬件方案对比](./open-source-humanoid-hardware.md)
- [Locomotion 任务](../tasks/locomotion.md)

## 参考来源

- [RoboParty RP1 官方产品与文档核查](../../sources/sites/roboparty_rp1_official.md)
- [RP1 官方仓库体系归档](../../sources/repos/roboparty_rp1.md)
- [RoboParty GitHub 组织](https://github.com/Roboparty)
- [RPO 官方聚合仓库](https://github.com/Roboparty/roboto_origin)

## 推荐继续阅读

- [RP1 官方技术文档](https://roboparty.com/documents/roboto-01)
- [RP1 GitHub 总览](https://github.com/Roboparty/RP1)
- [RP1 硬件设计文件](https://github.com/Roboparty/rp1_hardware)
