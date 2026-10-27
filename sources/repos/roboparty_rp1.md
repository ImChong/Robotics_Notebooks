# RoboParty RP1（ROBOTO 01）

> 来源归档（ingest · 官方代码仓库）

- **标题：** RP1: An OpenSource Full-Stack Platform for Agentic Humanoids
- **类型：** repo（官方仓库总览及分模块仓库）
- **来源：** RoboParty（GitHub 组织）
- **链接：** https://github.com/Roboparty/RP1
- **官方产品页：** https://roboparty.com/products/RP1
- **技术文档：** https://roboparty.com/documents/roboto-01
- **入库日期：** 2026-10-27
- **许可证：** 总览仓库 GPL-3.0；子仓库各自维护许可证。硬件仓库声明 CERN-OHL-W-2.0，文档 CC BY-SA 4.0。
- **一句话说明：** RP1（ROBOTO 01）是 RoboParty 的开放式人形机器人平台；总览仓库导航到机械电气、关节模组、模型、固件、ROS 2 部署与训练等独立仓库。
- **沉淀到 wiki：** 是（`wiki/entities/roboparty-rp1.md`）

## 官方仓库职责

| 仓库 | 官方说明 |
|---|---|
| [RP1](https://github.com/Roboparty/RP1) | 全栈平台总览；索引各模块，具体版本、配置与贡献入口以子仓库为准 |
| [rp1_hardware](https://github.com/Roboparty/rp1_hardware) | 机械源文件、电子设计、PCB、线束、BOM、测试与装配 SOP |
| [ROMOMO](https://github.com/Roboparty/ROMOMO) | 关节模组、固件与主机端工具 |
| [RP1_description](https://github.com/Roboparty/RP1_description) | URDF、MJCF、网格与机器人配置 |
| [roboparty_deploy](https://github.com/Roboparty/roboparty_deploy) | RPO / RP1 ROS 2 部署框架、驱动与推理；README 当前将 RP1 部署指南标为 Coming soon |
| [RP1_firmware](https://github.com/Roboparty/RP1_firmware) | Linux BSP、USB-CAN bridge、底层 SDK 与软件包 |
| [roboparty_train](https://github.com/Roboparty/roboparty_train) | Isaac Lab 训练工作区；总览说明主要写 RPO locomotion policies，RP1 任务应核对实际配置 |
| [RP1_TEST](https://github.com/Roboparty/RP1_TEST) | RP1 整机测试 |
| [roboparty_joystick](https://github.com/Roboparty/roboparty_joystick) | 手柄与 Android App 控制接口 |
| [RP Hand](https://github.com/Roboparty/RP_Hand) | 灵巧手文档、接线、CAD、机器人模型与 SDK 入口 |

## 可复现入口

官方 RP1 文档提供三条起步路径：

1. **仿真查看**：安装 MuJoCo，克隆 `RP1_description`，加载 MJCF 模型。
2. **策略训练与验证**：在支持的工作站按官方训练文档配置 Isaac Lab，再用 MuJoCo 做 Sim2Sim 验证。
3. **真机运行**：文档以 24 轴 RP1、Ubuntu 22.04 + ROS 2 Humble 和游戏手柄举例；首次标定与推理需要保护架和急停。

不要把同名文件或工作区混用：RP1_description 中的 `rp1_flat.xml` 与训练仓库中的 Sim2Sim 场景可能针对不同自由度配置。

## 维护与许可边界

- 官方总览说明贡献应提交到对应子仓库，不向总览仓库提模块 PR。
- 硬件仓库 README 明确硬件设计使用 CERN-OHL-W-2.0，文档为 CC BY-SA 4.0。
- RP1 总览仓库声明 GPL-3.0；其它子仓库、Git 子模块、二进制和第三方资产须分别阅读许可说明。
- RP1 产品页所述开放范围是厂商声明。能否复现某项能力仍取决于对应机器人配置、发布文件、权重/数据可用性和许可。

## 关联来源

- [RP1 官方产品与文档核查](../sites/roboparty_rp1_official.md)
- [RoboParty RP1 开源状态快照（2026-10-15）](../sites/roboparty_rp1_opensource_status_2026-10-15.md)
- [Roboto Origin 聚合仓库](./roboto_origin.md)
