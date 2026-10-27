# RoboParty RP1 官方产品与文档

> 来源归档（ingest · 官方产品页、技术文档与开源状态）

- **日期：** 2026-10-27（核查日期）
- **机构：** 上海萝博派对科技有限公司（RoboParty）
- **类型：** site（产品页与技术文档）
- **产品页：** https://roboparty.com/products/RP1
- **技术文档：** https://roboparty.com/documents/roboto-01
- **GitHub 总览：** https://github.com/Roboparty/RP1
- **一句话说明：** RoboParty 将 ROBOTO 01（RP1）定位为全栈开放的人形机器人平台，官方页面提供产品参数、子仓库索引及从 MuJoCo 仿真到真机的入门路径。
- **源码：** https://github.com/Roboparty/RP1（总览；具体实现分布在下属仓库，许可证因仓库而异）
- **沉淀到 wiki：** 是（`wiki/entities/roboparty-rp1.md`）

## 官方资料摘录

- 产品页标注身高 1.35 m、24/30 主动自由度、约 39/41 kg（依版本），100 mm 关节模组峰值扭矩 160 N·m、10 Ah 电池和约 3 小时续航。页面同时注明运动表现与续航会随负载、地形、电池和软件配置变化；这些属于厂商规格，不是独立测评结果。
- 产品页称硬件与软件开放，列出结构图、BOM、接口、PCB、3D 模型、操作系统、电机驱动、步态算法、ROS 2、仿真及训练代码等开放范围。
- GitHub 总览把实现拆分为硬件（`rp1_hardware`）、关节模组（`ROMOMO`）、机器人描述（`RP1_description`）、部署（`roboparty_deploy`）、固件（`RP1_firmware`）、训练（`roboparty_train`）等仓库；具体配置与发布内容以各仓库当前分支/Release 为准。
- 官方入门文档提供 MuJoCo 模型浏览、Isaac Lab 策略训练、Sim2Sim 与真机运行路径；真机章节以 24 轴配置为例，并要求使用承重保护架、保持急停可触及。
- 发布说明存在需要持续核对的边界：`roboparty_deploy` README 中 RP1 部署指南仍标注 Coming soon，而技术文档已列出 RP1 24 轴真机快速开始步骤。引用时应以具体配置和当前代码为准。
- 训练仓库的官方总览描述为 RPO locomotion policy 的 Isaac Lab 工作区；仓库含 RP1 相关支持的说法应按具体任务配置验证，不能把 RPO 策略资产自动算作 RP1 可运行策略。

## 开源与许可核查

RP1 总览仓库链接到多个独立子仓库，不能仅凭“全栈开源”概括为所有文件采用同一许可证。当前可核实的例子包括：总览仓库 GPL-3.0、`rp1_hardware` 的硬件设计 CERN-OHL-W-2.0（文档 CC BY-SA 4.0）、ROMOMO 公开资料 CC BY 4.0（该仓库说明不含软件/固件源码）；其它仓库及第三方内容应逐仓核对 LICENSE / LICENSES.md。

## 官方入口

- [RP1 总览仓库](https://github.com/Roboparty/RP1)
- [RP1 硬件设计](https://github.com/Roboparty/rp1_hardware)
- [机器人描述与仿真模型](https://github.com/Roboparty/RP1_description)
- [ROS 2 部署框架](https://github.com/Roboparty/roboparty_deploy)
- [RP1 固件](https://github.com/Roboparty/RP1_firmware)
- [Isaac Lab 训练仓库](https://github.com/Roboparty/roboparty_train)

## 使用注意

以上是 RoboParty 自有产品页面、文档与代码仓库的归档。产品参数、跑步速度、坡度和续航等宣传指标不是独立实验结果；项目配置、代码支持状态和授权以对应仓库文件与版本为准。RP1 与同一公司的 RPO（Roboto Origin）是不同产品/项目，不应混用其参数、固件或训练结论。
