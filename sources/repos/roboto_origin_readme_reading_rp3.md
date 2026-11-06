# roboto_origin 仓库 README 通读

> 来源归档（原文通读 · 支撑 `wiki/entities/roboparty.md`）

- **标题：** ROBOTO_ORIGIN - Fully Open-Source DIY Humanoid Robot
- **类型：** repo / readme / open-source-humanoid
- **原文：** <https://github.com/Roboparty/roboto_origin>（2026-11 经 raw.githubusercontent.com 获取 main 分支 README.md 全文，约 11 KB）
- **一句话说明：** RoboParty 的 Roboto Origin 聚合仓 README：公司自述、全栈开源声明、子仓库分工、贡献规则、许可与免责声明。
- **核对范围：** 通读 README.md 全文（About Us、Documentation、Community、About This Repository、贡献表、Resource Guide、Quick Start、FAQ、行为准则、免责声明与许可）。未逐个进入 `modules/` 下各子仓库源码，也未阅读 Know-How 文档站正文、BOM 表与 LICENSES.md；这些内容页面仅引用既有归档。
- **通读：** 已完成

## 核心摘录（均出自原文）

- **公司自述：** 上海萝博派对科技有限公司，2025-02-21 成立，同年 4 月启动人形研发，四个月完成原型 ROBOTO_ORIGIN；结构、电子、训练、部署的完整研发过程开源。
- **可复现声明：** 可通过淘宝采购 + 嘉立创打样自行组装；配合开源训练与部署代码可实现行走与奔跑；后续会逐步加入该机器人上实现的运控算法，并将上线创意工坊。
- **聚合仓性质：** 每日更新的核心子仓快照聚合，仅用于集中拉取；模块级 issue / PR 应提交到各子仓，本仓只处理快照同步、根文档、链接、元数据等聚合层问题。
- **子仓分工（8 个）：** rpo_hardware（结构 / CAD / PCB / BOM）、roboparty_deploy（ROS2 部署框架，IMU / 电机 / 推理）、roboparty_train（IsaacLab 直接训练，高透明度、低重构难度，支持 Sim2Sim 到 MuJoCo）、rpo_description（URDF / MJCF）、roboparty_firmware（固件、USB2CAN、OrangePi / RDK 构建、守护进程）、rpo_appearance（仅静态展示，不建议打印）、roboparty_navigation（实验性导航定位）、roboparty_xr_teleop（XR 遥操作）。
- **技术栈徽章：** ROS2 Humble、Isaac Sim 4.5.0、Isaac Lab 2.1.1、Python 3.10、C++17、Linux；RL 链接指向 rsl_rl。
- **命名兼容：** `modules/...` 快照采用公开仓名，内部包名 / 文件名 / 路径可能仍含历史 Atom01 命名。
- **入门门槛（FAQ）：** 有 Linux 与 ROS2 基础更有帮助，文档面向初学者。
- **许可与免责：** 聚合层材料 GPL v3，`modules/` 快照与第三方材料沿用各自许可；软件输出命令作用于实体硬件，物理后果由操作者负责；二次开发风险（人身伤害、参数配置不当损坏设备、安全测试不足事故）由用户承担。

## 对 wiki 的映射

- [`wiki/entities/roboparty.md`](../../wiki/entities/roboparty.md)
- [`wiki/entities/roboto-origin.md`](../../wiki/entities/roboto-origin.md)
