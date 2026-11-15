# 第02期开源机器人索引：官方入口原始资料复核

- **类型：** site / 官方导航与技术说明
- **核对日期：** 2026-11-15
- **核对范围：** 本索引的来源发现与选型边界：Reachy2-sdk根README、InMoov Build Yours、StanfordDoggoProject根README、Pupper v3 monorepo根README、myAGV根README、Franka FCI Overview、PAROL6手册入口全文；复用下列已有TidyBot2通读归档及本轮Poppy/myCobot/Kinova核心文档。不宣称阅读所有项目的完整手册、CAD、固件或论文。
- **通读：** 已完成
- **验证范围：** 公开资料阅读与入口核查，没有购置、装配或执行任何机器人。

## 官方原始资料与索引用法

| 项目 | 原始资料 | 本页可据此确认的边界 |
|---|---|---|
| Reachy2 | [SDK README](https://github.com/pollen-robotics/reachy2-sdk) | SDK安装与例程；离线CI测试不需要gRPC连接，在线/相机测试另需环境 |
| Poppy | [本轮官方技术资料](../repos/poppy-humanoid-runtime-reading-2026-11-15.md) | 家族配置、Humanoid运行路径与软硬件分开许可 |
| InMoov | [Build Yours](https://inmoov.fr/build-yours/) | 3D打印、Arduino/MyRobotLab、分模块装配；可先做finger starter |
| Doggo/Pupper | [Doggo README](https://github.com/Nate711/StanfordDoggoProject)、[Pupper v3](https://github.com/Nate711/pupperv3-monorepo) | Doggo已声明停止支持；Pupper v3有独立系统镜像、ROS2工作区和子模块流程，两者配置不通用 |
| myCobot320 | [M5技术文档与源码](../repos/mycobot320-core-docs-reading-2026-11-15.md) | 控制器、固件、串口与SDK须配套 |
| myAGV | [README](https://github.com/elephantrobotics/myAGV)、[既有官方文档通读](../repos/elephantrobotics_myagv.md) | 麦克纳姆底盘、建图导航与机械臂组合入口；不把文档仓视为全部驱动源码 |
| TidyBot2 | [已完成主论文/硬件文档](../papers/tidybot2_arxiv_2412_10447.md)、[已完成源码归档](../repos/jimmyyhwu_tidybot2.md) | 移动操作硬件、遥操作与控制架构由原文和官方代码支撑；本轮复用通读记录 |
| Kinova Gen3 | [ROS1驱动与例程](../repos/kinova-gen3-driver-reading-2026-11-15.md) | 本体/自由度/夹爪及ROS分支，不推断整机硬件开放 |
| Franka Research3 | [FCI Overview](https://frankarobotics.github.io/docs/overview.html) | FCI经以太网提供1kHz控制与反馈；ROS/ROS2通过libfranka集成，网络和控制接口有前提 |
| PAROL6 | [官方手册入口](https://source-robotics.github.io/PAROL-docs/latest/) | 读取到SR-1002、release1.1、PythonAPI0.9.1、Revision2026.C与控制板v2.0适用范围；入口标有版本提示，不能仅凭latest路径断言所有设备兼容 |

## 跨项目归纳

“开源”需要按硬件设计、驱动/API、训练/策略和资产分别核对。协作臂驱动可用于算法实验，不代表本体CAD已开放；可打印平台仍需制造、校准与电气工程。控制器版本、消息单位和工作空间比清单标题更能决定复现成本。该索引服务发现与初筛，具体控制路径回到各实体页和官方技术文档。

## 对 wiki 的映射

- [第02期实体索引与选型步骤](../../wiki/overview/robot-open-source-wechat-issue02-curator.md)
