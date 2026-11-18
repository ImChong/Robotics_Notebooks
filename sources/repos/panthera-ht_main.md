# Panthera-HT_Main：高擎开源六轴机械臂官方总入口

- **标题：** Panthera-HT — Open-Source Six-Axis Robotic Arm
- **类型：** repo / 官方项目总入口
- **机构：** 高擎动力（HighTorque Robotics）；原始社区项目 Ragtime-LAB / Ragtime_Panthera
- **链接：** https://github.com/HighTorque-Robotics/Panthera-HT_Main
- **项目主页：** https://hightorque.cn/Panthera-HT_Hub/
- **许可证：** 主仓 README 标注 MIT；具体配套仓库的 LICENSE 须分别核查
- **入库日期：** 2026-09-06；一手核对更新：2026-11-18
- **核对范围：** 通读官方主仓 README.md（中英介绍、开源定位、能力矩阵、仓库索引、快速开始、供电与安全提示、未来规划）；与官网 Hub、Model、SDK、ROS2、LeRobot 官方资料交叉检查。
- **通读：** 已完成
- **一句话说明：** Panthera-HT 是采用行星关节模组的桌面级六轴机械臂，公开机械结构与 SDK，接入 ROS 2、LeRobot、主从遥操作和教学研究数据采集。
- **沉淀到 wiki：** [Panthera-HT 独立项目节点](../../wiki/entities/cn-os-panthera-ht-main.md)

## 关键事实（依据官方 README）

- **项目身份：** 主仓是文档和各子仓入口，**不是单独的 URDF/MJCF/USD 全量模型仓**；机械 CAD / STEP / BOM 见 Panthera-HT_Model。
- **结构：** 钣金框架、3D 打印、CNC 加工和高擎行星关节模组；官方表示可更换关节模组、修改结构和外观。
- **控制能力：** SDK 包含位置、速度、力矩、阻抗、重力与摩擦补偿、关节及笛卡尔运动能力；README 列有主从遥操与拖动示教。
- **工具链：** C++ / Python SDK、ROS 2 驱动与 MoveIt、LeRobot 数据采集与策略推理、RoboTwin 2.0 仿真采集；分别由独立仓库维护。
- **供电：** 自行组装时官方建议 24 V / 15 A 稳定供电；掉电防坠落与机械限位须按实物配置保护。
- **研发阶段区分：** 官方「未来规划」中的更多视觉任务和 Pi0 / Pi0.5 集成不是已发布、已复现的性能承诺。
- **力矩控制 ≠ 六维力传感：** 当前官方功能说明指关节力矩/阻抗控制，并未据此验证独立六维力/力矩传感器或末端力控精度。

## 开源状态与归档

| 组件 | 开放状态 | 官方入口 |
|---|---|---|
| 主项目文档及仓库导航 | 已公开 | https://github.com/HighTorque-Robotics/Panthera-HT_Main |
| 机械 CAD、STEP、BOM | 已公开 | https://github.com/HighTorque-Robotics/Panthera-HT_Model |
| Python / C++ 控制 SDK | 已公开 | https://github.com/HighTorque-Robotics/Panthera-HT_SDK |
| ROS 2 / MoveIt 适配 | 已公开 | https://github.com/HighTorque-Robotics/Panthera-HT_ROS2 |
| LeRobot 主从臂/训练适配 | 已公开 | https://github.com/HighTorque-Robotics/Panthera-HT_lerobot |
| RoboTwin 仿真适配 | 主仓列出官方仓库 | https://github.com/HighTorque-Robotics/Panthera-HT_RoboTwin |

## 对 wiki 的映射

- [Panthera-HT 项目总览](../../wiki/entities/cn-os-panthera-ht-main.md)
- [Panthera-HT SDK](../../wiki/entities/cn-os-panthera-ht-sdk.md)
- [Panthera-HT ROS 2](../../wiki/entities/cn-os-panthera-ht-ros2.md)

## 备注

本次按 2026-11-18 可读的一手 README 和源码路径编译，不声称在本机编译、驱动真机或实测力矩性能。
