# hunter_bipedal_control 官方项目页

> 官方一手项目介绍归档

- **标题：** An open source bipedal robot control framework, based on non-linear MPC and WBC
- **类型：** site / official-project-page
- **机构：** 桥介数物（BridgeDP Robotics）
- **链接：** https://bridgedp.github.io/hunter_bipedal_control/
- **代码：** https://github.com/bridgedp/hunter_bipedal_control
- **归档日期：** 2026-11-18
- **核对范围：** 已通读官方项目主页的 Hardware components、Control components、Demo video 与 Code 入口；技术参数按项目页陈述记录，未独立实测。
- **通读：** 已完成
- **对应源码归档：** [hunter_bipedal_control](../repos/hunter_bipedal_control.md)
- **沉淀到 wiki：** [hunter_bipedal_control](../../wiki/entities/cn-os-hunter-bipedal-control.md)

## 原始信息要点

- 项目面向 **EC-Hunter80-V01**，页面称机器人采用无上肢的双足结构，**10 自由度**、约 **12 kg**。
- 控制方法明确为 **MPC + WBC**，涉及定位/估计、步态规划与地面反力计算；页面未给出独立可下载的训练权重或数据集。
- 控制硬件描述包括 NUCWSKI7、EtherCAT-to-CAN 转换及 IMU 同总线接入；这是**项目指定硬件**，并非通用二足平台的必要条件。
- 页首提供 [Code](https://github.com/bridgedp/hunter_bipedal_control) 和演示视频入口，代码开放状态可由 GitHub 仓库进一步确认。

## 阅读边界

官方页主要描述目标机器人及总体技术路线；详细编译、运行方式以源码仓库 README 和 launch/C++ 实现为准。项目页中的规格与通信速率为发布方提供的说明，**本次未独立验证真机性能**。
