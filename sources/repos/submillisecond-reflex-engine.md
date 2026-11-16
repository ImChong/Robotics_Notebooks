# Sub-Millisecond Reflexes for Continuous Physical Human-Robot Interaction — 官方代码仓库

- **官方仓库：** https://github.com/RoboRef/franka_reflex_engine
- **论文：** https://arxiv.org/abs/2610.21913
- **项目：** 论文官方原文
- **开放结论：** 公开 ROS 2 / Franka reflex engine，README 说明依赖 multipanda_ros2；实际部署需核对硬件/版本。
- **核对范围：** 仓库首页 README、项目开放标识；真实源码运行入口和调用链仍待逐项核实
- **通读：** 未完成

## 研究对应关系

ROS 2 兼容 multimode controller 将控制行为预备于实时循环中；危险接触、关节极限和速度阈值触发 reflex 状态机，通过无锁指针切换活跃控制器而非停机重启。

## 复现说明

当前未声明已实际运行。先按仓库 README 核对模型、依赖、数据和入口，再补运行时序图。

## 对 wiki 的映射

- [论文详情](../../wiki/entities/paper-submillisecond-reflex-engine.md)
