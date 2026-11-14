# Sub-Millisecond Reflexes for Continuous Physical Human-Robot Interaction

> 来源归档（ingest），来自 2026-11-13「具身智能小站」18 篇论文阅读清单；以原始 arXiv 及官方项目和仓库为核查依据。

## 元数据

- **标题：** Sub-Millisecond Reflexes for Continuous Physical Human-Robot Interaction
- **中文：** 亚毫秒反射：连续人机物理交互的无锁控制切换
- **arXiv：** 2610.21913
- **版本：** 2026-11-14 检查 arXiv HTML；具体修订号待逐项补齐
- **入库日期：** 2026-11-14
- **核对范围：** 已阅读 arXiv HTML 的摘要、背景、方法、实验、结论、局限及可用附录；公开源码仅核查官方 README 与实际入口，未独立运行。
- **通读：** 已完成
- **论文：** https://arxiv.org/abs/2610.21913
- **HTML：** https://arxiv.org/html/2610.21913
- **PDF：** https://arxiv.org/pdf/2610.21913
- **仓库：** https://github.com/RoboRef/franka_reflex_engine

## 论文机制

ROS 2 兼容 multimode controller 将控制行为预备于实时循环中；危险接触、关节极限和速度阈值触发 reflex 状态机，通过无锁指针切换活跃控制器而非停机重启。

## 实验与证据边界

Franka Panda 的控制模式切换延迟低于 0.1 ms；多阶段任务执行耗时降低 3.5%，峰值 jerk 下降超过三倍。0.1 ms 是控制模式切换耗时，非接触后机械系统卸力的总时延。

## 复现与开放状态

公开 ROS 2 / Franka reflex engine，README 说明依赖 multipanda_ros2；实际部署需核对硬件/版本。

## 局限

结果针对所测 Franka 与控制软件栈；实际安全性能仍取决于传感、驱动、执行器动态与设备安全机制。

## 对 wiki 的映射

- [亚毫秒反射：连续人机物理交互的无锁控制切换](../../wiki/entities/paper-submillisecond-reflex-engine.md)

## 参考来源

- [原论文 HTML](https://arxiv.org/html/2610.21913)
- [转载阅读线索（仅作索引）](https://mp.weixin.qq.com/s/gatO-d6IxsemBSk62EvS1w)
