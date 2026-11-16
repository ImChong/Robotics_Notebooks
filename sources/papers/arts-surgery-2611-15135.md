# ARTS: An Open-Source, Reproducible Three-Arm Platform for Autonomy Research in Minimally Invasive Surgery｜ARTS：可重建的三机械臂微创手术研究平台

- **类型：** 论文
- **原文：** https://arxiv.org/abs/2611.15135（v1）
- **PDF：** https://arxiv.org/pdf/2611.15135
- **HTML：** 未发布，按原始 PDF v1 阅读
- **代码：** https://github.com/MRE-Lab-UMD/arts
- **核对范围：** arXiv 原文方法、系统实现边界、主要实验与作者局限；ARTS 使用 7 页原 PDF 并核对系统结构及主结果。未做模型训练或实际硬件实验。
- **通读：** 已完成
- **入库日期：** 2026-11-16
- **wiki：** [论文实体](../../wiki/entities/paper-arts-surgery-2611-15135.md)

## 原文摘录与 Wiki 映射

- **机制：** 两台 UR5e 操作 da Vinci X/Xi 器械，一台 UR5 托举双目镜；ROS 2 Jazzy/ros2_control 分臂控制，末端位姿目标进入 RCM 约束 QP 和 Ruckig 轨迹约束；FLIR 硬件时间同步并用 MCAP 录制。
- **实验：** 总物料约 137,320 美元；两台 PSM 500 Hz、ECM 125 Hz；双目相机曝光偏差 12 微秒以内、手到臂约 58 毫秒（作者台架结果）。
- **局限：** 论文测试为台架、遥操作与少量训练任务，不能证明临床安全性；无人从独立实验室完全复现安装链，部分摄像头 SDK 为专有依赖。
- **对 wiki 的映射：** 独立论文详情的核心原理、评测及工程实践。
