# Robot Navigation and Obstacle Avoidance from Randomly Positioned External Cameras｜VINE：任意外部摄像头视角下的机器人导航

- **类型：** 论文
- **原文：** https://arxiv.org/abs/2611.16282（v1）
- **PDF：** https://arxiv.org/pdf/2611.16282
- **HTML：** https://arxiv.org/html/2611.16282v1
- **代码：** https://github.com/raulsteinmetz/vine
- **核对范围：** arXiv 原文方法、系统实现边界、主要实验与作者局限；ARTS 使用 7 页原 PDF 并核对系统结构及主结果。未做模型训练或实际硬件实验。
- **通读：** 已完成
- **入库日期：** 2026-11-16
- **wiki：** [论文实体](../../wiki/entities/paper-vine-exocentric-nav-2611-16282.md)

## 原文摘录与 Wiki 映射

- **机制：** 单/多个外部 RGB 视角与摄像头/房间几何送入带 FiLM 的障碍地图网络，得到 64×64 统一占据图；按机器人位姿变换成 ego 地图，独立于视觉网络的 PPO 或 A* 做动作决策。
- **实验：** Isaac Lab 随机相机位姿布局、多个相机与近照片级场景测试；障碍地图 IoU 与导航成功率须分别汇报。
- **局限：** 依赖机器人位姿、近似摄像头位姿和房间尺寸；缺少这些输入不能声称任意布置的摄像头均无需标定。
- **对 wiki 的映射：** 独立论文详情的核心原理、评测及工程实践。
