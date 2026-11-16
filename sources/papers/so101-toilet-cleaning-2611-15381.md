# Low-Cost Mobile Bimanual Manipulation for Long-Horizon Toilet Cleaning｜低成本移动双臂马桶清洁长程操作

- **类型：** 论文
- **论文：** https://arxiv.org/abs/2611.15381（v1）
- **HTML：** https://arxiv.org/html/2611.15381v1
- **PDF：** https://arxiv.org/pdf/2611.15381
- **项目：** https://so101-turtlebot4.github.io/mobile-bimanual-cleaning/
- **核对范围：** 原始论文任务定义、核心方法、主实验与定量/定性指标、结论及其局限。部分官方网页和附录代码无法在当前接口直接访问，需另行核实；不声称执行论文中的真实实验。
- **通读：** 已完成
- **入库日期：** 2026-11-16
- **wiki：** [独立论文节点](../../wiki/entities/paper-so101-toilet-cleaning-2611-15381.md)

## 原文对应知识归纳

- **核心贡献：** 双 SO-101 与 TurtleBot4 通过 Zenoh–ROS 2 桥和 LeRobot 采集 250 段真机演示；对 π0、π0.5、SmolVLA、GR00T 分别微调，另外测试进度信号 SARM 和短记忆 MEM 对阶段切换的作用。
- **评测读法：** π0 在该任务下严格自主成功 60%、宽松完成 70%、有人短暂干预时 90%；进度/记忆增强没有稳定优于对应基线。
- **边界：** 结果集中于同一个硬件、房间和五阶段马桶清洁任务；严格成功与人工介入成功分母不同，代码和数据在项目页仍标 Coming Soon。
