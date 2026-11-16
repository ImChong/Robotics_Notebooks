# MOSAIC: Modular Obstacle and Spatial-Aware Framework for Manipulation In Cluttered Scenes｜MOSAIC：杂乱场景目标、落点与避障空间建模

- **类型：** 论文
- **论文：** https://arxiv.org/abs/2611.15937（v1）
- **HTML：** https://arxiv.org/html/2611.15937v1
- **PDF：** https://arxiv.org/pdf/2611.15937
- **项目：** https://mmahdavian.github.io/MOSAIC/
- **核对范围：** 原始论文任务定义、核心方法、主实验与定量/定性指标、结论及其局限。部分官方网页和附录代码无法在当前接口直接访问，需另行核实；不声称执行论文中的真实实验。
- **通读：** 已完成
- **入库日期：** 2026-11-16
- **wiki：** [独立论文节点](../../wiki/entities/paper-mosaic-clutter-2611-15937.md)

## 原文对应知识归纳

- **核心贡献：** VLA 骨干输出目标/落点语义 mask，冻结的三维解码器提供深度和置信度；针对目标及落点生成各 17 个几何 token，后层动作专家接收这些几何条件；障碍物使用专门的邻近注意力引导。
- **评测读法：** 主要定量结果来自 RoboPRO 杂乱操作仿真，真机部分以定性展示为主；不能当作大样本真机成功率提升。
- **边界：** 对深度置信度、语义 mask 和障碍物检测敏感，真实几何误差与动作控制延迟仍需验证；项目网页抓取未成功。
