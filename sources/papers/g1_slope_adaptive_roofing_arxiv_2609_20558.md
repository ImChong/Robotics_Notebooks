# G1 斜坡全身作业 Locomotion（arXiv:2609.20558）

> 来源归档（paper）

- **标题：** Learning Slope-Adaptive Whole-Body Locomotion for Humanoid Robots in Roofing Construction
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2609.20558>
- **PDF：** <https://arxiv.org/pdf/2609.20558>
- **入库日期：** 2026-09-19
- **一句话说明：** 屋顶施工场景：PICO 人体演示→G1 重定向→metric 屋顶 mesh 上轨迹级支撑/作业语义优化→Isaac Lab 相位门控 clearance/非穿透 RL 跟踪；钉枪/锤击/侧推 clearance 0.256–0.531 cm，真机 MPJPE <80 mm。

## 开源状态

- **确认未开源**（步骤 2.5，2026-09-19）：无项目页/GitHub。

## 核心摘录

1. **问题：** 直接 retarget 保动作外观但脚/手相对坡面几何错误；需 scene-grounded 支撑与作业关系。
2. **参考优化：** 屋顶对齐坐标系、推断支撑区间、任务相位与手–面距离；多点支撑锚定、作业 clearance、体 mesh 非穿透、保形与平滑。
3. **RL：** Isaac Lab PPO 4096 并行、29-DoF G1；相位门控 task-clearance 与 mesh-nonpenetration 奖励。
4. **消融：** 钉枪五档 A/M/B/C/D（raw retarget → 手动 offset → 支撑 → 参考任务 → 执行感知 RL）。
5. **机构：** 佛罗里达大学（University of Florida）土木与海岸工程系；作者 Songyang Liu、Shuai Li。

**对 wiki 的映射**

- [paper-g1-slope-adaptive-roofing-locomotion](../../wiki/entities/paper-g1-slope-adaptive-roofing-locomotion.md)
- [humanoid-locomotion](../../wiki/tasks/humanoid-locomotion.md)
- [unitree-g1](../../wiki/entities/unitree-g1.md)
