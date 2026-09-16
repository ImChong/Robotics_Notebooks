# VLBiMan++: Expanding the Generalization Boundary of Vision-Language Anchored One-Shot Bimanual Manipulation

> 来源归档（ingest）

- **标题：** VLBiMan++: Expanding the Generalization Boundary of Vision-Language Anchored One-Shot Bimanual Manipulation
- **简称：** VLBiMan++
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2609.14310>
- **PDF：** <https://arxiv.org/pdf/2609.14310>
- **代码：** <https://github.com/hnuzhy/BiRoMan>
- **项目页：** <https://hnuzhy.github.io/projects/VLBiManPlus/>
- **入库日期：** 2026-09-15
- **索引来源：** [具身智能小站 9+EffVLA 盘点](../blogs/wechat_embodied_station_9_papers_resources_effvla_2026-09-15.md)
- **一句话说明：** 单次双臂示范拆技能 + VLA 锚定 + 轻量轨迹优化；覆盖铰接/可变形物体与异构双臂。

## 开源状态（步骤 2.5，2026-09-15）

**结论：已开源**

## 核心摘录

### 摘录 1

单次双臂示范拆技能 + VLA 锚定 + 轻量轨迹优化；覆盖铰接/可变形物体与异构双臂。

**对 wiki 的映射：** [paper-vlbiman-plus](../../wiki/entities/paper-vlbiman-plus.md)

### 摘录 2（官方 abstract 要点，2026-09-15 补录）

- **目标：** 可泛化双臂操作需要一个 **可复用的任务先验**，以避免大规模遥操作示范与策略重训的成本。
- **起点：** 从 **单次人类示范** 出发，做 **task-aware decomposition** 识别可复用/可适配的技能组件。
- **迁移机制：** **vision-language grounded geometric adaptation** 把技能迁到新配置，**无需重训**。
- **五个泛化维度（论文自列）：**
  1. **任务**：多样与长时序技能组合；
  2. **物体**：未见类别、几何变化、更复杂的 **铰接/可变形** 物体；
  3. **场景**：杂乱、遮挡、动态干扰；
  4. **本体**：异构双臂平台；
  5. **部署**：反复外部扰动下的 **长时闭环执行**。
- **新增机制：** **object-state-aware adaptation** + **轻量轨迹优化**，以容纳超出刚体 6-DoF 位姿变化的情形，同时保持双臂协调可靠。
- **评测形态：** **大量真机实验**；论文报告在上述逐级变难的设定下保持任务成功与适配能力（abstract 未给聚合数字）。

**对 wiki 的映射：** 同上（补入该页「核心原理（方法）」「实验与评测」「与其他工作对比」三节）

## 当前提炼状态

- [x] 项目页/仓库核查
- [x] wiki 映射
