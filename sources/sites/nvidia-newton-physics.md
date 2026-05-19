# NVIDIA Developer：Newton Physics

> 来源归档

- **标题：** Newton Physics（NVIDIA Developer）
- **类型：** site（厂商产品页）
- **来源：** NVIDIA
- **链接：** https://developer.nvidia.com/newton-physics
- **入库日期：** 2026-05-19
- **一句话说明：** NVIDIA 对 Newton 的产品叙事：开源可扩展物理引擎、Warp + OpenUSD 底座、服务机器人学习与工业接触丰富任务。
- **沉淀到 wiki：** 是 → [`wiki/entities/newton-physics.md`](../../wiki/entities/newton-physics.md)

---

## 官方要点摘录

### 定位

- 开源、可扩展物理引擎，基于 **NVIDIA Warp** 与 **OpenUSD**。
- 由 NVIDIA、Google DeepMind、Disney Research 发起，**Linux Foundation** 管理，推进机器人学习与开发。

### 物理建模范围

- 质量与动量守恒、刚体/软体动力学、接触与摩擦、执行器与传感器建模等，用于预测多体系统（含机器人）行为。

### 生态集成

| 组件 | 角色 |
|------|------|
| **OpenUSD** | 灵活数据模型与组合引擎，聚合机器人与环境描述 |
| **NVIDIA Warp** | Python 侧构建与加速仿真、空间计算 |
| **Isaac Sim / Isaac Lab** | 兼容的机器人仿真与学习框架（高保真物理、sim-to-sim） |
| **MuJoCo Warp** | GPU 优化 MuJoCo，Newton 的**关键求解器**之一 |
| **MuJoCo Playground** | 官方列出的兼容机器人学习框架 |

### 宣传特性

- **GPU 加速**：Warp 上 CUDA 级吞吐，无需手写底层 CUDA。
- **可扩展**：可插拔自定义求解器，支持丰富多物理。
- **可微**：加速训练、设计优化与系统辨识。
- **开源**：Apache 2.0，社区协作迭代。

### 延伸阅读（页内链接）

- [Newton 文档](https://newton-physics.github.io/newton/guide/overview.html)
- [Newton 1.0 博客](https://developer.nvidia.com/blog/newton-adds-contact-rich-manipulation-and-locomotion-capabilities-for-industrial-robotics)（接触丰富操作与 locomotion）
- [Isaac Lab + Newton 四足教程](https://developer.nvidia.com/blog/train-a-quadruped-locomotion-policy-and-simulate-cloth-manipulation-with-nvidia-isaac-lab-and-newton/)
- Isaac Lab 集成分支：<https://github.com/isaac-sim/IsaacLab/tree/feature/newton>

## 对 wiki 的映射

- 产品层叙事与 Isaac / Playground 对接 → [`wiki/entities/newton-physics.md`](../../wiki/entities/newton-physics.md)
- Omniverse / Isaac 底座 → [`wiki/entities/nvidia-omniverse.md`](../../wiki/entities/nvidia-omniverse.md)
