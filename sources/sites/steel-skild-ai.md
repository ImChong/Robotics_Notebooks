# Steel — Skild AI × NVIDIA 官方技术博客（来源归档）

- **标题：** Steel: Scaling Simulation Across Robots, Tasks, and Worlds
- **来源：** https://skild.ai/blogs/steel
- **机构：** Skild AI、英伟达（NVIDIA）
- **类型：** 官方技术博客 / 求解器项目
- **官方源码：** https://github.com/newton-physics/newton/tree/main/newton/_src/solvers/steel
- **公告仓：** https://github.com/skild-ai/steel
- **WebGPU 移植：** https://github.com/skild-ai/steel-webgpu
- **版本 / 核验日期：** 2026-11-09 发布；2026-11-10 核验
- **核对范围：** 阅读官方博客介绍、Newton 集成、刚体与可变形耦合示例、异构批次实验、性能脚注和测量条件；另检查两个 GitHub README 与 Newton SolverSteel 入口及阶段代码。
- **通读：** 已完成

## 可核实的主张

1. **定位：** Steel（Scalable Throughput-optimized Engine for Embodied Learning）是集成在 Newton 的 GPU 刚体动力学求解器；支持跨不同机器人、任务和世界的异构环境批处理；不是独立 VLA 训练平台。
   - 对 wiki 的映射：[Steel 实体](../../wiki/entities/steel-newton-solver.md)。
2. **主实验：** 20 个机器人学习任务中，按两引擎各自最佳测量 batch 大小的吞吐几何均值，为 PhysX 的 **3.8 倍**，其中 19/20 任务更快。
   - 对 wiki 的映射：实体「实验与评测」。
3. **异构实验：** 混合 5、10、15、20 种任务，最高为 PhysX 峰值测量吞吐的 **13.0 倍**。针对 20 种任务、61,440 环境，相同 batch 对比为 **12.0 倍吞吐与低 62% GPU 内存**；不能将两种口径混用。
   - 对 wiki 的映射：实体「评测口径与局限」。
4. **硬件及边界：** 每次运行一张 NVIDIA RTX PRO 6000 Blackwell；数字是仿真 throughput，**不是端到端策略训练速度或现实 Sim2Real 成功率**。博客的“多样性扩展”属于研究目标，而非已经测得的泛化能力提升。
   - 对 wiki 的映射：实体「局限与风险」。

## 开源核查

- **已开源：** 主求解器位于 [newton-physics/newton](https://github.com/newton-physics/newton/tree/main/newton/_src/solvers/steel)，对外接口 `newton.solvers.SolverSteel`。
- [skild-ai/steel](https://github.com/skild-ai/steel) 是公告与入口导航，真正的求解器实现位于 Newton；许可均声明 Apache-2.0。
- **另行发布：** [steel-webgpu](https://github.com/skild-ai/steel-webgpu) 提供浏览器 WebGPU 移植，并非同一 Python CUDA 求解路径。
- **源码归档：** [Newton SolverSteel](../repos/steel-newton.md)。
